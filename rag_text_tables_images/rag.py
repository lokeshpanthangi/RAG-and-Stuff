from config import model, text_splitter, vector_store, vision_model
from langchain_core.documents import Document
from fastapi import File, UploadFile
import pandas as pd
import fitz # PyMuPDF
import base64
from PIL import Image
import io




def analyze_image(image_bytes: bytes):

    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    msg = [
        (
            "user",
            [
                {"type": "text", "text": "Describe this image in detail for retrieval purposes. Focus on charts, graphs, or text inside the image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        )
    ]
    response = vision_model.invoke(msg)
    return {"base64": base64_image, "summary": response.content}



def upload_file_multimodal(file: UploadFile = File(...)):
    full_text_content = ""
    documents_to_add = []
    
    # We keep a running buffer of the text from the CURRENT page 
    # to provide "overlap" context for images found on that page.
    page_text_buffer = "" 

    file_bytes = file.file.read()

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            page_num = page.number + 1
            
            # --- 1. EXTRACT TEXT FIRST (To create context for images) ---
            # We extract text first so we can give the image some context of what was written above it
            raw_page_text = page.get_text()
            full_text_content += raw_page_text
            
            # Update local buffer for this page
            page_text_buffer = raw_page_text

            # --- 2. EXTRACT TABLES (Your existing logic) ---
            tabs = page.find_tables()
            if tabs.tables:
                for tab in tabs:
                    table_data = tab.extract()
                    if table_data and len(table_data) >= 2:
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        rows_per_chunk = 10
                        for i in range(0, len(df), rows_per_chunk):
                            chunk_df = df.iloc[i : i + rows_per_chunk]
                            markdown_chunk = chunk_df.to_markdown(index=False)
                            table_context = f"Table from Page {page_num}, Rows {i} to {i+len(chunk_df)}"
                            final_text = f"{table_context}\n{markdown_chunk}"
                            doc = Document(
                                page_content=final_text, 
                                metadata={"source": file.filename, "type": "table", "page": page_num}
                            )
                            documents_to_add.append(doc)

            # --- 3. EXTRACT IMAGES ---
            image_list = page.get_images(full=True)
            
            if image_list:
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # FILTER: Ignore tiny images (icons, lines, logos)
                    # We load it into PIL just to check dimensions
                    try:
                        pil_img = Image.open(io.BytesIO(image_bytes))
                        width, height = pil_img.size
                        
                        # Only process if image is larger than 100x100 pixels
                        if width > 100 and height > 100:
                            
                            # ANALYZE
                            analysis_result = analyze_image(image_bytes)
                            summary = analysis_result["summary"]
                            
                            # CREATE CONTEXT (Overlap)
                            # Grab the last 200 chars of text from the page buffer to put before the image summary
                            prev_context = page_text_buffer[-200:].replace("\n", " ")
                            
                            final_image_text = (
                                f"Context from Page {page_num} text: ...{prev_context}...\n"
                                f"[IMAGE SUMMARY START]\n{summary}\n[IMAGE SUMMARY END]"
                            )

                            # CREATE DOCUMENT
                            img_doc = Document(
                                page_content=final_image_text,
                                metadata={
                                    "source": file.filename,
                                    "type": "image",
                                    "page": page_num,
                                    "image_index": img_index
                                    # You can store 'base64_data': analysis_result["base64"] here if needed, 
                                    # but it will make metadata huge.
                                }
                            )
                            documents_to_add.append(img_doc)
                            print(f"Processed Image on page {page_num}")
                            
                    except Exception as e:
                        print(f"Skipping image on page {page_num}: {e}")

    # --- 4. PROCESS REGULAR TEXT ---
    text_chunks = text_splitter.split_text(full_text_content)
    text_documents = [
        Document(page_content=t, metadata={"source": file.filename, "type": "text"}) 
        for t in text_chunks
    ]
    
    # Combine everything
    all_documents = documents_to_add + text_documents
    
    # Upload
    if all_documents:
        vector_store.add_documents(all_documents)
    
    file.file.close()
    return {
        "filename": file.filename, 
        "status": "uploaded", 
        "total_chunks": len(all_documents),
        "images_found": len(documents_to_add) # Note: this includes tables in this count variable, adjust as needed
    }