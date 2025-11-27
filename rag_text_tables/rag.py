from config import text_splitter, vector_store
from fastapi import File, UploadFile
import fitz # PyMuPDF
import pandas as pd
from langchain_core.documents import Document




def upload_file2(file: UploadFile = File(...)):
    documents_to_add = []
    text = ""
    with fitz.open(stream=file.file.read(), filetype="pdf") as pdf:
        for page in pdf:

            tabs = page.find_tables()
            
            if tabs.tables:
                for tab in tabs:
                    table_data = tab.extract()

                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    
                    rows_per_chunk = 10
                    
                    for i in range(0, len(df), rows_per_chunk):

                        chunk_df = df.iloc[i : i + rows_per_chunk]
                        
                        markdown_chunk = chunk_df.to_markdown(index=False)
                        
                        table_context = f"Table from Page {page.number}, Rows {i} to {i+len(chunk_df)}"
                        final_text = f"{table_context}\n{markdown_chunk}"
                        
                        doc = Document(page_content=final_text, metadata={"source": file.filename, "type": "table", "page": page.number})
                        documents_to_add.append(doc)

            text += page.get_text()

    text_chunks = text_splitter.split_text(text)
    print(text_chunks)
    text_documents = [Document(page_content=t, metadata={"source": file.filename, "type": "text"}) for t in text_chunks]
    all_documents = documents_to_add + text_documents
    vector_store.add_documents(all_documents)
    
    file.file.close()
    return {"filename": file.filename, "status": "uploaded", "chunks_created": len(all_documents)}