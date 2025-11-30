from fastapi import APIRouter, File, UploadFile
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_query import embeddings
import fitz  # PyMuPDF
import tempfile
from dotenv import load_dotenv
import os

load_dotenv()

upload = APIRouter()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

# @upload.post("/upload/pdf")
# async def upload_pdf(file: UploadFile = File(...)):
#     # Step 1: Save uploaded file to temporary storage
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         temp_path = tmp_file.name
#         content = await file.read()
#         tmp_file.write(content)

#     # Step 2: Load the saved PDF file with PyMuPDF
#     pdf_document = fitz.open(temp_path)

#     # Step 3: Extract all text
#     full_text = ""
#     for page_num in range(len(pdf_document)):
#         page = pdf_document.load_page(page_num)
#         full_text += page.get_text()
#     pdf_document.close()

#     # Step 4: Split text into chunks
    
#     texts = text_splitter.split_text(full_text)

#     # Step 6: Add directly to Qdrant (assuming your Qdrant wrapper takes texts + vectors)
#     vector_store = QdrantVectorStore.from_documents(
#         documents=texts,
#         url="http://localhost:6333",  # URL of the Qdrant server
#         collection_name="advrag-c",  # name of the collection in Qdrant
#         embedding=embeddings,
#     )

#     return {"status": "success", "message": f"Uploaded and processed {file.filename}"}
