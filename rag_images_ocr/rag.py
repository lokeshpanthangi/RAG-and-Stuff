from fastapi import FastAPI, File, UploadFile
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
import os
import base64
import fitz
import io
from PIL import Image

load_dotenv()

app = FastAPI()

parser = StrOutputParser()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
)   


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)
vision_model = ChatOpenAI(model_name="gpt-4o", temperature=0)


pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc = pinecone.Index("blurgs")
vector_store = PineconeVectorStore(index=pc, embedding=embeddings)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 

chain = prompt | llm | parser


@app.get("/answer")
async def answer_question(question: str):
    docs = vector_store.similarity_search(question, k=3)
    response = chain.invoke({
        "context": docs,
        "question": question
    })
    return response


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type == "application/pdf":
        contents = await file.read()
        pdf_document = fitz.open(stream=contents, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        vector_store.add_texts([text])
        return {"filename": file.filename, "status": "PDF processed and text added to vector store."}
    
    elif file.content_type.startswith("image/"):
        contents = await file.read()
        image = Image.open(file.file)
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        

        message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}
        human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Give me a exact detailed MD of the image content."},
    {"type": "image", "base64": img_str, "mime_type": file.content_type},
])
        response = vision_model.invoke([human_message])
        print(response.content)
        vector_store.add_texts([response.content])
        return {"filename": file.filename, "status": "Image processed and description added to vector store."}