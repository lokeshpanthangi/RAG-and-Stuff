from langchain_openai import ChatOpenAI, embeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from fastapi import File, UploadFile
import fitz
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
import os


load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Specific model is usually safer`
embed_model = embeddings.OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# pinecone 
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("ragtest")
vector_store = PineconeVectorStore(index=index, embedding=embed_model)

#prompt 

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question.
    
    Context: {context}
    
    Question: {question}
    
    Answer in a concise manner.
    """
)


parser = StrOutputParser()

chain = prompt | model | parser


def get_response(query):
    docs = vector_store.similarity_search(query, k=3)
    result = chain.invoke({"context": docs, "question": query})
    return result


def upload_file(file: UploadFile = File(...)):
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk,metadata={"filename": file.filename,"page": i}) for i, chunk in enumerate(chunks)]
        vector_store.add_documents(documents)
    return {"message": f"Uploaded and processed {file.filename}"}