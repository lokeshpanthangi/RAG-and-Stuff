from fastapi import APIRouter, File, UploadFile
from langchain_docling.loader import DoclingLoader
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
import tempfile


load_dotenv()


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)
index_name = "rag-test"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


prompt = PromptTemplate(
    template="Given the context: {context}, answer the question: {question} and never mention about the context , and also analyze the context Before answering and Answer to the point if there are no relavent chunks for the question, you can use Your Knowledge to answer the question",
    input_variables=["context", "question"]
)

parser = StrOutputParser()


chain = prompt | model | parser

docs_router = APIRouter()


@docs_router.post("/upload_pdf")
def add_document(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())

    loader = DoclingLoader(tmp.name)
    docs = loader.load()
    docs = text_splitter.split_documents(docs)
    
    # Clean metadata to ensure compatibility with Pinecone
    for d in docs:
        # Keep only simple metadata fields that Pinecone can handle
        cleaned_metadata = {}
        for key, value in d.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned_metadata[key] = value
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                cleaned_metadata[key] = value
            else:
                # Convert complex objects to string representation
                cleaned_metadata[key] = str(value)
        d.metadata = cleaned_metadata
        vector_store.add_documents([d])
    return {"status": "success"}


@docs_router.post("/query")
def query_document(query: str):
    results = vector_store.similarity_search(query)
    response = chain.invoke({"context": results, "question": query})
    return {"response": response}



