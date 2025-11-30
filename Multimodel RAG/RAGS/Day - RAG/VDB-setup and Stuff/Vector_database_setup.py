from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os


load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
embeddings = OpenAIEmbeddings()

#chromadb Setup
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

#Pinecone Setup
index = pc.Index("wisechoice")
vector_store_pinecone = PineconeVectorStore(index=index, embedding=embeddings)


