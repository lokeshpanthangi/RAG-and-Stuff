from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
vision_model = ChatOpenAI(model_name="gpt-4o-vision-preview", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)


# Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "ragtest"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

parser = StrOutputParser()