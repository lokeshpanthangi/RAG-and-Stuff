from fastapi import APIRouter
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1536)
model = ChatOpenAI(model="gpt-4", temperature=0.3)


QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "advrag-dv")


qdrant = QdrantVectorStore.from_existing_collection(
    collection_name=QDRANT_COLLECTION,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    embedding=embeddings,
    vector_name="advrag-dv",
)


prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""You are an AI assistant that helps users by providing information based on the context provided.
If the context does not contain relevant information, politely inform the user that you don't have the answer
here is the context:
{context}
and this is the user query:
{query}"""
)

parser = StrOutputParser()

chain = prompt | model | parser

query = APIRouter()


@query.get("/query/health")
def health_check():
    return {"status": "healthy"}


@query.post("/query/vector-query")
def vector_query(query  : str):
    docs = qdrant.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    response = chain.invoke({"context": context, "query": query})
    return {"response": response}

