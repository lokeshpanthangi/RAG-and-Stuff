from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from pydantic_schemas import payloadModel
from langchain_core.output_parsers import StrOutputParser

# Setup
load_dotenv()
app = FastAPI()

model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("startnew")
vectorstore = PineconeVectorStore(index, embeddings)

prompt = PromptTemplate(
    template="Given the following context: {context}\n\nAnswer the question: {question}",
    input_variables=["context", "question"],
)
parser = StrOutputParser()

# -----------------
# Nodes
# ----------------    

def retrieve(state: dict):
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {**state, "context": docs}

def generate(state: dict):
    chain = prompt | model | parser
    result = chain.invoke({"context": state["context"], "question": state["question"]})
    return {**state, "messages": [result]}

# -----------------
# Graph
# -----------------
graph = StateGraph(dict)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

rag_graph = graph.compile()

# -----------------
# FastAPI route
# -----------------
@app.post("/chat")
def chat(payload: payloadModel):
    final_state = rag_graph.invoke(payload.dict())
    return {"response": final_state["messages"][-1]}
