from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic_schemas import payloadModel

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
# -----------------
def orchestrator(state: dict) -> dict:
    """Decide whether to use RAG or not"""
    decision_prompt = f"""
    You are an orchestrator. 
    If the question needs external knowledge (facts, definitions, data), return "rag" the RAG contains data about the USER and Machine Learning the user name is Lokesh.
    If it's simple, conversational, or opinion-based, return "direct".
    
    Question: {state["question"]}
    """
    response = model.invoke(decision_prompt).content
    if "rag" in response:
        state["route"] = "rag"
    else:
        state["route"] = "direct"
    return state

def retrieve(state: dict) -> dict:
    docs = vectorstore.similarity_search(state["question"], k=3)
    context = "\n\n".join([d.page_content for d in docs])
    return {**state, "context": context}

def rag_generate(state: dict) -> dict:
    chain = prompt | model | parser
    result = chain.invoke({"context": state["context"], "question": state["question"]})
    return {**state, "messages": [result]}

def direct_generate(state: dict) -> dict:
    result = model.invoke(state["question"]).content
    return {**state, "messages": [result]}

# -----------------
# Graph
# -----------------
graph = StateGraph(dict)

graph.add_node("orchestrator", orchestrator)
graph.add_node("retrieve", retrieve)
graph.add_node("rag_generate", rag_generate)
graph.add_node("direct_generate", direct_generate)

# Entry point
graph.add_edge(START, "orchestrator")

# Branching
graph.add_conditional_edges(
    "orchestrator",
    lambda state: state["route"],  # routing function
    {
        "rag": "retrieve",
        "direct": "direct_generate",
    },
)

# RAG flow
graph.add_edge("retrieve", "rag_generate")
graph.add_edge("rag_generate", END)

# Direct flow
graph.add_edge("direct_generate", END)

rag_graph = graph.compile()

# -----------------
# FastAPI route
# -----------------
@app.post("/chat")
def chat(payload: payloadModel):
    final_state = rag_graph.invoke(payload.dict())
    return {"response": final_state["messages"][-1], "route": final_state["route"]}
