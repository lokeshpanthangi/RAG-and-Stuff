from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI

app = FastAPI()

model = ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    messages: Annotated[list[str], add_messages]

@app.post("/chat", response_model=State)
def chatbot(state: State):
    result = model.invoke(state["messages"]).content
    state["messages"].append(result)
    return {"messages": state["messages"]}



graph = StateGraph(State)

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
graph_1 = graph.compile()



