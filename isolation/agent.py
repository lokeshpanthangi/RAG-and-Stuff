from rag import get_response
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()


memory = MemorySaver()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


class State(TypedDict):
    messages: Annotated[list, add_messages]



@tool
def search_vector_store(query: str) -> str:
    """
    Search the vector store using the provided query.
    """
    return get_response(query)


search_tool = DuckDuckGoSearchRun(region="us-en")


tools = [search_tool, search_vector_store]

llm_with_tools = model.bind_tools(tools)



#nodes 

def chat_node(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    state["messages"].append(response)
    return state


tool_node = ToolNode(tools)



graph = StateGraph(State)

graph.add_node("chat_node",chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools", "chat_node")

app = graph.compile(checkpointer=memory)


config = {"configurable" : {"thread_id":"1"}}

while True:
    user_input = input("User: ")
    result = app.invoke({"messages": [user_input]},config=config)
    print("Bot:", result["messages"][-1].content)