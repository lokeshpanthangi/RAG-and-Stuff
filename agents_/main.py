# backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver # <--- Added for Memory
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Specific model is usually safer

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    # Use 'add_messages' so the bot remembers history (appends instead of overwrites)
    messages: Annotated[list, add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# Logic: If tool call -> tools, else -> END
graph.add_conditional_edges("chat_node", tools_condition) 
graph.add_edge('tools', 'chat_node')

# We need memory to persist the conversation across the while loop
memory = MemorySaver()

# Compile with checkpointer
chatbot = graph.compile(checkpointer=memory)

# -------------------
# 8. Test
# -------------------

# Create a thread ID to track this specific conversation history
config = {"configurable": {"thread_id": "1"}}

while True:
    query = input("User: ")
    if query.lower() == "exit":
        break
    
    # Send a HumanMessage object, not just a string
    input_message = HumanMessage(content=query)
    
    # Invoke with config (thread_id) so it remembers previous context
    result = chatbot.invoke({"messages": [input_message]}, config=config)
    
    # Get the last message from the AI
    last_msg = result["messages"][-1]
    print(f"AI: {last_msg.content}")