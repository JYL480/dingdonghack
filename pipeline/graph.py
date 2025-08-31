from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import Literal, List, Annotated, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from IPython.display import display, Image

# --- Mock implementations for demonstration purposes (now synchronous) ---
def return_llm_output(user_input: str) -> str:
    # A simple mock LLM response
    if "green" in user_input.lower():
        return "YES"
    return "NO"

# This mock will return "red" on the first call and "green" on the second
call_count = 0
def return_llm_image_cap_output(user_input: str, image_path: str):
    global call_count
    print(f"Mock VLM processing image at {image_path} with prompt: {user_input}")
    if call_count == 0:
        call_count += 1
        return "The button appears to be red with a medium confidence score."
    else:
        return "The button appears to be green with a high confidence score."

# 1. Define the state using Pydantic's BaseModel
class State(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    image_path: Optional[str] = None
    question: Optional[str] = None
    text: Optional[str] = None
    decision: Optional[Literal["YES", "NO"]] = None

# 2. Define the nodes (now synchronous)
def capture_agent(state: State) -> dict:
    """Simulates screen capture and adds an initial message."""
    print("---EXECUTING CAPTURE AGENT---")
    return {
        "messages": [AIMessage("Starting the process. Screen captured.")],
        "image_path": "/path/to/simulated/image.png",
        "question": "Is the button green?"
    }

def vlm_agent(state: State) -> dict:
    """Processes the image and question using the VLM model."""
    print("---EXECUTING VLM AGENT---")
    image_path = state.image_path
    question = state.question
    vlm_response = return_llm_image_cap_output(question, image_path)
    return {
        "messages": [AIMessage("VLM analysis complete.")],
        "text": vlm_response
    }

def verification_agent(state: State) -> dict:
    """Verifies the VLM's output using another LLM."""
    print("---EXECUTING VERIFICATION AGENT---")
    text_to_verify = state.text
    decision_response = return_llm_output(f"Analyze this text and tell me if it indicates a button is blue. Just answer YES or NO. Text: {text_to_verify}")
    print(f"LLM decision: {decision_response}")
    return {
        "messages": [AIMessage("Verification complete.")],
        "decision": decision_response
    }

def back_track(state: State) -> dict:
    """Backtrack to the VLM agent for re-evaluation."""
    print("---BACKTRACKING TO VLM AGENT---")
    return {
        "messages": [AIMessage("Backtracking to VLM agent for re-evaluation.")],
        "text": None  # Clear previous text to force re-evaluation
    }

# 3. Define the router function
def verification_router(state: State) ->str:
    """Inspects the 'decision' and routes accordingly."""
    print("---ROUTING BASED ON VERIFICATION---")
    if state.decision == "NO":
        print("Decision is NO. Routing to backtrack.")
        return "back_track"
    else:
        print("Decision is YES. Routing to end.")
        return "end"

# --- Graph Definition ---
builder = StateGraph(State)

# Add the nodes
builder.add_node("capture_agent", capture_agent)
builder.add_node("vlm_agent", vlm_agent)
builder.add_node("verification_agent", verification_agent)
builder.add_node("back_track", back_track)

# Set a single entry point
builder.set_entry_point("capture_agent")

# Define the flow
builder.add_edge("capture_agent", "vlm_agent")
builder.add_edge("vlm_agent", "verification_agent")

# Add the conditional edge after the verification node
builder.add_conditional_edges(
    "verification_agent",
    verification_router,
    {
        "end": END,
        "back_track": "back_track"
    }
)

# Connect the backtrack node back to the VLM agent to form a loop
builder.add_edge("back_track", "vlm_agent")

# Compile the graph
graph = builder.compile()

# --- Graph Invocation and Display (now synchronous) ---
def main():
    print("Invoking graph...")
    # Call the synchronous invoke method
    final_state = graph.invoke({})
    
    print("\n---FINAL STATE---")
    print(final_state)
    
    try:
        # Display the graph diagram after the synchronous invocation
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"Error displaying graph: {e}")


if __name__ == "__main__":
    main()