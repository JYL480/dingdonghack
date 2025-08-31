import os
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

from data.State import DeploymentState
from utils.main_utils import return_llm_image_cap_output, take_screenshot_ui, record_video

load_dotenv()
ADB_DEVICE = "emulator-5554"

# --- 1. MODIFY THE TOOLS TO BE SYNCHRONOUS ---
@tool
def check_element_colour(input_test: str) -> str:
    """
    Analyzes a screenshot to determine the color of a specified UI element.
    """
    print(f"--- TOOL: check_element_colour ---")
    print(f"Analyzing question: '{input_test}'")
    screenshot_path = take_screenshot_ui(device=ADB_DEVICE, app_name="example_app", step=1)
    return return_llm_image_cap_output(input_test, screenshot_path)

@tool 
def check_element_present(input_test: str) -> str:
    """
    Checks if a specific UI element is present on the screen.
    """
    print(f"--- TOOL: check_element_present ---")
    print(f"Analyzing question: '{input_test}'")
    screenshot_path = take_screenshot_ui(device=ADB_DEVICE, app_name="example_app", step=2)
    return return_llm_image_cap_output(input_test, screenshot_path)

@tool
def check_element_animation (input_test: str) -> str:
    """
    Checks if a specific UI element is animated on the screen.
    """
    print(f"--- TOOL: check_element_animation ---")
    print(f"Analyzing question: '{input_test}'")

    video_path = record_video(device=ADB_DEVICE, app_name="example_app", duration=5, step=3)


# --- 2. CREATE THE AGENT EXECUTOR ONCE ---
# This is efficient. It's created when the module is loaded and reused.
tools = [check_element_colour, check_element_present]
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # Use 0 for deterministic tool use
    openai_api_key=os.getenv("CHATGPT_KEY"),
)
agent_executor = create_react_agent(
    model=model,
    tools=tools,
    prompt="You are a UI tester agent."
)


# --- 3. DEFINE THE LANGGRAPH NODE --- 
# This function is now a proper, dynamic, and synchronous node.

def ui_test_agent_node(state: DeploymentState) -> DeploymentState:
    """
    A LangGraph node that uses a ReAct agent to answer a question about the UI.
    """
    print("--- NODE: UI Test Agent ---")
    
    # Get the task dynamically from the current state
    task = state.get("task")
    if not task:
        raise ValueError("Task is missing from the state.")

    try:
        # Prepare the input for the agent
        agent_input = {"messages": [HumanMessage(content=task)]}
        
        # Synchronously invoke the agent
        response = agent_executor.invoke(agent_input)
        
        # Get the final answer
        final_answer = response['messages'][-1].content
        print(f"  ↪️  Agent's final answer: {final_answer}")
        for message in response['messages']:
            message.pretty_print()

        # Return the ENTIRE state, but with our new information added.
        return {
            **state,
            "messages": response['messages'],
            "agent_response": final_answer,  # Store the clean answer
            "execution_status": "success",
            "completed": True,
        }
    except Exception as e:
        print(f"--- ERROR in UI Test Agent Node: {e} ---")
        return {
            **state,
            "agent_response": f"An error occurred: {e}",
            "execution_status": "error",
            "completed": True,  # End the flow on error
        }


if __name__ == "__main__":
    # pass
    # Define a blocking function to run the agent.
    def run_agent():
        print("--- Invoking Agent ---")
        
        # The agent uses a synchronous tool now, so we can use 'invoke' directly.
        response = agent_executor.invoke({
            "messages": [HumanMessage(content="Is the tiktok app present?")]
        })
        
        for message in response['messages']:
            message.pretty_print()

    # Run the agent synchronously.
    run_agent()
