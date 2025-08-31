# pipeline/ui_test_agent.py

import asyncio
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
from utils.main_utils import return_llm_image_cap_output, take_screenshot_ui

load_dotenv()
ADB_DEVICE = "emulator-5554"

# def pre_process_in;put(state: DeploymentState) -> Dict[str, Any]:
#     """
#     Pre-process the input state to extract necessary information for the agent.
#     """
#     task = state.get("task", "No task provided")
#     return {"messages": [HumanMessage(content=task)]}

@tool
async def check_element_colour(input_test: str) -> str:
    """
    Analyzes a screenshot to determine the color of a specified UI element.
    """
    print(f"--- TOOL: check_element_colour ---")
    print(f"Analyzing question: '{input_test}'")
    screenshot_path = await asyncio.to_thread(
        take_screenshot_ui, device=ADB_DEVICE, app_name="example_app", step=1
    )
    return await return_llm_image_cap_output(input_test, screenshot_path)


@tool 
async def check_element_present(input_test: str) -> str:
    """
    Checks if a specific UI element is present on the screen.
    """
    print(f"--- TOOL: check_element_present ---")
    print(f"Analyzing question: '{input_test}'")
    screenshot_path = await asyncio.to_thread(
        take_screenshot_ui, device=ADB_DEVICE, app_name="example_app", step=2
    )
    return await return_llm_image_cap_output(input_test, screenshot_path)


# --- 2. CREATE THE AGENT EXECUTOR ONCE ---
# This is efficient. It's created when the module is loaded and reused.
tools = [check_element_colour, check_element_present]
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0, # Use 0 for deterministic tool use
    openai_api_key=os.getenv("CHATGPT_KEY"),
)
agent_executor = create_react_agent(
    model=model,
    tools=tools,
    prompt="You are a UI tester agent."
)


# --- 3. DEFINE THE LANGGRAPH NODE ---
# This function is now a proper, dynamic, and asynchronous node.

async def ui_test_agent_node(state: DeploymentState) -> DeploymentState:
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
        
        # Asynchronously invoke the agent (this is non-blocking!)
        response = await agent_executor.ainvoke(agent_input)
        
        # Get the final answer
        final_answer = response['messages'][-1].content
        print(f"  ↪️  Agent's final answer: {final_answer}")
        for message in response['messages']:
            message.pretty_print()

        # Return the ENTIRE state, but with our new information added.
        return {
            **state,
            "messages": response['messages'],
            "agent_response": final_answer, # Store the clean answer
            "execution_status": "success",
            "completed": True,
        }
    except Exception as e:
        print(f"--- ERROR in UI Test Agent Node: {e} ---")
        return {
            **state,
            "agent_response": f"An error occurred: {e}",
            "execution_status": "error",
            "completed": True, # End the flow on error
        }


if __name__ == "__main__":
    # pass
    # Define an async function to run the agent.
    async def run_agent():
        print("--- Invoking Agent ---")
        
        # The agent uses an async tool, so we must use 'ainvoke'.
        # The input is a dictionary, with the user's message under the key "messages".
        response = await agent_executor.ainvoke({
            "messages": [HumanMessage(content="Is the tiktok app present?")]
        })
        
        for message in response['messages']:
            message.pretty_print()

    # Use asyncio.run() to start the asynchronous agent execution.
    asyncio.run(run_agent())

