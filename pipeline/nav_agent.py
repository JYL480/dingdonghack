from data.State import DeploymentState, ElementMatch
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
import os
import base64
import datetime
from time import sleep
import subprocess
from ppadb.client import Client as AdbClient
from dotenv import load_dotenv
from tool.coordinate_converter import convert_element_to_click_coordinates, get_element_bounds
from tool.click_visualizer import draw_click_marker, save_action_visualization
from tool.img_tool import *
from tool.screen_content import *
from typing import Dict, Any
# Load environment variables
load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("CHATGPT_KEY"),
    streaming=True
)

def capture_screen_node(state: DeploymentState) -> DeploymentState:
    # Silently capture screen without verbose logging
    task_folder = state.get("task_folder")

    state_dict = dict(state)
    # Make sure task_folder is in the dict
    if task_folder:
        state_dict["task_folder"] = task_folder
        
    updated_state = capture_and_parse_screen(state_dict)

    # Update state
    for key, value in updated_state.items():
        if key in state:
            state[key] = value
    
    # Ensure task_folder is preserved
    if task_folder:
        state["task_folder"] = task_folder

    if not state["current_page"]["screenshot"]:
        state["should_fallback"] = True
        print("❌ Unable to capture screen")

    return state

def capture_and_parse_screen(state: DeploymentState) -> DeploymentState:
    """
    Capture current screen and parse elements, update state

    Args:
        state: Deployment state

    Returns:
        Updated deployment state
    """
    try:
        # Use task folder if available, otherwise use "deployment"
        app_name = state.get("task_folder", "deployment")
        
        # 1. Take screenshot
        screenshot_path = take_screenshot.invoke(
            {
                "device": state["device"],
                "app_name": app_name,
                "step": state["current_step"],
            }
        )

        if not screenshot_path or not os.path.exists(screenshot_path):
            print("❌ Screenshot failed")
            return state

        # 2. Parse screen elements
        screen_result = screen_element.invoke({"image_path": screenshot_path})

        if "error" in screen_result:
            print(f"❌ Screen element parsing failed: {screen_result['error']}")
            return state

        # 3. Update current page information
        state["current_page"]["screenshot"] = screenshot_path
        state["current_page"]["elements_json"] = screen_result[
            "parsed_content_json_path"
        ]

        # 4. Load element data
        with open(
            screen_result["parsed_content_json_path"], "r", encoding="utf-8"
        ) as f:
            state["current_page"]["elements_data"] = json.load(f)

        # Silent - already shown in console
        return state

    except Exception as e:
        print(f"❌ Error capturing and parsing screen: {str(e)}")
        return state

def ui_test_agent(state: DeploymentState) -> DeploymentState:
    """
    Fall back to React mode when template execution fails

    Args:
        state: Execution state

    Returns:
        Updated execution state
    """
    import re
    import time
    
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 20)
    print(f"\n📍 Step {current_step + 1}/{max_steps}: Executing action...")
    
    task = state["task"]
    
    # Add a small delay to let the screen stabilize after previous action
    time.sleep(0.5)

    # Import smart_screen_action for accurate clicking
    from tool.screen_content import smart_screen_action
    


    # Create action_agent for page operation decisions with smart_screen_action
    action_agent = create_react_agent(
        model=model,
        tools=[smart_screen_action],
        prompt="You are a UI navigation agent."
    )

    # Initialize React mode
    if not state["messages"]:
        # Set system prompt
        system_message = SystemMessage(
            content="""You are an intelligent smartphone operation assistant who will help users complete tasks on mobile devices.
You can help users by observing the screen and performing various operations (clicking, typing text, swiping, etc.).
Analyze the current screen content, determine the best next action, and use the appropriate tools to execute it.
Each step of the operation should move toward completing the user's goal task."""
        )

        state["messages"].append(system_message)

        # Add user task
        user_message = HumanMessage(
            content=f"I need to complete the following task on a mobile device: {task}"
        )
        state["messages"].append(user_message)

    # Capture current screen
    state = capture_and_parse_screen(state)
    if not state["current_page"]["screenshot"]:
        state["execution_status"] = "error"
        print("Unable to capture or parse screen")
        return state

    # Prepare screen information
    screenshot_path = state["current_page"]["screenshot"]
    elements_json_path = state["current_page"]["elements_json"]
    device = state["device"]
    device_size = get_device_size.invoke(device)

    # Find the LABELED image, not the raw screenshot!
    labeled_image_path = None
    if screenshot_path:
        import os
        processed_dir = os.path.join(os.path.dirname(screenshot_path), "processed_images")
        if os.path.exists(processed_dir):
            base_name = os.path.basename(screenshot_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            labeled_path = os.path.join(processed_dir, f"labeled_{base_name}")
            if os.path.exists(labeled_path):
                labeled_image_path = labeled_path
            else:
                labeled_image_path = screenshot_path
    
    if not labeled_image_path:
        labeled_image_path = screenshot_path
    
    # Load the LABELED image as base64
    with open(labeled_image_path, "rb") as f:
        image_data = f.read()
        image_data_base64 = base64.b64encode(image_data).decode("utf-8")

    # Load element JSON data
    with open(elements_json_path, "r", encoding="utf-8") as f:
        elements_data = json.load(f)

    elements_text = json.dumps(elements_data, ensure_ascii=False, indent=2)

    # Build messages
    messages = [
        SystemMessage(
            content=f"""Below is the current page information and user intent. Please analyze comprehensively and recommend the next reasonable action (please complete only one step),
and complete it by calling the smart_screen_action tool.

IMPORTANT: Use the smart_screen_action tool with the element_id parameter for accurate clicking. The element_id corresponds to the index in the JSON array (0-based).
Example: To click element at index 5, use: smart_screen_action(device="{device}", action="tap", element_id=5, json_path="{elements_json_path}")
For text input: smart_screen_action(device="{device}", action="text", element_id=5, input_str="your text", json_path="{elements_json_path}")

BEFORE SELECTING AN ELEMENT:
1. First explain what you're looking for based on the task
2. List the relevant elements you found and their IDs
3. Explain which element you chose and WHY
4. Then execute the action

All tool calls must include:
- device: specify the operating device
- element_id: the index of the element to interact with (from the JSON array)
- json_path: path to the elements JSON file
Only execute one tool call."""
        ),
        HumanMessage(
            content=f"The current device is: {device}, the device screen size is {device_size}. The user's current task intent is: {task}"
        ),
        HumanMessage(
            content="Below is the current page's parsed JSON data. Each element has an index (starting from 0) that you should use as element_id. PAY ATTENTION TO THE 'content' field to find the right element:\n"
            + elements_text
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": "Below is the LABELED screenshot with numbered bounding boxes. Each number corresponds to an element ID in the JSON. Look at the numbers on the image to identify which element to click:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"},
                },
            ],
        ),
    ]

    # Add these messages to state
    state["messages"].extend(messages)

    # Call action_agent for decision making and action execution
    action_result = action_agent.invoke({"messages": state["messages"][-4:]})

    # Parse results
    final_messages = action_result.get("messages", [])
    if final_messages:
        # Add AI reply to message history
        ai_message = final_messages[-1]
        state["messages"].append(ai_message)

        # Extract recommended action from final_message
        recommended_action = ai_message.content.strip()
        
        # Show AI reasoning (truncate if too long)
        if len(recommended_action) > 150:
            print(f"   🤖 AI: {recommended_action[:150]}...")
        else:
            print(f"   🤖 AI: {recommended_action}")

        # Update execution status
        state["current_step"] += 1
        state["history"].append(
            {
                "step": state["current_step"],
                "screenshot": screenshot_path,
                "elements_json": elements_json_path,
                "action": "react_mode",
                "recommended_action": recommended_action,
                "status": "success",
            }
        )

        state["execution_status"] = "success"
        # Action feedback based on what was done
        if "tap" in recommended_action.lower() or "click" in recommended_action.lower():
            # Extract what was clicked if mentioned
            if "element" in recommended_action.lower():
                elem_match = re.search(r'element\s+(\d+)', recommended_action.lower())
                if elem_match:
                    print(f"   ✓ Clicked element {elem_match.group(1)}")
                else:
                    print(f"   ✓ Clicked")
            else:
                print(f"   ✓ Clicked")
        elif "text" in recommended_action.lower() or "type" in recommended_action.lower():
            # Try to extract what was typed
            if '"' in recommended_action:
                text_match = re.search(r'"([^"]+)"', recommended_action)
                if text_match:
                    print(f"   ✓ Typed: \"{text_match.group(1)}\"")
                else:
                    print(f"   ✓ Typed text")
            else:
                print(f"   ✓ Typed text")
        else:
            print(f"   ✓ Action completed")
    else:
        error_msg = "React mode execution failed: No messages returned"
        print(f"❌ {error_msg}")

        # Update execution status
        state["history"].append(
            {
                "step": state["current_step"],
                "screenshot": screenshot_path,
                "elements_json": elements_json_path,
                "action": "react_mode",
                "status": "error",
                "error": error_msg,
            }
        )

        state["execution_status"] = "error"

    return state

def check_task_completion(state: DeploymentState) -> DeploymentState:
    """
    Determine if task is completed

    Args:
        state: Execution state

    Returns:
        Updated execution state with task completion status
    """
    # Check if we've reached max steps
    max_steps = state.get("max_steps", 20)
    if state["current_step"] >= max_steps:
        print(f"⚠️ Reached maximum steps ({max_steps}), stopping execution")
        state["completed"] = True
        state["execution_status"] = "max_steps_reached"
        return state
    
    # Don't check completion too early - let it run for a few steps
    if state["current_step"] < 3:
        print(f"📝 Step {state['current_step']}: Continuing task execution...")
        state["completed"] = False
        return state

    print("🔍 Evaluating if task is completed...")

    # Get task description
    task = state["task"]

    # Step 1: Generate task completion criteria
    completion_messages = [
        SystemMessage(
            content="You are an assistant that will help analyze task completion criteria. Please carefully read the following user task:"
        ),
        HumanMessage(
            content=f"The user's task is: {task}\nPlease describe clear, checkable task completion criteria. For example: 'When certain elements or states appear on the page, it indicates the task is complete.'"
        ),
    ]

    completion_response = model.invoke(completion_messages)
    completion_criteria = completion_response.content

    # Collect recent screenshots
    recent_screenshots = []
    for step in state["history"][-3:]:
        if "screenshot" in step and step["screenshot"]:
            recent_screenshots.append(step["screenshot"])

    if not recent_screenshots:
        if state["current_page"]["screenshot"]:
            recent_screenshots.append(state["current_page"]["screenshot"])

    if not recent_screenshots:
        print("⚠️ No screenshots available, cannot determine if task is complete")
        return state

    # Build image messages
    image_messages = []
    for idx, img_path in enumerate(recent_screenshots, start=1):
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            image_messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"Here is data for screenshot {idx}:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                        },
                    ]
                )
            )

    # Step 2: Determine if task is complete
    judgement_messages = [
        SystemMessage(
            content="You are a page assessment assistant that will determine if a task is complete based on completion criteria and current page screenshots. Be strict - only say 'yes' if the ENTIRE task is fully completed."
        ),
        HumanMessage(
            content=f"Original task: {task}\n"
            f"Completion criteria: {completion_criteria}\n\n"
            f"Based on the following screenshots, determine if the ENTIRE task is complete.\n"
            f"- If the task is partially done but not finished, respond 'no'\n"
            f"- Only respond 'yes' if the full goal has been achieved\n"
            f"- For example, if the task is 'search for blue cheese', just opening the search bar is NOT complete - the search must be performed\n"
            f"Respond with ONLY 'yes' or 'no'."
        ),
    ]

    # Combine all messages
    all_messages = judgement_messages + image_messages

    # Call LLM for judgment
    judgement_response = model.invoke(all_messages)
    judgement_answer = judgement_response.content.strip()

    # Update task completion status
    if "yes" in judgement_answer.lower() or "complete" in judgement_answer.lower():
        state["completed"] = True
        state["execution_status"] = "completed"
        print(f"✓ Task completed: {judgement_answer}")
    else:
        state["completed"] = False
        print(f"⚠️ Task not completed: {judgement_answer}")

    # Add to history
    state["history"].append(
        {
            "step": state["current_step"],
            "action": "task_completion_check",
            "completion_criteria": completion_criteria,
            "judgement": judgement_answer,
            "status": "success",
            "completed": state["completed"],
        }
    )

    return state

def is_task_completed(state: DeploymentState) -> str:
    """
    Check if task is completed
    """
    if state["completed"]:
        return "end"
    return "continue"

def run_multiple_tasks(tasks: list, device: str = "emulator-5554") -> Dict[str, Any]:
    """
    Execute multiple tasks in sequence
    
    Args:
        tasks: List of task descriptions
        device: Device ID
        
    Returns:
        Combined execution results
    """
    import datetime
    import re
    
    # Create a master run folder for all tasks
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_folder = f"multi_run_{timestamp}"
    
    print(f"📁 Master folder for all tasks: ./log/screenshots/{master_folder}/")
    print(f"📋 Executing {len(tasks)} tasks in sequence")
    
    results = []
    for idx, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"📌 Task {idx}/{len(tasks)}: {task}")
        print(f"{'='*60}")
        
        # Create subfolder for this specific task
        clean_task = re.sub(r'[^\w\s-]', '', task)[:30].strip().replace(' ', '_')
        task_subfolder = f"{master_folder}/task_{idx}_{clean_task}"
        
        # Run the task with the specific subfolder
        result = run_task_with_folder(task, device, task_subfolder)
        results.append({
            "task": task,
            "result": result,
            "folder": task_subfolder
        })
        
        # Brief pause between tasks
        if idx < len(tasks):
            print(f"\n⏸️ Pausing 2 seconds before next task...")
            import time
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"✅ Completed all {len(tasks)} tasks")
    print(f"📁 All results saved in: ./log/screenshots/{master_folder}/")
    
    return {
        "total_tasks": len(tasks),
        "results": results,
        "master_folder": master_folder
    }

def run_task_with_folder(task: str, device: str = "emulator-5554", folder: str = None) -> Dict[str, Any]:
    """
    Execute a single task with a specific folder
    """
    print(f"🚀 Starting task execution: {task}")

    try:
        # Use provided folder or create a new one
        if not folder:
            import datetime
            import re
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_task = re.sub(r'[^\w\s-]', '', task)[:50].strip().replace(' ', '_')
            folder = f"run_{timestamp}_{clean_task}"
        
        print(f"📁 Saving screenshots to: ./log/screenshots/{folder}/")
        
        # Initialize state
        from data.State import create_deployment_state
        state = create_deployment_state(
            task=task,
            device=device,
            max_retries=3,
        )
        
        state["task_folder"] = folder
        state["max_steps"] = 20
        
        # Execute task using workflow
        workflow = build_workflow()
        app = workflow.compile()
        result = app.invoke(state)
        
        if "task_folder" not in result:
            result["task_folder"] = folder
            
        # Display final status
        if result["execution_status"] == "success":
            print(f"✅ Task completed successfully in {result.get('current_step', 0)} steps")
        elif result["execution_status"] == "max_steps_reached":
            print(f"⚠️ Task stopped after reaching max steps ({state['max_steps']})")
        else:
            print(f"❌ Task failed: {result.get('execution_status', 'unknown')}")
            
        return {
            "status": result["execution_status"],
            "message": "Task execution completed",
            "steps_completed": result.get("current_step", 0),
            "folder": folder
        }
        
    except Exception as e:
        print(f"❌ Error executing task: {str(e)}")
        return {
            "status": "error",
            "message": f"Error executing task: {str(e)}",
            "error": str(e)
        }

def build_workflow() -> StateGraph:
    """
    Build simplified workflow state graph without GraphDB
    """
    workflow = StateGraph(DeploymentState)

    # Add only necessary nodes for React mode
    workflow.add_node("capture_screen", capture_screen_node)
    workflow.add_node("fallback", ui_test_agent)
    workflow.add_node("check_completion", check_task_completion)

    # Simple flow: capture -> react mode -> check completion
    workflow.set_entry_point("capture_screen")
    
    # Always go to fallback (React mode) after capture
    workflow.add_edge("capture_screen", "fallback")
    
    # Check task completion after fallback
    workflow.add_edge("fallback", "check_completion")
    
    # Loop or end based on completion
    workflow.add_conditional_edges(
        "check_completion",
        is_task_completed,
        {"end": END, "continue": "capture_screen"},
    )

    return workflow
