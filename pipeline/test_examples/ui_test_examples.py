"""
UI Test Examples for Button Color and Presence Checks
These examples show how to integrate UI tests into subtasks
"""

# Example 1: Testing button color in a subtask
button_color_test_subtask = {
    "id": "1",
    "task": "Verify the submit button is blue",
    "description": "Check that the submit button has the correct blue color (#007AFF)",
    "action_type": "test",
    "requires_test": True,
    "test_config": {
        "test_type": "check_element_colour",
        "element": "submit button",
        "expected": "blue color (#007AFF)"
    },
    "status": "pending"
}

# Example 2: Testing button presence in a subtask
button_presence_test_subtask = {
    "id": "2", 
    "task": "Ensure the login button is present on screen",
    "description": "Verify that the login button exists and is visible",
    "action_type": "test",
    "requires_test": True,
    "test_config": {
        "test_type": "check_element_present",
        "element": "login button",
        "expected": "visible on screen"
    },
    "status": "pending"
}

# Example 3: Complex flow with testing
login_flow_with_tests = [
    {
        "id": "1",
        "task": "Navigate to login page",
        "description": "Open the app and go to login screen",
        "action_type": "navigate",
        "requires_test": False,
        "status": "pending"
    },
    {
        "id": "2",
        "task": "Verify login button is green before interaction",
        "description": "Check that the login button is green (#00FF00) initially",
        "action_type": "test",
        "requires_test": True,
        "test_config": {
            "test_type": "check_element_colour",
            "element": "login button",
            "expected": "green (#00FF00)"
        },
        "status": "pending"
    },
    {
        "id": "3",
        "task": "Enter username",
        "description": "Type username into the username field",
        "action_type": "input",
        "target": "username field",
        "requires_test": False,
        "status": "pending"
    },
    {
        "id": "4",
        "task": "Check password field is present",
        "description": "Verify password input field exists",
        "action_type": "test",
        "requires_test": True,
        "test_config": {
            "test_type": "check_element_present",
            "element": "password input field",
            "expected": "visible and accessible"
        },
        "status": "pending"
    },
    {
        "id": "5",
        "task": "Enter password",
        "description": "Type password into the password field",
        "action_type": "input",
        "target": "password field",
        "requires_test": False,
        "status": "pending"
    },
    {
        "id": "6",
        "task": "Tap login button",
        "description": "Press the login button to submit",
        "action_type": "tap",
        "target": "login button",
        "requires_test": False,
        "status": "pending"
    },
    {
        "id": "7",
        "task": "Verify successful login by checking home button presence",
        "description": "Confirm we're on home screen by checking for home button",
        "action_type": "test",
        "requires_test": True,
        "test_config": {
            "test_type": "check_element_present",
            "element": "home button or dashboard element",
            "expected": "visible indicating successful login"
        },
        "status": "pending"
    }
]

# Example 4: How the supervisor would generate these test subtasks
def generate_test_subtask_examples(user_task: str):
    """
    Examples of how supervisor generates subtasks with tests based on user input
    """
    
    if "test button color" in user_task.lower():
        return {
            "id": "test_1",
            "task": f"Test the color of the button as requested",
            "description": "Use UI test agent to verify button color",
            "action_type": "test",
            "requires_test": True,
            "test_config": {
                "test_type": "check_element_colour",
                "query": user_task  # Pass full user query for context
            },
            "status": "pending"
        }
    
    elif "verify button exists" in user_task.lower() or "check button present" in user_task.lower():
        return {
            "id": "test_2",
            "task": f"Verify button presence as requested",
            "description": "Use UI test agent to check if button exists",
            "action_type": "test", 
            "requires_test": True,
            "test_config": {
                "test_type": "check_element_present",
                "query": user_task
            },
            "status": "pending"
        }
    
    # Default non-test subtask
    return {
        "id": "action_1",
        "task": user_task,
        "description": "Execute user requested action",
        "action_type": "execute",
        "requires_test": False,
        "status": "pending"
    }

# Example 5: How orchestrator would handle these test subtasks
def orchestrator_test_handling_example(subtask, screenshot_path, device):
    """
    Example of how orchestrator processes test subtasks
    """
    if subtask.get("requires_test", False):
        print(f"[TEST] Running UI test for: {subtask.get('task')}")
        
        # Import the test runner
        from pipeline.utils.nich_utils import run_ui_test_for_subtask
        
        # Run the test
        test_result = run_ui_test_for_subtask(
            subtask=subtask,
            screenshot_path=screenshot_path,
            device=device
        )
        
        # Process result
        if test_result.get("tested"):
            print(f"[TEST] Result: {test_result.get('result')}")
            if test_result.get("success"):
                print("[TEST] ✓ Test passed")
                subtask["status"] = "completed"
            else:
                print("[TEST] ✗ Test failed")
                subtask["test_failed"] = True
        
        return test_result
    
    return {"tested": False}

# Example 6: Test keywords that trigger requires_test flag
TEST_TRIGGER_KEYWORDS = [
    'test', 'verify', 'check', 'ensure', 'validate', 'confirm',
    'assert', 'inspect', 'examine', 'review', 'audit'
]

def should_add_test_flag(task_description: str) -> bool:
    """
    Determine if a subtask should have requires_test flag based on keywords
    """
    task_lower = task_description.lower()
    return any(keyword in task_lower for keyword in TEST_TRIGGER_KEYWORDS)

# Example usage scenarios
if __name__ == "__main__":
    # Scenario 1: User asks to test button color
    user_request_1 = "Go to settings and test if the save button is blue"
    
    # Supervisor would generate:
    subtasks_1 = [
        {
            "id": "1",
            "task": "Navigate to settings",
            "requires_test": False,
            "status": "pending"
        },
        {
            "id": "2", 
            "task": "Test if the save button is blue",
            "requires_test": True,  # Auto-detected from 'test' keyword
            "test_config": {
                "test_type": "check_element_colour",
                "element": "save button",
                "expected": "blue"
            },
            "status": "pending"
        }
    ]
    
    # Scenario 2: User asks to verify element presence
    user_request_2 = "Open the app and verify the login button is present"
    
    # Supervisor would generate:
    subtasks_2 = [
        {
            "id": "1",
            "task": "Open the app",
            "requires_test": False,
            "status": "pending"
        },
        {
            "id": "2",
            "task": "Verify the login button is present",
            "requires_test": True,  # Auto-detected from 'verify' keyword
            "test_config": {
                "test_type": "check_element_present",
                "element": "login button"
            },
            "status": "pending"
        }
    ]
    
    print("Test examples created successfully")
    print(f"Example 1 - Button Color Test: {button_color_test_subtask}")
    print(f"Example 2 - Button Presence Test: {button_presence_test_subtask}")