"""
Supervisor Agent V2 - With Vision Capabilities
Can see the screen when needed to make better decisions
"""

import json
import re
import base64
from typing import Dict, List, Any, Optional
from langchain.schema.messages import HumanMessage, SystemMessage

from .prompts import (
    TASK_DECOMPOSITION_PROMPT,
    PROGRESS_TRACKING_PROMPT,
    STUCK_RECOVERY_PROMPT,
    COMPLETION_CHECK_PROMPT
)


class SupervisorAgentV2:
    """
    Enhanced Supervisor Agent with vision capabilities
    - Can see screen when Worker gets stuck
    - Dynamically adjusts subtasks based on actual screen state
    - Makes informed decisions, not blind ones
    """
    
    def __init__(self, model):
        """
        Initialize Supervisor with vision-capable model
        
        Args:
            model: LangChain ChatOpenAI model with vision support
        """
        self.model = model  # Now uses vision model
        self.subtasks = []
        self.current_subtask_idx = 0
        self.stuck_counter = 0
        self.action_history = []
        self.max_retries = 3
        self.last_screen_analysis = None
        self.last_correction = None  # Store correction from validation
        self.original_task = ""  # Remember the original directive
        self.task_context = {}  # Store important values found in task
        
    def decompose_task(self, task: str, initial_screenshot: str = None) -> List[Dict[str, Any]]:
        """
        Break down task into subtasks, optionally using initial screen context
        
        Args:
            task: The user's task description
            initial_screenshot: Optional screenshot to understand starting point
            
        Returns:
            List of subtask dictionaries
        """
        print("\n[SUPERVISOR V2] Analyzing task and creating subtasks...")
        
        # Store the original task for reference
        self.original_task = task
        
        # First, decompose the full task using LLM
        self._decompose_full_task(task)
        
        # If we have a screenshot, refine based on current screen
        if initial_screenshot:
            self._refine_subtasks_from_screen(task, initial_screenshot)
        
        return self.subtasks
    
    def _decompose_full_task(self, task: str):
        """
        Use LLM to break down the full task into all necessary steps
        """
        try:
            messages = [
                SystemMessage(content="""Break down this Android task into ALL necessary atomic steps.
                
Each step should be a SINGLE action (tap, type, swipe).
DO NOT include wait steps - the system handles waiting automatically.
Think through the ENTIRE flow from start to finish.

SPECIAL: Mark ALL verification/check tasks with "requires_test": true
- Any task with "check", "verify", "test", "ensure", "validate", "confirm" needs "requires_test": true
- These tasks MUST ALWAYS execute - never skip them even if screen looks right
- Test tasks can trigger backtracking if they fail

For example, "open playstore and install tiktok" requires:
1. Open Play Store app
2. Tap search bar
3. Type "tiktok"
4. Tap search button
5. Select TikTok from results
6. Tap Install button

For tasks with checks: "go to phone app and check if keypad button is red, click it, check for green call button":
1. {"task": "Open Phone app", "requires_test": false}
2. {"task": "Locate and verify keypad button on bottom right is red", "requires_test": true}
3. {"task": "Tap keypad button on bottom right", "requires_test": false}
4. {"task": "Verify green call button is visible on keypad page", "requires_test": true}

IMPORTANT: Verification tasks must happen BEFORE their associated actions!
- "Check/Verify X then tap/click Y" means verify FIRST, then act
- Don't skip actions just because the end state is visible
- Respect the sequence: locate → verify → act

Output JSON format:
{
  "subtasks": [
    {"id": 1, "task": "Open Play Store app", "requires_test": false},
    {"id": 2, "task": "Tap search bar", "requires_test": false},
    ...
  ]
}"""),
                HumanMessage(content=f"Task: {task}")
            ]
            
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                if result.get("subtasks"):
                    self.subtasks = []
                    for st in result["subtasks"]:
                        st["status"] = "pending"
                        st["attempts"] = 0
                        
                        # Ensure requires_test flag is set correctly
                        if "requires_test" not in st:
                            # If LLM didn't set it, auto-detect based on task text
                            task_text = st.get('task', '').lower()
                            test_keywords = ['check', 'verify', 'test', 'ensure', 'validate', 'confirm', 'locate and verify']
                            st["requires_test"] = any(keyword in task_text for keyword in test_keywords)
                            
                            if st["requires_test"]:
                                print(f"[SUPERVISOR V2] Auto-detected test task: '{st['task']}'")
                        elif st["requires_test"]:
                            print(f"[SUPERVISOR V2] LLM marked as test task: '{st['task']}'")
                        
                        self.subtasks.append(st)
                    
                    print(f"[SUPERVISOR V2] Decomposed into {len(self.subtasks)} subtasks")
                    for st in self.subtasks:
                        print(f"  {st['id']}. {st['task']}")
                        
        except Exception as e:
            print(f"[SUPERVISOR V2] Error decomposing task: {e}")
            # Fallback to single task
            self.subtasks = [
                {
                    "id": 1,
                    "task": task,
                    "status": "pending",
                    "attempts": 0
                }
            ]
    
    def _refine_subtasks_from_screen(self, task: str, screenshot_path: str):
        """
        Look at the screen and add any screen-specific context to existing subtasks
        """
        try:
            with open(screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            messages = [
                SystemMessage(content=f"""You are looking at an Android screen.
                
Describe what you see to provide context for task execution.
Do NOT create new subtasks - we already have them planned.
Just describe the current state of the screen.

Output JSON format:
{{
  "current_screen": "detailed description of what screen this is and what elements are visible"
}}"""),
                HumanMessage(content=f"Task: {task}"),
                HumanMessage(content=[
                    {"type": "text", "text": "Current screen:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}}
                ])
            ]
            
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                print(f"[SUPERVISOR V2] Screen shows: {result.get('current_screen', 'unknown')}")
                
                # Store screen context but don't override subtasks
                self.last_screen_analysis = result.get('current_screen', '')
                    
        except Exception as e:
            print(f"[SUPERVISOR V2] Error analyzing screen: {e}")
    
    def analyze_stuck_situation(self, worker_feedback: Dict, screenshot_path: str) -> Dict[str, Any]:
        """
        When stuck, look at the screen to understand what's wrong
        
        Args:
            worker_feedback: Last feedback from Worker
            screenshot_path: Current screenshot
            
        Returns:
            Recovery strategy based on visual analysis
        """
        print("\n[SUPERVISOR V2] Analyzing screen to understand stuck situation...")
        
        try:
            with open(screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            current_subtask = self.get_current_subtask()
            
            messages = [
                SystemMessage(content="""You are debugging why a task is stuck.
Look at the screen and understand:
1. What screen is currently showing
2. Why the worker's action might have failed
3. What should be done instead

The worker has tried the same action multiple times without success.

Output JSON:
{
  "screen_analysis": "what you see",
  "problem": "why it's stuck",
  "solution": "what to do instead",
  "new_subtask": "specific action to take now"
}"""),
                HumanMessage(content=f"""
Current subtask: {current_subtask.get('task', 'unknown')}
Worker tried: {worker_feedback.get('decision', 'unknown')}
Result: {worker_feedback.get('result', 'unknown')}

Task goal: {self.get_original_task()}
"""),
                HumanMessage(content=[
                    {"type": "text", "text": "Current screen:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}}
                ])
            ]
            
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                print(f"[SUPERVISOR V2] Screen shows: {analysis.get('screen_analysis', '')}")
                print(f"[SUPERVISOR V2] Problem: {analysis.get('problem', '')}")
                print(f"[SUPERVISOR V2] Solution: {analysis.get('solution', '')}")
                
                # Create new subtask based on analysis
                if analysis.get("new_subtask"):
                    # Insert new subtask
                    new_task = {
                        "id": len(self.subtasks) + 1,
                        "task": analysis["new_subtask"],
                        "type": "recovery",
                        "status": "pending",
                        "attempts": 0,
                        "specific_value": getattr(self, 'specific_value', None)  # Pass along specific value
                    }
                    
                    # Mark current as skipped and add new one
                    current_subtask["status"] = "skipped"
                    current_subtask["skip_reason"] = analysis.get("problem", "")
                    
                    # Insert after current
                    idx = self.subtasks.index(current_subtask)
                    self.subtasks.insert(idx + 1, new_task)
                    
                    return {
                        "action": "execute",
                        "subtask": new_task,
                        "context": {"recovery": True, "reason": analysis.get("solution", "")}
                    }
                
        except Exception as e:
            print(f"[SUPERVISOR V2] Error in visual analysis: {e}")
        
        # Fallback to skip
        return {
            "action": "skip",
            "reason": "Unable to recover from stuck state"
        }
    
    def verify_subtask_completion(self, subtask: Dict, screenshot_path: str) -> bool:
        """
        Visually verify if a subtask is actually complete
        Also checks if future subtasks are already satisfied
        
        Args:
            subtask: The subtask to verify
            screenshot_path: Current screenshot
            
        Returns:
            True if visually confirmed complete
        """
        try:
            with open(screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Get list of upcoming subtasks for context
            current_idx = self.subtasks.index(subtask)
            upcoming_subtasks = self.subtasks[current_idx+1:current_idx+3] if current_idx < len(self.subtasks)-1 else []
            upcoming_tasks_str = "\n".join([f"- {st['task']}" for st in upcoming_subtasks if st['status'] == 'pending'])
            
            # Check if this is a test/verification task
            is_test_task = subtask.get('requires_test', False)
            test_result = subtask.get('test_result_details', {})
            
            validation_prompt = f"""Look at the screen and determine if the worker's action achieved the subtask goal.

ORIGINAL TASK CONTEXT:
"{self.original_task}"

CURRENT SUBTASK TO VALIDATE:
"{subtask.get('task', '')}"
"""
            
            if is_test_task and test_result:
                validation_prompt += f"""
TEST EXECUTION RESULT:
The UI test agent reported: "{test_result.get('result', '')}"

CRITICAL: This is a VERIFICATION/TEST task. You must assess whether the test condition is actually met:
- If the task asks to "verify X is Y" and the test reports "X is not Y", mark as NOT COMPLETE
- If the task asks to "check if X exists" and test reports "X exists", mark as COMPLETE
- Consider the INTENT: Is this checking a specific condition (fail if wrong) or just confirming presence (pass if exists)?
- Example: "Verify button is green" with response "button is blue" = NOT COMPLETE (wrong color)
- Example: "Check if button exists" with response "button exists" = COMPLETE (presence confirmed)
"""
            else:
                validation_prompt += f"""
UPCOMING SUBTASKS:
{upcoming_tasks_str if upcoming_tasks_str else "None"}

INSTRUCTIONS:
1. Look at what's currently on the screen
2. Determine if the subtask goal has been achieved
3. Consider if the screen has progressed beyond what the subtask required
4. Check if any upcoming subtasks are ALREADY SATISFIED by the current screen state
5. If the screen shows evidence that the subtask was already completed (even if in a previous step), mark as complete
"""

            validation_prompt += """
VALIDATION RULES:
- For TEST tasks: Mark COMPLETE only if the specific condition is met (not just acknowledged)
- For ACTION tasks: Mark COMPLETE if the action was performed or screen shows it's done
- For typing tasks, check if the correct value appears where it should
- Consider the overall flow - don't repeat completed actions

Output JSON:
{{
  "complete": true/false,
  "evidence": "what you see on screen that supports your decision",
  "correction": "if not complete, what specific action is needed",
  "current_context": "brief description of current screen state",
  "already_satisfied_subtasks": ["list of upcoming subtask descriptions that are already done based on current screen"]
}}"""

            messages = [
                SystemMessage(content=validation_prompt),
                HumanMessage(content=f"Subtask: {subtask.get('task', '')}"),
                HumanMessage(content=[
                    {"type": "text", "text": "Current screen after worker action:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}}
                ])
            ]
            
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                evidence = result.get('evidence', '')
                is_complete = result.get("complete", False)
                
                if is_complete:
                    print(f"[SUPERVISOR V2] ✓ Action correct: {evidence}")
                    
                    # Don't refine task list for test tasks - they handle their own completion
                    is_test_task = subtask.get('requires_test', False)
                    if not is_test_task:
                        # Check if any future subtasks are already satisfied
                        already_done = result.get('already_satisfied_subtasks', [])
                        if already_done:
                            print(f"[SUPERVISOR V2] Future subtasks already satisfied: {already_done}")
                            # Be very careful about marking tasks as already_completed
                            for st in self.subtasks:
                                # Check if this is a test/verification task
                                is_test_task = st.get('requires_test', False)
                                if not is_test_task:
                                    # Also check by keywords if requires_test flag is missing
                                    task_lower = st.get('task', '').lower()
                                    test_keywords = ['check', 'verify', 'test', 'ensure', 'validate', 'confirm', 'locate']
                                    is_test_task = any(keyword in task_lower for keyword in test_keywords)
                                
                                # Check if there's a test task that depends on this action not being done yet
                                has_dependent_test = False
                                for other_st in self.subtasks:
                                    if (other_st.get('requires_test', False) and 
                                        other_st['status'] == 'pending' and
                                        other_st.get('id', 999) < st.get('id', 0)):
                                        # There's a test task before this action that needs to run first
                                        has_dependent_test = True
                                        break
                                
                                # Only skip non-test tasks that don't have pending tests before them
                                if (st['status'] == 'pending' and not is_test_task and not has_dependent_test and 
                                    any(done_task in st['task'] for done_task in already_done)):
                                    st['status'] = 'already_completed'
                                    st['skip_reason'] = 'Screen shows this was already done'
                                    print(f"[SUPERVISOR V2] Marking '{st['task']}' as already_completed")
                                elif st['status'] == 'pending' and is_test_task and any(done_task in st['task'] for done_task in already_done):
                                    print(f"[SUPERVISOR V2] Keeping test task '{st['task']}' active for verification")
                                elif has_dependent_test:
                                    print(f"[SUPERVISOR V2] Not skipping '{st['task']}' - has pending test before it")
                        
                        # Refine task list after successful validation of non-test tasks
                        self.refine_task_list(screenshot_path, {"validation": "success", "evidence": evidence})
                    else:
                        print(f"[SUPERVISOR V2] Test task validated - skipping refinement to preserve test results")
                else:
                    print(f"[SUPERVISOR V2] ✗ Action incorrect: {evidence}")
                    if result.get('correction'):
                        print(f"[SUPERVISOR V2] Correction needed: {result['correction']}")
                        # Store correction for next attempt
                        self.last_correction = result['correction']
                
                return is_complete
                
        except Exception as e:
            print(f"[SUPERVISOR V2] Error verifying completion: {e}")
            
        return False
    
    def get_correction_action(self) -> Dict[str, Any]:
        """
        Get correction action based on last validation
        
        Returns:
            Correction action for worker
        """
        if self.last_correction:
            current_subtask = self.get_current_subtask()
            return {
                "action": "execute",
                "subtask": current_subtask,
                "context": {
                    "correction": self.last_correction,
                    "retry": True,
                    "reason": "Previous action didn't achieve goal"
                }
            }
        return {"action": "continue"}
    
    def get_next_action(self, worker_feedback: Dict, screenshot_path: str = None) -> Dict[str, Any]:
        """
        Decide next action based on worker feedback AND visual confirmation
        
        Args:
            worker_feedback: Results from worker's last action
            screenshot_path: Current screenshot for visual verification
            
        Returns:
            Dictionary with next action decision
        """
        current_subtask = self.get_current_subtask()
        
        if not current_subtask:
            return {"action": "complete", "reason": "All subtasks completed"}
        
        # Add to history
        self.action_history.append(worker_feedback)
        
        # Check if stuck
        if self._is_stuck():
            self.stuck_counter += 1
            print(f"[SUPERVISOR V2] Detected stuck situation (counter: {self.stuck_counter}/{self.max_retries})")
            
            # Refine the task list when stuck
            if screenshot_path:
                self.refine_task_list(screenshot_path, worker_feedback)
                
                # After refinement, get the current subtask
                current_subtask = self.get_current_subtask()
                if current_subtask:
                    return {
                        "action": "execute",
                        "subtask": current_subtask,
                        "context": {"refined": True, "reason": "Task list refined after stuck detection"}
                    }
            
            # If still stuck after max retries, mark as failed
            if self.stuck_counter >= self.max_retries:
                current_subtask["status"] = "failed"
                current_subtask["failure_reason"] = "Unable to complete after multiple attempts"
                print(f"[SUPERVISOR V2] Marking subtask as failed after {self.stuck_counter} attempts")
                
                # Try next subtask
                next_subtask = self.get_current_subtask()
                if next_subtask:
                    return {
                        "action": "execute",
                        "subtask": next_subtask,
                        "context": {"skipped_previous": True}
                    }
                else:
                    return {"action": "complete", "reason": "Task ended with failures"}
        else:
            self.stuck_counter = 0
        
        # Check if current subtask is a test/verification task
        if current_subtask.get('requires_test', False):
            # Only handle test results if the test was actually executed
            if worker_feedback.get('is_test', False):
                # This is a test task result - handle its results specially
                result = self.handle_test_task_result(current_subtask, worker_feedback, screenshot_path)
                
                # If test passed, ensure the subtask is marked completed in main list
                if worker_feedback.get('success', False) or worker_feedback.get('test_passed', False):
                    current_subtask["status"] = "completed"
                    current_subtask["test_result"] = "passed"
                    print(f"[SUPERVISOR V2] Test task {current_subtask['id']} marked as completed")
                
                return result
            else:
                # Test task hasn't been executed yet - return it for execution
                return {
                    "action": "execute",
                    "subtask": current_subtask,
                    "context": {"is_test": True}
                }
        
        # If worker reports success, verify visually if possible
        if worker_feedback.get("success") and screenshot_path:
            if self.verify_subtask_completion(current_subtask, screenshot_path):
                current_subtask["status"] = "completed"
                print(f"[SUPERVISOR V2] Subtask {current_subtask['id']} visually confirmed complete")
                
                # Get next subtask
                next_subtask = self.get_current_subtask()
                if next_subtask:
                    # Analyze screen to provide better context for next subtask
                    try:
                        context = self._get_visual_context(next_subtask, screenshot_path) if screenshot_path else {}
                    except Exception as e:
                        print(f"[SUPERVISOR V2] Error getting visual context: {e}")
                        context = {}
                    
                    return {
                        "action": "execute",
                        "subtask": next_subtask,
                        "context": context
                    }
                else:
                    return {"action": "complete", "reason": "All subtasks done"}
            else:
                # Worker thought it succeeded but visual check says no
                print(f"[SUPERVISOR V2] Worker reported success but visual check failed")
                return {
                    "action": "execute",
                    "subtask": current_subtask,
                    "context": {"retry": True, "reason": "Visual verification failed"}
                }
        
        # Default: continue with current subtask
        return {
            "action": "execute",
            "subtask": current_subtask,
            "context": {}
        }
    
    def _get_visual_context(self, subtask: Dict, screenshot_path: str) -> Dict[str, Any]:
        """
        Look at screen to provide specific hints for the subtask
        """
        if not subtask or not screenshot_path:
            return {}
            
        try:
            with open(screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            messages = [
                SystemMessage(content="""Look at the screen and provide specific hints for completing the subtask.
Identify relevant elements and their locations.

Output JSON:
{
  "screen_type": "what kind of screen this is",
  "relevant_elements": "elements related to the subtask",
  "specific_hint": "precise instruction based on what you see"
}"""),
                HumanMessage(content=f"Subtask: {subtask.get('task', '')}"),
                HumanMessage(content=[
                    {"type": "text", "text": "Current screen:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}}
                ])
            ]
            
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                hints = json.loads(json_match.group())
                return {
                    "visual_hints": hints.get("specific_hint", ""),
                    "screen_context": hints.get("screen_type", "")
                }
                
        except Exception as e:
            print(f"[SUPERVISOR V2] Error getting visual context: {e}")
            
        return {}
    
    def get_current_subtask(self) -> Optional[Dict[str, Any]]:
        """Get the current pending subtask, skipping already_completed ones (except unexecuted test tasks)"""
        for subtask in self.subtasks:
            if subtask["status"] in ["pending", "in_progress"]:
                # Mark as in_progress when we start working on it
                if subtask["status"] == "pending":
                    subtask["status"] = "in_progress"
                return subtask
            elif subtask["status"] == "already_completed":
                # Check if this is a test task that should still run
                if subtask.get('requires_test', False) and not subtask.get('test_result'):
                    print(f"[SUPERVISOR V2] Found unexecuted test task marked as already_completed, reactivating: '{subtask['task']}'")
                    subtask["status"] = "in_progress"
                    return subtask
                # Skip and log tasks (test or non-test) that are done
                if subtask.get("skip_reason"):
                    print(f"[SUPERVISOR V2] Skipping '{subtask['task']}': {subtask['skip_reason']}")
                elif subtask.get('test_result'):
                    print(f"[SUPERVISOR V2] Skipping completed test '{subtask['task']}': test {subtask['test_result']}")
                continue
            elif subtask["status"] == "completed":
                # Skip completed tasks
                continue
        return None
    
    def get_original_task(self) -> str:
        """Get the original task description"""
        if self.subtasks and len(self.subtasks) > 0:
            return self.subtasks[0].get("task", "").replace("Complete: ", "")
        return ""
    
    def _is_stuck(self) -> bool:
        """Check if worker is stuck repeating same failed actions"""
        if len(self.action_history) < 2:
            return False
        
        # Get last 2-3 actions
        recent_actions = self.action_history[-2:]
        
        # Extract what was attempted and what happened
        recent_attempts = []
        for action in recent_actions:
            worker_result = action.get("worker_result", {})
            attempt = {
                "decision": worker_result.get("decision", ""),
                "result": worker_result.get("result", ""),
                "success": worker_result.get("success", False),
                "subtask": worker_result.get("subtask", "")
            }
            recent_attempts.append(attempt)
        
        # Check if same action with same failure
        if len(recent_attempts) >= 2:
            # Same decision and both failed
            if (recent_attempts[-1]["decision"] == recent_attempts[-2]["decision"] and
                not recent_attempts[-1]["success"] and not recent_attempts[-2]["success"]):
                print(f"[SUPERVISOR V2] Stuck: Same failed action repeated")
                return True
            
            # Typing same wrong value multiple times
            if "typed" in recent_attempts[-1]["result"].lower():
                typed_texts = [a["result"] for a in recent_attempts if "typed" in a["result"].lower()]
                if len(typed_texts) >= 2 and typed_texts[-1] == typed_texts[-2]:
                    print(f"[SUPERVISOR V2] Stuck: Typing same value repeatedly")
                    return True
        
        return False
    
    def handle_test_task_result(self, test_task: Dict, test_result: Dict, screenshot_path: str = None) -> Dict[str, Any]:
        """
        Handle test/verification task results with potential backtracking
        
        Args:
            test_task: The test subtask that was executed
            test_result: Results from the test execution
            screenshot_path: Current screenshot
            
        Returns:
            Next action to take based on test results
        """
        test_passed = test_result.get("success", False) or test_result.get("test_passed", False)
        
        if test_passed:
            print(f"[SUPERVISOR V2] ✓ Test passed: {test_task['task']}")
            test_task["status"] = "completed"
            test_task["test_result"] = "passed"
            
            # Mark in the main subtasks list too
            for st in self.subtasks:
                if st.get('id') == test_task.get('id'):
                    st["test_result"] = "passed"
                    break
            
            # Continue to next subtask
            next_subtask = self.get_current_subtask()
            if next_subtask:
                return {
                    "action": "execute",
                    "subtask": next_subtask,
                    "context": {"previous_test_passed": True}
                }
            else:
                return {"action": "complete", "reason": "All tasks including tests completed"}
        else:
            print(f"[SUPERVISOR V2] ✗ Test failed: {test_task['task']}")
            test_task["status"] = "failed"
            test_task["test_result"] = "failed"
            test_task["failure_reason"] = test_result.get("error", "Test condition not met")
            
            # Check if we need to navigate back to perform the test
            # The test might have failed because we're on the wrong screen
            task_lower = test_task['task'].lower()
            
            # If we're trying to check something that's not visible, we might need to go back
            if 'keypad button' in task_lower and 'bottom right' in task_lower:
                # We're likely on the keypad screen but need to be on the main phone screen
                print(f"[SUPERVISOR V2] Test requires different screen - navigating back")
                
                # Create a special navigation subtask
                back_subtask = {
                    "id": -1,  # Special ID for navigation
                    "task": "Go back to previous screen",
                    "requires_test": False,
                    "is_navigation": True
                }
                
                return {
                    "action": "execute", 
                    "subtask": back_subtask,
                    "context": {
                        "navigation": "back",
                        "reason": f"Need to go back to see '{test_task['task']}'",
                        "next_test": test_task
                    }
                }
            
            # Otherwise, identify which previous action needs to be corrected
            action_to_fix = None
            for i, st in enumerate(self.subtasks):
                # Find the action task before this test task
                if st['id'] < test_task['id'] and 'check' not in st['task'].lower() and 'verify' not in st['task'].lower():
                    action_to_fix = st
            
            if action_to_fix:
                print(f"[SUPERVISOR V2] Backtracking to fix: {action_to_fix['task']}")
                # Reset the action task to retry
                action_to_fix["status"] = "in_progress"
                action_to_fix["attempts"] = action_to_fix.get("attempts", 0) + 1
                
                # Also reset the test task for re-verification
                test_task["status"] = "pending"
                
                return {
                    "action": "execute",
                    "subtask": action_to_fix,
                    "context": {
                        "retry": True,
                        "reason": f"Test '{test_task['task']}' failed, retrying action",
                        "test_requirement": test_result.get("requirement", "Meet test condition")
                    }
                }
            else:
                # Can't identify what to fix, try to provide guidance
                return {
                    "action": "execute",
                    "subtask": test_task,
                    "context": {
                        "retry": True,
                        "reason": "Test failed, attempting to meet test requirements",
                        "test_requirement": test_result.get("requirement", "Meet test condition")
                    }
                }
    
    def refine_task_list(self, current_screen_path: str, worker_feedback: Dict) -> None:
        """
        Dynamically refine the entire task list based on current situation
        Can skip, merge, remove subtasks that are no longer needed
        """
        print("\n[SUPERVISOR V2] Refining task list based on current situation...")
        
        try:
            with open(current_screen_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Prepare current state summary
            current_state = {
                "completed": [st for st in self.subtasks if st["status"] == "completed"],
                "pending": [st for st in self.subtasks if st["status"] in ["pending", "in_progress"]],
                "already_completed": [st for st in self.subtasks if st["status"] == "already_completed"],
                "failed": [st for st in self.subtasks if st["status"] == "failed"]
            }
            
            messages = [
                SystemMessage(content=f"""You are refining the task list to achieve the original goal more efficiently.
                
Based on the current screen state, intelligently modify the subtask list:
- SKIP subtasks that are no longer needed (mark as 'already_completed')
- MERGE subtasks if the screen shows we can do multiple things at once
- REMOVE redundant subtasks
- ADAPT subtasks to match what's actually on screen
- Keep the original goal in mind

ORIGINAL TASK TO REFERENCE:
"{self.original_task}"

IMPORTANT RULES:
1. If screen already shows what a subtask wants to achieve, mark it 'already_completed' UNLESS it's a verification/test task
2. NEVER skip verification tasks (check/verify/test/ensure/validate/locate) - they must ALWAYS execute
3. NEVER skip an action if there's a pending verification task that should happen first
4. Respect task ordering - if task 2 is "verify X" and task 3 is "tap Y", don't skip task 3 just because Y is visible
5. If typing "tiktok" already brought up TikTok in results, you can skip "Tap search button"
6. Look at what's ACTUALLY on screen, not what you expect
7. Extract actual values from original task (emails, passwords, etc)

Output JSON:
{{
  "reasoning": "why these changes are needed",
  "refined_subtasks": [
    {{"id": 1, "task": "specific action", "status": "completed/pending/already_completed", "skip_reason": "if already_completed, why"}},
    ...
  ]
}}"""),
                HumanMessage(content=f"""Original task: {self.original_task}
                
Current subtasks:
{json.dumps(self.subtasks, indent=2)}

Recent feedback: {worker_feedback}

Based on the current screen, what subtasks should be skipped or modified?"""),
                HumanMessage(content=[
                    {"type": "text", "text": "Current screen:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}}
                ])
            ]
            
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                print(f"[SUPERVISOR V2] Reasoning: {result.get('reasoning', '')}")
                
                if result.get("refined_subtasks"):
                    # Store old subtasks to preserve flags and test results
                    old_subtasks_by_id = {st.get('id'): st for st in self.subtasks}
                    old_count = len([st for st in self.subtasks if st["status"] == "pending"])
                    
                    # Store test results before replacing
                    test_results = {}
                    for st in self.subtasks:
                        if st.get('test_result'):
                            test_results[st.get('id')] = st.get('test_result')
                    
                    # Replace entire subtask list with refined version
                    self.subtasks = result["refined_subtasks"]
                    
                    # Ensure all subtasks have required fields and preserve test flags
                    for st in self.subtasks:
                        if "status" not in st:
                            st["status"] = "pending"
                        if "attempts" not in st:
                            st["attempts"] = 0
                        
                        # Restore test results if this task was already tested
                        if st.get('id') in test_results:
                            st['test_result'] = test_results[st.get('id')]
                            # If it has a test result, it should be marked as completed
                            if st['status'] == 'already_completed':
                                st['status'] = 'completed'
                        
                        # Preserve requires_test flag from original subtask if it exists
                        old_st = old_subtasks_by_id.get(st.get('id'))
                        if old_st and 'requires_test' in old_st:
                            st['requires_test'] = old_st['requires_test']
                        elif 'requires_test' not in st:
                            # Check if this is a test-related subtask based on keywords
                            task_text = st.get('task', '').lower()
                            test_keywords = ['test', 'verify', 'check', 'ensure', 'validate', 'confirm']
                            st['requires_test'] = any(keyword in task_text for keyword in test_keywords)
                        
                        # CRITICAL: Test tasks should only be reverted if they haven't been executed yet
                        if st.get('requires_test', False) and st['status'] == 'already_completed':
                            # Check against the original subtask list to see if it was executed
                            original_st = old_subtasks_by_id.get(st.get('id'))
                            
                            # Check if test was executed (either in refined list or original)
                            test_was_executed = st.get('test_result') or (original_st and original_st.get('test_result'))
                            
                            if not test_was_executed:
                                # Test wasn't executed, revert to pending
                                st['status'] = 'pending'
                                st['skip_reason'] = None
                                print(f"[SUPERVISOR V2] Reverting unexecuted test task '{st['task']}' from already_completed to pending")
                            else:
                                # Test was executed, keep it as completed and preserve result
                                if original_st and original_st.get('test_result'):
                                    st['test_result'] = original_st['test_result']
                                st['status'] = 'completed'  # Ensure it's marked completed, not already_completed
                                print(f"[SUPERVISOR V2] Keeping executed test task '{st['task']}' as completed (result: {st.get('test_result')})")
                    
                    new_count = len([st for st in self.subtasks if st["status"] == "pending"])
                    skipped_count = len([st for st in self.subtasks if st["status"] == "already_completed"])
                    
                    print(f"[SUPERVISOR V2] Refined: {old_count} pending → {new_count} pending, {skipped_count} skipped")
                    
                    # Reset stuck counter since we have a new plan
                    self.stuck_counter = 0
                    
        except Exception as e:
            print(f"[SUPERVISOR V2] Error refining task list: {e}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        total = len(self.subtasks)
        # Count both completed and already_completed as completed work
        completed = sum(1 for st in self.subtasks if st["status"] in ["completed", "already_completed"])
        skipped = sum(1 for st in self.subtasks if st["status"] == "skipped")
        
        # Calculate actual work done (completed + skipped)
        work_done = completed + skipped
        
        return {
            "total_subtasks": total,
            "completed": work_done,  # Report total work done
            "skipped": skipped,
            "in_progress": self.get_current_subtask(),
            "progress_percentage": (work_done / total * 100) if total > 0 else 0
        }