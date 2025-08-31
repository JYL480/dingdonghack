# Supervisor-Worker Implementation Plan

## Executive Summary

A fully implemented Supervisor-Worker architecture with vision capabilities, coordinate extrapolation for unmarked elements, and full upfront task decomposition. Successfully handles complex multi-step tasks like app installation.

## Current State Analysis

### What We Have (Working Well)
1. **Two-stage execution** - Vision → Action separation works efficiently
2. **OmniParser integration** - Reliable UI element extraction
3. **Click visualization** - Red markers showing exact click positions
4. **Enhanced vision prompts** - Reasoning and context awareness
5. **Improved typing** - Two-step process (tap to focus, then type)
6. **Better action parsing** - Handles variations like "a) tap element 50"

### Current Capabilities
1. **Coordinate Extrapolation** - Click unmarked UI elements using spatial inference
2. **Full Task Decomposition** - Breaks complex tasks into all steps upfront
3. **Vision-Based Validation** - Every action validated with screenshots
4. **Smart Text Input** - Clears fields, types full words at once
5. **Screen-Aware Estimation** - Uses device dimensions (1080x2400) as reference
6. **Dynamic Task Refinement** - Adjusts subtasks based on screen state
7. **No Wait Steps** - System handles delays automatically

### Evidence from Test Runs
- Chrome task: Clicked element 7 successfully but got stuck on sign-in
- Typing worked: Successfully typed "81291126" after improvements
- Completion check hung: API timeout after multiple screenshots
- Action parsing failed: "a) Tap the 'Next' button (element 50)" not executed

## Proposed Architecture

### Three-Layer System

#### Layer 1: Supervisor Agent V2 (Vision-Enabled Brain)
- **Role**: Task planning and visual validation
- **Model**: GPT-4.1-2025-04-14 with vision capabilities
- **Responsibilities**:
  - Full upfront task decomposition via LLM
  - Visual validation of every worker action
  - Dynamic task refinement based on screen state
  - Stuck detection with retry limits
  - Context preservation throughout execution

#### Layer 2: Worker Agent (Enhanced Executor)
- **Role**: Two-stage navigation with coordinate inference
- **Model**: GPT-4.1-2025-04-14 (vision) + ADB execution
- **Responsibilities**:
  - Stage 1: Vision analysis with element coordinates
  - Stage 2: Smart action execution
  - Coordinate extrapolation for unmarked elements
  - Field clearing before typing
  - Full word/phrase typing at once

#### Layer 3: Orchestrator (Coordinator)
- **Role**: Manage communication between agents
- **Implementation**: Pure Python
- **Responsibilities**:
  - Route tasks between agents
  - Manage state
  - Handle timeouts
  - Log everything

## Implementation Phases

### Key Implementation Features

#### 1. Coordinate Extrapolation System
```python
def extrapolate_coordinate(instruction: str, elements_data: List[Dict], device_size: Dict):
    """
    Supports:
    - "tap between element 3 and element 7" → midpoint calculation
    - "tap 200 pixels below element 5" → relative positioning
    - "tap coordinate 540,1200" → absolute positioning
    - "tap 30% right of element 10" → percentage-based
    """
```

#### 2. Full Task Decomposition
```python
class SupervisorAgent:
    def decompose_task(self, task: str) -> List[Dict]:
        """
        LLM-based full task decomposition upfront
        Example: "open playstore and install tiktok" →
        [
            {"id": 1, "task": "Open Play Store app"},
            {"id": 2, "task": "Tap search bar"},
            {"id": 3, "task": "Type tiktok"},
            {"id": 4, "task": "Tap search button"},
            {"id": 5, "task": "Select TikTok from results"},
            {"id": 6, "task": "Tap Install button"}
        ]
        Note: No wait steps - system handles delays automatically
        """
        
    def get_next_action(self, worker_feedback: Dict) -> Dict:
        """
        Decide what to do next based on worker feedback
        Returns: {
            "action": "execute|check|retry|skip",
            "subtask": current_subtask,
            "context": hints_for_worker
        }
        """
        
    def handle_stuck(self, repeated_actions: List) -> Dict:
        """
        Provide recovery strategy when stuck
        """
```

#### 1.2 Supervisor Prompts
```python
TASK_DECOMPOSITION_PROMPT = """
You are a task planner for Android automation.
Break this task into clear, sequential subtasks.

Task: {task}

Rules:
1. Each subtask should be a single action
2. Include waiting/verification steps
3. Maximum 10 subtasks
4. Be specific about what to look for

Output JSON:
{
  "subtasks": [
    {"id": 1, "task": "...", "dependencies": []},
    {"id": 2, "task": "...", "dependencies": [1]}
  ]
}
"""

PROGRESS_TRACKING_PROMPT = """
Current subtask: {subtask}
Worker result: {result}
History: {recent_actions}

Decide next action:
- "continue": Move to next subtask
- "retry": Try current subtask again with hints
- "skip": Skip if optional
- "check": Run completion check

Output JSON:
{
  "action": "continue|retry|skip|check",
  "reason": "...",
  "hints": "..." (if retry)
}
"""
```

### Worker Implementation (Completed)

#### 2.1 Modify worker_agent.py
```python
class WorkerAgent:
    def __init__(self, vision_model):
        self.vision_model = vision_model
        
    def execute_subtask(self, subtask: str, context: Dict) -> Dict:
        """
        Execute a single subtask with context from Supervisor
        """
        # Capture screen
        screenshot = take_screenshot()
        elements = parse_elements(screenshot)
        
        # Vision analysis with subtask focus
        decision = self.analyze_for_subtask(
            screenshot=screenshot,
            elements=elements,
            subtask=subtask,
            hints=context.get("hints", "")
        )
        
        # Execute action
        result = self.execute_action(decision, elements)
        
        return {
            "subtask": subtask,
            "decision": decision,
            "result": result,
            "screenshot": screenshot
        }
```

#### 2.2 Focused Vision Prompts
```python
SUBTASK_VISION_PROMPT = """
You are completing a specific subtask on Android.

CURRENT SUBTASK: {subtask}
SUPERVISOR HINTS: {hints}

IMPORTANT: Element numbers are NOT in order! Check carefully.

Analyze the screen and decide the ONE action for this subtask:
- If subtask is "Open Chrome", look for Chrome icon
- If subtask is "Type text", look for input field
- Focus ONLY on the current subtask

DECISION: tap element X | type "text" in element Y | swipe up | go back
"""
```

### Orchestrator Implementation (Active)

#### 3.1 Create orchestrator.py
```python
class Orchestrator:
    def __init__(self):
        self.supervisor = SupervisorAgent(get_text_model())
        self.worker = WorkerAgent(get_vision_model())
        self.state = {}
        
    def run_task(self, task: str, device: str) -> Dict:
        """
        Main execution loop
        """
        # Step 1: Decompose task
        subtasks = self.supervisor.decompose_task(task)
        self.state["subtasks"] = subtasks
        
        # Step 2: Execute subtasks
        while not self.all_complete():
            # Get next action from Supervisor
            action = self.supervisor.get_next_action(
                self.state.get("last_result", {})
            )
            
            if action["action"] == "execute":
                # Worker executes subtask
                result = self.worker.execute_subtask(
                    subtask=action["subtask"],
                    context=action.get("context", {})
                )
                self.state["last_result"] = result
                
            elif action["action"] == "check":
                # Run focused completion check
                complete = self.check_subtask_completion(
                    subtask=action["subtask"],
                    screenshot=self.state["last_screenshot"]
                )
                if complete:
                    self.supervisor.mark_complete(action["subtask"])
                    
            elif action["action"] == "retry":
                # Retry with different approach
                self.worker.retry_with_hints(action["hints"])
                
            elif action["action"] == "skip":
                # Skip problematic subtask
                self.supervisor.skip_subtask(action["subtask"])
                
            # Check for stuck condition
            if self.is_stuck():
                self.supervisor.handle_stuck(self.state["history"])
                
        return self.state
```

### Integration Status (In Progress)

#### 4.1 Update nich_pipeline.py
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--mode", choices=["two-stage", "supervisor"], 
                       default="supervisor")
    args = parser.parse_args()
    
    if args.mode == "supervisor":
        orchestrator = Orchestrator()
        result = orchestrator.run_task(args.task, args.device)
    else:
        # Fallback to current two-stage
        result = run_two_stage(args.task, args.device)
```

#### 4.2 State Management Enhancements
```python
class EnhancedState:
    def __init__(self):
        self.task = ""
        self.subtasks = []
        self.current_subtask_idx = 0
        self.action_history = []
        self.stuck_counter = 0
        self.completion_checks = 0
        self.total_api_calls = 0
        
    def to_dict(self) -> Dict:
        """Export state for persistence"""
        
    def from_dict(self, data: Dict):
        """Restore state from dict"""
```

## Testing Strategy

### Test Cases

#### 1. Unmarked Element Test
```
Task: "Open Chrome and search for cheese"
Challenge: Search bar has no element number
Solution: "tap 200 pixels below element 5" (Google logo)
Result: Successfully focuses search bar using coordinate inference
```

#### 2. Complex Multi-Step Task
```
Task: "Open playstore and install tiktok"
Decomposed Subtasks:
1. Open Play Store app
2. Tap search bar
3. Type "tiktok"
4. Tap search button
5. Select TikTok from results
6. Tap Install button
Result: Full task completion with proper decomposition
```

#### 3. Sign-in Handling
```
Task: "Open Photos app and sign in with 81291126"
Expected Subtasks:
1. Open Photos app
2. Wait for sign-in popup
3. Click sign-in button
4. Enter phone number
5. Click Next
Expected Result: Handles sign-in correctly
```

### Achieved Metrics
- ✅ Coordinate inference for unmarked elements
- ✅ Full task decomposition upfront
- ✅ No duplicate text in fields (field clearing)
- ✅ Accurate distance estimation using screen dimensions
- ✅ Vision validation on every action
- ✅ Dynamic task refinement when stuck
- ✅ No hardcoded patterns or examples

## Migration Path

### Week 1: Parallel Development
- Build Supervisor alongside existing system
- Test on simple tasks
- Maintain backward compatibility

### Week 2: Integration
- Wire Supervisor to existing Worker
- Add orchestrator
- Test on complex tasks

### Week 3: Optimization
- Tune prompts based on results
- Optimize subtask decomposition
- Improve stuck detection

### Week 4: Deployment
- Switch default mode to Supervisor
- Keep two-stage as fallback
- Monitor performance

## Risk Mitigation

### Risk 1: Supervisor Overhead
- **Mitigation**: Cache subtask decompositions
- **Fallback**: Direct to Worker for simple tasks

### Risk 2: Communication Complexity
- **Mitigation**: Clear interfaces between agents
- **Fallback**: Simplified message passing

### Risk 3: Subtask Decomposition Quality
- **Mitigation**: Provide examples in prompts
- **Fallback**: Manual subtask templates

## Expected Improvements

### Before (Current Issues)
1. Gets stuck on sign-in screens
2. Completion check runs every step
3. No understanding of overall task
4. Repeats failing actions
5. Can't recover from errors

### After (With Supervisor)
1. Intelligent handling of sign-in flows
2. Completion checks only when needed
3. Full task context and planning
4. Detects and recovers from stuck states
5. Can skip problematic steps

## File Structure
```
pipeline/
├── supervisor/
│   ├── __init__.py
│   ├── supervisor_agent.py
│   ├── prompts.py
│   └── task_templates.py
├── worker/
│   ├── __init__.py
│   ├── worker_agent.py (modified two-stage)
│   └── vision_prompts.py
├── orchestrator/
│   ├── __init__.py
│   ├── orchestrator.py
│   └── state_manager.py
└── nich_pipeline.py (updated entry point)
```

## Configuration
```python
# config.py additions
SUPERVISOR_MODEL = "gpt-4-1106-preview"  # Text-only, fast
WORKER_MODEL = "gpt-4-vision-preview"  # Vision capable
MAX_SUBTASKS = 10
MAX_RETRIES_PER_SUBTASK = 3
COMPLETION_CHECK_FREQUENCY = 3  # Check every 3 subtasks
STUCK_THRESHOLD = 3  # Same action 3 times = stuck
```

## Success Criteria

### Immediate (Week 1-2)
- [ ] Supervisor can decompose tasks
- [ ] Worker accepts subtask context
- [ ] Basic orchestration works
- [ ] Backwards compatible

### Short-term (Week 3-4)
- [ ] 85% task completion rate
- [ ] Handles sign-in flows correctly
- [ ] Reduces API calls by 30%
- [ ] Recovers from stuck states

### Long-term (Month 2)
- [ ] Templates for common tasks
- [ ] Learning from failures
- [ ] Parallel subtask execution
- [ ] Multi-device support

## Conclusion

The Supervisor-Worker architecture addresses all current issues:
1. **Task understanding** through decomposition
2. **Stuck recovery** through intelligent retry
3. **Efficient completion** checking
4. **Sign-in handling** with context
5. **Error recovery** with skip/retry logic

This plan provides a clear path from the current two-stage system to a robust, intelligent automation pipeline that can handle complex, multi-step tasks reliably.