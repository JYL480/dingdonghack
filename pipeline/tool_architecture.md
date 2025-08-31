# Tool Architecture - Simplified Pipeline

## 🎯 Philosophy: Let AI Do What It's Good At

We've simplified the tool architecture by removing unnecessary complexity and letting the vision model understand context naturally.

## 🛠️ Current Tool Stack

```
┌─────────────────────────────────────────────────────┐
│                   EXTERNAL APIs                      │
│  • OmniParser (UI element extraction)                │
│  • OpenAI GPT-4o-mini (vision + text)                │
└─────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────┐
│                    CORE TOOLS                        │
│  • ADB Commands (device control)                     │
│  • Screenshot Capture                                │
│  • Click Visualization                               │
└─────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────┐
│                  AGENT TOOLS                         │
│  Supervisor: Vision validation, stuck detection      │
│  Worker: Two-stage navigation, action execution      │
└─────────────────────────────────────────────────────┘
```

## 📦 Core Tools (What We Actually Use)

### 1. **OmniParser** (`nich_utils.py::call_omniparser_api`)
```python
def call_omniparser_api(screenshot_path: str) -> Dict:
    """Parse UI elements from screenshot"""
    # Returns: {"elements": [...], "labeled_image": "path"}
```
- **Purpose**: Extract clickable elements with numbers
- **Used by**: Every screen analysis
- **No fallback**: Critical dependency

### 2. **ADB Execution** (`nich_utils.py::execute_adb`)
```python
def execute_adb(adb_command: str) -> str:
    """Execute ADB command with path correction"""
    # Auto-fixes Windows path: C:/Users/.../adb.exe
```
- **Purpose**: All device interactions
- **Actions**: tap, swipe, type, back, home
- **Smart**: Auto-corrects ADB path on Windows

### 3. **Screenshot Tool** (`nich_utils.py::take_screenshot_adb`)
```python
def take_screenshot_adb(device: str, filename: str) -> str:
    """Capture current screen"""
    # Saves to: log/screenshots/run_*/
```
- **Purpose**: Capture screen state
- **Used by**: Both Supervisor and Worker
- **Organized**: Timestamped folders per task

### 4. **Click Visualization** (`nich_utils.py::visualize_click`)
```python
def visualize_click(screenshot_path: str, x: int, y: int) -> str:
    """Draw red marker where clicked"""
    # Saves as: *_clicked.png
```
- **Purpose**: Debug where system clicked
- **Visual**: Red dot at click location

## 🤖 Agent-Specific Tools

### Supervisor Tools (Vision-Based)

#### Task Decomposition
```python
def decompose_task(self, task: str, screenshot: str = None):
    # Extract specific values (regex for numbers)
    # Create subtasks with specific_value field
    # Use initial screenshot for context
```

#### Visual Validation (Every Action)
```python
def verify_subtask_completion(self, subtask: Dict, screenshot: str) -> bool:
    # AI looks at screenshot with context
    # Understands screen progression
    # Returns: True/False + correction if needed
    # No hardcoded patterns - pure AI understanding
```

#### Stuck Detection
```python
def _is_stuck(self) -> bool:
    # Check if typing same value repeatedly
    # Check if same actions repeated
    # Max 3 retries then mark failed
```

### Worker Tools (Execution)

#### Two-Stage Navigation
```python
def navigate_to_task(task: str, step_num: int, max_steps: int):
    # Stage 1: Vision decides what to do
    # Stage 2: Clear field if needed, then execute
    # Waits for screen transitions
    # Return: decision, result, screenshot
```

#### Specific Value Handling
```python
# Gets specific_value from supervisor
if specific_value and "type" in task:
    task = f"Type '{specific_value}' in the field"
    # Types actual value, not examples
```

## 🚫 What We Removed (And Why)

### ❌ Complex Screen Detection
**Before:**
```python
indicators = {
    "home": ["Phone", "Messages", "Chrome"],
    "chrome": ["address bar", "tabs"],
    # ... 20+ hardcoded patterns
}
```
**Now:**
```python
return "current_screen"  # Supervisor figures it out
```
**Why**: The supervisor can SEE the screen. It doesn't need our hints.

### ❌ Hardcoded App Patterns
**Before:** Try to detect every app by keywords
**Now:** Let the vision model understand
**Why**: AI is better at understanding context than our patterns

### ❌ Screen Context Tracking
**Before:** Complex state management for screen transitions
**Now:** Simple - supervisor knows where we are
**Why**: Unnecessary complexity that wasn't helping

## 🔧 Tool Configuration

### Environment Variables
```python
# From config.py
LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKEN = 4096
Omni_URI = "https://teen-alt-clocks-athletes.trycloudflare.com"
```

### Tool Timeouts
```python
OMNIPARSER_TIMEOUT = 30  # API call
ADB_TIMEOUT = 10         # Device action  
SCREENSHOT_TIMEOUT = 5   # Quick operation
VISION_TIMEOUT = 30      # Model inference
```

## 🎯 Tool Usage Flow

### 1. Screen Analysis
```
Screenshot → OmniParser → Labeled Image → Vision Model
     ↓           ↓              ↓              ↓
   Image     Elements      Numbers         Decision
```

### 2. Action Execution
```
Decision → Parse → Get Coordinates → ADB Command
    ↓         ↓           ↓              ↓
"tap 7"   Extract    (x, y)          adb tap
```

### 3. Validation
```
Action → Screenshot → Supervisor Vision → Result
   ↓          ↓              ↓              ↓
Execute    Capture       Validate       Pass/Fail
```

## 🚀 Adding New Tools

### Step 1: Identify Need
Ask: Can the supervisor's vision handle this?
- If yes: Don't add a tool
- If no: Continue

### Step 2: Create Tool
```python
def your_tool(param1: type) -> ReturnType:
    """What it does"""
    # Simple, single purpose
    # Return success/failure + data
```

### Step 3: Integrate
```python
# In appropriate agent
if need_special_handling:
    result = your_tool(params)
```

## 💡 Best Practices

### DO:
- ✅ Let vision model understand context
- ✅ Provide element coordinates for better estimation
- ✅ Use screen dimensions as reference
- ✅ Decompose tasks fully upfront (no wait steps)
- ✅ Clear text fields before typing
- ✅ Log coordinate calculations

### DON'T:
- ❌ Hardcode app-specific patterns
- ❌ Duplicate what vision can do
- ❌ Create complex detection logic
- ❌ Try to be smarter than AI
- ❌ Add tools "just in case"

## 🐛 Common Issues

### Issue: "Element not found"
**Solution**: OmniParser returned empty - retry or check screenshot

### Issue: "ADB command failed"
**Solution**: Path auto-correction handles most cases

### Issue: "Duplicate text in fields"
**Solution**: Clear field before typing with clear_text action

### Issue: "Wrong value in wrong field"
**Solution**: AI recognizes screen context and implicit task completion

### Issue: "Stuck in retry loop"
**Solution**: Limited to 3 retries, then marks failed and moves on

## 📊 Tool Performance

| Tool | Latency | Reliability | Critical? |
|------|---------|-------------|-----------|
| OmniParser | ~2-3s | 95% | Yes |
| ADB Commands | <1s | 99% | Yes |
| Screenshot | ~1s | 99% | Yes |
| Vision Model | ~2-4s | 90% | Yes |

## 🔮 Future Considerations

### Keep Simple
- Voice commands? Let supervisor decide when needed
- Gesture recording? Only if supervisor can't handle
- Network monitoring? Only if task requires

### Remember
The supervisor with vision is powerful. Before adding tools, ask:
**"Can the supervisor just look and understand?"**

Usually, the answer is yes.

## 📝 Tool Documentation Template

When adding a new tool (rarely needed):

```python
def tool_name(required_param: type) -> Dict[str, Any]:
    """
    One line description.
    
    Args:
        required_param: What it is
        
    Returns:
        {"success": bool, "data": Any, "error": str}
    """
    try:
        # Do one thing well
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## 🎯 Key Takeaway

**Let AI do the thinking, not hardcoded rules.**

- No hardcoded screen detection patterns
- No hardcoded transition logic
- AI understands context through vision
- Clear fields before typing to prevent issues
- Wait for screens to load before validation

Before adding anything, ask: "Can the AI figure this out on its own?"