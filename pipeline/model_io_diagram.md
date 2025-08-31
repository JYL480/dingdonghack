# Model I/O Diagram - Android Automation Pipeline

## 🎯 Current System Overview

An intelligent Android automation pipeline that uses vision-capable AI models to understand screens and execute complex multi-step tasks through a Supervisor-Worker architecture.

```
┌─────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                             │
│                  (orchestrator.py)                           │
│                                                              │
│  • Main execution loop                                       │
│  • Coordinates Supervisor ↔ Worker communication             │
│  • Handles timeouts and max steps                           │
│  • Creates task folders for screenshots                      │
└────────────┬───────────────────────────┬────────────────────┘
             │                           │
             ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│    SUPERVISOR V2 👁️      │  │       WORKER 🤖             │
│ (supervisor_agent_v2.py) │  │    (worker_agent.py)         │
│                          │  │                              │
│ WITH VISION:             │  │ EXECUTES:                    │
│ • Sees screenshots       │  │ • Two-stage navigation       │
│ • Creates subtasks       │  │ • Stage 1: Vision (what)     │
│ • Validates every action │  │ • Stage 2: Action (do it)    │
│ • Provides corrections   │  │ • Takes screenshots          │
│ • Detects stuck states   │  │ • Executes ADB commands      │
└──────────────────────────┘  └──────────────────────────────┘
             │                           │
             └───────────┬───────────────┘
                         │
                    [gpt-4o-mini]
                    Vision Model
                         │
                    ┌────▼────┐
                    │ ANDROID │
                    │ DEVICE  │
                    └─────────┘
```

## 📊 Enhanced Data Flow

### 1. Task Decomposition (Full Upfront)
```
User: "Open playstore and install tiktok"
                    ↓
Supervisor: [LLM decomposes entire task]
   → Subtask 1: Open Play Store app
   → Subtask 2: Tap search bar
   → Subtask 3: Type "tiktok"
   → Subtask 4: Tap search button
   → Subtask 5: Select TikTok from results
   → Subtask 6: Tap Install button
```

### 2. Execution Loop
```
For each subtask:
   Worker executes → Screenshot → Supervisor validates
      ↓                              ↓
   If correct: ✓ Next              If wrong: Correction
```

### 3. Coordinate Extrapolation (New!)
```
Worker Vision: "Search bar has no element number"
               "Element 5 (Google logo) at (352, 234)"
               "Search bar is ~200 pixels below"
                    ↓
Worker Decision: "tap 200 pixels below element 5"
                    ↓
System: Calculates (352, 434) and taps there
```

### 4. Validation (Every Action)
```
Worker: "Tapped coordinate (352, 434)"
Supervisor: [Validates with screenshot]
   → ✓ "Search bar is now focused" OR
   → ✗ "Wrong location, try 250 pixels below"
```

## 🔧 Key Components & Changes

### Orchestrator (`orchestrator/orchestrator.py`)
**No changes** - Still the main coordinator
- Manages execution flow
- Tracks progress (1/3 subtasks complete, etc.)
- Handles timeouts (5 min default)

### Supervisor V2 (`supervisor/supervisor_agent_v2.py`)
**Key Features:**
- ✅ Full task decomposition via LLM (breaks complex tasks into atomic steps)
- ✅ Vision-based validation of every worker action
- ✅ Dynamic task refinement based on screen state
- ✅ Intelligent stuck detection (max 3 retries then skip)
- ✅ Context-aware corrections without hardcoding
- ✅ Tracks original task throughout execution
- ✅ Can mark subtasks as failed and continue

**Key Methods:**
```python
decompose_task(task, screenshot)  # Creates subtasks with specific values
verify_subtask_completion(subtask, screenshot)  # Validates with vision
analyze_stuck_situation(feedback, screenshot)  # Understands why stuck
_is_stuck()  # Detects retry loops (enhanced)
```

### Worker (`worker/worker_agent.py`)
**Key Features:**
- ✅ Two-stage navigation (Vision → Action)
- ✅ Coordinate extrapolation for unmarked elements
- ✅ Clears text fields before typing
- ✅ Uses actual element coordinates for estimation
- ✅ Types full words/phrases at once
- ✅ Screen-aware distance calculations

**Key Flow:**
```python
# Gets subtask with specific_value: "81291126"
# Types actual value, not "your_email@example.com"
```

### Two-Stage Navigator (`utils/nich_utils_twostage.py`)
**Core Capabilities:**
- ✅ Enhanced vision prompts with element coordinates
- ✅ Coordinate inference for unmarked UI elements
- ✅ Smart decision parsing (preserves coordinate instructions)
- ✅ Screen dimension-based distance estimation
- ✅ Supports: tap between elements, pixel offsets, screen positions
- ✅ No hardcoded patterns or examples

## 🚀 What We Have Now

### Strengths
1. **AI-Driven Understanding** - Supervisor sees and understands screens
2. **Specific Value Handling** - Types "81291126" not generic examples
3. **Robust Stuck Detection** - Stops after 3 retries, marks as failed
4. **Immediate Validation** - Every action is checked visually
5. **Simple Architecture** - No duplicate screen detection logic

### How It Works
```python
# Example Flow
Task: "Type 81291126 in the field"
→ Supervisor extracts: specific_value = "81291126"
→ Worker receives: subtask with specific_value
→ Worker types: "81291126" (not example text)
→ Supervisor validates: Checks if typed correctly
→ If error: Supervisor detects stuck after 3 tries
→ Marks failed and moves to next subtask
```

### Error Handling
```python
# Stuck Detection
if typing same value 3 times:
    mark subtask as "failed"
    move to next subtask
    log: "Unable to complete after 3 attempts"

# Screen Transition Delays
time.sleep(3)  # Wait for screens to fully load before validation

# Field Clearing
clear_text()  # Clear existing text before typing new values
```

## 🎯 Design Philosophy

**Before:** Try to predict what screen we're on with patterns
**Now:** Let the AI see and understand

**Before:** Complex screen detection with hardcoded apps
**Now:** Supervisor looks at screenshot and knows

**Before:** Infinite retry loops
**Now:** Max 3 retries then skip

**Before:** Generic example values
**Now:** Actual requested values

## 📁 File Structure (Current)
```
pipeline/
├── orchestrator/
│   ├── orchestrator.py         # Main loop (unchanged)
│   └── orchestrator_async.py   # Async version
├── supervisor/
│   └── supervisor_agent_v2.py  # Vision supervisor (enhanced)
├── worker/
│   └── worker_agent.py         # Task executor (enhanced)
├── utils/
│   ├── nich_utils.py           # Core utilities
│   └── nich_utils_twostage.py  # Two-stage nav (simplified)
└── nich_pipeline.py            # Entry point
```

## 🔄 Recent Improvements

1. **AI-Driven Validation**
   - No hardcoded screen transitions
   - Vision model understands context
   - Recognizes implicit task completion

2. **Screen Transition Handling**
   - Added delays for screen loading
   - Fresh screenshots for validation
   - Prevents wrong-field typing

3. **Field Management**
   - Clear text before typing
   - Prevents duplicate text
   - Proper value validation

4. **Validation on Every Action**
   - Not just when stuck
   - Immediate correction
   - Better success rate

## 💡 Key Insight

**The supervisor with vision doesn't need our help to understand screens.**

We were duplicating effort - trying to detect screens with patterns when the AI can just look and understand. Now the system is simpler and more robust.