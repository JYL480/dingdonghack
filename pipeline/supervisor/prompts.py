"""
Prompts for the Supervisor Agent
"""

TASK_DECOMPOSITION_PROMPT = """You are a task planner for Android automation.
Break this task into clear, sequential subtasks.

Task: {task}

Rules:
1. Each subtask should be a single, specific action
2. Be LITERAL - if user says "click sign in", that means click a sign-in button, not navigate elsewhere
3. Maximum 10 subtasks
4. Don't add unnecessary steps - focus on what the user explicitly asked
5. If user mentions typing into a specific field, that's the goal

IMPORTANT: 
- If task mentions "sign in", user wants to use the sign-in page, not bypass it
- If task mentions typing a number/text into a field, that's a key goal
- Don't add navigation steps unless explicitly mentioned

Examples:
- "Open Chrome, click sign in and type 81291126" means:
  1. Open Chrome
  2. Handle any setup screens to get to browser
  3. Click sign in button/link
  4. Type 81291126 in the email/phone field
  5. Click Next/Submit

Output ONLY valid JSON in this format:
{{
  "subtasks": [
    {{"id": 1, "task": "Find and tap Chrome app icon", "type": "action"}},
    {{"id": 2, "task": "Handle Chrome setup if needed", "type": "conditional"}},
    {{"id": 3, "task": "Look for and tap sign in button", "type": "action"}},
    {{"id": 4, "task": "Type given text in email/phone field", "type": "input"}},
    {{"id": 5, "task": "Tap Next or Submit button", "type": "action"}}
  ]
}}"""

PROGRESS_TRACKING_PROMPT = """You are managing task execution progress.

Current subtask: {subtask}
Worker result: {result}
Recent history: {recent_actions}
Stuck counter: {stuck_counter}

Decide the next action based on the worker's result:
- "continue": Subtask completed successfully, move to next
- "retry": Try again with different approach (provide hints)
- "skip": Skip if optional or problematic
- "check": Run completion check for this subtask

Consider:
- If worker reports success, usually continue
- If same action failed 3+ times, consider skip or different approach
- If worker seems stuck, provide specific hints

Output ONLY valid JSON:
{{
  "action": "continue|retry|skip|check",
  "reason": "Brief explanation",
  "hints": "Specific guidance for retry (if applicable)",
  "confidence": "high|medium|low"
}}"""

STUCK_RECOVERY_PROMPT = """The worker is stuck repeating the same actions.

Recent actions:
{action_history}

Current subtask: {subtask}
Failed attempts: {attempts}

Provide recovery strategy:
1. Alternative approach to try
2. Whether to skip this subtask
3. Specific hints for the worker

Output ONLY valid JSON:
{{
  "strategy": "retry|skip|alternative",
  "hints": "Specific guidance",
  "alternative_action": "If different approach needed",
  "skip_reason": "If skipping, why"
}}"""

COMPLETION_CHECK_PROMPT = """Determine if the subtask is complete.

Subtask: {subtask}
Evidence from worker: {evidence}
Screenshot available: {has_screenshot}

Be strict - only mark complete if fully done.

Output ONLY valid JSON:
{{
  "complete": true|false,
  "reason": "Why complete or not",
  "missing": "What's still needed (if not complete)"
}}"""