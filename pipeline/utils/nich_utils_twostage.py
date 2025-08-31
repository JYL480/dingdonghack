#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-Stage Navigation System with Enhanced Vision
Stage 1: Vision model analyzes screen with reasoning and context
Stage 2: Direct Python execution with precise coordinates
"""

import json
import base64
import os
import time
import re
from typing import Dict, Any, List, Optional
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import config

# Enhanced Vision Prompt Template
ENHANCED_VISION_PROMPT = """You are an Android navigation assistant with vision capabilities.

CURRENT CONTEXT:
- Current Screen: {current_screen_type}
- Current App: {current_app_name}
- Previous Action: {last_action}
- Previous Result: {last_result}
- Task Progress: {progress_summary}

YOUR TASK: {original_task}

ACTIONS TAKEN SO FAR:
{action_history}

⚠️ RULER ALERT: If you see COLORED RULER LINES with DISTANCE LABELS on the screenshot:
- These are MEASUREMENTS to help you tap unmarked elements!
- READ THE LABELS: "100px RIGHT", "150px BELOW", etc.
- USE THESE EXACT MEASUREMENTS in your decision!
- Example: Chrome icon at "100px RIGHT" marker → "tap 100 pixels right of element 0"

IMPORTANT: Element numbers are NOT sequential or in order! Element 3 might be at the top while element 10 is at the bottom.
Always look at the actual number labels on the screenshot, not their position.

⚠️ CRITICAL POSITIONING WARNING:
- ALWAYS check element COORDINATES (shown below) before using "between"!
- If Y coordinates differ by >500px, elements are FAR APART VERTICALLY - DON'T use "between"!
- If X coordinates differ by >500px, elements are FAR APART HORIZONTALLY - DON'T use "between"!
- "Between" should ONLY be used for ADJACENT elements (like buttons in a row)
- For Chrome address bar: Usually just tap element 1 directly, or use coordinates!

ANALYZE THE SCREEN AND PROVIDE:

1. OBSERVATION: Describe what you see
   - Identify key apps by their icons (Chrome=multicolor circle, Settings=gear, etc.)
   - Note which screen you're on (home/app/settings)
   - List relevant numbered elements WITH their actual numbers from the image

2. OPTIONS: Provide 2-3 possible actions with reasoning
   Format: 
   a) [action] - [why this helps] - [what you expect to happen]
   b) [action] - [why this helps] - [what you expect to happen]
   
3. DECISION: Your chosen action
   
   CRITICAL RULE: 
   ⚠️ IF THE TARGET HAS NO ELEMENT NUMBER, YOU MUST USE COORDINATE INFERENCE!
   ⚠️ DO NOT tap a nearby element hoping it will work!
   
   FOR NUMBERED ELEMENTS:
   - tap element X (ONLY if X is the actual target)
   - type "text" in element X (ONLY if X is the input field)
   
   FOR TYPING TEXT:
   - If keyboard is visible and field is focused: type "full text at once"
   - Do NOT tap individual keyboard keys one by one
   - Example: type "cheese" (not tap c, tap h, tap e...)
   
   FOR UNMARKED ELEMENTS (NO NUMBER ON TARGET):
   USE THESE COORDINATE-BASED FORMATS:
   
   1. EDGE POSITIONING (target adjacent to element):
      - "tap just below element X" (at bottom edge + small padding)
      - "tap directly above element X" (at top edge)
      - "tap to the right of element X" (at right edge)
   
   2. PIXEL DISTANCE (specific offset from element edge):
      - "tap 150 pixels below element X" (from bottom edge of bbox)
      - "tap 200 pixels right of element X" (from right edge of bbox)
      Note: Distance starts from element's EDGE, not center
   
   3. BETWEEN ELEMENTS (ONLY for nearby elements!):
      - "tap between element X and element Y" (calculates midpoint)
      - ⚠️ ONLY use this if elements are CLOSE TOGETHER (adjacent buttons, side-by-side icons)
      - NEVER use "between" for elements far apart vertically or on opposite sides!
   
   4. ABSOLUTE POSITION (when you know exact coordinates):
      - "tap coordinate 540,1200" (direct screen position)
   
   HOW TO CALCULATE:
   - Use provided element coordinates and bounding boxes
   - Element bbox gives you [x1,y1,x2,y2] corners
   - Calculate distances from edges, not centers
   
   ⚠️ CRITICAL: IF YOU SEE RULER LINES ON THE SCREENSHOT:
   - YOU MUST USE THE RULER MEASUREMENTS - THEY ARE THERE TO HELP YOU!
   - RED ruler = distance to the RIGHT (look for "100px RIGHT" labels)
   - MAGENTA ruler = distance to the LEFT (look for "100px LEFT" labels)
   - GREEN ruler = distance BELOW (look for "100px BELOW" labels)
   - ORANGE ruler = distance ABOVE (look for "100px ABOVE" labels)
   - READ THE EXACT PIXEL DISTANCES FROM THE LABELS!
   - Example: If Chrome is at the "100px RIGHT" marker from element 0 → "tap 100 pixels right of element 0"
   - DO NOT use vague terms like "just right of" when rulers show exact distances!
   
   Example: If element 5 is at (350, 230) and your target is below it:
   → Calculate: target appears ~200 pixels down → "tap 200 pixels below element 5"
   
   Example: If two elements are at (300, 500) and (700, 500):
   → Target between them → "tap between element X and element Y"
   
   SCREEN REFERENCE (typical Android):
   - Width: 1080 pixels
   - Height: 2400 pixels
   - Use these to estimate distances
   
   ⚠️ NEVER tap the wrong element just because it's nearby!
   ⚠️ ALWAYS use coordinate inference for unmarked targets!
   
4. REASONING: Why you chose this over other options
   
5. EXPECTATION: What should happen after this action

RESPONSE FORMAT:
OBSERVATION: [what you see]
OPTIONS:
  a) [option 1]
  b) [option 2]
  c) [option 3]
DECISION: [your choice]
REASONING: [why this choice]
EXPECTATION: [what should happen]

EXAMPLES - USING COORDINATES:

SCENARIO: Target has no element number but you see nearby elements
- Check coordinates of nearby elements
- Calculate distance to target
- Use: "tap [distance] pixels [direction] of element X"

SCENARIO: Target is between two elements
- Note both element positions
- Use: "tap between element X and element Y"

SCENARIO: Target in specific screen area
- Use screen dimensions (width × height)
- Calculate position
- Use: "tap coordinate X,Y"

REMEMBER: Never tap the wrong element just because it's nearby!

APP IDENTIFICATION TIPS:
- Chrome: Multicolored circular icon (red, green, yellow, blue) - NOT the Google 'G'
- Settings: Gray gear/cog icon
- Google Search: Multicolored 'G' logo or search bar
- Play Store: Multicolored triangle play button
- Gmail: Red and white envelope/M
- YouTube: Red rectangle with white play button
- Messages: Blue/green speech bubble
- Phone: Green phone handset
- Camera: Camera lens icon
- Photos: Multicolored pinwheel or flower icon

CRITICAL: Element numbers are assigned by OmniParser and are NOT in visual order!
Element 3 could be anywhere on screen. Always check the actual number label.

IMPORTANT CHROME SETUP:
- If Chrome shows "Add account to device" or sign-in, look for "Use without an account" or "Skip"
- Prefer skipping sign-in to quickly access browser
- If stuck on sign-in page, use "go back" or "go home" to restart

ACTION FORMATS:
- 'tap element X' (where X is the element number)
- 'swipe up' / 'swipe down' / 'swipe left' / 'swipe right'
- 'go back' (Android back button)
- 'go home' (go to home screen)
- 'type "text" in element X' (tap element X first, then type)
- 'type "text"' (type in already focused field)
- 'element between X and Y' (tap between elements)
- 'coordinate X,Y' (tap specific coordinates)

FOR TYPING:
- If you see an input field (email, phone, text box), use: 'type "your text" in element X'
- This will first tap element X to focus it, then type the text
- The keyboard will appear after tapping the input field
"""

def detect_screen_context(elements_data: List[Dict], previous_context: Optional[Dict] = None) -> str:
    """Simple screen context - let the supervisor figure out what screen we're on"""
    # Just return a basic indicator - the supervisor with vision will understand the actual context
    return "current_screen"

def format_action_history(state: Dict[str, Any]) -> str:
    """Format action history for model context"""
    
    history_text = ""
    action_history = state.get('action_history', [])
    
    # Get last 5 actions for context
    for i, action in enumerate(action_history[-5:]):
        history_text += f"Step {action['step']}: {action['action']}"
        if action.get('element_description'):
            history_text += f" ({action['element_description']})"
        history_text += f" -> {action.get('result', 'Unknown result')}"
        if not action.get('success', True):
            history_text += " [FAILED]"
        history_text += "\n"
    
    return history_text if history_text else "No actions taken yet"

def parse_vision_response(vision_response: str) -> Dict[str, Any]:
    """Parse structured vision response with reasoning"""
    
    response_parts = {
        "observation": "",
        "options": [],
        "decision": "",
        "reasoning": "",
        "expectation": ""
    }
    
    # Parse each section
    lines = vision_response.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower()
        
        if 'observation:' in line_lower:
            current_section = 'observation'
            response_parts['observation'] = line.split(':', 1)[1].strip() if ':' in line else ''
        elif 'options:' in line_lower:
            current_section = 'options'
        elif 'decision:' in line_lower:
            current_section = 'decision'
            response_parts['decision'] = line.split(':', 1)[1].strip() if ':' in line else ''
        elif 'reasoning:' in line_lower:
            current_section = 'reasoning'
            response_parts['reasoning'] = line.split(':', 1)[1].strip() if ':' in line else ''
        elif 'expectation:' in line_lower:
            current_section = 'expectation'
            response_parts['expectation'] = line.split(':', 1)[1].strip() if ':' in line else ''
        elif current_section == 'options' and line.strip() and line.strip()[0] in 'abc)':
            response_parts['options'].append(line.strip())
        elif current_section and line.strip():
            # Continue previous section
            if current_section != 'options':
                response_parts[current_section] += ' ' + line.strip()
    
    # Clean up decision - remove letter prefixes like "a)" or "b)" if present
    decision = response_parts['decision'].strip()
    if decision and decision[0] in 'abc' and ')' in decision[:3]:
        decision = decision.split(')', 1)[1].strip()
        response_parts['decision'] = decision
    
    # Normalize decision to extract element numbers
    decision_lower = decision.lower()
    
    # IMPORTANT: Check for coordinate inference patterns FIRST
    # Don't override coordinate-based decisions!
    if any(keyword in decision_lower for keyword in ['pixels', 'between element', 'coordinate', '%', 'screen position']):
        # This is a coordinate inference decision - keep it as-is
        response_parts['decision'] = decision
    # Check if this is a sequence of letter taps (typing individual keys)
    elif re.search(r"tap.*\(['\"]?\w['\"]?\)", decision_lower):
        # Extract all letters being tapped in sequence
        letters = re.findall(r"\(['\"]?(\w)['\"]?\)", decision)
        if letters:
            # Convert individual key taps to a type command
            word = ''.join(letters)
            response_parts['decision'] = f'type "{word}"'
            print(f"[PARSER] Converted individual key taps to: type \"{word}\"")
    # Handle various tap/click formats
    elif 'tap' in decision_lower or 'click' in decision_lower:
        # Only extract simple element taps (not coordinate inference)
        if 'tap element' in decision_lower or 'click element' in decision_lower:
            match = re.search(r'(tap|click)\s+element\s+(\d+)(?!\s+(pixels|and))', decision_lower)
            if match:
                response_parts['decision'] = f"tap element {match.group(2)}"
        else:
            # Look for element in parentheses like "Next button (element 50)"
            match = re.search(r'\(element\s+(\d+)\)', decision_lower)
            if match:
                response_parts['decision'] = f"tap element {match.group(1)}"
    elif 'type' in decision_lower:
        # Handle typing - extract text and element if specified
        text_match = re.search(r'["\']([^"\'\']+)["\']', decision)
        element_match = re.search(r'element\s+(\d+)', decision_lower)
        if text_match:
            text = text_match.group(1)
            if element_match:
                response_parts['decision'] = f'type "{text}" in element {element_match.group(1)}'
            else:
                response_parts['decision'] = f'type "{text}"'
    elif 'swipe' in decision_lower:
        for direction in ['up', 'down', 'left', 'right']:
            if direction in decision_lower:
                response_parts['decision'] = f"swipe {direction}"
                break
    elif 'back' in decision_lower:
        response_parts['decision'] = "go back"
    elif 'home' in decision_lower:
        response_parts['decision'] = "go home"
    
    # If no structured response, extract decision from raw text
    if not response_parts['decision'] and vision_response:
        # Look for action patterns in the response
        response_lower = vision_response.lower()
        if 'tap element' in response_lower or 'click element' in response_lower:
            match = re.search(r'(tap|click)\s+element\s+(\d+)', response_lower)
            if match:
                response_parts['decision'] = f"tap element {match.group(2)}"
        elif 'swipe' in response_lower:
            for direction in ['up', 'down', 'left', 'right']:
                if direction in response_lower:
                    response_parts['decision'] = f"swipe {direction}"
                    break
        elif 'go back' in response_lower or 'back' in response_lower:
            response_parts['decision'] = "go back"
        elif 'go home' in response_lower or 'home' in response_lower:
            response_parts['decision'] = "go home"
    
    return response_parts

def extract_element_description(parsed_response: Dict[str, Any], elements_data: List[Dict], element_id: Optional[int] = None) -> str:
    """Extract description of the element being acted upon"""
    
    if element_id is not None and 0 <= element_id < len(elements_data):
        elem = elements_data[element_id]
        if isinstance(elem, dict):
            text = elem.get('text', '')
            if text:
                return text
    
    # Try to extract from observation
    observation = parsed_response.get('observation', '')
    if observation:
        # Look for app names
        for app in ['Chrome', 'Settings', 'Gmail', 'YouTube', 'Play Store']:
            if app.lower() in observation.lower():
                return app
    
    return "UI element"

def two_stage_navigation(state: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Enhanced Two-stage navigation with reasoning and context
    Stage 1: Vision model analyzes screen with full context and reasoning
    Stage 2: Direct Python execution with precise coordinates
    """
    from .nich_utils import (
        capture_and_parse_screen, 
        get_device_size,
        smart_screen_action,
        save_action_visualization
    )
    
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 20)
    print(f"\n[STEP] Step {current_step + 1}/{max_steps}: Two-stage execution...")
    
    task = state["task"]
    device = state["device"]
    
    # Small delay for screen stability
    time.sleep(0.5)
    
    # Initialize enhanced context if first time
    if 'screen_context' not in state:
        state['screen_context'] = {
            'type': 'home',
            'app_name': None,
            'previous_screen': None,
            'previous_app': None
        }
    
    if 'action_history' not in state:
        state['action_history'] = []
    
    if 'task_progress' not in state:
        state['task_progress'] = {
            'original_task': task,
            'current_objective': 'Start task',
            'stuck_counter': 0
        }
    
    # Initialize message history if first time
    if not state.get("messages"):
        state["messages"] = []
    
    # Capture screen if needed
    if not state.get("current_page") or not state["current_page"].get("screenshot"):
        state = capture_and_parse_screen(state)
        if not state["current_page"]["screenshot"]:
            state["execution_status"] = "error"
            print("[ERROR] Unable to capture or parse screen")
            return state
    
    # Get screen info
    screenshot_path = state["current_page"]["screenshot"]
    elements_json_path = state["current_page"]["elements_json"]
    device_size = get_device_size(device)
    
    # Load elements JSON for Stage 2
    with open(elements_json_path, "r", encoding="utf-8") as f:
        elements_data = json.load(f)
    
    # Find labeled image
    labeled_image_path = screenshot_path
    processed_dir = os.path.join(os.path.dirname(screenshot_path), "processed_images")
    if os.path.exists(processed_dir):
        base_name = os.path.basename(screenshot_path)
        labeled_path = os.path.join(processed_dir, f"labeled_{base_name}")
        if os.path.exists(labeled_path):
            labeled_image_path = labeled_path
    
    # Check if task might need coordinate inference
    needs_ruler = False
    ruler_element_ids = []
    
    # Detect if task needs relative positioning
    task_lower = task.lower()
    
    # Only show rulers if there's actual relative positioning needed
    # Check for relative/spatial keywords that indicate unmarked elements
    relative_keywords = [
        # Relative positioning terms
        "near", "next", "beside", "between", "around", "close", "adjacent",
        "above", "below", "under", "over", "beneath", 
        "left of", "right of", "to the left", "to the right",
        
        # Apps that are often unmarked (especially Chrome)
        "chrome", "browser",
        
        # UI elements that are often unmarked
        "search bar", "address bar", "url bar", "input field", "text field",
        
        # Indicators of unmarked elements
        "unmarked", "without number", "no number", "not numbered",
        
        # Screen regions that need positioning
        "corner", "edge", "side", "dock", "bottom row", "top row"
    ]
    
    # Only show rulers if relative positioning is likely needed
    if any(keyword in task_lower for keyword in relative_keywords):
        needs_ruler = True
    
    # Also check if this is a second attempt where we might need rulers
    # (e.g., if previous attempt mentioned Chrome or other unmarked elements)
    if state.get('action_history'):
        last_actions = state['action_history'][-2:] if len(state['action_history']) >= 2 else state['action_history']
        for action in last_actions:
            action_text = str(action.get('action', '')).lower() + str(action.get('observation', '')).lower()
            if any(keyword in action_text for keyword in relative_keywords):
                needs_ruler = True
                print("[RULER] Enabling rulers based on previous action history")
                break
        
        # Find multiple relevant elements to use as references
        # Prioritize elements at edges and corners for better triangulation
        important_elements = []
        
        # Look for elements at bottom (dock apps)
        for i, elem in enumerate(elements_data[:20]):
            elem_text = str(elem.get('text', '')).lower() if elem.get('text') else ''
            if 'bbox' in elem:
                bbox = elem['bbox']
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    y_pos = bbox[1]
                    # Elements at bottom of screen (dock area)
                    if y_pos > device_size['height'] * 0.85:
                        important_elements.append((i, "bottom"))
                    # Elements at top of screen
                    elif y_pos < device_size['height'] * 0.15:
                        important_elements.append((i, "top"))
                        
            # Also include any elements with relevant text
            if any(word in elem_text for word in ["play store", "chrome", "gmail", "photos", "google"]):
                if i not in [e[0] for e in important_elements]:
                    important_elements.append((i, "relevant"))
        
        # Select up to 3 diverse reference elements
        if important_elements:
            # Get elements from different regions
            bottom_elems = [e for e in important_elements if e[1] == "bottom"][:2]
            top_elems = [e for e in important_elements if e[1] == "top"][:1]
            relevant_elems = [e for e in important_elements if e[1] == "relevant"][:2]
            
            ruler_element_ids = [e[0] for e in (bottom_elems + top_elems + relevant_elems)[:3]]
            
            if not ruler_element_ids and elements_data:
                # Fallback: use first, middle, and last visible elements
                ruler_element_ids = [0, min(5, len(elements_data)-1), min(10, len(elements_data)-1)]
            
            print(f"[RULER] Using elements {ruler_element_ids} as references for coordinate help")
    
    # Create ruler-annotated screenshot if needed
    image_for_vision = labeled_image_path
    if needs_ruler and ruler_element_ids and elements_data:
        from .nich_utils import draw_ruler_on_screenshot
        
        # Draw rulers from multiple reference elements
        annotated_image = draw_ruler_on_screenshot(
            labeled_image_path,
            ruler_element_ids,  # Pass list of elements
            elements_data,
            target_description=f"Task: {task}"
        )
        if annotated_image and annotated_image != labeled_image_path:
            image_for_vision = annotated_image
            print(f"[RULER] Using annotated screenshot with pixel rulers for vision")
    
    # Load the appropriate image as base64 for vision model
    with open(image_for_vision, "rb") as f:
        image_data = f.read()
        image_data_base64 = base64.b64encode(image_data).decode("utf-8")
    
    # Simple screen context - supervisor handles understanding
    if 'screen_context' not in state:
        state['screen_context'] = {}
    state['screen_context']['type'] = 'current_screen'
    
    # Format context for prompt
    last_action = state['action_history'][-1] if state['action_history'] else {'action': 'None', 'result': 'None'}
    context_data = {
        'current_screen_type': state['screen_context']['type'],
        'current_app_name': state['screen_context'].get('app_name', 'None'),
        'last_action': last_action.get('action', 'None'),
        'last_result': last_action.get('result', 'None'),
        'progress_summary': state['task_progress'].get('current_objective', 'Start task'),
        'original_task': task,
        'action_history': format_action_history(state)
    }
    
    # ========================================
    # STAGE 1: Vision Analysis (What to click)
    # ========================================
    print("\n" + "="*60)
    print("STAGE 1: VISION ANALYSIS")
    print("="*60)
    
    # Extract key element coordinates for reference
    element_coords_text = f"\nScreen size: {device_size['width']}x{device_size['height']} pixels\n"
    element_coords_text += "Key element positions (for distance estimation):\n"
    
    # Show coordinates of first 15 elements for reference with position hints
    for i, elem in enumerate(elements_data[:15]):
        if 'coordinates' in elem:
            x, y = elem['coordinates']
            text = elem.get('text', '')[:20]  # Truncate long text
            
            # Add position hint
            pos_hint = ""
            if y < device_size['height'] * 0.15:
                pos_hint = "[TOP]"
            elif y > device_size['height'] * 0.85:
                pos_hint = "[BOTTOM]"
            
            if x < device_size['width'] * 0.2:
                pos_hint += "[LEFT]"
            elif x > device_size['width'] * 0.8:
                pos_hint += "[RIGHT]"
                
            element_coords_text += f"Element {i}: ({x}, {y}) {pos_hint} - {text}\n"
    
    # Build enhanced vision prompt
    vision_messages = [
        SystemMessage(content=ENHANCED_VISION_PROMPT.format(**context_data)),
        HumanMessage(content=f"Task: {task}\n\nWhat should I do?"),
        HumanMessage(
            content=[
                {"type": "text", "text": f"Current screen with numbered elements (numbers are NOT in order - check carefully):\n{element_coords_text}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_data_base64}",
                    "detail": "high"  # Use high detail for better element recognition
                }}
            ]
        )
    ]
    
    print(f"Task: {task}")
    print("Analyzing screen with vision model...")
    
    # Call vision model
    try:
        vision_response = model.invoke(vision_messages)
        raw_response = vision_response.content.strip()
        parsed_response = parse_vision_response(raw_response)
        
        # Log reasoning
        print("\n" + "="*60)
        print("VISION ANALYSIS")
        print("="*60)
        if parsed_response['observation']:
            print(f"OBSERVATION: {parsed_response['observation'][:200]}..." if len(parsed_response['observation']) > 200 else f"OBSERVATION: {parsed_response['observation']}")
        if parsed_response['options']:
            print("OPTIONS:")
            for opt in parsed_response['options'][:3]:
                print(f"  {opt}")
        print(f"DECISION: {parsed_response['decision']}")
        if parsed_response['reasoning']:
            print(f"REASONING: {parsed_response['reasoning'][:200]}..." if len(parsed_response['reasoning']) > 200 else f"REASONING: {parsed_response['reasoning']}")
        if parsed_response['expectation']:
            print(f"EXPECTATION: {parsed_response['expectation']}")
        
        vision_decision = parsed_response['decision'].lower() if parsed_response['decision'] else raw_response.lower()
        
    except Exception as e:
        print(f"[ERROR] Vision model failed: {str(e)}")
        state["execution_status"] = "error"
        return state
    
    # ========================================
    # STAGE 2: Action Execution (How to execute)
    # ========================================
    print("\n" + "="*60)
    print("STAGE 2: ACTION EXECUTION")
    print("="*60)
    
    # Parse vision decision and execute
    action_result = None
    action_summary = ""
    element_id_executed = None
    
    try:
        # First check if this requires coordinate extrapolation (unmarked elements)
        # Be VERY inclusive about what might need coordinate extrapolation
        coord_keywords = [
            'pixels', 'between', 'coordinate', 
            'below', 'above', 'left', 'right',
            'near', 'next', 'beside', 'close', 'adjacent',
            'directly', 'immediately', 'just',
            'to the', 'of element', 'from element',
            'under', 'over', 'beneath', 'past', 'beyond'
        ]
        
        # Check if it's NOT a simple "tap element X" command
        is_simple_tap = re.match(r'^(tap|click)\s+element\s+\d+$', vision_decision.strip(), re.IGNORECASE)
        
        if not is_simple_tap and any(keyword in vision_decision.lower() for keyword in coord_keywords):
            # Use coordinate extrapolation
            from .nich_utils import extrapolate_coordinate
            
            coordinates = extrapolate_coordinate(vision_decision, elements_data, device_size)
            if coordinates:
                x, y = coordinates
                print(f"[COORDINATE] Extrapolated position: ({x}, {y}) from instruction: {vision_decision}")
                
                # Execute tap at extrapolated coordinates
                action_result = smart_screen_action.invoke({
                    "device": device,
                    "action": "tap",
                    "coordinate": [x, y]
                })
                
                if action_result:
                    action_summary = f"Tapped at ({x}, {y})"
                    print(f"\n[ACTION EXECUTED] {action_summary}")
                    
                    # Save visualization
                    viz_result = save_action_visualization(
                        labeled_image_path, 
                        (x, y),
                        {
                            "step": current_step + 1,
                            "action": "tap",
                            "target": f"coordinate ({x}, {y})",
                            "instruction": vision_decision
                        }
                    )
                    
                    if viz_result and "clicked_image" in viz_result:
                        state["current_page"]["clicked_image"] = viz_result["clicked_image"]
            else:
                print(f"[WARNING] Could not extrapolate coordinates from: {vision_decision}")
        
        # Standard element tap
        elif "tap element" in vision_decision or "click element" in vision_decision:
            match = re.search(r'element\s+(\d+)', vision_decision)
            if match:
                element_id = int(match.group(1))
                if 0 <= element_id < len(elements_data):
                    element = elements_data[element_id]
                    # Get coordinates from element
                    if 'coordinates' in element:
                        x, y = element['coordinates']
                    elif 'bbox' in element:
                        bbox = element['bbox']
                        x = bbox['x'] + bbox['width'] // 2
                        y = bbox['y'] + bbox['height'] // 2
                    else:
                        x, y = 500, 1000  # Fallback center
                    
                    print(f"Tapping element {element_id} at ({x}, {y})")
                    
                    # Execute tap with visualization
                    action_result = smart_screen_action.invoke({
                        "device": device,
                        "action": "tap",
                        "x": x,
                        "y": y,
                        "element_id": element_id,
                        "json_path": elements_json_path,
                        "screenshot_path": screenshot_path,
                        "step": current_step
                    })
                    action_summary = f"Tapped element {element_id}"
                    element_id_executed = element_id
        
        # Parse swipe actions
        elif "swipe" in vision_decision:
            direction = None
            if "up" in vision_decision:
                direction = "up"
            elif "down" in vision_decision:
                direction = "down"
            elif "left" in vision_decision:
                direction = "left"
            elif "right" in vision_decision:
                direction = "right"
            
            if direction:
                print(f"Swiping {direction}")
                action_result = smart_screen_action.invoke({
                    "device": device,
                    "action": "swipe",
                    "direction": direction
                })
                action_summary = f"Swiped {direction}"
        
        # Parse navigation actions
        elif "back" in vision_decision:
            print("Going back")
            action_result = smart_screen_action.invoke({
                "device": device,
                "action": "back"
            })
            action_summary = "Pressed back"
            
        elif "home" in vision_decision:
            print("Going home")
            action_result = smart_screen_action.invoke({
                "device": device,
                "action": "home"
            })
            action_summary = "Went to home screen"
        
        # Parse text input - improved to handle element specification
        elif "type" in vision_decision or "text" in vision_decision:
            # Check if this is a request to type the full word/phrase
            # Look for patterns like: type "cheese" or finish typing "cheese"
            text_match = re.search(r"(?:type|typing|enter)\s+['\"]([^'\"]+)['\"]", vision_decision, re.IGNORECASE)
            if not text_match:
                # Try alternate patterns like: type cheese (without quotes)
                text_match = re.search(r"(?:type|typing|enter)\s+([\w@.]+)", vision_decision, re.IGNORECASE)
            
            if text_match:
                text_to_type = text_match.group(1)
                print(f"Text to type: {text_to_type}")
                
                # Check if element ID is specified in the decision
                element_match = re.search(r"element\s+(\d+)", vision_decision)
                if element_match:
                    # Use specified element
                    input_element_id = int(element_match.group(1))
                    print(f"Typing in element {input_element_id}")
                    
                    # First tap the element to focus it
                    if 0 <= input_element_id < len(elements_data):
                        element = elements_data[input_element_id]
                        if 'coordinates' in element:
                            x, y = element['coordinates']
                        elif 'bbox' in element:
                            bbox = element['bbox']
                            x = bbox['x'] + bbox['width'] // 2
                            y = bbox['y'] + bbox['height'] // 2
                        else:
                            x, y = 500, 500
                        
                        print(f"First tapping element {input_element_id} at ({x}, {y}) to focus")
                        # Tap to focus the input field
                        smart_screen_action.invoke({
                            "device": device,
                            "action": "tap",
                            "x": x,
                            "y": y
                        })
                        time.sleep(1)  # Wait for keyboard to appear
                    
                    # Clear any existing text first
                    print(f"Clearing existing text in element {input_element_id}")
                    
                    # Use clear_text action to clear the field
                    smart_screen_action.invoke({
                        "device": device,
                        "action": "clear_text"
                    })
                    time.sleep(0.5)
                    
                    # Now type the text (will replace selected text)
                    print(f"Typing: {text_to_type}")
                    action_result = smart_screen_action.invoke({
                        "device": device,
                        "action": "text",
                        "input_str": text_to_type
                    })
                    action_summary = f"Typed '{text_to_type}' in element {input_element_id}"
                    element_id_executed = input_element_id
                else:
                    # No element specified, just type (assumes field is already focused)
                    print(f"Typing (no element specified): {text_to_type}")
                    action_result = smart_screen_action.invoke({
                        "device": device,
                        "action": "text",
                        "input_str": text_to_type
                    })
                    action_summary = f"Typed: {text_to_type}"
        
        # Parse coordinate-based tap
        elif "coordinate" in vision_decision:
            coord_match = re.search(r'(\d+)\s*,\s*(\d+)', vision_decision)
            if coord_match:
                x = int(coord_match.group(1))
                y = int(coord_match.group(2))
                print(f"Tapping coordinates ({x}, {y})")
                action_result = smart_screen_action.invoke({
                    "device": device,
                    "action": "tap",
                    "x": x,
                    "y": y
                })
                action_summary = f"Tapped at ({x}, {y})"
        
        # Parse "between elements" case
        elif "between" in vision_decision:
            between_match = re.search(r'(\d+)\s+and\s+(\d+)', vision_decision)
            if between_match:
                elem1_id = int(between_match.group(1))
                elem2_id = int(between_match.group(2))
                if 0 <= elem1_id < len(elements_data) and 0 <= elem2_id < len(elements_data):
                    elem1 = elements_data[elem1_id]
                    elem2 = elements_data[elem2_id]
                    
                    # Calculate midpoint between elements
                    if 'coordinates' in elem1 and 'coordinates' in elem2:
                        x1, y1 = elem1['coordinates']
                        x2, y2 = elem2['coordinates']
                    else:
                        bbox1 = elem1.get('bbox', {})
                        bbox2 = elem2.get('bbox', {})
                        x1 = bbox1.get('x', 0) + bbox1.get('width', 0) // 2
                        y1 = bbox1.get('y', 0) + bbox1.get('height', 0) // 2
                        x2 = bbox2.get('x', 0) + bbox2.get('width', 0) // 2
                        y2 = bbox2.get('y', 0) + bbox2.get('height', 0) // 2
                    
                    x = (x1 + x2) // 2
                    y = (y1 + y2) // 2
                    
                    print(f"Tapping between elements {elem1_id} and {elem2_id} at ({x}, {y})")
                    action_result = smart_screen_action.invoke({
                        "device": device,
                        "action": "tap",
                        "x": x,
                        "y": y,
                        "screenshot_path": screenshot_path,
                        "step": current_step
                    })
                    action_summary = f"Tapped between elements {elem1_id} and {elem2_id}"
        
        # Parse coordinate-based instructions (pixels, percentage, direct coordinates)
        elif any(keyword in vision_decision.lower() for keyword in ["pixel", "coordinate", "%"]):
            from .nich_utils import extrapolate_coordinate, get_device_size
            
            # Get device size if not already available
            if 'device_size' not in locals():
                device_size = get_device_size(device)
            
            # Extract coordinates using the new function
            coords = extrapolate_coordinate(vision_decision, elements_data, device_size)
            
            if coords:
                x, y = coords
                print(f"Tapping extrapolated coordinate at ({x}, {y})")
                
                action_result = smart_screen_action.invoke({
                    "device": device,
                    "action": "tap",
                    "x": x,
                    "y": y,
                    "screenshot_path": screenshot_path,
                    "step": current_step
                })
                
                # Create summary based on instruction type
                if "pixel" in vision_decision.lower():
                    action_summary = f"Tapped coordinate ({x}, {y}) via pixel offset"
                elif "%" in vision_decision:
                    action_summary = f"Tapped coordinate ({x}, {y}) via percentage offset"
                else:
                    action_summary = f"Tapped coordinate ({x}, {y})"
            else:
                print(f"[WARNING] Could not extrapolate coordinates from: {vision_decision}")
                action_summary = "Failed to extrapolate coordinates"
        
    except Exception as e:
        print(f"[ERROR] Action execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Update state with results
    if action_result:
        result_message = f"{action_summary} successfully"
        success = True
    else:
        result_message = f"Failed to execute: {vision_decision}"
        success = False
        print(f"\n[WARNING] No action executed for decision: {vision_decision}")
    
    # Update action history with enhanced tracking
    state['action_history'].append({
        'step': current_step + 1,
        'action': action_summary if action_summary else vision_decision,
        'element_description': extract_element_description(parsed_response, elements_data, element_id_executed),
        'expectation': parsed_response.get('expectation', ''),
        'result': result_message,
        'success': success,
        'screen_context': state['screen_context']['type'],
        'vision_decision': vision_decision,
        'reasoning': parsed_response.get('reasoning', '')
    })
    
    # Check if stuck (same action repeated)
    if len(state['action_history']) >= 2:
        last_two = state['action_history'][-2:]
        if last_two[0]['action'] == last_two[1]['action']:
            state['task_progress']['stuck_counter'] += 1
            if state['task_progress']['stuck_counter'] > 2:
                print("\n[WARNING] Stuck on same action, may need alternative approach")
                # Update objective to try something different
                state['task_progress']['current_objective'] = "Try alternative approach"
        else:
            state['task_progress']['stuck_counter'] = 0
    
    # Update regular history for compatibility
    if "history" not in state:
        state["history"] = []
    
    state["history"].append({
        "step": current_step + 1,
        "screenshot": screenshot_path,
        "elements_json": elements_json_path,
        "vision_decision": vision_decision,
        "action": action_summary if action_summary else vision_decision,
        "status": "success" if success else "failed"
    })
    
    # Update execution status
    if action_result:
        print(f"\n[ACTION EXECUTED] {action_summary}")
        state["current_step"] += 1
        state["execution_status"] = "success"
    else:
        state["execution_status"] = "no_action"
    
    # Add decisions to message history for context
    state["messages"].append(SystemMessage(content=f"Vision: {vision_decision}"))
    state["messages"].append(SystemMessage(content=f"Action: {action_summary if action_summary else 'No action'}"))
    
    return state