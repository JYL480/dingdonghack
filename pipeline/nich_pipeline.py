#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nicholas Pipeline - Main Android navigation pipeline
Extracted from deployment.py demo.py
Runs tasks without UI, directly from command line
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain.schema.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from pipeline.utils.nich_utils import (
    # LLM Configuration
    get_model,
    
    # ADB Commands
    list_all_devices,
    get_device_size,
    
    # From tool/screen_content.py (as per document)
    take_screenshot,
    screen_element,
    smart_screen_action,
    screen_action,
    
    # From deployment.py (as per document) 
    capture_and_parse_screen,  # deployment.py:153
    fallback_to_react,  # deployment.py:670
    check_task_completion,  # deployment.py:2232
    build_workflow,  # deployment.py:2202
    
    # Main execution functions (as per document)
    execute_deployment_task,  # demo.py:1189
    run_task,  # deployment.py:1197
    run_task_with_folder,  # deployment.py:1137
    run_multiple_tasks,
    
    # From tool/coordinate_converter.py
    convert_element_to_click_coordinates,
    get_element_bounds,
    
    # From tool/click_visualizer.py
    draw_click_marker,
    save_action_visualization,
    
    # State and folder management
    create_deployment_state,
    create_task_folder,
    
    # Additional helpers
    execute_screen_action
)

# Import screen tools if available
try:
    from tool.screen_content import screen_element, take_screenshot as tool_take_screenshot
    from tool.screen_content import smart_screen_action as tool_smart_screen_action
    TOOLS_AVAILABLE = True
except ImportError:
    print("Warning: Tool imports not available, using simplified versions")
    TOOLS_AVAILABLE = False

# =====================================================
# State Definition
# =====================================================

class DeploymentState(TypedDict):
    """State machine for deployment execution with enhanced vision context"""
    # Task related
    task: str
    completed: bool
    current_step: int
    total_steps: int
    execution_status: str
    retry_count: int
    max_retries: int
    max_steps: int
    
    # Device related
    device: str
    task_folder: str
    
    # Page information
    current_page: Dict
    
    # Enhanced context fields
    screen_context: Dict[str, Any]  # Current screen/app context
    action_history: List[Dict]  # Detailed action history with reasoning
    task_progress: Dict[str, Any]  # Task progress tracking
    
    # Execution related
    matched_elements: List[Dict]
    messages: list
    history: List[Dict]
    
    # Flow control
    should_fallback: bool
    should_execute_shortcut: bool

# =====================================================
# Main React Navigation Function
# =====================================================

# Removed execute_react_navigation - now using two_stage_navigation

# =====================================================
# Workflow Node Functions
# =====================================================

def capture_screen_node(state: DeploymentState) -> DeploymentState:
    """Capture and parse screen"""
    task_folder = state.get("task_folder")
    state = capture_and_parse_screen(state)
    
    if task_folder:
        state["task_folder"] = task_folder
    
    if not state["current_page"]["screenshot"]:
        state["should_fallback"] = True
        print("[ERROR] Unable to capture screen")
    
    return state

def fallback_node(state: DeploymentState) -> DeploymentState:
    """Execute React navigation"""
    # Use two-stage navigation instead of React agent
    from pipeline.utils.nich_utils_twostage import two_stage_navigation
    model = get_model()
    state = two_stage_navigation(state, model)
    state["completed"] = False  # Let check_completion decide
    return state

def check_completion_node(state: DeploymentState) -> DeploymentState:
    """Check if task is completed"""
    model = get_model()
    state = check_task_completion(state, model)
    return state

# =====================================================
# Routing Functions
# =====================================================

def is_task_completed(state: DeploymentState) -> str:
    """Check if task is completed for routing"""
    if state.get("completed", False):
        return "end"
    return "continue"

# The build_workflow function is now imported from nich_utils

# =====================================================
# Main Execution Functions
# =====================================================

# The run_task function is now imported from nich_utils

# The run_task_with_folder function is now imported from nich_utils

# The run_multiple_tasks function is now imported from nich_utils

# =====================================================
# Command Line Interface
# =====================================================

def main():
    """Main entry point for command line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nicholas Pipeline - Android Navigation")
    parser.add_argument("task", nargs='?', default=None, help="Task description to execute")
    parser.add_argument("-d", "--device", default="emulator-5554", help="Device ID (default: emulator-5554)")
    parser.add_argument("-m", "--max-steps", type=int, default=20, help="Maximum steps (default: 20)")
    parser.add_argument("--mode", choices=["supervisor", "two-stage"], default="supervisor",
                       help="Execution mode: supervisor (new) or two-stage (legacy)")
    parser.add_argument("--async", action="store_true", help="Use async execution (experimental)")
    parser.add_argument("--list-devices", action="store_true", help="List available devices")
    parser.add_argument("--debug", action="store_true", help="Enable debug output (verbose logging)")
    
    args = parser.parse_args()
    
    # Set debug mode
    from pipeline.utils.logger import set_debug_mode
    set_debug_mode(args.debug)
    
    if args.list_devices:
        devices = list_all_devices()
        if devices:
            print("Available devices:")
            for d in devices:
                print(f"  - {d}")
        else:
            print("No devices found")
        return
    
    # Check if task was provided when not listing devices
    if not args.task:
        parser.error("task is required when not using --list-devices")
    
    # Check device connection
    devices = list_all_devices()
    if args.device not in devices:
        print(f"[ERROR] Device {args.device} not found!")
        print(f"Available devices: {devices}")
        return
    
    # Run task based on mode
    if args.mode == "supervisor":
        # Use new Supervisor-Worker architecture
        if getattr(args, 'async', False):
            # Try async version
            print("[INFO] Using Async Supervisor-Worker architecture")
            try:
                import asyncio
                from pipeline.orchestrator.orchestrator_async import OrchestratorAsync
                
                orchestrator = OrchestratorAsync(device=args.device, max_steps=args.max_steps)
                result = asyncio.run(orchestrator.run_task(args.task))
            except ImportError:
                print("[WARNING] Async orchestrator not available, falling back to sync")
                from pipeline.orchestrator import Orchestrator
                orchestrator = Orchestrator(device=args.device, max_steps=args.max_steps)
                result = orchestrator.run_task(args.task)
        else:
            print("[INFO] Using Supervisor-Worker architecture")
            from pipeline.orchestrator import Orchestrator
            
            # Set environment variable for debug mode that components can check
            if args.debug:
                os.environ['PIPELINE_DEBUG'] = 'true'
                print("[INFO] Debug mode enabled")
            else:
                os.environ['PIPELINE_DEBUG'] = 'false'
            
            orchestrator = Orchestrator(device=args.device, max_steps=args.max_steps)
            result = orchestrator.run_task(args.task)
        
        # Print detailed summary for supervisor mode
        print("\n" + "="*60)
        print("EXECUTION SUMMARY (SUPERVISOR-WORKER)")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"Task: {result.get('task', 'N/A')}")
        print(f"Subtasks: {result.get('subtasks_completed', 0)}/{result.get('subtasks_total', 0)} completed")
        if result.get('subtasks_skipped', 0) > 0:
            print(f"Skipped: {result['subtasks_skipped']} subtasks")
        print(f"Steps taken: {result.get('steps_taken', 0)}")
        print(f"Execution time: {result.get('execution_time', 'N/A')}")
        print(f"Folder: {result.get('folder', 'N/A')}")
        
        # Show subtask details if available
        if result.get('subtask_details'):
            print("\nSubtask Details:")
            for st in result['subtask_details']:
                # Show checkmark for both completed and already_completed
                if st['status'] in ['completed', 'already_completed']:
                    status_icon = "✓"
                elif st['status'] == 'pending':
                    status_icon = "○"
                elif st['status'] == 'skipped':
                    status_icon = "⊙"
                else:
                    status_icon = "⊘"
                print(f"  {status_icon} {st['id']}. {st['task']} [{st['status']}]")
    else:
        # Use legacy two-stage system
        print("[INFO] Using legacy two-stage system")
        result = run_task(args.task, args.device)
        
        # Print summary
        print("\n" + "="*60)
        print("EXECUTION SUMMARY (TWO-STAGE)")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"Steps: {result.get('steps_completed', 0)}")
        print(f"Folder: {result.get('folder', 'N/A')}")

if __name__ == "__main__":
    main()