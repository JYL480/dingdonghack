#!/usr/bin/env python
"""
Clean output runner for the pipeline
Filters verbose debug output to show only essential information
"""

import sys
import subprocess
import re

def run_clean(task, device="emulator-5554"):
    """Run pipeline with filtered output"""
    
    # Run the pipeline command
    cmd = [sys.executable, "nich_pipeline.py", task, "-d", device]
    
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}\n")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Patterns to always show
    always_show = [
        r"✓",  # Success marks
        r"Test result:",  # Test results
        r"Progress:",  # Progress updates
        r"EXECUTION SUMMARY",  # Final summary
        r"Status:",  # Status updates
        r"Subtasks:",  # Subtask counts
        r"Time:",  # Execution time
        r"\[Task \d+\]",  # Task headers
        r"→",  # Action arrows
    ]
    
    # Patterns to skip in normal mode
    skip_patterns = [
        r"\[ADB\]",  # ADB commands
        r"\[INFO\] Calling OmniParser",  # API calls
        r"\[STEP\]",  # Step details
        r"\[WORKER\]",  # Worker details
        r"\[SUPERVISOR",  # Supervisor details
        r"\[ORCHESTRATOR\] Step",  # Step details
        r"STAGE \d:",  # Stage headers
        r"VISION ANALYSIS",  # Vision details
        r"OBSERVATION:",  # Vision observations
        r"OPTIONS:",  # Vision options
        r"DECISION:",  # Vision decisions
        r"REASONING:",  # Vision reasoning
        r"EXPECTATION:",  # Vision expectations
        r"ACTION EXECUTED",  # Detailed action logs
        r"Analyzing screen",  # Screen analysis
        r"Tapping element",  # Element tapping details
        r"screenshot",  # Screenshot paths
        r"=== Human Message",  # Agent messages
        r"=== Ai Message",  # Agent messages
        r"Tool Calls:",  # Tool call details
    ]
    
    # Buffer for multiline handling
    buffer = []
    in_summary = False
    
    for line in process.stdout:
        line = line.rstrip()
        
        # Check if we're in the summary section
        if "EXECUTION SUMMARY" in line:
            in_summary = True
            
        # Always show summary section
        if in_summary:
            print(line)
            continue
            
        # Check if line should be shown
        should_show = False
        
        # Always show certain patterns
        for pattern in always_show:
            if re.search(pattern, line):
                should_show = True
                break
                
        # Skip verbose patterns unless they match always_show
        if not should_show:
            skip = False
            for pattern in skip_patterns:
                if re.search(pattern, line):
                    skip = True
                    break
            if not skip and line.strip():  # Show non-empty lines that don't match skip patterns
                # Additional filtering for cleaner output
                if not line.startswith("[") or line.startswith("[Task"):
                    print(line)
    
    process.wait()
    return process.returncode

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_clean.py 'task description' [--device device_id]")
        sys.exit(1)
    
    task = sys.argv[1]
    device = "emulator-5554"
    
    if len(sys.argv) > 3 and sys.argv[2] == "--device":
        device = sys.argv[3]
    
    sys.exit(run_clean(task, device))