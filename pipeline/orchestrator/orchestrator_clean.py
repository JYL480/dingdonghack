"""
Clean output version of the orchestrator
Uses the logger for simpler, more readable output
"""

from pipeline.orchestrator.orchestrator import Orchestrator
from pipeline.utils.logger import logger

class CleanOrchestrator(Orchestrator):
    """Orchestrator with clean logging output"""
    
    def run_task(self, task: str):
        """Run task with clean output in normal mode"""
        # In normal mode, show simplified output
        if not logger.debug:
            print(f"\n{'='*50}")
            print(f"TASK: {task}")
            print(f"{'='*50}")
            
            # Run the task
            result = super().run_task(task)
            
            # Show clean summary
            print(f"\n{'='*50}")
            print("EXECUTION SUMMARY")
            print(f"{'='*50}")
            
            if result['status'] == 'success':
                print(f"✓ Task completed successfully")
            else:
                print(f"✗ Task {result['status']}")
                
            print(f"Time: {result.get('execution_time', 'N/A')}")
            print(f"Steps: {result.get('steps_taken', 0)}")
            
            # Show subtask summary
            if result.get('subtask_details'):
                print(f"\nSubtasks completed: {result.get('subtasks_completed', 0)}/{result.get('subtasks_total', 0)}")
                
                # Only show failed or pending tasks
                for st in result['subtask_details']:
                    if st['status'] not in ['completed', 'already_completed']:
                        print(f"  ⚠ {st['task']} - {st['status']}")
                    elif logger.debug:
                        # In debug mode, show all
                        status_icon = '✓' if st['status'] in ['completed', 'already_completed'] else '○'
                        print(f"  {status_icon} {st['task']}")
            
            return result
        else:
            # In debug mode, use original verbose output
            return super().run_task(task)