"""
Logging configuration for the pipeline
Supports different verbosity levels for cleaner output
"""

import os
from enum import Enum
from typing import Optional

class LogLevel(Enum):
    """Logging levels for pipeline output"""
    QUIET = 0      # Minimal output - only results
    NORMAL = 1     # Standard output - key actions and progress
    VERBOSE = 2    # Detailed output - includes decisions
    DEBUG = 3      # Full debug - all details including API calls

class PipelineLogger:
    """Centralized logger for the pipeline"""
    
    def __init__(self, level: LogLevel = LogLevel.NORMAL):
        self.level = level
        self.indent = 0
        
    def set_level(self, level: LogLevel):
        """Set logging level"""
        self.level = level
        
    def debug(self, message: str):
        """Debug level logging - only shown in DEBUG mode"""
        if self.level.value >= LogLevel.DEBUG.value:
            self._print(f"[DEBUG] {message}", color="\033[90m")  # Gray
            
    def verbose(self, message: str):
        """Verbose logging - shown in VERBOSE and DEBUG modes"""
        if self.level.value >= LogLevel.VERBOSE.value:
            self._print(message, color="\033[37m")  # Light gray
            
    def info(self, message: str):
        """Info logging - shown in NORMAL and above"""
        if self.level.value >= LogLevel.NORMAL.value:
            self._print(message)
            
    def progress(self, current: int, total: int, task: str = ""):
        """Progress updates - always shown except in QUIET mode"""
        if self.level.value >= LogLevel.NORMAL.value:
            bar_length = 20
            filled = int(bar_length * current / total)
            bar = "█" * filled + "░" * (bar_length - filled)
            percent = (current / total) * 100
            self._print(f"\r[{bar}] {percent:.0f}% - {task}", end="", color="\033[36m")  # Cyan
            if current == total:
                print()  # New line when complete
                
    def success(self, message: str):
        """Success messages - shown except in QUIET mode"""
        if self.level.value >= LogLevel.NORMAL.value:
            self._print(f"✓ {message}", color="\033[32m")  # Green
            
    def warning(self, message: str):
        """Warnings - always shown"""
        self._print(f"⚠ {message}", color="\033[33m")  # Yellow
        
    def error(self, message: str):
        """Errors - always shown"""
        self._print(f"✗ {message}", color="\033[31m")  # Red
        
    def task(self, message: str):
        """Task headers - shown in NORMAL and above"""
        if self.level.value >= LogLevel.NORMAL.value:
            self._print(f"\n→ {message}", color="\033[35m")  # Magenta
            
    def result(self, message: str):
        """Final results - always shown"""
        self._print(message, color="\033[32m", bold=True)  # Bold green
        
    def _print(self, message: str, color: str = "", bold: bool = False, end: str = "\n"):
        """Internal print with formatting"""
        indent = "  " * self.indent
        reset = "\033[0m"
        bold_code = "\033[1m" if bold else ""
        print(f"{indent}{bold_code}{color}{message}{reset}", end=end)
        
    def indent_push(self):
        """Increase indentation level"""
        self.indent += 1
        
    def indent_pop(self):
        """Decrease indentation level"""
        self.indent = max(0, self.indent - 1)

# Global logger instance
logger = PipelineLogger()

def set_log_level(level: str):
    """Set log level from string"""
    level_map = {
        "quiet": LogLevel.QUIET,
        "normal": LogLevel.NORMAL,
        "verbose": LogLevel.VERBOSE,
        "debug": LogLevel.DEBUG
    }
    if level.lower() in level_map:
        logger.set_level(level_map[level.lower()])
    else:
        logger.warning(f"Unknown log level: {level}, using NORMAL")
        logger.set_level(LogLevel.NORMAL)