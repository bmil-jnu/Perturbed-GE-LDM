"""
Timing utilities.
"""

import logging
from datetime import timedelta
from functools import wraps
from time import time
from typing import Any, Callable, Optional


def timeit(logger_name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """
    Decorator factory that creates a timer decorator.
    
    The decorator wraps a function and prints/logs the elapsed time
    after the function completes.
    
    Args:
        logger_name: Name of logger to use. If None, uses print().
        
    Returns:
        Decorator function
        
    Example:
        >>> @timeit(logger_name="my_logger")
        ... def slow_function():
        ...     time.sleep(2)
        >>> slow_function()  # Logs "Elapsed time = 0:00:02"
    """
    def timeit_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            elapsed = timedelta(seconds=round(time() - start_time))
            
            message = f"Elapsed time for {func.__name__}: {elapsed}"
            
            if logger_name is not None:
                logger = logging.getLogger(logger_name)
                logger.info(message)
            else:
                print(message)
            
            return result
        return wrapper
    return timeit_decorator


class Timer:
    """
    Context manager for timing code blocks.
    
    Example:
        >>> with Timer("my operation"):
        ...     slow_operation()
        my operation took 2.34s
    """
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        """
        Initialize Timer.
        
        Args:
            name: Name to display in timing message
            logger: Optional logger (uses print if None)
        """
        self.name = name
        self.logger = logger
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time()
        return self
    
    def __exit__(self, *args) -> None:
        self.elapsed = time() - self.start_time
        message = f"{self.name} took {self.elapsed:.2f}s"
        
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)


class ProgressTimer:
    """
    Timer that tracks progress through iterations.
    
    Example:
        >>> timer = ProgressTimer(total=100)
        >>> for i in range(100):
        ...     # do work
        ...     timer.step()
        ...     print(timer.eta)  # Estimated time remaining
    """
    
    def __init__(self, total: int):
        """
        Initialize ProgressTimer.
        
        Args:
            total: Total number of iterations
        """
        self.total = total
        self.current = 0
        self.start_time = time()
    
    def step(self, n: int = 1) -> None:
        """Increment counter."""
        self.current += n
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time() - self.start_time
    
    @property
    def eta(self) -> float:
        """Get estimated time remaining in seconds."""
        if self.current == 0:
            return float("inf")
        rate = self.current / self.elapsed
        remaining = self.total - self.current
        return remaining / rate
    
    @property
    def eta_str(self) -> str:
        """Get ETA as formatted string."""
        eta = self.eta
        if eta == float("inf"):
            return "unknown"
        return str(timedelta(seconds=round(eta)))
    
    @property
    def progress(self) -> float:
        """Get progress as fraction (0-1)."""
        return self.current / self.total
    
    @property
    def progress_pct(self) -> float:
        """Get progress as percentage."""
        return self.progress * 100
