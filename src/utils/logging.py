"""
Logging utilities.
"""

import logging
import os
from typing import Optional


def create_logger(
    name: str,
    save_dir: Optional[str] = None,
    quiet: bool = False,
) -> logging.Logger:
    """
    Create a logger with stream and file handlers.
    
    Creates a logger with:
    - A stream handler for console output
    - File handlers for verbose.log and quiet.log (if save_dir provided)
    
    Args:
        name: Logger name
        save_dir: Directory to save log files
        quiet: Whether to suppress debug output to console
        
    Returns:
        Configured logger
    """
    # Return existing logger if already created
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if quiet else logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handlers
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Verbose log (all messages)
        fh_v = logging.FileHandler(os.path.join(save_dir, "verbose.log"))
        fh_v.setLevel(logging.DEBUG)
        fh_v.setFormatter(formatter)
        logger.addHandler(fh_v)
        
        # Quiet log (info and above)
        fh_q = logging.FileHandler(os.path.join(save_dir, "quiet.log"))
        fh_q.setLevel(logging.INFO)
        fh_q.setFormatter(formatter)
        logger.addHandler(fh_q)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class that provides logging functionality.
    
    Classes that inherit from this mixin get access to
    debug(), info(), warning(), and error() methods.
    """
    
    _logger: Optional[logging.Logger] = None
    
    @property
    def logger(self) -> logging.Logger:
        """Get or create logger."""
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
