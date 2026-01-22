"""
I/O utilities.
"""

import json
import os
from typing import Any, Dict


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Create directory, including parent directories.
    
    Args:
        path: Path to directory (or file if isfile=True)
        isfile: If True, creates parent directory of the file
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Path to JSON file
        indent: Indentation level
    """
    makedirs(path, isfile=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def save_text(text: str, path: str) -> None:
    """
    Save text to file.
    
    Args:
        text: Text to save
        path: Path to file
    """
    makedirs(path, isfile=True)
    with open(path, "w") as f:
        f.write(text)


def load_text(path: str) -> str:
    """
    Load text from file.
    
    Args:
        path: Path to file
        
    Returns:
        File contents
    """
    with open(path, "r") as f:
        return f.read()
