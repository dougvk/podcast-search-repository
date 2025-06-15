"""
Web interface module for podcast search repository.

Contains the web application for manual search and result display.
"""

from .app import create_web_app

__all__ = [
    "create_web_app"
]