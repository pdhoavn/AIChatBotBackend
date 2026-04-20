"""
Test configuration and fixtures for the test suite
"""
import pytest
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all fixtures from test_config
from tests.test_config import *
