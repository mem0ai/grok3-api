"""
Grok Client Package
===================

This package provides client utilities for interacting with a Grok API. 
It includes:
- `GrokClient`: A low-level client for sending messages directly to the Grok service.
- `GrokOpenAIClient`: An OpenAI-compatible client that wraps the Grok API, 
  allowing it to be used with tools and libraries designed for the OpenAI API structure.
- Custom error classes for more specific error handling.
- A FastAPI server (`server.py`) that exposes an OpenAI-compatible API endpoint 
  backed by the Grok service.

The main components intended for direct use are typically `GrokClient` or 
`GrokOpenAIClient`.
"""
from .client import GrokClient
from .grok_openai_client import GrokOpenAIClient
from .errors import GrokClientError, GrokAPIError, AuthenticationError, NetworkError, ConfigurationError

__version__ = "0.1.0"
"""The version of the grok_client package."""

__all__ = [
    'GrokClient', 
    'GrokOpenAIClient',
    'GrokClientError', 
    'GrokAPIError', 
    'AuthenticationError', 
    'NetworkError', 
    'ConfigurationError'
]
"""Publicly exposed names from the grok_client package."""