"""
Custom exception classes for the Grok client.

This module defines a hierarchy of custom exceptions to provide more specific error
information when interacting with the Grok API or the client library itself.
"""

class GrokClientError(Exception):
    """
    Base class for all custom exceptions raised by the Grok client library.
    
    This exception can be used to catch any error originating from the Grok client,
    allowing for a general way to handle client-specific issues.
    """
    pass

class GrokAPIError(GrokClientError):
    """
    Raised when the Grok API returns an error response.
    
    This typically indicates a problem on the server-side or an issue with the
    request that the API itself has identified (e.g., invalid parameters,
    rate limits, server errors within Grok's infrastructure). The error message
    will usually contain details from the API's error response.
    """
    pass

class AuthenticationError(GrokClientError):
    """
    Raised for authentication-related failures.
    
    This can occur if SSO tokens (cookies) are missing, invalid, or expired,
    preventing successful authentication with the Grok API. It may also indicate
    permission issues if the authenticated user does not have access to a
    requested resource or model.
    """
    pass

class ConfigurationError(GrokClientError):
    """
    Raised for errors related to client or environment configuration.
    
    This includes issues such as missing or invalid essential environment variables
    (e.g., API host, port if not using defaults), incorrect cookie paths, or other
    setup problems that prevent the client from initializing or operating correctly.
    """
    pass

class NetworkError(GrokClientError):
    """
    Raised for network-related issues encountered while communicating with the Grok API.
    
    This can include problems like connection timeouts, DNS resolution failures,
    or other issues preventing the client from reaching the Grok API servers.
    It generally suggests a problem with the network connection between the client
    and the API endpoint.
    """
    pass
