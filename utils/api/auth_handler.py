"""Authentication handler module for managing JWT tokens."""

import os
import jwt
from datetime import datetime, timedelta

class AuthHandler:
    """Handles JWT token generation and verification."""
    
    def __init__(self):
        """Initialize authentication handler with configuration."""
        self.JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')  # In production, use environment variable
        self.MAX_TOKEN_AGE_HOURS = 2  # Token expires after 2 hours
    
    def generate_download_token(self, session_id, task_id):
        """Generate a JWT token for file download.
        
        Args:
            session_id (str): The session ID associated with the task
            task_id (str): The ID of the task
            
        Returns:
            str: The generated JWT token
        """
        expiration = datetime.utcnow() + timedelta(hours=self.MAX_TOKEN_AGE_HOURS)
        payload = {
            'session_id': session_id,
            'task_id': task_id,
            'exp': expiration
        }
        
        try:
            # Try using jwt.encode from PyJWT
            token = jwt.encode(payload, self.JWT_SECRET, algorithm='HS256')
            # In PyJWT 2.x, encode returns a string, in older versions it returns bytes
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            return token
        except AttributeError:
            # Fallback to python-jwt if PyJWT is not available
            import python_jwt as pyjwt
            token = pyjwt.generate_jwt(payload, self.JWT_SECRET, 'HS256', timedelta(hours=self.MAX_TOKEN_AGE_HOURS))
            return token
    
    def verify_download_token(self, token):
        """Verify JWT token and return payload if valid.
        
        Args:
            token (str): The JWT token to verify
            
        Returns:
            dict: The token payload if valid, None otherwise
        """
        try:
            try:
                # Try using jwt.decode from PyJWT
                payload = jwt.decode(token, self.JWT_SECRET, algorithms=['HS256'])
                return payload
            except AttributeError:
                # Fallback to python-jwt if PyJWT is not available
                import python_jwt as pyjwt
                header, claims = pyjwt.verify_jwt(token, self.JWT_SECRET, ['HS256'])
                return claims
        except Exception:
            # Any other errors (expired token, invalid signature, etc.)
            return None 