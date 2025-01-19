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
        token = jwt.encode({
            'session_id': session_id,
            'task_id': task_id,
            'exp': expiration
        }, self.JWT_SECRET, algorithm='HS256')
        return token
    
    def verify_download_token(self, token):
        """Verify JWT token and return payload if valid.
        
        Args:
            token (str): The JWT token to verify
            
        Returns:
            dict: The token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.JWT_SECRET, algorithms=['HS256'])
            return payload
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None 