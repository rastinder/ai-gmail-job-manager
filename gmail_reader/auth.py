"""Gmail authentication module."""

import os.path
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

class GmailAuthenticator:
    """Handles Gmail API authentication."""
    
    # If modifying these scopes, delete the token.pickle file.
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',  # Read emails
        'https://www.googleapis.com/auth/gmail.labels',    # Manage labels
        'https://www.googleapis.com/auth/gmail.modify'     # Modify emails (apply labels)
    ]
    
    def __init__(self):
        """Initialize authenticator with credentials path."""
        self.credentials_path = 'credentials.json'
        self.token_path = 'token.pickle'
        
    def authenticate(self):
        """
        Authenticate with Gmail API.
        
        Returns:
            service: Authenticated Gmail API service
        """
        creds = None
        
        # Load existing token if present
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
                
        # Refresh token if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Failed to refresh token: {str(e)}")
                creds = None
                
        # Create new token if none exists
        if not creds:
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(
                    f"Credentials file not found at {self.credentials_path}"
                )
                
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path,
                    self.SCOPES
                )
                creds = flow.run_local_server(port=0)
                
                # Save token
                with open(self.token_path, 'wb') as token:
                    pickle.dump(creds, token)
                    
            except Exception as e:
                logger.error(f"Failed to create new token: {str(e)}")
                raise
                
        try:
            # Build Gmail API service
            service = build('gmail', 'v1', credentials=creds)
            return service
            
        except Exception as e:
            logger.error(f"Failed to build Gmail service: {str(e)}")
            raise