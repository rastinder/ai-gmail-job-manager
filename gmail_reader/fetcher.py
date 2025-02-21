"""Gmail email fetching module."""

import base64
from typing import List, Dict, Optional
from datetime import datetime
from email.mime.text import MIMEText

class GmailFetcher:
    """Handles fetching and processing emails from Gmail."""
    
    def __init__(self, service):
        """
        Initialize Gmail fetcher.
        
        Args:
            service: Authenticated Gmail API service
        """
        self.service = service

    def list_messages(self, 
                     max_results: int = 100, 
                     label_ids: Optional[List[str]] = None,
                     query: str = "") -> List[Dict]:
        """
        List messages from Gmail with specified criteria.
        
        Args:
            max_results (int): Maximum number of messages to return
            label_ids (List[str]): List of label IDs to filter by
            query (str): Gmail search query string
            
        Returns:
            List[Dict]: List of message metadata
        """
        try:
            messages = []
            request = self.service.users().messages().list(
                userId='me',
                labelIds=label_ids,
                q=query,
                maxResults=max_results
            )
            
            while request is not None:
                response = request.execute()
                messages.extend(response.get('messages', []))
                
                request = self.service.users().messages().list_next(
                    request, response)
                
                if len(messages) >= max_results:
                    messages = messages[:max_results]
                    break
                    
            return messages
        except Exception as e:
            raise Exception(f"Failed to list messages: {str(e)}")

    def get_message(self, msg_id: str) -> Dict:
        """
        Get full message details by ID.
        
        Args:
            msg_id (str): Message ID
            
        Returns:
            Dict: Full message details
        """
        try:
            return self.service.users().messages().get(
                userId='me', 
                id=msg_id, 
                format='full'
            ).execute()
        except Exception as e:
            raise Exception(f"Failed to get message {msg_id}: {str(e)}")

    def get_message_content(self, message: Dict) -> str:
        """
        Extract text content from message.
        
        Args:
            message (Dict): Full message object
            
        Returns:
            str: Extracted text content
        """
        parts = []
        if 'payload' not in message:
            return ""
            
        def get_parts(payload):
            if 'parts' in payload:
                for part in payload['parts']:
                    if part.get('mimeType') == 'text/plain':
                        if 'data' in part['body']:
                            text = base64.urlsafe_b64decode(
                                part['body']['data']).decode('utf-8')
                            parts.append(text)
                    else:
                        get_parts(part)
            elif payload.get('mimeType') == 'text/plain':
                if 'data' in payload['body']:
                    text = base64.urlsafe_b64decode(
                        payload['body']['data']).decode('utf-8')
                    parts.append(text)
                    
        get_parts(message['payload'])
        return '\n'.join(parts)

    def get_headers(self, message: Dict) -> Dict[str, str]:
        """
        Extract headers from message.
        
        Args:
            message (Dict): Full message object
            
        Returns:
            Dict[str, str]: Dictionary of headers
        """
        headers = {}
        for header in message['payload']['headers']:
            headers[header['name'].lower()] = header['value']
        return headers

    def get_messages_batch(self, 
                          max_results: int = 10, 
                          label_ids: Optional[List[str]] = None,
                          query: str = "") -> List[Dict]:
        """
        Get batch of full message content with metadata.
        
        Args:
            max_results (int): Maximum number of messages to return
            label_ids (List[str]): List of label IDs to filter by
            query (str): Gmail search query string
            
        Returns:
            List[Dict]: List of processed messages with content
        """
        messages_meta = self.list_messages(max_results, label_ids, query)
        processed_messages = []
        
        for msg_meta in messages_meta:
            full_msg = self.get_message(msg_meta['id'])
            headers = self.get_headers(full_msg)
            content = self.get_message_content(full_msg)
            
            processed_messages.append({
                'id': msg_meta['id'],
                'threadId': msg_meta['threadId'],
                'subject': headers.get('subject', ''),
                'from': headers.get('from', ''),
                'to': headers.get('to', ''),
                'date': headers.get('date', ''),
                'content': content
            })
            
        return processed_messages