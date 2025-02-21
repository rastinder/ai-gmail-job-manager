"""Email content parsing and processing module."""

import re
from typing import Dict, Any, List
from datetime import datetime
from email.utils import parsedate_to_datetime

class EmailParser:
    """Handles parsing and processing of email content."""
    
    def __init__(self):
        """Initialize the EmailParser."""
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text (str): Raw text content
            
        Returns:
            str: Cleaned text
        """
        # Remove email signatures
        lines = text.split('\n')
        cleaned_lines = []
        signature_markers = ['--', 'Best regards', 'Regards', 'Thanks,', 'Thank you,']
        
        for line in lines:
            if any(marker in line for marker in signature_markers):
                break
            cleaned_lines.append(line)
            
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove quoted text
        text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text content.
        
        Args:
            text (str): Text content
            
        Returns:
            List[str]: List of extracted URLs
        """
        return self.url_pattern.findall(text)
    
    def parse_date(self, date_str: str) -> datetime:
        """
        Parse email date string to datetime object.
        
        Args:
            date_str (str): Date string from email headers
            
        Returns:
            datetime: Parsed datetime object
        """
        try:
            return parsedate_to_datetime(date_str)
        except Exception:
            return None
    
    def prepare_for_llm(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare email content for LLM processing.
        
        Args:
            email_data (Dict[str, Any]): Raw email data
            
        Returns:
            Dict[str, Any]: Processed email data ready for LLM
        """
        processed_content = self.clean_text(email_data['content'])
        urls = self.extract_urls(processed_content)
        parsed_date = self.parse_date(email_data['date'])
        
        return {
            'id': email_data['id'],
            'metadata': {
                'subject': email_data['subject'],
                'from': email_data['from'],
                'to': email_data['to'],
                'date': parsed_date.isoformat() if parsed_date else None,
                'urls': urls
            },
            'content': processed_content,
            'summary_prompt': (
                f"Please analyze the following email:\n\n"
                f"Subject: {email_data['subject']}\n"
                f"From: {email_data['from']}\n"
                f"Date: {email_data['date']}\n\n"
                f"{processed_content}\n\n"
                f"Please provide:\n"
                f"1. A brief summary\n"
                f"2. Key points or action items\n"
                f"3. Sentiment analysis\n"
                f"4. Any important dates or deadlines mentioned"
            )
        }
    
    def batch_process(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of emails.
        
        Args:
            emails (List[Dict[str, Any]]): List of raw email data
            
        Returns:
            List[Dict[str, Any]]: List of processed emails
        """
        return [self.prepare_for_llm(email) for email in emails]