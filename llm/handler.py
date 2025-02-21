"""LLM handler for managing email analysis and actions."""

import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import yaml
from .providers import create_provider, BaseLLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailLogger:
    """Handles logging of email operations."""
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize loggers.
        
        Args:
            config (Dict[str, str]): Logging configuration
        """
        os.makedirs('logs', exist_ok=True)
        
        self.check_logger = self._setup_logger(
            config['email_check_log'], 'email_check'
        )
        self.deleted_logger = self._setup_logger(
            config['deleted_emails_log'], 'deleted'
        )
        self.selected_logger = self._setup_logger(
            config['selected_emails_log'], 'selected'
        )
        
    def _setup_logger(self, log_file: str, name: str) -> logging.Logger:
        """Set up a specific logger."""
        logger = logging.getLogger(name)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def log_check(self, email_id: str, subject: str, analysis: Dict[str, Any]):
        """Log email check details."""
        self.check_logger.info(
            f"Email ID: {email_id} | Subject: {subject} | "
            f"Analysis: {analysis}"
        )
        
    def log_deletion(self, email_id: str, subject: str, reason: str):
        """Log email deletion."""
        self.deleted_logger.info(
            f"Email ID: {email_id} | Subject: {subject} | "
            f"Reason: {reason}"
        )
        
    def log_selected(self, email_id: str, subject: str):
        """Log selected job email."""
        self.selected_logger.info(
            f"Email ID: {email_id} | Subject: {subject}"
        )

class LLMHandler:
    """Handles LLM operations and email management."""
    
    def __init__(self, gmail_service=None):
        """
        Initialize LLM handler.
        
        Args:
            gmail_service: Gmail API service instance
        """
        config = self._load_config()
        self.provider = create_provider()
        self.logger = EmailLogger(config['logging'])
        self.gmail_service = gmail_service
        self.labels = config['labels']
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from secret.yaml."""
        config_path = os.path.join('llm', 'secret.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def ensure_labels_exist(self):
        """Ensure required Gmail labels exist."""
        try:
            # List all labels
            results = self.gmail_service.users().labels().list(
                userId='me'
            ).execute()
            existing_labels = {label['name']: label['id'] 
                             for label in results['labels']}
            
            # Create 'Selected' label if it doesn't exist
            if self.labels['selected'] not in existing_labels:
                label = {
                    'name': self.labels['selected'],
                    'labelListVisibility': 'labelShow',
                    'messageListVisibility': 'show'
                }
                self.gmail_service.users().labels().create(
                    userId='me',
                    body=label
                ).execute()
                
        except Exception as e:
            logger.error(f"Failed to ensure labels: {str(e)}")
            
    def apply_label(self, email_id: str, label_name: str):
        """Apply label to an email."""
        try:
            # Get label ID
            results = self.gmail_service.users().labels().list(
                userId='me'
            ).execute()
            label_id = None
            for label in results['labels']:
                if label['name'] == label_name:
                    label_id = label['id']
                    break
                    
            if not label_id:
                raise Exception(f"Label not found: {label_name}")
                
            # Apply label
            self.gmail_service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'addLabelIds': [label_id]}
            ).execute()
            
        except Exception as e:
            logger.error(f"Failed to apply label: {str(e)}")
            
    def delete_email(self, email_id: str):
        """Move email to trash."""
        try:
            self.gmail_service.users().messages().trash(
                userId='me',
                id=email_id
            ).execute()
        except Exception as e:
            logger.error(f"Failed to delete email: {str(e)}")
            
    def process_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process email content through LLM.
        
        Args:
            email_data (Dict[str, Any]): Email data including content
            
        Returns:
            Dict[str, Any]: Analysis results with actions taken
        """
        try:
            # Analyze email
            analysis = self.provider.analyze_job_email(email_data['content'])
            
            # Log the check
            self.logger.log_check(
                email_data['id'],
                email_data['subject'],
                analysis
            )
            
            # Perform actions based on analysis
            if analysis['is_job_related']:
                if analysis['type'] == 'rejection':
                    self.delete_email(email_data['id'])
                    self.logger.log_deletion(
                        email_data['id'],
                        email_data['subject'],
                        "Job rejection"
                    )
                elif analysis['type'] in ['selection', 'next_steps']:
                    self.apply_label(
                        email_data['id'],
                        self.labels['selected']
                    )
                    self.logger.log_selected(
                        email_data['id'],
                        email_data['subject']
                    )
                    
            return {
                'email_id': email_data['id'],
                'analysis': analysis,
                'action_taken': analysis.get('action', 'keep')
            }
            
        except Exception as e:
            logger.error(f"Failed to process email: {str(e)}")
            return {
                'email_id': email_data['id'],
                'error': str(e),
                'action_taken': 'none'
            }
            
    def process_batch(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple emails.
        
        Args:
            emails (List[Dict[str, Any]]): List of email data
            
        Returns:
            List[Dict[str, Any]]: List of analysis results
        """
        # Ensure required labels exist
        self.ensure_labels_exist()
        
        results = []
        for email in emails:
            try:
                result = self.process_email(email)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process email {email['id']}: {str(e)}")
                results.append({
                    'email_id': email['id'],
                    'error': str(e),
                    'action_taken': 'none'
                })
                
        return results