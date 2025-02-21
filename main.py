"""Script to fetch and analyze emails from Gmail."""

import logging
import argparse
import os
import sys
from datetime import datetime, timedelta
from gmail_reader import GmailAuthenticator, GmailFetcher, EmailParser
from googleapiclient.errors import HttpError
from llm.providers import create_provider
from typing import Dict, Tuple
import re

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to stdout with default UTF-8 encoding
    ]
)
logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging directory with UTF-8 encoding."""
    os.makedirs('logs', exist_ok=True)
    
    # Configure specific formatters for each log type
    deleted_formatter = logging.Formatter(
        '%(asctime)s - Email Process Summary\n'
        'Status: %(status)s\n'
        'From: %(sender)s\n'  # Changed from 'from' to avoid keyword conflict
        'Subject: %(subject)s\n'
        'Type: %(type)s\n'
        'Action: %(action)s\n'
        'Confidence: %(confidence).2f\n'
        'Reason: %(reason)s\n'
        'Preview: %(preview)s\n'  # Changed from 'content' to be more descriptive
        '----------------------------------------\n'
    )
    
    selected_formatter = logging.Formatter(
        '%(asctime)s - Selected Email\n'
        'From: %(sender)s\n'
        'Subject: %(subject)s\n'
        'Type: %(type)s\n'
        'Confidence: %(confidence).2f\n'
        'Reason: %(reason)s\n'
        '----------------------------------------\n'
    )
    
    check_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s\n'
        'Email ID: %(email_id)s\n'
        'Subject: %(subject)s\n'
        'Analysis: %(analysis)s\n'
        'Status: %(status)s\n'
        '----------------------------------------\n'
    )
    
    # Create and configure handlers
    handlers = {
        'email_checks.log': (logging.getLogger('email_checks'), check_formatter),
        'deleted_emails.log': (logging.getLogger('deleted_emails'), deleted_formatter),
        'selected_emails.log': (logging.getLogger('selected_emails'), selected_formatter)
    }
    
    for log_file, (logger, formatter) in handlers.items():
        handler = logging.FileHandler(os.path.join('logs', log_file), encoding='utf-8')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent duplicate logs
        logger.propagate = False

def build_date_query(days: int = None) -> str:
    """Build Gmail search query for date range."""
    if not days:
        return ""
        
    today = datetime.now()
    start_date = today - timedelta(days=days)
    return f"after:{start_date.strftime('%Y/%m/%d')}"

def build_search_query(base_query: str = None, days: int = None) -> str:
    """Build complete Gmail search query."""
    queries = []
    
    # Add date range if specified
    date_query = build_date_query(days)
    if date_query:
        queries.append(date_query)
    
    # Add base query if provided
    if base_query:
        queries.append(f"({base_query})")
    
    # Add job-related terms if no specific query provided
    if not base_query:
        job_query = ("subject:(application OR interview OR job OR position OR "
                    "opportunity OR career OR recruitment)")
        queries.append(job_query)
    
    return " ".join(queries)

def get_label_id(service, label_name: str) -> str:
    """Get Gmail label ID by name."""
    try:
        results = service.users().labels().list(userId='me').execute()
        for label in results.get('labels', []):
            if label['name'].lower() == label_name.lower():
                logger.info(f"Found existing label: {label_name}")
                return label['id']
        return None
    except Exception as e:
        logger.error(f"Error getting label {label_name}: {str(e)}")
        raise

def create_label(service, label_name: str) -> str:
    """Create a new Gmail label."""
    try:
        label_object = {
            'name': label_name,
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show'
        }
        created_label = service.users().labels().create(
            userId='me',
            body=label_object
        ).execute()
        logger.info(f"Created new label: {label_name}")
        return created_label['id']
    except HttpError as e:
        if e.resp.status == 409:  # Label exists
            return get_label_id(service, label_name)
        raise
    except Exception as e:
        logger.error(f"Error creating label {label_name}: {str(e)}")
        raise

def get_or_create_label(service, label_name: str) -> str:
    """Get label ID if exists, create if it doesn't."""
    label_id = get_label_id(service, label_name)
    if label_id:
        return label_id
    return create_label(service, label_name)

def apply_label(service, email_id: str, label_id: str):
    """Apply label to an email."""
    try:
        service.users().messages().modify(
            userId='me',
            id=email_id,
            body={'addLabelIds': [label_id]}
        ).execute()
        logger.info(f"Successfully applied label to email {email_id}")
    except Exception as e:
        logger.error(f"Failed to apply label: {str(e)}")
        raise

def delete_email(service, email_id: str):
    """Move email to trash."""
    try:
        service.users().messages().trash(
            userId='me',
            id=email_id
        ).execute()
        logger.info(f"Successfully moved email {email_id} to trash")
    except Exception as e:
        logger.error(f"Failed to delete email: {str(e)}")
        raise

def get_lifetime_stats() -> Tuple[int, int, int]:
    """Get lifetime statistics from log files."""
    try:
        deleted_count = 0
        selected_count = 0
        ignored_count = 0
        
        # Count deleted emails
        deleted_log = os.path.join('logs', 'deleted_emails.log')
        if os.path.exists(deleted_log):
            try:
                with open(deleted_log, 'r', encoding='utf-8', errors='ignore') as f:
                    deleted_count = sum(1 for line in f if 'Status: DELETED' in line)
            except Exception as e:
                logger.error(f"Error reading deleted_emails.log: {str(e)}")
        
        # Count selected emails
        selected_log = os.path.join('logs', 'selected_emails.log')
        if os.path.exists(selected_log):
            try:
                with open(selected_log, 'r', encoding='utf-8', errors='ignore') as f:
                    selected_count = sum(1 for line in f if 'Selected Email' in line)
            except Exception as e:
                logger.error(f"Error reading selected_emails.log: {str(e)}")
        
        # Count ignored emails (from email_checks.log)
        checks_log = os.path.join('logs', 'email_checks.log')
        if os.path.exists(checks_log):
            try:
                with open(checks_log, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Count actions marked as "No action taken"
                    ignored_count = content.count('Status: No action taken')
            except Exception as e:
                logger.error(f"Error reading email_checks.log: {str(e)}")
        
        logger.info(f"Lifetime Stats - Deleted: {deleted_count}, Selected: {selected_count}, Ignored: {ignored_count}")
        return deleted_count, selected_count, ignored_count
    except Exception as e:
        logger.error(f"Error getting lifetime stats: {str(e)}")
        return 0, 0, 0

def print_stats_summary(current_stats: Dict[str, int], lifetime_stats: Tuple[int, int, int]):
    """Print detailed statistics summary."""
    deleted_total, selected_total, ignored_total = lifetime_stats
    total_processed = deleted_total + selected_total + ignored_total
    
    print("\nCurrent Session Statistics:")
    print("=" * 50)
    print(f"Emails processed: {current_stats['processed']}")
    print(f"Emails labeled as Selected: {current_stats['labeled']}")
    print(f"Emails deleted: {current_stats['deleted']}")
    print(f"Emails ignored: {current_stats['processed'] - current_stats['labeled'] - current_stats['deleted']}")
    
    print("\nLifetime Statistics:")
    print("=" * 50)
    print(f"Total emails deleted: {deleted_total}")
    print(f"Total emails selected: {selected_total}")
    print(f"Total emails ignored: {ignored_total}")
    print(f"Total emails processed: {total_processed}")
    if total_processed > 0:
        print(f"\nProcessing Rates:")
        print(f"Selected Rate: {(selected_total/total_processed)*100:.1f}%")
        print(f"Deleted Rate: {(deleted_total/total_processed)*100:.1f}%")
        print(f"Ignored Rate: {(ignored_total/total_processed)*100:.1f}%")
    print("=" * 50)

def process_emails(
    max_count: int = 10,
    label: str = None,
    query: str = None,
    days: int = None
):
    """Fetch and process emails."""
    try:
        # Initialize components
        authenticator = GmailAuthenticator()
        service = authenticator.authenticate()
        fetcher = GmailFetcher(service)
        parser = EmailParser()
        llm_provider = create_provider()

        # Ensure 'Selected' label exists
        selected_label_id = get_or_create_label(service, 'Selected')
        logger.info("Successfully initialized 'Selected' label")

        # Prepare search query
        label_ids = [label] if label else None
        final_query = build_search_query(query, days)
        
        # Fetch emails
        logger.info(f"Fetching {max_count} recent emails...")
        if label:
            logger.info(f"Filtering by label: {label}")
        if final_query:
            logger.info(f"Using search query: {final_query}")
            
        emails = fetcher.get_messages_batch(
            max_results=max_count,
            label_ids=label_ids,
            query=final_query
        )
        
        if not emails:
            logger.info("No emails found matching the criteria.")
            return
            
        processed = 0
        labeled = 0
        deleted = 0
        
        # Process emails
        for i, email in enumerate(emails, 1):
            print(f"\nEmail {i} of {len(emails)}")
            print("=" * 50)
            print(f"Subject: {email['subject']}")
            print(f"From: {email['from']}")
            print(f"To: {email['to']}")
            print(f"Date: {email['date']}")
            print("-" * 50)
            
            # Clean content
            cleaned_content = parser.clean_text(email['content'])
            print("Content:")
            print(cleaned_content)
            
            # Analyze email using LLM
            email_data = {
                "subject": email['subject'],
                "from": email['from'],
                "content": cleaned_content
            }
            analysis = llm_provider.analyze_job_email(email_data)
            
            print("\nAnalysis Results:")
            print(f"Is Job Related: {analysis['is_job_related']}")
            print(f"Type: {analysis['type']}")
            print(f"Recommended Action: {analysis['action']}")
            print(f"Confidence: {analysis.get('confidence', 0):.2f}")
            print(f"Reason: {analysis.get('reason', 'No reason provided')}")
            
            # Take action based on analysis
            action_status = "No action taken"
            try:
                if analysis['action'] == 'delete':
                    delete_email(service, email['id'])
                    deleted_logger = logging.getLogger('deleted_emails')
                    deleted_logger.info("", extra={
                        'status': 'DELETED',
                        'sender': email['from'],
                        'subject': email['subject'],
                        'type': analysis['type'],
                        'action': analysis['action'],
                        'confidence': analysis.get('confidence', 0),
                        'reason': analysis.get('reason', 'No reason provided'),
                        'preview': cleaned_content[:500]
                    })
                    deleted += 1
                    action_status = "Deleted"
                elif analysis['action'] == 'label_selected':
                    apply_label(service, email['id'], selected_label_id)
                    selected_logger = logging.getLogger('selected_emails')
                    selected_logger.info("", extra={
                        'sender': email['from'],
                        'subject': email['subject'],
                        'type': analysis['type'],
                        'confidence': analysis.get('confidence', 0),
                        'reason': analysis.get('reason', 'No reason provided')
                    })
                    labeled += 1
                    action_status = "Labeled as Selected"
                    
                processed += 1
                
            except Exception as e:
                logger.error(f"Failed to process action for email {email['id']}: {str(e)}")
                action_status = f"Error: {str(e)}"
                continue
            
            # Log email check with all details
            check_logger = logging.getLogger('email_checks')
            check_logger.info("", extra={
                'email_id': email['id'],
                'subject': email['subject'],
                'analysis': str(analysis),
                'status': action_status
            })
            
            print("=" * 50)
            
        # Get lifetime statistics
        lifetime_stats = get_lifetime_stats()
        
        # Current session statistics
        current_stats = {
            'processed': processed,
            'labeled': labeled,
            'deleted': deleted
        }
        
        # Print detailed summary
        print_stats_summary(current_stats, lifetime_stats)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

def main():
    """Parse arguments and run email processor."""
    parser = argparse.ArgumentParser(
        description="Fetch and analyze emails from Gmail"
    )
    parser.add_argument(
        "-n", "--num-emails",
        type=int,
        default=50,
        help="Number of emails to fetch (default: 50)"
    )
    parser.add_argument(
        "-l", "--label",
        type=str,
        help="Gmail label to filter by (e.g., 'INBOX', 'SENT')"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Search query to filter emails"
    )
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=30,
        help="Number of days to look back (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Ensure logging directory exists
    setup_logging()
    
    # Process emails
    process_emails(
        max_count=args.num_emails,
        label=args.label,
        query=args.query,
        days=args.days
    )

if __name__ == "__main__":
    main()