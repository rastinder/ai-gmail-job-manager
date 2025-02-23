# AI Gmail Job Manager

An intelligent email processor that uses AI to automatically manage your job-related emails in Gmail. The system analyzes incoming emails to identify job opportunities, interview requests, and rejections, then takes appropriate actions like labeling important emails or removing rejections.

## Features

- 🤖 AI-powered email analysis using Large Language Models
- 📧 Automatic processing of job-related emails
- 🏷️ Smart labeling of important emails (interviews, selections)
- 🗑️ Automatic cleanup of rejection emails
- 📊 Detailed logging and statistics
- 🔒 Secure Gmail API integration

## Prerequisites

- Python 3.8+
- Gmail API credentials
- OpenAI API key or other supported LLM provider

## Setup

1. Clone the repository:
```bash
git clone https://github.com/rastinder/ai-gmail-job-manager.git
cd ai-gmail-job-manager
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Gmail API:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download credentials and save as `credentials.json` in project root

4. Configure LLM provider:
   - Copy `llm/secret.yaml.example` to `llm/secret.yaml`
   - Add your API keys and configuration

## Configuration

### LLM Configuration (llm/secret.yaml)

Note: We have updated our configuration format. The example files might contain old configuration formats.

```yaml
# Active Provider Configuration
active_provider: 'codestral'  # Options: 'codestral', 'openai', 'llama'

# Feature Flags
features:
  enable_delete_rejection: true
  enable_delete_job_application: true

# Codestral Configuration
codestral:
  api_keys:
    - 'your-codestral-key-here'
  llm_api_url: 'https://codestral.mistral.ai/v1'
  llm_model: 'codestral-latest'
  max_tpm: 500000
  max_monthly_tokens: 1000000000
  max_rps: 1
  max_tokens: 256768

# OpenAI Configuration  
openai:
  api_keys:
    - 'your-openai-key-here'
  llm_model: 'gpt-4-turbo-preview'
  max_tokens: 128000
  max_rps: 3
  temperature: 0.7

# Llama Configuration
llama:
  api_keys:
    - 'your-llama-key-here'
  llm_api_url: 'http://localhost:11434'
  llm_model: 'llama2:latest'
  max_tokens: 4096
  temperature: 0.7

# Logging Configuration
logging:
  email_check_log: 'logs/email_checks.log'
  deleted_emails_log: 'logs/deleted_emails.log'
  selected_emails_log: 'logs/selected_emails.log'

# Gmail Labels
labels:
  selected: 'Selected'
```

## Usage

Basic usage with default settings (processes last 50 emails from last 30 days):
```bash
python main.py
```

Process specific number of emails:
```bash
python main.py --num-emails 10
```

Process emails with specific label:
```bash
python main.py --label "INBOX" --days 7
```

Custom search query:
```bash
python main.py --query "subject:interview" --num-emails 20
```

### Command Line Arguments

- `-n, --num-emails`: Number of emails to process (default: 50)
- `-l, --label`: Gmail label to filter by (e.g., 'INBOX', 'SENT')
- `-q, --query`: Custom search query
- `-d, --days`: Number of days to look back (default: 30)

## Example Output

```
Email 1 of 10
==================================================
Subject: Interview Invitation - Software Engineer Position
From: hr@company.com
To: you@gmail.com
Date: 2024-02-20T10:30:00Z
--------------------------------------------------
Content:
Dear Candidate,

We are pleased to invite you for an interview...

Analysis Results:
Is Job Related: True
Type: next_steps
Recommended Action: label_selected
Confidence: 0.95
Reason: Interview invitation for software engineering position
==================================================
```

## Stats and Logging

The system maintains detailed logs and statistics:

- `logs/email_checks.log`: All processed emails and their analysis
- `logs/deleted_emails.log`: Records of deleted emails (rejections)
- `logs/selected_emails.log`: Important emails that were labeled

View statistics at the end of each run:
```
Current Session Statistics:
==================================================
Emails processed: 50
Emails labeled as Selected: 5
Emails deleted: 10
Emails ignored: 35

Lifetime Statistics:
==================================================
Total emails deleted: 150
Total emails selected: 75
Total emails ignored: 275
Total emails processed: 500

Processing Rates:
Selected Rate: 15.0%
Deleted Rate: 30.0%
Ignored Rate: 55.0%
==================================================
```

## Security

- Uses OAuth 2.0 for Gmail API authentication
- Sensitive credentials stored in separate configuration files
- All API keys and secrets should be kept private
- Logs don't store full email content

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
