"""LLM provider implementations with key rotation and rate limiting."""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import httpx
from datetime import datetime, timedelta
import logging
import yaml
import os
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_rps: int = 1):
        self.max_rps = max_rps
        self.last_call = datetime.now() - timedelta(seconds=1)
        
    def wait(self):
        """Wait if needed to respect rate limits."""
        now = datetime.now()
        time_since_last = (now - self.last_call).total_seconds()
        
        if time_since_last < 1.0 / self.max_rps:
            sleep_time = (1.0 / self.max_rps) - time_since_last
            time.sleep(sleep_time)
            
        self.last_call = datetime.now()

class APIKey:
    """API key with rate limiting."""
    
    def __init__(self, key: str, max_rps: int):
        self.key = key
        self.rate_limiter = RateLimiter(max_rps)

class KeyRotator:
    """Manages API key rotation."""
    
    def __init__(self, keys: List[APIKey]):
        self.keys = keys
        self.current_index = 0
        
    def get_next_key(self) -> APIKey:
        """Get next available API key."""
        key = self.keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.keys)
        return key

def get_email_patterns() -> str:
    """Get the list of rejection and acknowledgment patterns."""
    return (
        "Email Pattern Categories:\n\n"
        "1. Rejection Patterns:\n"
        "- Direct rejection phrases:\n"
        "  * 'regret to inform'\n"
        "  * 'unfortunately'\n"
        "  * 'not selected'\n"
        "  * 'position has been filled'\n"
        "  * 'decided to move forward with other candidates'\n"
        "- Standard rejection indicators:\n"
        "  * 'wish you success'\n"
        "  * 'best of luck'\n"
        "  * 'future opportunities'\n"
        "\n2. Application Acknowledgment Patterns:\n"
        "- Confirmation phrases:\n"
        "  * 'thank you for your application'\n"
        "  * 'application received'\n"
        "  * 'application has been processed'\n"
        "  * 'we will keep your resume on file'\n"
        "  * 'will contact you if'\n"
        "- Automated response patterns:\n"
        "  * 'this is an automated response'\n"
        "  * 'do not reply'\n"
        "  * 'no-reply'\n"
        "\nClassification Rules:\n"
        "- Mark as 'rejection' ONLY if rejection patterns are found\n"
        "- Mark as 'acknowledgment' for application confirmation patterns\n"
        "- Consider email context and tone, not just individual phrases"
    )

def get_system_prompt() -> str:
    """Get common system prompt for email analysis."""
    return (
        "You are an AI assistant analyzing job-related emails. Analyze the content carefully to categorize emails properly.\n\n"
        f"Email Classification Guide:\n"
        "1. Rejection Analysis:\n"
        "   - Look for explicit rejection language and patterns\n"
        "   - Mark as 'rejection' ONLY if clear rejection is found\n"
        "   - Rejection must be explicit and direct\n\n"
        "2. Acknowledgment Analysis:\n"
        "   - Check for application confirmation patterns\n"
        "   - Mark as 'acknowledgment' for standard responses\n"
        "   - Application receipts and automated responses\n\n"
        f"{get_email_patterns()}\n\n"
        "3. Selection Analysis:\n"
        "   - ONLY mark as 'selection' if you find:\n"
        "     * Direct interview invitations\n"
        "     * Specific next steps in hiring process\n"
        "     * Clear positive responses about application\n"
        "   - Must have concrete action items or requests\n\n"
        "4. Other Categories:\n"
        "   - Mark as 'other_job' if job-related but unclear status\n"
        "   - Mark as 'not_job' for non-job-related emails\n\n"
        "Action Rules:\n"
        "1. For type 'rejection': set action: delete (if enabled)\n"
        "2. For type 'acknowledgment': set action: delete (if enabled)\n"
        "3. For type 'selection': set action: label_selected\n"
        "4. For all others: set action: keep\n\n"
        "Return JSON with these fields:\n"
        "{\n"
        "  \"is_job_related\": boolean,\n"
        "  \"type\": string (rejection/acknowledgment/selection/other_job/not_job),\n"
        "  \"action\": string (delete/label_selected/keep),\n"
        "  \"confidence\": number (0-1),\n"
        "  \"reason\": string (brief explanation with matched patterns)\n"
        "}"
    )

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with config."""
        self.features = config.get('features', {})
        
    @abstractmethod
    def process_text(self, prompt: str) -> str:
        """Process text through LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling various formats."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*({[^}]+})\s*```', response)
            if json_match:
                json_str = json_match.group(1)
                json_str = re.sub(r'([{\s,])([a-zA-Z_][a-zA-Z0-9_]*):',
                                r'\1"\2":', json_str)  # Quote unquoted keys
                json_str = re.sub(r'([^\\])"([^"]*)\n([^"]*)"', 
                                r'\1"\2 \3"', json_str)  # Fix newlines in values
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse cleaned JSON: {e}")
                    logger.error(f"Cleaned JSON string: {json_str}")
            
            logger.error(f"Could not extract valid JSON from response: {response}")
            return None

    def analyze_job_email(self, email_content: str) -> Dict[str, Any]:
        """Analyze job-related email content."""
        try:
            response = self.process_text(email_content)
            logger.debug(f"LLM Response: {response}")
            
            default_response = {
                "is_job_related": False,
                "type": "unknown",
                "action": "keep",
                "confidence": 0.0,
                "reason": "Failed to analyze email"
            }
            
            analysis = self._extract_json_from_response(response)
            if not analysis:
                return default_response
                
            required_fields = ['is_job_related', 'type', 'action', 'confidence']
            if not all(field in analysis for field in required_fields):
                logger.error(f"Missing required fields in analysis: {analysis}")
                return default_response

            # Apply feature flags and action rules based on email type
            if analysis['type'] == 'rejection':
                if not self.features.get('enable_delete_rejection', False):
                    logger.info("Delete rejection disabled - keeping rejection email")
                    analysis['action'] = 'keep'
                else:
                    analysis['action'] = 'delete'

            elif analysis['type'] == 'acknowledgment':
                if not self.features.get('enable_delete_job_application', False):
                    logger.info("Delete job application disabled - keeping acknowledgment email")
                    analysis['action'] = 'keep'
                else:
                    analysis['action'] = 'delete'

            elif analysis['type'] == 'selection':
                analysis['action'] = 'label_selected'
            
            elif analysis['type'] in ['other_job', 'not_job']:
                logger.info(f"Setting keep action for {analysis['type']} type")
                analysis['action'] = 'keep'
            
            # Validate non-job emails are always kept
            if analysis['type'] == 'not_job' and analysis['action'] != 'keep':
                logger.warning("non_job type must have keep action - correcting")
                analysis['action'] = 'keep'
                
            # Only accept high confidence classifications for delete actions
            if analysis['action'] == 'delete' and analysis['confidence'] < 0.8:
                logger.info(f"Low confidence ({analysis['confidence']}) for deletion - defaulting to keep")
                analysis['action'] = 'keep'
                
            return analysis
            
        except Exception as e:
            logger.error(f"Email analysis failed: {str(e)}")
            return {
                "is_job_related": False,
                "type": "error",
                "action": "keep",
                "confidence": 0.0,
                "reason": f"Analysis error: {str(e)}"
            }

class CodestralProvider(BaseLLMProvider):
    """Codestral LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Codestral provider."""
        super().__init__(config)
        max_rps = config.get('max_rps', 1)
        api_keys = [
            APIKey(key=key, max_rps=max_rps) 
            for key in config.get('api_keys', [])
        ]
        self.key_rotator = KeyRotator(api_keys)
        
        self.api_url = config['llm_api_url'].rstrip('/')
        self.model = config['llm_model']
        self.max_tokens = config.get('max_tokens', 4096)
        
        logger.info(f"Initialized Codestral provider with model: {self.model}")
        
    def process_text(self, prompt: str) -> str:
        """Process text through Codestral API with rate limiting."""
        api_key = self.key_rotator.get_next_key()
        api_key.rate_limiter.wait()
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key.key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this email for job-related content:\n\nSubject: {prompt['subject']}\nFrom: {prompt['from']}\n\nContent:\n{prompt['content']}"
                    }
                ],
                "temperature": 0.3,
                "max_tokens": self.max_tokens
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
            
            content = response.json()
            return content['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"Error during LLM processing: {str(e)}")
            raise
            
    def is_available(self) -> bool:
        """Check if Codestral service is available."""
        try:
            api_key = self.key_rotator.get_next_key()
            headers = {
                "Authorization": f"Bearer {api_key.key}",
                "Accept": "application/json"
            }
            
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.api_url}/models",
                    headers=headers
                )
                response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Service availability check failed: {str(e)}")
            return False

class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        max_rps = config.get('max_rps', 3)
        api_keys = [
            APIKey(key=key, max_rps=max_rps)
            for key in config.get('api_keys', [])
        ]
        self.key_rotator = KeyRotator(api_keys)
        self.model = config['llm_model']
        self.max_tokens = config.get('max_tokens', 128000)
        self.temperature = config.get('temperature', 0.7)
        
        logger.info(f"Initialized OpenAI provider with model: {self.model}")

    def process_text(self, prompt: str) -> str:
        api_key = self.key_rotator.get_next_key()
        api_key.rate_limiter.wait()

        try:
            headers = {
                "Authorization": f"Bearer {api_key.key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this email for job-related content:\n\nSubject: {prompt['subject']}\nFrom: {prompt['from']}\n\nContent:\n{prompt['content']}"
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()

            content = response.json()
            return content['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"Error during OpenAI processing: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if OpenAI service is available."""
        try:
            api_key = self.key_rotator.get_next_key()
            headers = {
                "Authorization": f"Bearer {api_key.key}",
                "Content-Type": "application/json"
            }
            
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    "https://api.openai.com/v1/models",
                    headers=headers
                )
                response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Service availability check failed: {str(e)}")
            return False

class LlamaProvider(BaseLLMProvider):
    """Llama provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        max_rps = config.get('max_rps', 1)
        api_keys = [
            APIKey(key=key, max_rps=max_rps)
            for key in config.get('api_keys', [])
        ]
        self.key_rotator = KeyRotator(api_keys)
        self.api_url = config['llm_api_url'].rstrip('/')
        self.model = config['llm_model']
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 0.7)
        
        logger.info(f"Initialized Llama provider with model: {self.model}")

    def process_text(self, prompt: str) -> str:
        api_key = self.key_rotator.get_next_key()
        api_key.rate_limiter.wait()

        try:
            headers = {
                "Authorization": f"Bearer {api_key.key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this email for job-related content:\n\nSubject: {prompt['subject']}\nFrom: {prompt['from']}\n\nContent:\n{prompt['content']}"
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()

            content = response.json()
            return content['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"Error during Llama processing: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if Llama service is available."""
        try:
            api_key = self.key_rotator.get_next_key()
            headers = {
                "Authorization": f"Bearer {api_key.key}",
                "Content-Type": "application/json"
            }
            
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.api_url}/models",
                    headers=headers
                )
                response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Service availability check failed: {str(e)}")
            return False

def load_llm_config() -> Dict[str, Any]:
    """Load LLM configuration from secret.yaml."""
    config_path = os.path.join('llm', 'secret.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load LLM config: {str(e)}")
        raise

def create_provider(provider_type: Optional[str] = None) -> BaseLLMProvider:
    """Factory function to create LLM provider instance."""
    config = load_llm_config()
    
    # Use active_provider from config if provider_type not specified
    if provider_type is None:
        provider_type = config.get('active_provider', 'codestral')

    # Add feature flags to provider config
    provider_config = config.get(provider_type, {})
    provider_config['features'] = config.get('features', {})

    if provider_type == 'codestral':
        return CodestralProvider(provider_config)
    elif provider_type == 'openai':
        return OpenAIProvider(provider_config)
    elif provider_type == 'llama':
        return LlamaProvider(provider_config)
    
    raise ValueError(f"Unsupported provider type: {provider_type}")