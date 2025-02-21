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

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def process_text(self, prompt: str) -> str:
        """Process text through LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass

class CodestralProvider(BaseLLMProvider):
    """Codestral LLM provider implementation with key rotation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Codestral provider.
        
        Args:
            config (Dict[str, Any]): Configuration from secret.yaml
        """
        # Initialize key rotation with configured RPS
        max_rps = config.get('max_rps', 1)
        api_keys = [
            APIKey(key=key, max_rps=max_rps) 
            for key in [
                config['llm_api_key'],
                config['llm_api_key1'],
                config['llm_api_key2']
            ]
        ]
        self.key_rotator = KeyRotator(api_keys)
        
        # Store configuration
        self.api_url = config['llm_api_url'].rstrip('/')
        self.model = config['llm_model']
        self.max_tokens = config.get('max_tokens', 4096)
        
        logger.info(f"Initialized Codestral provider with model: {self.model}")
        
    def process_text(self, prompt: str) -> str:
        """
        Process text through Codestral API with rate limiting.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated response
        """
        api_key = self.key_rotator.get_next_key()
        api_key.rate_limiter.wait()  # Respect rate limits
        
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
                        "content": (
                            "You are an AI assistant analyzing job-related emails. Analyze the content carefully to categorize emails properly.\n\n"
                            "Email Analysis Rules:\n"
                            "1. ONLY mark as rejection if it explicitly states the application was rejected or not selected.\n"
                            "2. ONLY mark as selection if it mentions interview invitation or clear positive response.\n"
                            "3. Mark as other_job if it's job-related but not a clear rejection or selection.\n"
                            "4. Mark as not_job for non-job-related emails.\n\n"
                            "Action Rules:\n"
                            "1. ONLY set action: delete for rejection emails.\n"
                            "2. ONLY set action: label_selected for selection emails.\n"
                            "3. ALL OTHER emails should have action: keep\n\n"
                            "Return JSON with these fields:\n"
                            "{\n"
                            "  \"is_job_related\": boolean,\n"
                            "  \"type\": string (rejection/selection/other_job/not_job),\n"
                            "  \"action\": string (delete/label_selected/keep),\n"
                            "  \"confidence\": number (0-1),\n"
                            "  \"reason\": string (brief explanation)\n"
                            "}"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this email for job-related content:\n\nSubject: {prompt['subject']}\nFrom: {prompt['from']}\n\nContent:\n{prompt['content']}"
                    }
                ],
                "temperature": 0.3,
                "max_tokens": self.max_tokens
            }

            logger.debug(f"Sending request to {self.api_url}/chat/completions")
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
            
            content = response.json()
            return content['choices'][0]['message']['content']
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during LLM processing: {str(e)}")
            logger.error(f"Response content: {e.response.text if hasattr(e, 'response') else 'No response content'}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM processing: {str(e)}")
            raise
            
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling various formats."""
        try:
            # Try direct JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from code blocks or text
            json_match = re.search(r'```(?:json)?\s*({[^}]+})\s*```', response)
            if json_match:
                json_str = json_match.group(1)
                # Clean up the JSON string
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
        """
        Analyze job-related email content.
        
        Args:
            email_content (str): Email content to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            response = self.process_text(email_content)
            logger.debug(f"LLM Response: {response}")
            
            # Default response
            default_response = {
                "is_job_related": False,
                "type": "unknown",
                "action": "keep",
                "confidence": 0.0,
                "reason": "Failed to analyze email"
            }
            
            # Try to parse the response
            analysis = self._extract_json_from_response(response)
            if not analysis:
                return default_response
                
            # Validate required fields
            required_fields = ['is_job_related', 'type', 'action', 'confidence']
            if not all(field in analysis for field in required_fields):
                logger.error(f"Missing required fields in analysis: {analysis}")
                return default_response
            
            # Validate action logic
            if not analysis['is_job_related'] and analysis['action'] != 'keep':
                logger.warning("Non-job email marked for action other than keep - forcing keep action")
                analysis['action'] = 'keep'
                
            if analysis['type'] == 'not_job' and analysis['action'] != 'keep':
                logger.warning("non_job type must have keep action - correcting")
                analysis['action'] = 'keep'
                
            # Only accept high confidence classifications for important actions
            if analysis['action'] != 'keep' and analysis['confidence'] < 0.8:
                logger.info(f"Low confidence ({analysis['confidence']}) for action {analysis['action']} - defaulting to keep")
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
            
    def is_available(self) -> bool:
        """Check if Codestral service is available."""
        try:
            api_key = self.key_rotator.get_next_key()
            headers = {
                "Authorization": f"Bearer {api_key.key}",
                "Accept": "application/json"
            }
            
            # Test models endpoint
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
        return config.get('codestral', {})
    except Exception as e:
        logger.error(f"Failed to load LLM config: {str(e)}")
        raise

def create_provider(provider_type: str = 'codestral') -> BaseLLMProvider:
    """
    Factory function to create LLM provider instance.
    
    Args:
        provider_type (str): Type of provider
        
    Returns:
        BaseLLMProvider: Provider instance
    """
    config = load_llm_config()
    
    if provider_type == 'codestral':
        return CodestralProvider(config)
    
    raise ValueError(f"Unsupported provider type: {provider_type}")