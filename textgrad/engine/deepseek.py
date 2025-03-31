import os
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import Optional
from .base import EngineLM, CachedEngine
import platformdirs
import json

class DeepSeekOpenRouterEngine(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = "deepseek/deepseek-chat-v3-0324:free",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize DeepSeek engine through OpenRouter
        
        :param model_string: Model identifier (e.g. deepseek/deepseek-chat-v3-0324:free)
        :param system_prompt: Default system prompt
        :param api_key: OpenRouter API key (required)
        :param site_url: Your site URL (required by OpenRouter)
        :param site_name: Your application name (required by OpenRouter)
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_deepseek_{model_string.replace('/', '_')}.db")
        super().__init__(cache_path=cache_path)

        self.model_string = model_string
        self.system_prompt = system_prompt
        
        # Validate required parameters
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Please provide api_key parameter or set OPENROUTER_API_KEY environment variable."
            )

        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url or "https://example.com",
            "X-Title": site_name or "TextGrad App"
        }
        
        # Default generation parameters
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 1.0),
        }

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        Generate response from DeepSeek model
        
        :param prompt: User input prompt
        :param system_prompt: Optional system prompt override
        :param kwargs: Additional generation parameters
        :return: Generated text response
        """
        sys_prompt = system_prompt or self.system_prompt
        
        # Check cache first
        cache_key = sys_prompt + prompt
        cached_response = self._check_cache(cache_key)
        if cached_response is not None:
            return cached_response

        # Prepare request payload
        payload = {
            "model": self.model_string,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            **self.default_params,
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            # Cache the response
            self._save_cache(cache_key, generated_text)
            return generated_text
            
        except requests.exceptions.RequestException as e:
            error_msg = f"DeepSeek API Error: {str(e)}"
            if hasattr(e, 'response') and e.response:
                error_msg += f" | Status: {e.response.status_code} | Response: {e.response.text}"
            raise RuntimeError(error_msg)

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)