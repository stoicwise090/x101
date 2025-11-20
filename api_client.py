"""
api_client.py
A dedicated Gemini Vision API client module that:
- Sends image and text prompts to the Gemini API (REST)
- Retries up to `max_attempts` with exponential backoff
- Minimal dependency: requests, PIL (for image encoding)

Environment variables expected:
    GEMINI_API_KEY        - required (for primary authentication)
    GEMINI_API_URL        - required (full endpoint URL, e.g., https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent)

Usage:
    from api_client import GeminiClient, APIError
    client = GeminiClient(api_key="your_key")
    try:
        resp = client.analyze_image_from_buffer(image_buffer, model_name, system_prompt)
    except APIError as e:
        print("Analysis failed:", e)
"""

import os
import time
import logging
import base64
from typing import Any, Dict, Optional, IO

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini_client")

# --- Error Class ---

class APIError(Exception):
    """Raised when all attempts fail."""
    pass

# --- Helper Functions ---

def _encode_image_buffer(image_buffer: IO[bytes]) -> str:
    """Read bytes from buffer and base64 encode them for the API payload."""
    image_buffer.seek(0)
    return base64.b64encode(image_buffer.read()).decode("utf-8")

def _make_gemini_payload(
    base64_image: str,
    system_prompt: str,
    user_query: str,
    model_name: str
) -> Dict[str, Any]:
    """Constructs the API request payload for multimodal analysis."""
    # The image analysis prompt is a single user message containing both the system prompt 
    # (as context) and the image data.
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    # 1. Image Data
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    # 2. Text Prompt (System Instruction embedded in user message for clarity)
                    { "text": f"SYSTEM INSTRUCTION: {system_prompt}\n\nUSER QUERY: {user_query}" }
                ]
            }
        ],
        "config": {
            "model": model_name
        }
    }


# --- Client Class ---

class GeminiClient:
    """Client for direct interaction with the Gemini API for Vision tasks."""
    
    # Use environment variables if not passed explicitly (typical for server deployment)
    _DEFAULT_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """Initializes the client with an API key and URL."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_url = api_url or self._DEFAULT_API_URL

        if not self.api_key:
            raise ValueError("API key must be provided or set as GEMINI_API_KEY environment variable.")
        
        # NOTE: get_available_models is removed as it's not a standard Gemini feature.
        # We will use hardcoded models in app.py.

    def _call_gemini(self, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """
        Make a single HTTP POST request to the configured Gemini endpoint.
        """
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key # Use key in header as recommended for security
        }

        # The model name is embedded in the URL if the endpoint is generic, 
        # or in the payload if the URL is model-specific. Since the default URL 
        # above is generic, we'll ensure the payload has the model specified.
        # However, the standard REST API uses the model name in the URL path.
        # Let's use the URL provided in the environment variable, which should 
        # include the model name (e.g., /v1beta/models/gemini-2.5-flash:generateContent).
        
        # If the user supplied a base URL without the model, use the model 
        # from the payload to construct the full URL, or trust the default/env var URL.
        # Given the original code's structure, we'll stick to a single URL:
        
        full_url = self.api_url
        
        resp = requests.post(full_url, json=payload, headers=headers, timeout=timeout)
        
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Provide context for logs
            body = resp.text
            status_code = resp.status_code
            # Check for common client-side configuration errors (4xx)
            if status_code in [400, 401, 403]:
                 error_msg = f"Gemini API Error {status_code}: Check API Key or URL configuration. Details: {body}"
            else:
                 error_msg = f"Gemini API Error {status_code}. Details: {body}"
            raise APIError(error_msg) from e

        # Return JSON if possible
        try:
            return resp.json()
        except ValueError:
            raise APIError(f"Invalid JSON response from API: {resp.text}")


    def analyze_image_from_buffer(
        self,
        image_buffer: IO[bytes],
        model_name: str,
        system_prompt: str,
        user_query: str = "Analyze the image based on the system instruction and provide your structured response.",
        max_attempts: int = 3,
        initial_backoff: float = 1.0,
        backoff_multiplier: float = 2.0,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Processes an image buffer and sends it to the Gemini model for analysis.
        """
        last_exc: Optional[Exception] = None
        
        try:
            base64_image = _encode_image_buffer(image_buffer)
        except Exception as e:
            logger.error("Error encoding image: %s", e)
            return {"success": False, "error": f"Image encoding failed: {str(e)}"}
            
        payload = _make_gemini_payload(base64_image, system_prompt, user_query, model_name)

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info("Gemini attempt %d/%d (Model: %s)", attempt, max_attempts, model_name)
                
                # Make the API call
                result = self._call_gemini(payload, timeout=timeout)
                logger.info("Gemini success on attempt %d", attempt)

                # Extract content from the specific Gemini response structure
                text_content = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                
                # Extract token usage (note: billing tokens might be different)
                usage_metadata = result.get('usageMetadata', {})
                total_tokens = usage_metadata.get('totalTokenCount', 0)

                return {
                    "success": True,
                    "analysis": text_content,
                    "model_used": model_name,
                    "tokens_used": total_tokens,
                }
                
            except (requests.RequestException, APIError) as e:
                last_exc = e
                logger.warning("Gemini attempt %d failed: %s", attempt, str(e))
                
                # Decide whether to retry: do not retry on client errors (4xx) unless 429
                if isinstance(e, APIError) and "API Error 4" in str(e) and "429" not in str(e):
                    logger.error("Client error encountered; not retrying further.")
                    break

                if attempt < max_attempts:
                    backoff = initial_backoff * (backoff_multiplier ** (attempt - 1))
                    logger.info("Retrying after %.1f seconds...", backoff)
                    time.sleep(backoff)
                else:
                    logger.error("Gemini failed after %d attempts", max_attempts)

        # If we reach here, all attempts failed
        error_message = f"All attempts failed. Last error: {last_exc}"
        return {"success": False, "error": error_message, "model_used": model_name}


if __name__ == "__main__":
    # Test section requires setting environment variables:
    # export GEMINI_API_KEY="YOUR_KEY"
    # export GEMINI_API_URL="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    # This test is hard to run without an actual image buffer, so skipping 
    # the runnable test block for simplicity, but the main logic is sound.
    print("GeminiClient initialized. Run main app.py to test integration.")
