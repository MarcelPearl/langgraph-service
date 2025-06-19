
from typing import Dict, Any, Optional, List
import asyncio
import aiohttp
import time
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class HuggingFaceRateLimiter:
    """Rate limiter for Hugging Face free tier"""

    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
        self.reset_time = 0

    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()

        if not settings.HUGGINGFACE_API_KEY:
            time_since_last = current_time - self.last_request_time
            if time_since_last < settings.HUGGINGFACE_FREE_TIER_DELAY:
                wait_time = settings.HUGGINGFACE_FREE_TIER_DELAY - time_since_last
                logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)

        self.last_request_time = time.time()


class HuggingFaceAPI:
    """Direct Hugging Face API client for advanced usage"""

    def __init__(self):
        self.rate_limiter = HuggingFaceRateLimiter()
        self.base_url = settings.HUGGINGFACE_INFERENCE_URL
        self.headers = {
            "Content-Type": "application/json"
        }

        if settings.HUGGINGFACE_API_KEY:
            self.headers["Authorization"] = f"Bearer {settings.HUGGINGFACE_API_KEY}"

    async def generate_text(
            self,
            model_name: str,
            prompt: str,
            parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate text using a Hugging Face model"""

        await self.rate_limiter.wait_if_needed()

        if parameters is None:
            parameters = {}

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": parameters.get("max_tokens", 500),
                "temperature": parameters.get("temperature", 0.7),
                "top_p": parameters.get("top_p", 0.9),
                "do_sample": parameters.get("do_sample", True),
                "return_full_text": parameters.get("return_full_text", False),
                **parameters
            },
            "options": {
                "wait_for_model": True,
                "use_cache": parameters.get("use_cache", True)
            }
        }

        url = f"{self.base_url}/{model_name}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=settings.AI_REQUEST_TIMEOUT)
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        return self._process_response(result, model_name)

                    elif response.status == 503:
                        error_data = await response.json()
                        estimated_time = error_data.get("estimated_time", 20)
                        logger.warning(f"Model {model_name} is loading, estimated time: {estimated_time}s")

                        if estimated_time < 60:  # Wait up to 1 minute
                            await asyncio.sleep(estimated_time + 5)
                            return await self.generate_text(model_name, prompt, parameters)
                        else:
                            raise Exception(f"Model loading time too long: {estimated_time}s")

                    elif response.status == 429:
                        logger.warning("Rate limited by Hugging Face API")
                        await asyncio.sleep(5)
                        raise Exception("Rate limited - try again later")

                    else:
                        error_text = await response.text()
                        logger.error(f"HF API error {response.status}: {error_text}")
                        raise Exception(f"Hugging Face API error: {response.status}")

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling Hugging Face model: {model_name}")
            raise Exception("Request timeout")
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {e}")
            raise

    def _process_response(self, response: Any, model_name: str) -> Dict[str, Any]:
        """Process Hugging Face API response"""

        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get("generated_text", "")
            return {
                "text": generated_text,
                "model": model_name,
                "provider": "huggingface"
            }
        elif isinstance(response, dict):
            return {
                "result": response,
                "model": model_name,
                "provider": "huggingface"
            }
        else:
            logger.warning(f"Unexpected response format from {model_name}: {response}")
            return {
                "text": str(response),
                "model": model_name,
                "provider": "huggingface"
            }

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a Hugging Face model"""

        url = f"https://huggingface.co/api/models/{model_name}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"Could not fetch model info: {response.status}"}
        except Exception as e:
            logger.error(f"Error fetching model info: {e}")
            return {"error": str(e)}

    async def list_available_models(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available models for a specific task"""

        url = "https://huggingface.co/api/models"
        params = {
            "filter": "inference",
            "sort": "downloads",
            "direction": -1,
            "limit": 50
        }

        if task:
            params["pipeline_tag"] = task

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        models = await response.json()
                        return [
                            {
                                "id": model.get("id"),
                                "pipeline_tag": model.get("pipeline_tag"),
                                "downloads": model.get("downloads", 0),
                                "likes": model.get("likes", 0),
                                "tags": model.get("tags", [])
                            }
                            for model in models
                        ]
                    else:
                        return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


hf_api = HuggingFaceAPI()