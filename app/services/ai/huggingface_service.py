import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, Optional, List
from app.core.config import settings

logger = logging.getLogger(__name__)


class HuggingFaceAPI:
    """Enhanced HuggingFace API client with multiple endpoint support"""

    def __init__(self):
        self.base_url = "https://api-inference.huggingface.co/models"
        self.last_request_time = 0
        self.min_delay = 2.0  # Rate limiting for free tier

        # Model-specific configurations
        self.model_configs = {
            "deepseek/deepseek-v3-0324": {
                "endpoint_type": "openai_compatible",
                "custom_endpoint": "https://router.huggingface.co/novita/v3/openai/chat/completions"
            },
            "microsoft/Phi-3-mini-4k-instruct": {
                "endpoint_type": "standard"
            },
            "deepset/roberta-base-squad2": {
                "endpoint_type": "qa"
            }
        }

    def _get_headers(self, use_auth=True, endpoint_type="standard"):
        """Get headers with proper authentication"""
        if endpoint_type == "openai_compatible":
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "FastAPI-HF-Client/1.0"
            }
            if use_auth and settings.HUGGINGFACE_API_KEY:
                headers["Authorization"] = f"Bearer {settings.HUGGINGFACE_API_KEY}"
        else:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "FastAPI-HF-Client/1.0"
            }
            if use_auth and settings.HUGGINGFACE_API_KEY:
                headers["Authorization"] = f"Bearer {settings.HUGGINGFACE_API_KEY}"

        return headers

    async def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()

    async def generate_text(
            self,
            model_name: str,
            prompt: str,
            parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate text with model-specific handling"""

        if parameters is None:
            parameters = {}

        model_config = self.model_configs.get(model_name, {"endpoint_type": "standard"})
        endpoint_type = model_config["endpoint_type"]

        if endpoint_type == "openai_compatible":
            return await self._generate_openai_compatible(model_name, prompt, parameters)
        elif endpoint_type == "qa":
            return await self._generate_qa(model_name, prompt, parameters)
        else:
            return await self._generate_standard(model_name, prompt, parameters)

    async def _generate_openai_compatible(self, model_name: str, prompt: str, parameters: Dict[str, Any]) -> Dict[
        str, Any]:
        """Generate using OpenAI-compatible endpoint (DeepSeek)"""
        await self._rate_limit()

        model_config = self.model_configs[model_name]
        url = model_config["custom_endpoint"]

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": parameters.get("temperature", 0.7),
            "max_tokens": min(parameters.get("max_tokens", 150), 500),
            "top_p": parameters.get("top_p", 0.9)
        }

        headers = self._get_headers(endpoint_type="openai_compatible")

        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        return {
                            "text": content,
                            "model": model_name,
                            "provider": "huggingface"
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI-compatible API error {response.status}: {error_text}")
                        return self._fallback_response(model_name, prompt)

        except Exception as e:
            logger.error(f"OpenAI-compatible endpoint error: {e}")
            return self._fallback_response(model_name, prompt)

    async def _generate_qa(self, model_name: str, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate using QA endpoint (RoBERTa)"""
        await self._rate_limit()

        url = f"{self.base_url}/{model_name}"

        # For QA models, we need context and question
        # Try to parse prompt or use as question with empty context
        parts = prompt.split("Context:", 1)
        if len(parts) == 2:
            context_and_q = parts[1].split("Question:", 1)
            if len(context_and_q) == 2:
                context = context_and_q[0].strip()
                question = context_and_q[1].strip()
            else:
                context = parts[1].strip()
                question = "What is this about?"
        else:
            context = ""
            question = prompt

        payload = {
            "inputs": {
                "question": question,
                "context": context if context else prompt
            }
        }

        headers = self._get_headers()

        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result.get("answer", "No answer found")
                        score = result.get("score", 0)
                        return {
                            "text": f"Answer: {answer} (Confidence: {score:.2f})",
                            "model": model_name,
                            "provider": "huggingface"
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"QA API error {response.status}: {error_text}")
                        return self._fallback_response(model_name, prompt)

        except Exception as e:
            logger.error(f"QA endpoint error: {e}")
            return self._fallback_response(model_name, prompt)

    async def _generate_standard(self, model_name: str, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate using standard HF inference API"""
        await self._rate_limit()

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": min(parameters.get("max_tokens", 100), 200),
                "temperature": parameters.get("temperature", 0.7),
                "do_sample": parameters.get("do_sample", True),
                "return_full_text": parameters.get("return_full_text", False),
            },
            "options": {
                "wait_for_model": True,
                "use_cache": True
            }
        }

        url = f"{self.base_url}/{model_name}"
        headers = self._get_headers()

        auth_methods = [
            (True, "with authentication"),
            (False, "without authentication (free tier)")
        ]

        if not settings.HUGGINGFACE_API_KEY:
            auth_methods = [(False, "free tier only")]

        for use_auth, method_desc in auth_methods:
            headers = self._get_headers(use_auth)

            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as response:

                        if response.status == 200:
                            result = await response.json()
                            return self._process_response(result, model_name)

                        elif response.status == 503:
                            error_data = await response.json()
                            estimated_time = error_data.get("estimated_time", 20)

                            if estimated_time < 60:
                                await asyncio.sleep(min(estimated_time + 5, 30))
                                async with session.post(url, headers=headers, json=payload) as retry_response:
                                    if retry_response.status == 200:
                                        result = await retry_response.json()
                                        return self._process_response(result, model_name)

                        elif response.status == 401 and use_auth:
                            continue  # Try without auth

            except Exception as e:
                logger.warning(f"Standard endpoint error {method_desc}: {e}")
                continue

        return self._fallback_response(model_name, prompt)

    def _process_response(self, response: Any, model_name: str) -> Dict[str, Any]:
        """Process HuggingFace API response"""
        if isinstance(response, list) and len(response) > 0:
            item = response[0]
            if isinstance(item, dict) and "generated_text" in item:
                return {
                    "text": item["generated_text"],
                    "model": model_name,
                    "provider": "huggingface"
                }

        return {
            "text": str(response),
            "model": model_name,
            "provider": "huggingface"
        }

    def _fallback_response(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Provide fallback response when API fails"""
        return {
            "text": f"I apologize, but I'm having trouble connecting to the {model_name} model right now. Please try again in a moment.",
            "model": model_name,
            "provider": "huggingface",
            "fallback": True
        }

    async def test_model_availability(self, model_name: str) -> bool:
        """Test if a model is available"""
        try:
            result = await self.generate_text(
                model_name=model_name,
                prompt="Hello",
                parameters={"max_tokens": 10}
            )
            return not result.get("fallback", False)
        except:
            return False


# Global instance
hf_api = HuggingFaceAPI()