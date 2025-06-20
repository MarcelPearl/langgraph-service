"""
Local HuggingFace Service using Transformers
Runs models locally - NO API CALLS NEEDED!
"""

import asyncio
import torch
import logging
from typing import Dict, Any, Optional, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class LocalHuggingFaceService:
    """Local HuggingFace service using a transformer library"""

    def __init__(self):
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chat_histories = {}

    async def load_model(self, model_name: str, task_type: str = "text-generation"):
        """Load model locally with proper authentication handling"""
        if model_name in self.loaded_models:
            return

        logger.info(f"Loading {model_name} locally for {task_type}...")

        try:
            # Configure authentication
            use_auth_token = None
            if settings.HUGGINGFACE_API_KEY and settings.HUGGINGFACE_API_KEY.startswith('hf_'):
                use_auth_token = settings.HUGGINGFACE_API_KEY
            else:
                # Try without authentication for public models
                use_auth_token = False
                logger.info("No valid HF token, trying without authentication...")

            if task_type == "question-answering":
                # Load QA model
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_auth_token=use_auth_token,
                    trust_remote_code=True
                )
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name,
                    use_auth_token=use_auth_token,
                    trust_remote_code=True
                )

                self.loaded_tokenizers[model_name] = tokenizer
                self.loaded_models[model_name] = model

            else:
                # Load generative model
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_auth_token=use_auth_token,
                    trust_remote_code=True
                )

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_auth_token=use_auth_token
                )

                model.to(self.device)
                model.eval()

                self.loaded_tokenizers[model_name] = tokenizer
                self.loaded_models[model_name] = model

            logger.info(f"✅ {model_name} loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")

            # Try alternative public models if the requested one fails
            alternative = self._get_alternative_model(model_name)
            if alternative and alternative != model_name:
                logger.info(f"Trying alternative model: {alternative}")
                return await self.load_model(alternative, task_type)
            else:
                raise

    def _get_alternative_model(self, failed_model: str) -> str:
        """Get alternative public models when authentication fails"""
        alternatives = {
            "microsoft/Phi-3-mini-4k-instruct": "microsoft/DialoGPT-medium",
            "deepseek/deepseek-v3-0324": "microsoft/DialoGPT-medium",
            "deepset/roberta-base-squad2": "distilbert-base-cased-distilled-squad"
        }
        return alternatives.get(failed_model, "gpt2")

    async def generate_text_local(
            self,
            model_name: str,
            prompt: str,
            parameters: Optional[Dict[str, Any]] = None,
            conversation_id: str = "default"
    ) -> Dict[str, Any]:
        """Generate text using local model"""

        if parameters is None:
            parameters = {}

        # Determine task type
        if "roberta" in model_name.lower() or "squad" in model_name.lower():
            task_type = "question-answering"
        else:
            task_type = "text-generation"

        # Load model if not loaded
        if model_name not in self.loaded_models:
            await self.load_model(model_name, task_type)

        try:
            if task_type == "question-answering":
                return await self._generate_qa_local(model_name, prompt, parameters)
            else:
                return await self._generate_text_local(model_name, prompt, parameters, conversation_id)

        except Exception as e:
            logger.error(f"Error generating text locally: {e}")
            return {
                "text": f"Local generation failed: {str(e)}",
                "model": model_name,
                "provider": "local_huggingface",
                "error": True
            }

    async def _generate_qa_local(self, model_name: str, prompt: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate QA response locally"""
        model = self.loaded_models[model_name]
        tokenizer = self.loaded_tokenizers[model_name]

        # Parse context and question from prompt
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
            context = prompt
            question = "What is this about?"

        # Use pipeline for easier QA
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == "cuda" else -1
        )

        result = qa_pipeline(question=question, context=context)

        return {
            "text": f"Answer: {result['answer']} (Confidence: {result['score']:.2f})",
            "model": model_name,
            "provider": "local_huggingface"
        }

    async def _generate_text_local(self, model_name: str, prompt: str, parameters: Dict[str, Any],
                                   conversation_id: str) -> Dict[str, Any]:
        """Generate text locally"""
        model = self.loaded_models[model_name]
        tokenizer = self.loaded_tokenizers[model_name]

        max_tokens = parameters.get("max_tokens", 100)
        temperature = parameters.get("temperature", 0.7)

        # Handle conversation history
        chat_history_ids = self.chat_histories.get(conversation_id, None)

        # Format prompt for Phi-3 if needed
        if "phi-3" in model_name.lower():
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        else:
            formatted_prompt = prompt

        # Encode input
        new_input_ids = tokenizer.encode(
            formatted_prompt + tokenizer.eos_token,
            return_tensors='pt'
        ).to(self.device)

        # Append to history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate
        with torch.no_grad():
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=min(bot_input_ids.shape[-1] + max_tokens, 1000),
                min_length=bot_input_ids.shape[-1] + 1,
                temperature=temperature,
                do_sample=parameters.get("do_sample", True),
                pad_token_id=tokenizer.eos_token_id,
                top_p=parameters.get("top_p", 0.9),
                repetition_penalty=1.1,
                num_return_sequences=1
            )

        # Store history
        self.chat_histories[conversation_id] = chat_history_ids

        # Decode response
        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        response = response.strip()
        if not response:
            response = "I'm not sure how to respond to that."

        return {
            "text": response,
            "model": model_name,
            "provider": "local_huggingface"
        }

    def clear_conversation(self, conversation_id: str = "default"):
        """Clear conversation history"""
        if conversation_id in self.chat_histories:
            del self.chat_histories[conversation_id]

    def get_loaded_models(self) -> List[str]:
        """Get a list of loaded models"""
        return list(self.loaded_models.keys())


local_hf_service = LocalHuggingFaceService()