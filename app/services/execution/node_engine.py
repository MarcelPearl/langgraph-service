# app/services/execution/node_engine.py
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
from abc import ABC, abstractmethod

from app.services.messaging.kafka_service import (
    kafka_service, NodeExecutionMessage, CompletionEvent
)
from app.services.context.redis_service import redis_context_service
from app.services.ai.model_factory import model_factory
from app.services.ai.huggingface_service import hf_api
from app.core.config import settings

logger = logging.getLogger(__name__)


class NodeExecutorBase(ABC):
    """Base class for node executors"""

    def __init__(self, node_type: str):
        self.node_type = node_type

    @abstractmethod
    async def execute(
            self,
            node_data: Dict[str, Any],
            context: Dict[str, Any],
            execution_id: str
    ) -> Dict[str, Any]:
        """Execute the node and return results"""
        pass


class AIDecisionExecutor(NodeExecutorBase):
    """Executor for AI decision making nodes"""

    def __init__(self):
        super().__init__("ai_decision")

    async def execute(
            self,
            node_data: Dict[str, Any],
            context: Dict[str, Any],
            execution_id: str
    ) -> Dict[str, Any]:
        """Execute AI decision node"""
        try:
            prompt = node_data.get("prompt", "")
            decision_options = node_data.get("options", [])
            ai_config = node_data.get("ai_config", {})

            # Build context-aware prompt
            context_prompt = self._build_context_prompt(prompt, context, decision_options)

            # Execute AI model
            provider = ai_config.get("provider", "huggingface")
            model_name = ai_config.get("model", settings.DEFAULT_HUGGINGFACE_MODEL)

            if provider.lower() == "huggingface":
                result = await hf_api.generate_text(
                    model_name=model_name,
                    prompt=context_prompt,
                    parameters={
                        "temperature": ai_config.get("temperature", 0.3),
                        "max_tokens": ai_config.get("max_tokens", 200)
                    }
                )
                response_text = result["text"]
            else:
                model = model_factory.get_model(
                    provider=provider,
                    model_name=model_name,
                    temperature=ai_config.get("temperature", 0.3)
                )
                response = await model.ainvoke(context_prompt)
                response_text = response.content

            # Parse decision from response
            decision = self._parse_decision(response_text, decision_options)

            return {
                "decision": decision,
                "reasoning": response_text,
                "options_evaluated": decision_options,
                "confidence": self._calculate_confidence(response_text, decision),
                "tokens_used": len(context_prompt.split()) + len(response_text.split())
            }

        except Exception as e:
            logger.error(f"AI decision execution failed: {e}")
            return {
                "error": str(e),
                "decision": decision_options[0] if decision_options else "default",
                "reasoning": f"Failed to execute AI decision: {e}"
            }

    def _build_context_prompt(
            self,
            base_prompt: str,
            context: Dict[str, Any],
            options: List[str]
    ) -> str:
        """Build context-aware prompt for decision making"""
        context_info = ""
        if context.get("previous_results"):
            context_info = f"\nPrevious results: {context['previous_results']}"

        options_text = "\n".join([f"- {option}" for option in options])

        return f"""{base_prompt}

Context: {context_info}

Available options:
{options_text}

Please analyze the situation and choose the best option. Explain your reasoning and state your final decision clearly."""

    def _parse_decision(self, response_text: str, options: List[str]) -> str:
        """Parse decision from AI response"""
        response_lower = response_text.lower()

        # Look for exact option matches
        for option in options:
            if option.lower() in response_lower:
                return option

        # Fallback to first option
        return options[0] if options else "unknown"

    def _calculate_confidence(self, response_text: str, decision: str) -> float:
        """Calculate confidence score based on response"""
        # Simple heuristic - can be enhanced with more sophisticated methods
        confidence_words = ["certain", "confident", "sure", "definitely", "clearly"]
        uncertainty_words = ["maybe", "perhaps", "possibly", "might", "unsure"]

        response_lower = response_text.lower()
        confidence_score = 0.5  # Base confidence

        for word in confidence_words:
            if word in response_lower:
                confidence_score += 0.1

        for word in uncertainty_words:
            if word in response_lower:
                confidence_score -= 0.1

        return max(0.0, min(1.0, confidence_score))


class AITextGeneratorExecutor(NodeExecutorBase):
    """Executor for AI text generation nodes"""

    def __init__(self):
        super().__init__("ai_text_generator")

    async def execute(
            self,
            node_data: Dict[str, Any],
            context: Dict[str, Any],
            execution_id: str
    ) -> Dict[str, Any]:
        """Execute AI text generation node"""
        try:
            prompt_template = node_data.get("prompt_template", "")
            ai_config = node_data.get("ai_config", {})

            # Fill template with context
            filled_prompt = self._fill_template(prompt_template, context)

            # Execute AI model
            provider = ai_config.get("provider", "huggingface")
            model_name = ai_config.get("model", settings.DEFAULT_HUGGINGFACE_MODEL)

            if provider.lower() == "huggingface":
                result = await hf_api.generate_text(
                    model_name=model_name,
                    prompt=filled_prompt,
                    parameters={
                        "temperature": ai_config.get("temperature", 0.7),
                        "max_tokens": ai_config.get("max_tokens", 500)
                    }
                )
                generated_text = result["text"]
            else:
                model = model_factory.get_model(
                    provider=provider,
                    model_name=model_name,
                    temperature=ai_config.get("temperature", 0.7)
                )
                response = await model.ainvoke(filled_prompt)
                generated_text = response.content

            return {
                "generated_text": generated_text,
                "prompt_used": filled_prompt,
                "model_used": model_name,
                "provider": provider,
                "tokens_used": len(filled_prompt.split()) + len(generated_text.split())
            }

        except Exception as e:
            logger.error(f"AI text generation failed: {e}")
            return {
                "error": str(e),
                "generated_text": f"Text generation failed: {e}",
                "prompt_used": prompt_template
            }

    def _fill_template(self, template: str, context: Dict[str, Any]) -> str:
        """Fill prompt template with context variables"""
        try:
            # Simple template filling - can be enhanced with Jinja2
            filled = template
            for key, value in context.items():
                placeholder = f"{{{key}}}"
                if placeholder in filled:
                    filled = filled.replace(placeholder, str(value))
            return filled
        except Exception as e:
            logger.warning(f"Template filling failed: {e}")
            return template


class AIDataProcessorExecutor(NodeExecutorBase):
    """Executor for AI-powered data processing nodes"""

    def __init__(self):
        super().__init__("ai_data_processor")

    async def execute(
            self,
            node_data: Dict[str, Any],
            context: Dict[str, Any],
            execution_id: str
    ) -> Dict[str, Any]:
        """Execute AI data processing node"""
        try:
            operation = node_data.get("operation", "analyze")
            data_source = node_data.get("data_source", "context")
            ai_config = node_data.get("ai_config", {})

            # Get data to process
            if data_source == "context":
                data = context.get("data", {})
            else:
                data = node_data.get("data", {})

            # Build processing prompt
            prompt = self._build_processing_prompt(operation, data)

            # Execute AI model
            provider = ai_config.get("provider", "huggingface")
            model_name = ai_config.get("model", settings.DEFAULT_HUGGINGFACE_MODEL)

            if provider.lower() == "huggingface":
                result = await hf_api.generate_text(
                    model_name=model_name,
                    prompt=prompt,
                    parameters={
                        "temperature": ai_config.get("temperature", 0.5),
                        "max_tokens": ai_config.get("max_tokens", 800)
                    }
                )
                analysis_result = result["text"]
            else:
                model = model_factory.get_model(
                    provider=provider,
                    model_name=model_name,
                    temperature=ai_config.get("temperature", 0.5)
                )
                response = await model.ainvoke(prompt)
                analysis_result = response.content

            # Parse structured results if possible
            processed_data = self._parse_analysis_result(analysis_result, operation)

            return {
                "operation": operation,
                "processed_data": processed_data,
                "analysis_result": analysis_result,
                "input_data_size": len(str(data)),
                "tokens_used": len(prompt.split()) + len(analysis_result.split())
            }

        except Exception as e:
            logger.error(f"AI data processing failed: {e}")
            return {
                "error": str(e),
                "operation": operation,
                "processed_data": None
            }

    def _build_processing_prompt(self, operation: str, data: Any) -> str:
        """Build prompt for data processing"""
        data_str = str(data)[:2000]  # Limit data size for prompt

        if operation == "analyze":
            return f"""Analyze the following data and provide insights:

Data: {data_str}

Please provide:
1. Key patterns or trends
2. Notable insights
3. Recommendations
4. Summary of findings"""

        elif operation == "summarize":
            return f"""Summarize the following data concisely:

Data: {data_str}

Provide a clear, structured summary highlighting the most important information."""

        elif operation == "extract":
            return f"""Extract key information from the following data:

Data: {data_str}

Extract and list:
- Key entities (people, places, organizations)
- Important dates and numbers
- Main topics or themes
- Critical facts"""

        else:
            return f"""Process the following data according to the operation '{operation}':

Data: {data_str}

Please provide appropriate processing results."""

    def _parse_analysis_result(self, result: str, operation: str) -> Dict[str, Any]:
        """Parse analysis result into structured format"""
        try:
            # Simple parsing - can be enhanced with NLP
            lines = result.split('\n')
            parsed = {
                "summary": lines[0] if lines else result[:200],
                "details": result,
                "operation": operation
            }

            # Try to extract structured information
            if "insights:" in result.lower():
                insights_section = result.split("insights:")[-1].split("\n")[0:3]
                parsed["insights"] = [line.strip() for line in insights_section if line.strip()]

            return parsed

        except Exception as e:
            logger.warning(f"Failed to parse analysis result: {e}")
            return {"raw_result": result}


class NodeExecutionEngine:
    """Main engine for executing AI workflow nodes"""

    def __init__(self):
        self.executors: Dict[str, NodeExecutorBase] = {}
        self.running = False
        self._register_executors()

    def _register_executors(self):
        """Register all node executors"""
        self.executors["ai_decision"] = AIDecisionExecutor()
        self.executors["ai_text_generator"] = AITextGeneratorExecutor()
        self.executors["ai_data_processor"] = AIDataProcessorExecutor()

        logger.info(f"Registered {len(self.executors)} node executors")

    async def start(self):
        """Start the node execution engine"""
        try:
            # Register message handler with Kafka service
            kafka_service.register_message_handler('ai_execution', self.handle_execution_message)

            self.running = True
            logger.info("Node execution engine started")

        except Exception as e:
            logger.error(f"Failed to start node execution engine: {e}")
            raise

    async def stop(self):
        """Stop the node execution engine"""
        self.running = False
        logger.info("Node execution engine stopped")

    async def handle_execution_message(self, message: NodeExecutionMessage):
        """Handle incoming node execution message"""
        start_time = datetime.utcnow()

        try:
            logger.info(f"Executing node {message.node_id} of type {message.node_type}")

            # Mark node as processing in Redis
            await redis_context_service.mark_processing(message.execution_id, message.node_id)

            # Get executor for node type
            executor = self.executors.get(message.node_type)
            if not executor:
                raise ValueError(f"No executor found for node type: {message.node_type}")

            # Execute the node
            result = await executor.execute(
                node_data=message.node_data,
                context=message.context,
                execution_id=message.execution_id
            )

            # Store result in Redis
            await redis_context_service.store_node_result(
                execution_id=message.execution_id,
                node_id=message.node_id,
                result=result
            )

            # Mark node as completed
            await redis_context_service.mark_completed(message.execution_id, message.node_id)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Send completion event
            completion_event = CompletionEvent(
                execution_id=message.execution_id,
                node_id=message.node_id,
                status="completed",
                result=result,
                next_possible_nodes=[],  # Will be determined by Spring Boot
                execution_time_ms=int(execution_time),
                service="fastapi"
            )

            await kafka_service.publish_completion_event(completion_event)

            # Update dependency graph (this will trigger Spring Boot to find next nodes)
            await self._update_dependencies(message.execution_id, message.node_id)

            logger.info(f"Node {message.node_id} completed successfully in {execution_time:.2f}ms")

        except Exception as e:
            logger.error(f"Node execution failed for {message.node_id}: {e}")

            # Mark as failed and send error event
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            completion_event = CompletionEvent(
                execution_id=message.execution_id,
                node_id=message.node_id,
                status="failed",
                result={"error": str(e)},
                next_possible_nodes=[],
                execution_time_ms=int(execution_time),
                service="fastapi",
                error=str(e)
            )

            await kafka_service.publish_completion_event(completion_event)

    async def _update_dependencies(self, execution_id: str, completed_node_id: str):
        """Update dependency counts after node completion"""
        try:
            # This is a simplified version - the actual dependency resolution
            # is handled by Spring Boot using Kahn's algorithm

            # Update workflow status
            await kafka_service.publish_state_update(
                execution_id=execution_id,
                status="node_completed",
                data={
                    "completed_node": completed_node_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Failed to update dependencies: {e}")

    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get current execution status"""
        try:
            context = await redis_context_service.get_execution_context(execution_id)
            processing_nodes = await redis_context_service.get_processing_nodes(execution_id)
            completed_nodes = await redis_context_service.get_completed_nodes(execution_id)
            workflow_status = await redis_context_service.get_workflow_status(execution_id)

            return {
                "execution_id": execution_id,
                "context": context,
                "processing_nodes": list(processing_nodes),
                "completed_nodes": list(completed_nodes),
                "workflow_status": workflow_status,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get execution status: {e}")
            return {
                "execution_id": execution_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def register_custom_executor(self, node_type: str, executor: NodeExecutorBase):
        """Register custom node executor"""
        self.executors[node_type] = executor
        logger.info(f"Registered custom executor for node type: {node_type}")


# Global instance
node_execution_engine = NodeExecutionEngine()