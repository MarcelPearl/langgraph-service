from typing import Dict, Any, List, Optional, Callable
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.services.langgraph.state import WorkflowState, StateManager
from app.services.ai.model_factory import model_factory
from app.services.langgraph.tools import tool_registry
from app.core.config import settings
import logging
import asyncio
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class BasicWorkflowEngine:
    """Basic workflow execution engine using LangGraph"""

    def __init__(self, db_session):
        self.db_session = db_session
        self.state_manager = StateManager(db_session)
        self.checkpointer = None
        self._setup_checkpointer()

    def _setup_checkpointer(self):
        """Setup PostgreSQL checkpointer for state persistence"""
        try:
            self.checkpointer = AsyncPostgresSaver.from_conn_string(
                settings.LANGGRAPH_CHECKPOINT_DB
            )
        except Exception as e:
            logger.error(f"Failed to setup checkpointer: {e}")
            self.checkpointer = None

    async def create_basic_workflow(self) -> StateGraph:
        """Create a basic AI workflow graph"""

        workflow = StateGraph(WorkflowState)

        workflow.add_node("process_input", self.process_input_node)
        workflow.add_node("execute_ai", self.execute_ai_node)
        workflow.add_node("handle_tools", self.handle_tools_node)
        workflow.add_node("finalize_output", self.finalize_output_node)

        workflow.add_edge(START, "process_input")
        workflow.add_edge("process_input", "execute_ai")
        workflow.add_conditional_edges(
            "execute_ai",
            self.should_use_tools,
            {
                "tools": "handle_tools",
                "finalize": "finalize_output"
            }
        )
        workflow.add_edge("handle_tools", "execute_ai")
        workflow.add_edge("finalize_output", END)

        if self.checkpointer:
            compiled_workflow = workflow.compile(checkpointer=self.checkpointer)
        else:
            compiled_workflow = workflow.compile()

        return compiled_workflow

    async def process_input_node(self, state: WorkflowState) -> WorkflowState:
        """Process initial input and setup workflow"""
        logger.info(f"Processing input for execution {state['execution_id']}")

        try:
            state = self.state_manager.update_state(state, {
                "current_step": "process_input",
            })

            input_text = state["input_data"].get("query", "")
            if input_text:
                human_msg = HumanMessage(content=input_text)
                state = self.state_manager.add_message(state, human_msg)

            state["execution_metadata"]["input_processed_at"] = datetime.utcnow().isoformat()

            logger.info(f"Input processed successfully for execution {state['execution_id']}")
            return state

        except Exception as e:
            logger.error(f"Error in process_input_node: {e}")
            return self.state_manager.handle_error(state, e, "process_input")

    async def execute_ai_node(self, state: WorkflowState) -> WorkflowState:
        """Execute AI model with current context"""
        logger.info(f"Executing AI model for execution {state['execution_id']}")

        try:
            state = self.state_manager.update_state(state, {
                "current_step": "execute_ai",
            })

            provider = state["ai_provider"].lower()

            if provider == "huggingface":
                response = await self._execute_huggingface_model(state)
            else:
                response = await self._execute_langchain_model(state)

            state = self.state_manager.add_message(state, response)

            if hasattr(response, "usage_metadata"):
                tokens = getattr(response.usage_metadata, "total_tokens", 0)
                state["tokens_used"] += tokens

            if hasattr(response, "tool_calls") and response.tool_calls:
                state["tool_calls"].extend([
                    {
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "args": tc.get("args"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    for tc in response.tool_calls
                ])

            logger.info(f"AI execution completed for execution {state['execution_id']}")
            return state

        except Exception as e:
            logger.error(f"Error in execute_ai_node: {e}")

            if not self.state_manager.is_max_retries_reached(state):
                state = self.state_manager.handle_error(state, e, "execute_ai")
                return state
            else:
                raise e

    async def _execute_langchain_model(self, state: WorkflowState) -> AIMessage:
        """Execute model using LangChain (OpenAI, Anthropic)"""
        model = model_factory.get_model(
            provider=state["ai_provider"],
            model_name=state["ai_model"],
            temperature=state["ai_config"].get("temperature", 0.7)
        )

        tools = tool_registry.get_all_tools()
        if tools:
            model = model.bind_tools(tools)

        response = await model.ainvoke(state["messages"])
        return response

    async def _execute_huggingface_model(self, state: WorkflowState) -> AIMessage:
        """Execute Hugging Face model directly"""
        from app.services.ai.huggingface_service import hf_api

        prompt = ""
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                prompt = message.content
                break

        if not prompt:
            prompt = "Hello"

        result = await hf_api.generate_text(
            model_name=state["ai_model"],
            prompt=prompt,
            parameters={
                "temperature": state["ai_config"].get("temperature", 0.7),
                "max_tokens": state["ai_config"].get("max_tokens", 500),
                "top_p": state["ai_config"].get("top_p", 0.9)
            }
        )

        response = AIMessage(content=result["text"])

        token_estimate = len(prompt.split()) + len(result["text"].split())
        state["tokens_used"] += token_estimate

        return response

    async def handle_tools_node(self, state: WorkflowState) -> WorkflowState:
        """Handle tool execution"""
        logger.info(f"Handling tools for execution {state['execution_id']}")

        try:
            state = self.state_manager.update_state(state, {
                "current_step": "handle_tools",
            })

            last_message = state["messages"][-1]
            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                logger.warning("No tool calls found in last message")
                return state

            tool_messages = []
            for tool_call in last_message.tool_calls:
                try:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id")

                    tool = tool_registry.get_tool(tool_name)
                    if not tool:
                        result = f"Error: Tool '{tool_name}' not found"
                    else:
                        if hasattr(tool, "_arun"):
                            result = await tool._arun(**tool_args)
                        else:
                            result = tool._run(**tool_args)

                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id
                    )
                    tool_messages.append(tool_message)


                    state["tool_results"].append({
                        "tool_name": tool_name,
                        "tool_id": tool_id,
                        "args": tool_args,
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    logger.info(f"Tool {tool_name} executed successfully")

                except Exception as tool_error:
                    logger.error(f"Tool execution error for {tool_name}: {tool_error}")
                    tool_message = ToolMessage(
                        content=f"Error executing tool: {str(tool_error)}",
                        tool_call_id=tool_call.get("id", "unknown")
                    )
                    tool_messages.append(tool_message)

            for tool_msg in tool_messages:
                state = self.state_manager.add_message(state, tool_msg)

            logger.info(f"Tools handled successfully for execution {state['execution_id']}")
            return state

        except Exception as e:
            logger.error(f"Error in handle_tools_node: {e}")
            return self.state_manager.handle_error(state, e, "handle_tools")

    async def finalize_output_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize workflow output"""
        logger.info(f"Finalizing output for execution {state['execution_id']}")

        try:
            state = self.state_manager.update_state(state, {
                "current_step": "finalize_output",
            })

            last_ai_message = None
            for message in reversed(state["messages"]):
                if isinstance(message, AIMessage):
                    last_ai_message = message
                    break

            if last_ai_message:
                final_output = {
                    "response": last_ai_message.content,
                    "tool_calls": len(state["tool_calls"]),
                    "tool_results": len(state["tool_results"]),
                    "tokens_used": state["tokens_used"],
                    "steps_completed": state["step_count"],
                    "execution_time": datetime.utcnow().isoformat()
                }
            else:
                final_output = {
                    "response": "No AI response generated",
                    "error": "No AI message found in conversation"
                }

            state["final_output"] = final_output

            logger.info(f"Output finalized for execution {state['execution_id']}")
            return state

        except Exception as e:
            logger.error(f"Error in finalize_output_node: {e}")
            return self.state_manager.handle_error(state, e, "finalize_output")

    def should_use_tools(self, state: WorkflowState) -> str:
        """Decide whether to use tools or finalize"""
        last_message = state["messages"][-1] if state["messages"] else None

        if (isinstance(last_message, AIMessage) and
                hasattr(last_message, "tool_calls") and
                last_message.tool_calls):
            return "tools"
        else:
            return "finalize"

    async def execute_workflow(
            self,
            workflow_id: str,
            execution_id: str,
            input_data: Dict[str, Any],
            ai_config: Dict[str, Any],
            user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a complete workflow"""
        logger.info(f"Starting workflow execution {execution_id}")

        try:
            workflow = await self.create_basic_workflow()

            initial_state = self.state_manager.create_initial_state(
                execution_id=execution_id,
                workflow_id=workflow_id,
                input_data=input_data,
                ai_config=ai_config,
                user_id=user_id
            )

            config = {
                "configurable": {
                    "thread_id": execution_id,
                    "checkpoint_id": f"{execution_id}_start"
                }
            }

            final_state = await workflow.ainvoke(initial_state, config=config)

            result = {
                "execution_id": execution_id,
                "status": "completed" if not final_state.get("error_state") else "failed",
                "output": final_state.get("final_output"),
                "error": final_state.get("error_state"),
                "metadata": {
                    "steps_completed": final_state["step_count"],
                    "tokens_used": final_state["tokens_used"],
                    "tool_calls": len(final_state["tool_calls"]),
                    "execution_time": final_state["last_updated"]
                }
            }

            logger.info(f"Workflow execution {execution_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Workflow execution failed for {execution_id}: {e}")
            return {
                "execution_id": execution_id,
                "status": "failed",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
