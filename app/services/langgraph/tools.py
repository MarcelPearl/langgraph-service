from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import asyncio
import aiohttp
import json
import math
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")


class Calculator(BaseTool):
    name = "calculator"
    description = "Performs mathematical calculations. Use for arithmetic, algebra, and basic math operations."
    args_schema = CalculatorInput

    def _run(self, expression: str) -> str:
        """Execute calculator tool"""
        try:
            allowed_chars = set('0123456789+-*/().% ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Expression contains invalid characters"

            result = eval(expression, {"__builtins__": {}, "math": math})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


class TextAnalyzerInput(BaseModel):
    text: str = Field(description="Text to analyze")
    analysis_type: str = Field(description="Type of analysis: sentiment, summary, keywords, or length")


class TextAnalyzer(BaseTool):
    name = "text_analyzer"
    description = "Analyzes text for sentiment, creates summaries, extracts keywords, or counts words/characters."
    args_schema = TextAnalyzerInput

    def _run(self, text: str, analysis_type: str) -> str:
        """Execute text analyzer tool"""
        try:
            if analysis_type == "length":
                words = len(text.split())
                chars = len(text)
                return f"Text length: {words} words, {chars} characters"

            elif analysis_type == "keywords":
                words = re.findall(r'\b\w+\b', text.lower())
                word_freq = {}
                for word in words:
                    if len(word) > 3:
                        word_freq[word] = word_freq.get(word, 0) + 1

                top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                keywords = [word for word, count in top_keywords]
                return f"Top keywords: {', '.join(keywords)}"

            elif analysis_type == "summary":
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                if len(sentences) <= 2:
                    return f"Summary: {text}"
                else:
                    summary = f"{sentences[0]}. {sentences[-1]}."
                    return f"Summary: {summary}"

            elif analysis_type == "sentiment":
                positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like",
                                  "happy", "joy"]
                negative_words = ["bad", "terrible", "awful", "hate", "dislike", "sad", "angry", "frustrated",
                                  "disappointed"]

                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)

                if pos_count > neg_count:
                    sentiment = "Positive"
                elif neg_count > pos_count:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                return f"Sentiment: {sentiment} (Positive words: {pos_count}, Negative words: {neg_count})"

            else:
                return f"Error: Unknown analysis type '{analysis_type}'"

        except Exception as e:
            return f"Error: {str(e)}"


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class WebSearch(BaseTool):
    name = "web_search"
    description = "Searches the web for information using a search engine."
    args_schema = WebSearchInput

    async def _arun(self, query: str, max_results: int = 5) -> str:
        """Execute web search tool (async)"""
        try:
            # For demo purposes, return mock search results
            # In production, integrate with actual search API (Google, Bing, etc.)
            mock_results = [
                {
                    "title": f"Search result {i + 1} for '{query}'",
                    "url": f"https://example.com/result-{i + 1}",
                    "snippet": f"This is a mock search result snippet for query '{query}'. "
                               f"This would contain relevant information from the web."
                }
                for i in range(min(max_results, 3))
            ]

            formatted_results = []
            for i, result in enumerate(mock_results, 1):
                formatted_results.append(
                    f"{i}. {result['title']}\n"
                    f"   URL: {result['url']}\n"
                    f"   {result['snippet']}"
                )

            return "Search Results:\n" + "\n\n".join(formatted_results)

        except Exception as e:
            return f"Error performing web search: {str(e)}"

    def _run(self, query: str, max_results: int = 5) -> str:
        """Sync wrapper for web search"""
        return asyncio.run(self._arun(query, max_results))


class DataProcessorInput(BaseModel):
    data: Dict[str, Any] = Field(description="Data to process")
    operation: str = Field(description="Operation to perform: filter, sort, aggregate, or transform")
    parameters: Dict[str, Any] = Field(default={}, description="Operation parameters")


class DataProcessor(BaseTool):
    name = "data_processor"
    description = "Processes data with operations like filtering, sorting, aggregation, and transformation."
    args_schema = DataProcessorInput

    def _run(self, data: Dict[str, Any], operation: str, parameters: Dict[str, Any] = None) -> str:
        """Execute data processor tool"""
        if parameters is None:
            parameters = {}

        try:
            if operation == "filter":
                filter_key = parameters.get("key")
                filter_value = parameters.get("value")

                if not filter_key:
                    return "Error: Filter operation requires 'key' parameter"

                if isinstance(data, list):
                    filtered = [item for item in data if item.get(filter_key) == filter_value]
                    return f"Filtered data: {json.dumps(filtered, indent=2)}"
                elif isinstance(data, dict):
                    if data.get(filter_key) == filter_value:
                        return f"Data matches filter: {json.dumps(data, indent=2)}"
                    else:
                        return "Data does not match filter criteria"

            elif operation == "sort":
                sort_key = parameters.get("key")
                reverse = parameters.get("reverse", False)

                if isinstance(data, list) and sort_key:
                    sorted_data = sorted(data, key=lambda x: x.get(sort_key, 0), reverse=reverse)
                    return f"Sorted data: {json.dumps(sorted_data, indent=2)}"
                else:
                    return "Error: Sort operation requires list data and 'key' parameter"

            elif operation == "aggregate":
                agg_key = parameters.get("key")
                agg_func = parameters.get("function", "sum")

                if isinstance(data, list) and agg_key:
                    values = [item.get(agg_key) for item in data if agg_key in item]
                    numeric_values = [v for v in values if isinstance(v, (int, float))]

                    if agg_func == "sum":
                        result = sum(numeric_values)
                    elif agg_func == "count":
                        result = len(values)
                    elif agg_func == "avg":
                        result = sum(numeric_values) / len(numeric_values) if numeric_values else 0
                    elif agg_func == "min":
                        result = min(numeric_values) if numeric_values else None
                    elif agg_func == "max":
                        result = max(numeric_values) if numeric_values else None
                    else:
                        return f"Error: Unknown aggregation function '{agg_func}'"

                    return f"Aggregation result ({agg_func} of {agg_key}): {result}"
                else:
                    return "Error: Aggregate operation requires list data and 'key' parameter"

            elif operation == "transform":
                # Simple data transformation
                transform_type = parameters.get("type", "json")

                if transform_type == "json":
                    return f"JSON representation: {json.dumps(data, indent=2)}"
                elif transform_type == "keys":
                    if isinstance(data, dict):
                        return f"Keys: {list(data.keys())}"
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        return f"Common keys: {list(data[0].keys())}"
                elif transform_type == "values":
                    if isinstance(data, dict):
                        return f"Values: {list(data.values())}"
                elif transform_type == "count":
                    if isinstance(data, (list, dict)):
                        return f"Count: {len(data)}"
                    else:
                        return f"Count: 1 (single item)"

                return f"Transformed data ({transform_type}): {str(data)}"

            else:
                return f"Error: Unknown operation '{operation}'"

        except Exception as e:
            return f"Error: {str(e)}"


class ToolRegistry:
    """Registry for managing AI tools"""

    def __init__(self):
        self.tools = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools"""
        default_tools = [
            Calculator(),
            TextAnalyzer(),
            WebSearch(),
            DataProcessor()
        ]

        for tool in default_tools:
            self.register_tool(tool)

    def register_tool(self, tool: BaseTool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools"""
        return list(self.tools.values())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for LLM function calling"""
        schemas = []
        for tool in self.tools.values():
            schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.model_json_schema() if tool.args_schema else {}
            }
            schemas.append(schema)
        return schemas


tool_registry = ToolRegistry()