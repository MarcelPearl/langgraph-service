# app/services/langgraph/tools.py
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
    name: str = "calculator"
    description: str = "Performs mathematical calculations. Use for arithmetic, algebra, and basic math operations."
    args_schema = CalculatorInput

    def _run(self, expression: str) -> str:
        """Execute calculator tool"""
        try:
            # Enhanced security for eval
            allowed_chars = set('0123456789+-*/().% ')
            allowed_names = {
                'abs', 'round', 'min', 'max', 'sum',
                'sqrt', 'pow', 'log', 'log10', 'exp',
                'sin', 'cos', 'tan', 'pi', 'e'
            }

            if not all(c in allowed_chars or c.isalpha() for c in expression):
                return "Error: Expression contains invalid characters"

            # Safe evaluation with math functions
            safe_dict = {
                "__builtins__": {},
                **{name: getattr(math, name) for name in allowed_names if hasattr(math, name)}
            }

            result = eval(expression, safe_dict)
            return f"Result: {result}"

        except Exception as e:
            return f"Error: {str(e)}"


class TextAnalyzerInput(BaseModel):
    text: str = Field(description="Text to analyze")
    analysis_type: str = Field(description="Type of analysis: sentiment, summary, keywords, or length")


class TextAnalyzer(BaseTool):
    name: str = "text_analyzer"
    description: str = "Analyzes text for sentiment, creates summaries, extracts keywords, or counts words/characters."
    args_schema = TextAnalyzerInput

    def _run(self, text: str, analysis_type: str) -> str:
        """Execute text analyzer tool"""
        try:
            if analysis_type == "length":
                words = len(text.split())
                chars = len(text)
                sentences = len(re.split(r'[.!?]+', text))
                return f"Text length: {words} words, {chars} characters, {sentences} sentences"

            elif analysis_type == "keywords":
                words = re.findall(r'\b\w+\b', text.lower())
                # Filter out common stop words
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                words = [word for word in words if len(word) > 3 and word not in stop_words]

                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1

                top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                keywords = [f"{word} ({count})" for word, count in top_keywords]
                return f"Top keywords: {', '.join(keywords)}"

            elif analysis_type == "summary":
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]

                if len(sentences) <= 2:
                    return f"Summary: {text}"
                else:
                    # Take first and last sentences, and middle if more than 4 sentences
                    summary_sentences = [sentences[0]]
                    if len(sentences) > 4:
                        summary_sentences.append(sentences[len(sentences) // 2])
                    summary_sentences.append(sentences[-1])
                    summary = ". ".join(summary_sentences) + "."
                    return f"Summary: {summary}"

            elif analysis_type == "sentiment":
                positive_words = [
                    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
                    "love", "like", "happy", "joy", "pleased", "satisfied", "awesome",
                    "brilliant", "perfect", "outstanding", "superb", "magnificent"
                ]
                negative_words = [
                    "bad", "terrible", "awful", "hate", "dislike", "sad", "angry",
                    "frustrated", "disappointed", "horrible", "disgusting", "poor",
                    "worst", "pathetic", "useless", "annoying", "boring"
                ]

                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)

                total_words = len(text.split())
                pos_ratio = pos_count / max(total_words, 1)
                neg_ratio = neg_count / max(total_words, 1)

                if pos_count > neg_count * 1.5:
                    sentiment = "Positive"
                elif neg_count > pos_count * 1.5:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                confidence = abs(pos_ratio - neg_ratio) * 100

                return f"Sentiment: {sentiment} (Confidence: {confidence:.1f}%, Positive: {pos_count}, Negative: {neg_count})"

            else:
                return f"Error: Unknown analysis type '{analysis_type}'. Available: sentiment, summary, keywords, length"

        except Exception as e:
            return f"Error: {str(e)}"


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class WebSearch(BaseTool):
    name: str = "web_search"
    description: str = "Searches the web for information using a search engine."
    args_schema = WebSearchInput

    async def _arun(self, query: str, max_results: int = 5) -> str:
        """Execute web search tool (async)"""
        try:
            # For demo/development - using DuckDuckGo instant answer API
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(search_url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()

                            results = []

                            # Check for instant answer
                            if data.get('Abstract'):
                                results.append({
                                    "title": data.get('Heading', 'Instant Answer'),
                                    "snippet": data['Abstract'],
                                    "url": data.get('AbstractURL', 'N/A')
                                })

                            # Check for related topics
                            for topic in data.get('RelatedTopics', [])[:max_results - len(results)]:
                                if isinstance(topic, dict) and 'Text' in topic:
                                    results.append({
                                        "title": topic.get('Text', '')[:50] + "...",
                                        "snippet": topic.get('Text', ''),
                                        "url": topic.get('FirstURL', 'N/A')
                                    })

                            if results:
                                formatted_results = []
                                for i, result in enumerate(results, 1):
                                    formatted_results.append(
                                        f"{i}. {result['title']}\n"
                                        f"   URL: {result['url']}\n"
                                        f"   {result['snippet'][:200]}..."
                                    )
                                return "Search Results:\n" + "\n\n".join(formatted_results)
                            else:
                                return f"No specific results found for '{query}'. You may want to try a more specific search query."

                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass

            # Fallback to mock results for development
            mock_results = [
                {
                    "title": f"Search result {i + 1} for '{query}'",
                    "url": f"https://example.com/result-{i + 1}",
                    "snippet": f"This is a mock search result snippet for query '{query}'. "
                               f"This would contain relevant information from the web about {query}."
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

            return "Search Results (Mock Data):\n" + "\n\n".join(formatted_results)

        except Exception as e:
            return f"Error performing web search: {str(e)}"

    def _run(self, query: str, max_results: int = 5) -> str:
        """Sync wrapper for web search"""
        return asyncio.run(self._arun(query, max_results))


class DataProcessorInput(BaseModel):
    data: Dict[str, Any] = Field(description="Data to process")
    operation: str = Field(description="Operation to perform: filter, sort, aggregate, transform, or validate")
    parameters: Dict[str, Any] = Field(default={}, description="Operation parameters")


class DataProcessor(BaseTool):
    name: str = "data_processor"
    description: str = "Processes data with operations like filtering, sorting, aggregation, transformation, and validation."
    args_schema = DataProcessorInput

    def _run(self, data: Dict[str, Any], operation: str, parameters: Dict[str, Any] = None) -> str:
        """Execute data processor tool"""
        if parameters is None:
            parameters = {}

        try:
            if operation == "filter":
                return self._filter_data(data, parameters)
            elif operation == "sort":
                return self._sort_data(data, parameters)
            elif operation == "aggregate":
                return self._aggregate_data(data, parameters)
            elif operation == "transform":
                return self._transform_data(data, parameters)
            elif operation == "validate":
                return self._validate_data(data, parameters)
            else:
                return f"Error: Unknown operation '{operation}'. Available: filter, sort, aggregate, transform, validate"

        except Exception as e:
            return f"Error: {str(e)}"

    def _filter_data(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Filter data based on criteria"""
        filter_key = parameters.get("key")
        filter_value = parameters.get("value")
        filter_condition = parameters.get("condition", "equals")  # equals, greater, less, contains

        if not filter_key:
            return "Error: Filter operation requires 'key' parameter"

        if isinstance(data, list):
            filtered = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                item_value = item.get(filter_key)
                if self._matches_condition(item_value, filter_value, filter_condition):
                    filtered.append(item)

            return f"Filtered data ({len(filtered)} items): {json.dumps(filtered, indent=2)}"

        elif isinstance(data, dict):
            if self._matches_condition(data.get(filter_key), filter_value, filter_condition):
                return f"Data matches filter: {json.dumps(data, indent=2)}"
            else:
                return "Data does not match filter criteria"

    def _matches_condition(self, item_value: Any, filter_value: Any, condition: str) -> bool:
        """Check if item value matches filter condition"""
        try:
            if condition == "equals":
                return item_value == filter_value
            elif condition == "greater":
                return float(item_value) > float(filter_value)
            elif condition == "less":
                return float(item_value) < float(filter_value)
            elif condition == "contains":
                return str(filter_value).lower() in str(item_value).lower()
            else:
                return item_value == filter_value
        except (ValueError, TypeError):
            return False

    def _sort_data(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Sort data by specified key"""
        sort_key = parameters.get("key")
        reverse = parameters.get("reverse", False)

        if isinstance(data, list) and sort_key:
            try:
                sorted_data = sorted(
                    data,
                    key=lambda x: x.get(sort_key, 0) if isinstance(x, dict) else 0,
                    reverse=reverse
                )
                return f"Sorted data ({len(sorted_data)} items): {json.dumps(sorted_data, indent=2)}"
            except Exception as e:
                return f"Error sorting data: {str(e)}"
        else:
            return "Error: Sort operation requires list data and 'key' parameter"

    def _aggregate_data(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Aggregate data using specified function"""
        agg_key = parameters.get("key")
        agg_func = parameters.get("function", "sum")

        if isinstance(data, list) and agg_key:
            values = [item.get(agg_key) for item in data if isinstance(item, dict) and agg_key in item]
            numeric_values = [v for v in values if isinstance(v, (int, float))]

            if agg_func == "sum":
                result = sum(numeric_values)
            elif agg_func == "count":
                result = len(values)
            elif agg_func == "avg" or agg_func == "average":
                result = sum(numeric_values) / len(numeric_values) if numeric_values else 0
            elif agg_func == "min":
                result = min(numeric_values) if numeric_values else None
            elif agg_func == "max":
                result = max(numeric_values) if numeric_values else None
            elif agg_func == "median":
                if numeric_values:
                    sorted_values = sorted(numeric_values)
                    n = len(sorted_values)
                    result = sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[
                        n // 2]) / 2
                else:
                    result = None
            else:
                return f"Error: Unknown aggregation function '{agg_func}'"

            return f"Aggregation result ({agg_func} of {agg_key}): {result}"
        else:
            return "Error: Aggregate operation requires list data and 'key' parameter"

    def _transform_data(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Transform data format"""
        transform_type = parameters.get("type", "json")

        if transform_type == "json":
            return f"JSON representation: {json.dumps(data, indent=2)}"
        elif transform_type == "keys":
            if isinstance(data, dict):
                return f"Keys: {list(data.keys())}"
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                all_keys = set()
                for item in data:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())
                return f"All keys: {list(all_keys)}"
        elif transform_type == "values":
            if isinstance(data, dict):
                return f"Values: {list(data.values())}"
        elif transform_type == "count":
            if isinstance(data, (list, dict)):
                return f"Count: {len(data)}"
            else:
                return f"Count: 1 (single item)"
        elif transform_type == "flatten":
            if isinstance(data, list):
                flattened = []
                for item in data:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                return f"Flattened data: {json.dumps(flattened, indent=2)}"

        return f"Transformed data ({transform_type}): {str(data)}"

    def _validate_data(self, data: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Validate data structure"""
        schema = parameters.get("schema", {})
        required_fields = parameters.get("required_fields", [])

        validation_results = []

        if isinstance(data, dict):
            # Check required fields
            for field in required_fields:
                if field not in data:
                    validation_results.append(f"Missing required field: {field}")
                elif data[field] is None or data[field] == "":
                    validation_results.append(f"Empty required field: {field}")

            # Check schema if provided
            for field, expected_type in schema.items():
                if field in data:
                    actual_type = type(data[field]).__name__
                    if actual_type != expected_type:
                        validation_results.append(
                            f"Type mismatch for {field}: expected {expected_type}, got {actual_type}")

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    for field in required_fields:
                        if field not in item:
                            validation_results.append(f"Item {i}: Missing required field: {field}")

        if validation_results:
            return f"Validation failed:\n" + "\n".join(validation_results)
        else:
            return "Validation passed: Data structure is valid"


class ToolRegistry:
    """Enhanced registry for managing AI tools"""

    def __init__(self):
        self.tools = {}
        self.tool_categories = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools"""
        default_tools = [
            (Calculator(), "computation"),
            (TextAnalyzer(), "text_processing"),
            (WebSearch(), "information_retrieval"),
            (DataProcessor(), "data_manipulation")
        ]

        for tool, category in default_tools:
            self.register_tool(tool, category)

    def register_tool(self, tool: BaseTool, category: str = "general"):
        """Register a tool with category"""
        self.tools[tool.name] = tool

        if category not in self.tool_categories:
            self.tool_categories[category] = []
        self.tool_categories[category].append(tool.name)

        logger.info(f"Registered tool: {tool.name} in category: {category}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category"""
        tool_names = self.tool_categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]

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
                "parameters": tool.args_schema.model_json_schema() if tool.args_schema else {},
                "category": self._get_tool_category(tool.name)
            }
            schemas.append(schema)
        return schemas

    def _get_tool_category(self, tool_name: str) -> str:
        """Get category for a tool"""
        for category, tools in self.tool_categories.items():
            if tool_name in tools:
                return category
        return "general"

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        return {
            "total_tools": len(self.tools),
            "categories": {cat: len(tools) for cat, tools in self.tool_categories.items()},
            "tool_list": list(self.tools.keys())
        }

    def search_tools(self, query: str) -> List[str]:
        """Search tools by name or description"""
        query_lower = query.lower()
        matching_tools = []

        for tool_name, tool in self.tools.items():
            if (query_lower in tool_name.lower() or
                    query_lower in tool.description.lower()):
                matching_tools.append(tool_name)

        return matching_tools


# Global instance
tool_registry = ToolRegistry()