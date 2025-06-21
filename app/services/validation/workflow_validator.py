# app/services/validation/workflow_validator.py
from typing import Dict, Any, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Supported node types"""
    AI_DECISION = "ai_decision"
    AI_TEXT_GENERATOR = "ai_text_generator"
    AI_DATA_PROCESSOR = "ai_data_processor"
    EMAIL = "email"
    WEBHOOK = "webhook"
    TRANSFORM = "transform"
    CONDITION = "condition"
    LOOP = "loop"
    JOIN = "join"
    START = "start"
    END = "end"


class AIProviderType(str, Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class ValidationResult:
    """Result of workflow validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    node_count: int
    edge_count: int
    estimated_complexity: int


class AIConfigSchema(BaseModel):
    """Schema for AI configuration"""
    provider: AIProviderType = Field(default=AIProviderType.HUGGINGFACE)
    model: str = Field(default="microsoft/Phi-3-mini-4k-instruct")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=1, le=4000)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    timeout: int = Field(default=30, ge=5, le=300)


class NodeSchema(BaseModel):
    """Base schema for workflow nodes"""
    id: str = Field(..., min_length=1, max_length=100)
    type: NodeType
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    dependencies: List[str] = Field(default=[])
    config: Dict[str, Any] = Field(default={})
    ai_config: Optional[AIConfigSchema] = None
    retry_count: int = Field(default=3, ge=0, le=10)
    timeout: int = Field(default=300, ge=5, le=3600)


class AIDecisionNodeSchema(NodeSchema):
    """Schema for AI decision nodes"""
    type: NodeType = Field(default=NodeType.AI_DECISION)
    config: Dict[str, Any] = Field(...)

    def __init__(self, **data):
        super().__init__(**data)
        # Validate required config fields
        required_fields = ["prompt", "options"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"AI decision node requires '{field}' in config")


class AITextGeneratorNodeSchema(NodeSchema):
    """Schema for AI text generator nodes"""
    type: NodeType = Field(default=NodeType.AI_TEXT_GENERATOR)
    config: Dict[str, Any] = Field(...)

    def __init__(self, **data):
        super().__init__(**data)
        # Validate required config fields
        required_fields = ["prompt_template"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"AI text generator node requires '{field}' in config")


class AIDataProcessorNodeSchema(NodeSchema):
    """Schema for AI data processor nodes"""
    type: NodeType = Field(default=NodeType.AI_DATA_PROCESSOR)
    config: Dict[str, Any] = Field(...)

    def __init__(self, **data):
        super().__init__(**data)
        # Validate required config fields
        required_fields = ["operation"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"AI data processor node requires '{field}' in config")


class EdgeSchema(BaseModel):
    """Schema for workflow edges"""
    id: str = Field(..., min_length=1, max_length=100)
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    condition: Optional[str] = None
    weight: float = Field(default=1.0, ge=0.0)


class WorkflowSchema(BaseModel):
    """Complete workflow schema"""
    id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    version: int = Field(default=1, ge=1)
    nodes: List[NodeSchema] = Field(..., min_items=1)
    edges: List[EdgeSchema] = Field(default=[])
    global_config: Dict[str, Any] = Field(default={})
    ai_config: AIConfigSchema = Field(default_factory=AIConfigSchema)
    max_execution_time: int = Field(default=3600, ge=60, le=86400)
    max_steps: int = Field(default=100, ge=1, le=1000)


class WorkflowValidator:
    """Comprehensive workflow validator"""

    def __init__(self):
        self.node_type_schemas = {
            NodeType.AI_DECISION: AIDecisionNodeSchema,
            NodeType.AI_TEXT_GENERATOR: AITextGeneratorNodeSchema,
            NodeType.AI_DATA_PROCESSOR: AIDataProcessorNodeSchema,
            # Other node types use base NodeSchema
        }

    def validate_workflow(self, workflow_data: Dict[str, Any]) -> ValidationResult:
        """Validate complete workflow"""
        errors = []
        warnings = []

        try:
            # Basic schema validation
            workflow = WorkflowSchema(**workflow_data)

            # Advanced validation
            node_errors, node_warnings = self._validate_nodes(workflow.nodes)
            edge_errors, edge_warnings = self._validate_edges(workflow.edges, workflow.nodes)
            graph_errors, graph_warnings = self._validate_graph_structure(workflow.nodes, workflow.edges)
            dependency_errors, dependency_warnings = self._validate_dependencies(workflow.nodes, workflow.edges)

            errors.extend(node_errors + edge_errors + graph_errors + dependency_errors)
            warnings.extend(node_warnings + edge_warnings + graph_warnings + dependency_warnings)

            # Calculate complexity
            complexity = self._calculate_complexity(workflow.nodes, workflow.edges)

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                node_count=len(workflow.nodes),
                edge_count=len(workflow.edges),
                estimated_complexity=complexity
            )

        except ValidationError as e:
            # Pydantic validation errors
            schema_errors = []
            for error in e.errors():
                field_path = " -> ".join(str(x) for x in error["loc"])
                schema_errors.append(f"Schema error in {field_path}: {error['msg']}")

            return ValidationResult(
                is_valid=False,
                errors=schema_errors,
                warnings=[],
                node_count=0,
                edge_count=0,
                estimated_complexity=0
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                node_count=0,
                edge_count=0,
                estimated_complexity=0
            )

    def _validate_nodes(self, nodes: List[NodeSchema]) -> Tuple[List[str], List[str]]:
        """Validate individual nodes"""
        errors = []
        warnings = []

        node_ids = set()
        start_nodes = []
        end_nodes = []

        for node in nodes:
            # Check for duplicate IDs
            if node.id in node_ids:
                errors.append(f"Duplicate node ID: {node.id}")
            node_ids.add(node.id)

            # Track special nodes
            if node.type == NodeType.START:
                start_nodes.append(node.id)
            elif node.type == NodeType.END:
                end_nodes.append(node.id)

            # Validate node-specific requirements
            try:
                if node.type in self.node_type_schemas:
                    schema_class = self.node_type_schemas[node.type]
                    schema_class(**node.dict())
            except ValueError as e:
                errors.append(f"Node {node.id}: {str(e)}")

            # Validate AI configuration
            if node.ai_config and node.type.value.startswith("ai_"):
                ai_warnings = self._validate_ai_config(node.ai_config, node.id)
                warnings.extend(ai_warnings)

        # Check for start and end nodes
        if len(start_nodes) == 0:
            warnings.append("No START node found. Workflow may need explicit entry point.")
        elif len(start_nodes) > 1:
            errors.append(f"Multiple START nodes found: {start_nodes}")

        if len(end_nodes) == 0:
            warnings.append("No END node found. Workflow may run indefinitely.")

        return errors, warnings

    def _validate_edges(self, edges: List[EdgeSchema], nodes: List[NodeSchema]) -> Tuple[List[str], List[str]]:
        """Validate workflow edges"""
        errors = []
        warnings = []

        node_ids = {node.id for node in nodes}
        edge_ids = set()

        for edge in edges:
            # Check for duplicate edge IDs
            if edge.id in edge_ids:
                errors.append(f"Duplicate edge ID: {edge.id}")
            edge_ids.add(edge.id)

            # Check if source and target nodes exist
            if edge.source not in node_ids:
                errors.append(f"Edge {edge.id}: Source node '{edge.source}' does not exist")

            if edge.target not in node_ids:
                errors.append(f"Edge {edge.id}: Target node '{edge.target}' does not exist")

            # Check for self-loops
            if edge.source == edge.target:
                warnings.append(f"Edge {edge.id}: Self-loop detected on node {edge.source}")

        return errors, warnings

    def _validate_graph_structure(self, nodes: List[NodeSchema], edges: List[EdgeSchema]) -> Tuple[
        List[str], List[str]]:
        """Validate overall graph structure"""
        errors = []
        warnings = []

        # Build adjacency lists
        outgoing = {node.id: [] for node in nodes}
        incoming = {node.id: [] for node in nodes}

        for edge in edges:
            if edge.source in outgoing and edge.target in incoming:
                outgoing[edge.source].append(edge.target)
                incoming[edge.target].append(edge.source)

        # Check for isolated nodes
        isolated_nodes = []
        for node in nodes:
            if (len(outgoing[node.id]) == 0 and len(incoming[node.id]) == 0 and
                    node.type not in [NodeType.START, NodeType.END]):
                isolated_nodes.append(node.id)

        if isolated_nodes:
            warnings.extend([f"Isolated node (no connections): {node_id}" for node_id in isolated_nodes])

        # Check for unreachable nodes
        reachable = self._find_reachable_nodes(nodes, edges)
        unreachable = [node.id for node in nodes if node.id not in reachable]

        if unreachable:
            warnings.extend([f"Unreachable node: {node_id}" for node_id in unreachable])

        # Check for cycles (except intentional loops)
        cycles = self._detect_cycles(nodes, edges)
        if cycles:
            warnings.extend([f"Cycle detected: {' -> '.join(cycle)}" for cycle in cycles])

        return errors, warnings

    def _validate_dependencies(self, nodes: List[NodeSchema], edges: List[EdgeSchema]) -> Tuple[List[str], List[str]]:
        """Validate node dependencies"""
        errors = []
        warnings = []

        node_ids = {node.id for node in nodes}

        for node in nodes:
            for dep in node.dependencies:
                if dep not in node_ids:
                    errors.append(f"Node {node.id}: Dependency '{dep}' does not exist")

        return errors, warnings

    def _validate_ai_config(self, ai_config: AIConfigSchema, node_id: str) -> List[str]:
        """Validate AI configuration"""
        warnings = []

        # Check temperature range
        if ai_config.temperature < 0.1:
            warnings.append(
                f"Node {node_id}: Very low temperature ({ai_config.temperature}) may produce repetitive results")
        elif ai_config.temperature > 1.5:
            warnings.append(
                f"Node {node_id}: High temperature ({ai_config.temperature}) may produce inconsistent results")

        # Check token limits
        if ai_config.max_tokens > 2000:
            warnings.append(f"Node {node_id}: High token limit ({ai_config.max_tokens}) may increase costs and latency")

        # Provider-specific warnings
        if ai_config.provider == AIProviderType.HUGGINGFACE and ai_config.max_tokens > 1000:
            warnings.append(f"Node {node_id}: HuggingFace free tier may have token limitations")

        return warnings

    def _find_reachable_nodes(self, nodes: List[NodeSchema], edges: List[EdgeSchema]) -> Set[str]:
        """Find all reachable nodes from start nodes"""
        # Build adjacency list
        adj = {node.id: [] for node in nodes}
        for edge in edges:
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        # Find start nodes
        start_nodes = [node.id for node in nodes if node.type == NodeType.START]
        if not start_nodes:
            # If no start nodes, consider first node as start
            start_nodes = [nodes[0].id] if nodes else []

        # DFS to find reachable nodes
        reachable = set()
        stack = start_nodes.copy()

        while stack:
            current = stack.pop()
            if current not in reachable:
                reachable.add(current)
                stack.extend(adj.get(current, []))

        return reachable

    def _detect_cycles(self, nodes: List[NodeSchema], edges: List[EdgeSchema]) -> List[List[str]]:
        """Detect cycles in the workflow graph"""
        # Build adjacency list
        adj = {node.id: [] for node in nodes}
        for edge in edges:
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        # DFS-based cycle detection
        white = set(node.id for node in nodes)  # Unvisited
        gray = set()  # Currently being processed
        black = set()  # Completely processed
        cycles = []

        def dfs(node, path):
            if node in gray:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in black:
                return

            white.remove(node)
            gray.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                dfs(neighbor, path.copy())

            gray.remove(node)
            black.add(node)

        for node in list(white):
            if node in white:
                dfs(node, [])

        return cycles

    def _calculate_complexity(self, nodes: List[NodeSchema], edges: List[EdgeSchema]) -> int:
        """Calculate estimated workflow complexity"""
        complexity = 0

        # Base complexity from node count
        complexity += len(nodes) * 10

        # Additional complexity from edges
        complexity += len(edges) * 5

        # AI nodes add more complexity
        ai_nodes = [node for node in nodes if node.type.value.startswith("ai_")]
        complexity += len(ai_nodes) * 20

        # Loop nodes add significant complexity
        loop_nodes = [node for node in nodes if node.type == NodeType.LOOP]
        complexity += len(loop_nodes) * 50

        # Conditional nodes add complexity
        condition_nodes = [node for node in nodes if node.type == NodeType.CONDITION]
        complexity += len(condition_nodes) * 15

        return complexity

    def validate_node_config(self, node_type: NodeType, config: Dict[str, Any]) -> ValidationResult:
        """Validate individual node configuration"""
        errors = []
        warnings = []

        try:
            if node_type == NodeType.AI_DECISION:
                if "prompt" not in config:
                    errors.append("AI decision node requires 'prompt' in config")
                if "options" not in config or not isinstance(config["options"], list):
                    errors.append("AI decision node requires 'options' list in config")
                elif len(config["options"]) < 2:
                    warnings.append("AI decision node should have at least 2 options")

            elif node_type == NodeType.AI_TEXT_GENERATOR:
                if "prompt_template" not in config:
                    errors.append("AI text generator node requires 'prompt_template' in config")

                # Check for template variables
                template = config.get("prompt_template", "")
                if "{" in template and "}" in template:
                    import re
                    variables = re.findall(r'\{([^}]+)\}', template)
                    if variables:
                        warnings.append(f"Template uses variables: {variables}. Ensure they're available in context.")

            elif node_type == NodeType.AI_DATA_PROCESSOR:
                if "operation" not in config:
                    errors.append("AI data processor node requires 'operation' in config")

                valid_operations = ["analyze", "summarize", "extract", "classify", "transform"]
                operation = config.get("operation")
                if operation and operation not in valid_operations:
                    warnings.append(f"Unknown operation '{operation}'. Valid operations: {valid_operations}")

        except Exception as e:
            errors.append(f"Config validation error: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            node_count=1,
            edge_count=0,
            estimated_complexity=10
        )


# Global validator instance
workflow_validator = WorkflowValidator()