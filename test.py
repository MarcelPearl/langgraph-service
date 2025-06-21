# test_integration.py
import asyncio
import sys
import json
import uuid
from datetime import datetime

sys.path.append('.')


async def test_enhanced_fastapi_service():
    """Comprehensive test of the enhanced FastAPI service"""
    print("🧪 Testing Enhanced FastAPI Workflow Service")
    print("=" * 60)

    try:
        # Test 1: Service Health Checks
        print("\n1. Testing Service Health Checks...")
        await test_health_checks()

        # Test 2: Redis Context Service
        print("\n2. Testing Redis Context Service...")
        await test_redis_context()

        # Test 3: Kafka Message Service
        print("\n3. Testing Kafka Message Service...")
        await test_kafka_messaging()

        # Test 4: Node Execution Engine
        print("\n4. Testing Node Execution Engine...")
        await test_node_execution()

        # Test 5: AI Model Integration
        print("\n5. Testing AI Model Integration...")
        await test_ai_models()

        # Test 6: Complete Workflow Simulation
        print("\n6. Testing Complete Workflow Simulation...")
        await test_complete_workflow()

        print("\n✅ All integration tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        raise


async def test_health_checks():
    """Test health check endpoints"""
    try:
        from app.services.context.redis_service import redis_context_service
        from app.services.messaging.kafka_service import kafka_service

        # Test Redis health
        await redis_context_service.connect()
        redis_health = await redis_context_service.health_check()
        print(f"   Redis Health: {redis_health['status']}")

        # Test Kafka health
        await kafka_service.start()
        kafka_health = await kafka_service.health_check()
        print(f"   Kafka Health: {kafka_health['status']}")

        print("   ✅ Health checks passed")

    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        raise


async def test_redis_context():
    """Test Redis context operations"""
    try:
        from app.services.context.redis_service import redis_context_service

        execution_id = str(uuid.uuid4())

        # Test context operations
        test_context = {
            "workflow_id": str(uuid.uuid4()),
            "status": "running",
            "variables": {"test": True}
        }

        await redis_context_service.set_execution_context(execution_id, test_context)
        retrieved_context = await redis_context_service.get_execution_context(execution_id)

        assert retrieved_context == test_context
        print("   ✅ Context storage/retrieval works")

        # Test dependency tracking
        dependencies = {"node1": 2, "node2": 1, "node3": 0}
        await redis_context_service.set_dependencies(execution_id, dependencies)

        ready_nodes = await redis_context_service.get_ready_nodes(execution_id)
        assert "node3" in ready_nodes
        print("   ✅ Dependency tracking works")

        # Test node state management
        await redis_context_service.mark_processing(execution_id, "node3")
        processing_nodes = await redis_context_service.get_processing_nodes(execution_id)
        assert "node3" in processing_nodes
        print("   ✅ Node state management works")

        # Cleanup
        await redis_context_service.cleanup_execution(execution_id)
        print("   ✅ Redis context tests passed")

    except Exception as e:
        print(f"   ❌ Redis context test failed: {e}")
        raise


async def test_kafka_messaging():
    """Test Kafka messaging operations"""
    try:
        from app.services.messaging.kafka_service import (
            kafka_service, NodeExecutionMessage, CompletionEvent
        )

        # Test message publishing
        test_message = NodeExecutionMessage(
            execution_id=str(uuid.uuid4()),
            workflow_id=str(uuid.uuid4()),
            node_id="test_node",
            node_type="ai_decision",
            node_data={
                "prompt": "Is this a test?",
                "options": ["yes", "no"]
            },
            context={"test": True},
            dependencies=[],
            timestamp=datetime.utcnow().isoformat()
        )

        await kafka_service.publish_node_execution(test_message)
        print("   ✅ Node execution message published")

        # Test completion event
        completion_event = CompletionEvent(
            execution_id=test_message.execution_id,
            node_id=test_message.node_id,
            status="completed",
            result={"decision": "yes", "confidence": 0.9},
            next_possible_nodes=["next_node"],
            execution_time_ms=1500,
            service="fastapi"
        )

        await kafka_service.publish_completion_event(completion_event)
        print("   ✅ Completion event published")

        # Test state update
        await kafka_service.publish_state_update(
            execution_id=test_message.execution_id,
            status="node_completed",
            data={"completed_node": test_message.node_id}
        )
        print("   ✅ State update published")

        print("   ✅ Kafka messaging tests passed")

    except Exception as e:
        print(f"   ❌ Kafka messaging test failed: {e}")
        raise


async def test_node_execution():
    """Test node execution engine"""
    try:
        from app.services.execution.node_engine import (
            AIDecisionExecutor, AITextGeneratorExecutor, AIDataProcessorExecutor
        )

        execution_id = str(uuid.uuid4())

        # Test AI Decision Executor
        decision_executor = AIDecisionExecutor()
        decision_result = await decision_executor.execute(
            node_data={
                "prompt": "Should we proceed?",
                "options": ["yes", "no"],
                "ai_config": {"provider": "huggingface", "temperature": 0.3}
            },
            context={"previous_step": "analysis_complete"},
            execution_id=execution_id
        )

        assert "decision" in decision_result
        print("   ✅ AI Decision Executor works")

        # Test AI Text Generator Executor
        text_executor = AITextGeneratorExecutor()
        text_result = await text_executor.execute(
            node_data={
                "prompt_template": "Generate a summary about {topic}",
                "ai_config": {"provider": "huggingface", "temperature": 0.7}
            },
            context={"topic": "artificial intelligence"},
            execution_id=execution_id
        )

        assert "generated_text" in text_result
        print("   ✅ AI Text Generator Executor works")

        # Test AI Data Processor Executor
        processor_executor = AIDataProcessorExecutor()
        processor_result = await processor_executor.execute(
            node_data={
                "operation": "analyze",
                "data_source": "context",
                "ai_config": {"provider": "huggingface", "temperature": 0.5}
            },
            context={"data": {"sales": [100, 200, 150], "region": "north"}},
            execution_id=execution_id
        )

        assert "processed_data" in processor_result
        print("   ✅ AI Data Processor Executor works")

        print("   ✅ Node execution tests passed")

    except Exception as e:
        print(f"   ❌ Node execution test failed: {e}")
        raise


async def test_ai_models():
    """Test AI model integrations"""
    try:
        from app.services.ai.huggingface_service import hf_api
        from app.services.ai.model_factory import model_factory

        # Test HuggingFace API
        test_prompt = "Hello, how are you?"

        hf_result = await hf_api.generate_text(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            prompt=test_prompt,
            parameters={"max_tokens": 50, "temperature": 0.7}
        )

        assert "text" in hf_result
        print("   ✅ HuggingFace API works")

        # Test model factory
        available_providers = model_factory.get_available_providers()
        assert "huggingface" in available_providers
        print(f"   ✅ Available providers: {available_providers}")

        # Test popular models
        popular_models = model_factory.get_popular_huggingface_models()
        assert len(popular_models) > 0
        print(f"   ✅ Popular models loaded: {len(popular_models)}")

        print("   ✅ AI model tests passed")

    except Exception as e:
        print(f"   ❌ AI model test failed: {e}")
        raise


async def test_complete_workflow():
    """Test complete workflow simulation"""
    try:
        from app.services.context.redis_service import redis_context_service
        from app.services.messaging.kafka_service import kafka_service, NodeExecutionMessage
        from app.services.execution.node_engine import node_execution_engine

        execution_id = str(uuid.uuid4())
        workflow_id = str(uuid.uuid4())

        # Setup execution context
        initial_context = {
            "workflow_id": workflow_id,
            "status": "running",
            "input_data": {"query": "What is artificial intelligence?"},
            "started_at": datetime.utcnow().isoformat()
        }

        await redis_context_service.set_execution_context(execution_id, initial_context)

        # Setup dependencies (simulate Kahn's algorithm result)
        dependencies = {
            "analysis_node": 0,  # Ready to execute
            "summary_node": 1,  # Depends on analysis_node
            "output_node": 1  # Depends on summary_node
        }

        await redis_context_service.set_dependencies(execution_id, dependencies)

        # Simulate node execution message from Spring Boot
        ai_message = NodeExecutionMessage(
            execution_id=execution_id,
            workflow_id=workflow_id,
            node_id="analysis_node",
            node_type="ai_text_generator",
            node_data={
                "prompt_template": "Analyze this query: {query}",
                "ai_config": {
                    "provider": "huggingface",
                    "model": "microsoft/Phi-3-mini-4k-instruct",
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            },
            context=initial_context,
            dependencies=[],
            timestamp=datetime.utcnow().isoformat()
        )

        # Process the message (simulate what happens when Kafka message is received)
        await node_execution_engine.handle_execution_message(ai_message)

        # Verify results
        result = await redis_context_service.get_node_result(execution_id, "analysis_node")
        assert result is not None
        assert "generated_text" in result
        print("   ✅ Node execution completed")

        # Check that node was marked as completed
        completed_nodes = await redis_context_service.get_completed_nodes(execution_id)
        assert "analysis_node" in completed_nodes
        print("   ✅ Node marked as completed")

        # Cleanup
        await redis_context_service.cleanup_execution(execution_id)
        print("   ✅ Complete workflow simulation passed")

    except Exception as e:
        print(f"   ❌ Complete workflow test failed: {e}")
        raise


async def cleanup_test_services():
    """Cleanup test services"""
    try:
        from app.services.context.redis_service import redis_context_service
        from app.services.messaging.kafka_service import kafka_service
        from app.services.execution.node_engine import node_execution_engine

        await node_execution_engine.stop()
        await kafka_service.stop()
        await redis_context_service.disconnect()

        print("✅ Test services cleaned up")

    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(test_enhanced_fastapi_service())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)
    finally:
        try:
            asyncio.run(cleanup_test_services())
        except:
            pass