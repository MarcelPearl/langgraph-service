#!/usr/bin/env python3
"""
Week 3 Integration Test Suite

Comprehensive tests for the complete Week 3 implementation:
- Kafka infrastructure
- Event-driven workflows
- AI integration
- End-to-end scenarios
"""

import pytest
import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.kafka.producer import event_producer
from app.services.kafka.consumer import WorkflowEventConsumer
from app.services.kafka.admin import kafka_admin
from app.schemas.events import (
    EventType, create_execution_event, create_node_event,
    create_ai_request_event, create_tool_call_event
)
from app.services.langgraph.workflow import BasicWorkflowEngine
from app.services.ai.model_factory import model_factory
from app.core.database import async_session_maker


class Week3IntegrationTest:
    """Complete Week 3 integration test suite"""

    def __init__(self):
        self.test_results = {}
        self.execution_id = str(uuid.uuid4())
        self.workflow_id = str(uuid.uuid4())

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("🧪 Starting Week 3 Integration Test Suite")
        print("=" * 50)

        test_methods = [
            self.test_kafka_infrastructure,
            self.test_event_publishing,
            self.test_event_consumption,
            self.test_workflow_execution,
            self.test_ai_integration,
            self.test_tool_execution,
            self.test_error_handling,
            self.test_performance,
            self.test_end_to_end_scenario
        ]

        overall_success = True

        for test_method in test_methods:
            test_name = test_method.__name__
            print(f"\n🔬 Running {test_name}...")

            try:
                start_time = time.time()
                result = await test_method()
                duration = time.time() - start_time

                self.test_results[test_name] = {
                    'status': 'passed' if result else 'failed',
                    'duration': duration,
                    'details': result if isinstance(result, dict) else {}
                }

                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"   {status} ({duration:.2f}s)")

                if not result:
                    overall_success = False

            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'error',
                    'duration': 0,
                    'error': str(e)
                }
                print(f"   ❌ ERROR: {e}")
                overall_success = False

        # Print summary
        self._print_test_summary(overall_success)

        return {
            'overall_success': overall_success,
            'test_results': self.test_results,
            'execution_id': self.execution_id
        }

    async def test_kafka_infrastructure(self) -> bool:
        """Test Kafka infrastructure setup"""
        try:
            # Test admin connectivity
            await kafka_admin.start()
            health = await kafka_admin.monitor_cluster_health()
            await kafka_admin.stop()

            # Check cluster health
            overall_status = health.get('overall_status')
            if overall_status not in ['healthy', 'degraded']:
                return False

            # Check topics
            topic_health = health.get('topic_health', {})
            required_topics = [
                'workflow-events',
                'execution-events',
                'node-events',
                'dead-letter-queue'
            ]

            for topic in required_topics:
                if topic not in topic_health:
                    print(f"   Missing topic: {topic}")
                    return False

            return True

        except Exception as e:
            print(f"   Kafka infrastructure error: {e}")
            return False

    async def test_event_publishing(self) -> bool:
        """Test event publishing functionality"""
        try:
            # Ensure producer is started
            if not event_producer.is_started:
                await event_producer.start()

            # Test different event types
            test_events = [
                create_execution_event(
                    event_type=EventType.EXECUTION_STARTED,
                    execution_id=self.execution_id,
                    workflow_id=self.workflow_id,
                    execution_status="running"
                ),
                create_node_event(
                    event_type=EventType.NODE_STARTED,
                    execution_id=self.execution_id,
                    workflow_id=self.workflow_id,
                    node_id="test_node",
                    node_type="test"
                ),
                create_ai_request_event(
                    event_type=EventType.AI_REQUEST_STARTED,
                    execution_id=self.execution_id,
                    workflow_id=self.workflow_id,
                    ai_provider="openai",
                    ai_model="gpt-4"
                )
            ]

            success_count = 0
            for event in test_events:
                success = await event_producer.publish_event(event)
                if success:
                    success_count += 1

            return success_count == len(test_events)

        except Exception as e:
            print(f"   Event publishing error: {e}")
            return False

    async def test_event_consumption(self) -> bool:
        """Test event consumption functionality"""
        try:
            # Create a test consumer
            async def mock_db_session():
                class MockSession:
                    async def execute(self, *args, **kwargs):
                        pass

                    async def commit(self, *args, **kwargs):
                        pass

                    async def rollback(self, *args, **kwargs):
                        pass

                return MockSession()

            consumer = WorkflowEventConsumer(mock_db_session)

            # Test event creation and parsing
            test_event_data = {
                "event_type": "execution.started",
                "event_id": str(uuid.uuid4()),
                "execution_id": self.execution_id,
                "workflow_id": self.workflow_id,
                "execution_status": "running",
                "timestamp": datetime.utcnow().isoformat()
            }

            # Test event payload creation
            event_payload = consumer._create_event_payload(
                EventType.EXECUTION_STARTED,
                test_event_data
            )

            return event_payload is not None

        except Exception as e:
            print(f"   Event consumption error: {e}")
            return False

    async def test_workflow_execution(self) -> bool:
        """Test basic workflow execution"""
        try:
            async with async_session_maker() as db:
                engine = BasicWorkflowEngine(db)

                # Test workflow creation
                workflow = await engine.create_basic_workflow()

                if workflow is None:
                    return False

                # Test state management
                state_manager = engine.state_manager
                initial_state = state_manager.create_initial_state(
                    execution_id=self.execution_id,
                    workflow_id=self.workflow_id,
                    input_data={"query": "test query"},
                    ai_config={"provider": "huggingface", "model": "gpt2"}
                )

                return initial_state is not None

        except Exception as e:
            print(f"   Workflow execution error: {e}")
            return False

    async def test_ai_integration(self) -> bool:
        """Test AI model integration"""
        try:
            # Test model factory
            providers = model_factory.get_available_providers()

            if not providers:
                print("   No AI providers available")
                return False

            # Test HuggingFace integration (always available)
            from app.services.ai.huggingface_service import hf_api

            test_result = await hf_api.generate_text(
                model_name="gpt2",
                prompt="Hello world",
                parameters={"max_tokens": 10}
            )

            return "text" in test_result

        except Exception as e:
            print(f"   AI integration error: {e}")
            return False

    async def test_tool_execution(self) -> bool:
        """Test tool execution functionality"""
        try:
            from app.services.langgraph.tools import tool_registry

            # Test calculator tool
            calculator = tool_registry.get_tool("calculator")
            if calculator is None:
                return False

            result = calculator._run("2 + 2")
            if "4" not in result:
                return False

            # Test text analyzer tool
            text_analyzer = tool_registry.get_tool("text_analyzer")
            if text_analyzer is None:
                return False

            result = text_analyzer._run("hello world", "length")
            if "2 words" not in result:
                return False

            return True

        except Exception as e:
            print(f"   Tool execution error: {e}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling mechanisms"""
        try:
            # Test error event publishing
            error_event = create_execution_event(
                event_type=EventType.EXECUTION_FAILED,
                execution_id=self.execution_id,
                workflow_id=self.workflow_id,
                execution_status="failed",
                error_data={
                    "error_type": "TestError",
                    "message": "Test error handling"
                }
            )

            success = await event_producer.publish_event(error_event)
            return success

        except Exception as e:
            print(f"   Error handling test error: {e}")
            return False

    async def test_performance(self) -> bool:
        """Test basic performance metrics"""
        try:
            # Test event publishing performance
            start_time = time.time()

            events = [
                create_execution_event(
                    event_type=EventType.EXECUTION_COMPLETED,
                    execution_id=str(uuid.uuid4()),
                    workflow_id=self.workflow_id,
                    execution_status="completed"
                ) for _ in range(10)
            ]

            results = await event_producer.publish_batch(events)

            end_time = time.time()
            duration = end_time - start_time

            # Should complete within reasonable time
            if duration > 10.0:  # 10 seconds for 10 events
                print(f"   Performance too slow: {duration:.2f}s for 10 events")
                return False

            # Should have high success rate
            success_rate = results['success'] / (results['success'] + results['failed'])
            if success_rate < 0.8:  # 80% success rate
                print(f"   Low success rate: {success_rate:.2%}")
                return False

            return True

        except Exception as e:
            print(f"   Performance test error: {e}")
            return False

    async def test_end_to_end_scenario(self) -> bool:
        """Test complete end-to-end workflow scenario"""
        try:
            # Simulate complete workflow execution with events
            scenario_id = str(uuid.uuid4())

            # 1. Workflow started
            start_event = create_execution_event(
                event_type=EventType.EXECUTION_STARTED,
                execution_id=scenario_id,
                workflow_id=self.workflow_id,
                execution_status="running",
                input_data={"query": "end-to-end test"}
            )

            success1 = await event_producer.publish_event(start_event)

            # 2. AI request
            ai_event = create_ai_request_event(
                event_type=EventType.AI_REQUEST_STARTED,
                execution_id=scenario_id,
                workflow_id=self.workflow_id,
                ai_provider="huggingface",
                ai_model="gpt2"
            )

            success2 = await event_producer.publish_event(ai_event)

            # 3. Tool execution
            tool_event = create_tool_call_event(
                event_type=EventType.TOOL_CALL_STARTED,
                execution_id=scenario_id,
                workflow_id=self.workflow_id,
                tool_name="calculator"
            )

            success3 = await event_producer.publish_event(tool_event)

            # 4. Workflow completed
            complete_event = create_execution_event(
                event_type=EventType.EXECUTION_COMPLETED,
                execution_id=scenario_id,
                workflow_id=self.workflow_id,
                execution_status="completed",
                output_data={"result": "end-to-end test completed"}
            )

            success4 = await event_producer.publish_event(complete_event)

            return all([success1, success2, success3, success4])

        except Exception as e:
            print(f"   End-to-end scenario error: {e}")
            return False

    def _print_test_summary(self, overall_success: bool):
        """Print comprehensive test summary"""
        print("\n" + "=" * 50)
        print("📊 WEEK 3 TEST SUMMARY")
        print("=" * 50)

        passed = sum(1 for r in self.test_results.values() if r['status'] == 'passed')
        failed = sum(1 for r in self.test_results.values() if r['status'] == 'failed')
        errors = sum(1 for r in self.test_results.values() if r['status'] == 'error')
        total = len(self.test_results)

        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errors: {errors}")
        print(f"📊 Total:  {total}")
        print()

        # Detailed results
        for test_name, result in self.test_results.items():
            status = result['status']
            duration = result.get('duration', 0)

            if status == 'passed':
                emoji = "✅"
            elif status == 'failed':
                emoji = "❌"
            else:
                emoji = "⚠️"

            print(f"{emoji} {test_name.replace('test_', '').replace('_', ' ').title()}: {status} ({duration:.2f}s)")

        print("\n" + "=" * 50)

        if overall_success:
            print("🎉 ALL TESTS PASSED - WEEK 3 SYSTEM IS READY!")
        else:
            print("❌ SOME TESTS FAILED - CHECK SYSTEM CONFIGURATION")

        print("=" * 50)


async def main():
    """Run the complete test suite"""
    test_suite = Week3IntegrationTest()

    try:
        results = await test_suite.run_all_tests()
        return 0 if results['overall_success'] else 1

    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)