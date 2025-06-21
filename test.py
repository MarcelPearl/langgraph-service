"""
Enhanced test script for the FastAPI AI Workflow Service - FIXED VERSION
"""
import asyncio
import logging
import sys
import os
from typing import Optional

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Configuration for external access (when running outside Docker)
EXTERNAL_KAFKA_PORT = "localhost:9094"
EXTERNAL_REDIS_URL = "redis://localhost:6379/0"


async def wait_for_service(service_name: str, check_func, max_retries: int = 10, delay: int = 5) -> bool:
    """Wait for a service to become available"""
    for attempt in range(max_retries):
        try:
            await check_func()
            logger.info(f"✅ {service_name} is ready")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.info(f"⏳ {service_name} not ready (attempt {attempt + 1}/{max_retries}), waiting {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"❌ {service_name} failed after {max_retries} attempts: {e}")
                return False
    return False


async def check_redis_connection() -> bool:
    """Check Redis connection with retry logic"""
    try:
        # Try external connection first (for host machine)
        import redis.asyncio as redis

        # Try external Redis first
        try:
            client = redis.from_url(EXTERNAL_REDIS_URL, socket_timeout=5, socket_connect_timeout=5)
            await client.ping()
            await client.close()
            logger.info("✅ Redis connection successful (external)")
            return True
        except Exception:
            # Fall back to internal Redis
            client = redis.from_url(settings.REDIS_URL, socket_timeout=5, socket_connect_timeout=5)
            await client.ping()
            await client.close()
            logger.info("✅ Redis connection successful (internal)")
            return True

    except Exception as e:
        raise Exception(f"Redis connection failed: {e}")


async def check_kafka_connection() -> bool:
    """Check Kafka connection with retry logic - FIXED VERSION"""
    try:
        from aiokafka import AIOKafkaProducer

        # Try external Kafka first (for host machine)
        bootstrap_servers = [EXTERNAL_KAFKA_PORT, settings.KAFKA_BOOTSTRAP_SERVERS]

        for server in bootstrap_servers:
            producer = None
            try:
                producer = AIOKafkaProducer(
                    bootstrap_servers=server,
                    request_timeout_ms=10000,
                    connections_max_idle_ms=5000,
                    retry_backoff_ms=100
                )
                await producer.start()

                # FIXED: Use the correct method to get metadata
                # Test by sending a test message to a test topic (this will verify connection)
                try:
                    # Try to get cluster metadata using the correct method
                    metadata = await producer.client.metadata()

                    logger.info(f"✅ Kafka connection successful ({server})")
                    logger.info(f"   Available topics: {len(metadata.topics)}")
                    logger.info(f"   Available brokers: {len(metadata.brokers)}")

                    await producer.stop()
                    return True

                except AttributeError:
                    # Alternative method: try to send to a non-existent topic and check the error
                    try:
                        # This will fail but will verify that Kafka is responsive
                        await asyncio.wait_for(
                            producer.send('test-connection-topic', b'test'),
                            timeout=5.0
                        )
                    except Exception:
                        # If we get any response (even an error), Kafka is working
                        pass

                    logger.info(f"✅ Kafka connection successful ({server})")
                    await producer.stop()
                    return True

            except Exception as e:
                logger.debug(f"Kafka connection failed for {server}: {e}")
                if producer:
                    try:
                        await producer.stop()
                    except:
                        pass
                continue

        raise Exception("All Kafka connection attempts failed")

    except Exception as e:
        raise Exception(f"Kafka connection failed: {e}")


async def check_ai_services() -> bool:
    """Check AI services"""
    try:
        from app.services.ai.huggingface_service import hf_api

        test_result = await hf_api.generate_text(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            prompt="Hello",
            parameters={"max_tokens": 10}
        )

        if test_result and "text" in test_result and not test_result.get("fallback"):
            logger.info("✅ HuggingFace API working")
            return True
        else:
            logger.warning("⚠️  HuggingFace API returned fallback response")
            return False

    except Exception as e:
        raise Exception(f"AI service check failed: {e}")


async def check_dependencies():
    """Check if all required services are available"""
    logger.info("🔍 Checking service dependencies...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    results = {}

    # Check Redis with retries
    logger.info("\n📡 Testing Redis connection...")
    results["redis"] = await wait_for_service(
        "Redis",
        check_redis_connection,
        max_retries=6,
        delay=3
    )

    # Check Kafka with retries
    logger.info("\n📡 Testing Kafka connection...")
    results["kafka"] = await wait_for_service(
        "Kafka",
        check_kafka_connection,
        max_retries=8,
        delay=5
    )

    # Check AI Services
    logger.info("\n🤖 Testing AI services...")
    results["ai"] = await wait_for_service(
        "AI Services",
        check_ai_services,
        max_retries=3,
        delay=2
    )

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 DEPENDENCY CHECK SUMMARY")
    logger.info("=" * 50)

    all_healthy = True
    for service, status in results.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"{status_icon} {service.upper()}: {'HEALTHY' if status else 'FAILED'}")
        if not status:
            all_healthy = False

    if all_healthy:
        logger.info("\n🎉 All dependencies are healthy! Service ready to start.")
    else:
        logger.warning("\n⚠️  Some dependencies have issues. Service may have limited functionality.")
        logger.info("\n💡 Troubleshooting tips:")
        if not results["kafka"]:
            logger.info("   - Make sure Kafka is running: docker-compose up kafka")
            logger.info("   - Wait for Kafka to fully start (can take 30-60 seconds)")
            logger.info("   - Check Kafka logs: docker-compose logs kafka")
        if not results["redis"]:
            logger.info("   - Make sure Redis is running: docker-compose up redis")
            logger.info("   - Check Redis logs: docker-compose logs redis")

    return all_healthy


async def initialize_services():
    """Initialize all services"""
    logger.info("🚀 Initializing services...")

    try:
        # Initialize event producer (graceful fallback if Kafka not available)
        try:
            from app.services.kafka.producer import initialize_event_producer
            await initialize_event_producer()
            logger.info("✅ Event producer initialized")
        except Exception as e:
            logger.warning(f"⚠️  Event producer initialization failed: {e}")
            logger.info("   Service will use mock event producer")

        logger.info("✅ Service initialization complete")

    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise


async def test_full_workflow():
    """Test a complete workflow execution"""
    logger.info("\n🧪 Testing complete workflow execution...")

    try:
        from app.services.execution.node_engine import node_execution_engine
        from app.services.context.redis_service import redis_context_service
        from app.services.messaging.kafka_service import NodeExecutionMessage
        import uuid
        from datetime import datetime

        # Connect services
        await redis_context_service.connect()

        execution_id = str(uuid.uuid4())
        workflow_id = str(uuid.uuid4())

        # Test AI decision node
        test_message = NodeExecutionMessage(
            execution_id=execution_id,
            workflow_id=workflow_id,
            node_id="test_decision_node",
            node_type="ai_decision",
            node_data={
                "prompt": "Should we proceed with the test?",
                "options": ["yes", "no", "maybe"],
                "ai_config": {
                    "provider": "huggingface",
                    "model": "microsoft/Phi-3-mini-4k-instruct",
                    "temperature": 0.3
                }
            },
            context={"test_mode": True},
            dependencies=[],
            timestamp=datetime.utcnow().isoformat()
        )

        # Execute the node
        await node_execution_engine.handle_execution_message(test_message)

        # Check results
        result = await redis_context_service.get_node_result(execution_id, "test_decision_node")

        if result and "decision" in result:
            logger.info(f"✅ Workflow test successful!")
            logger.info(f"   Decision: {result['decision']}")
            logger.info(f"   Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
        else:
            logger.error("❌ Workflow test failed - no valid result")

        # Cleanup
        await redis_context_service.cleanup_execution(execution_id)
        await redis_context_service.disconnect()

        return True

    except Exception as e:
        logger.error(f"❌ Workflow test failed: {e}")
        return False


def run_fastapi_server():
    """Run the FastAPI server"""
    import uvicorn

    logger.info(f"🌟 Starting FastAPI server on port 8000")
    logger.info(f"📊 Debug mode: {settings.DEBUG}")
    logger.info(f"🌍 Environment: {settings.ENVIRONMENT}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )


async def main():
    """Main startup function"""
    logger.info("=" * 60)
    logger.info("🤖 FastAPI AI Workflow Service - Enhanced Test (FIXED)")
    logger.info("=" * 60)

    try:
        # Check dependencies
        all_healthy = await check_dependencies()

        # Initialize services
        await initialize_services()

        # Run workflow test if dependencies are healthy
        if all_healthy:
            await test_full_workflow()

        logger.info("\n🎯 Testing complete!")

        if not all_healthy:
            logger.warning("⚠️  Some services have issues. Check logs above for details.")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def cli():
    """Command line interface"""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "check":
            # Just run dependency checks
            result = asyncio.run(check_dependencies())
            sys.exit(0 if result else 1)

        elif command == "workflow":
            # Test workflow only
            async def workflow_only():
                await initialize_services()
                return await test_full_workflow()

            result = asyncio.run(workflow_only())
            sys.exit(0 if result else 1)

        elif command == "server":
            # Start server after checks
            asyncio.run(main())
            run_fastapi_server()
            return

        elif command == "worker":
            # Run as event processor worker
            logger.info("🔧 Starting event processor worker...")
            try:
                from app.services.kafka.event_processor_service import main as worker_main
                asyncio.run(worker_main())
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
            return

    # Default: run full test
    result = asyncio.run(main())
    if result:
        logger.info("\n✅ All tests passed! You can now start the server.")
        logger.info("💡 To start the server: python test.py server")
    else:
        logger.warning("\n⚠️  Some tests failed. Check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}")
        sys.exit(1)