#!/usr/bin/env python3
"""
Week 3 Startup Script - Single Broker Setup

This script initializes the Week 3 system with single Kafka broker:
- Creates Kafka topics
- Tests connectivity
- Runs health checks
- Starts services
"""

import asyncio
import sys
import os
import time
import logging
from typing import Dict, Any

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.kafka.admin import kafka_admin
from app.services.kafka.producer import event_producer
from app.schemas.events import create_execution_event, EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Week3Initializer:
    """Initialize Week 3 system with single Kafka broker"""

    def __init__(self):
        self.services_ready = {
            'kafka_ready': False,
            'kafka_producer': False,
            'topics_created': False,
            'connectivity_tested': False
        }

    async def initialize_all(self) -> bool:
        """Initialize all Week 3 components"""
        logger.info("🚀 Starting Week 3 Single Broker Initialization")

        try:
            # 1. Wait for Kafka to be ready
            await self._wait_for_kafka()

            # 2. Create topics
            await self._create_topics()

            # 3. Initialize producer
            await self._initialize_producer()

            # 4. Test connectivity
            await self._test_connectivity()

            # 5. Run health checks
            await self._run_health_checks()

            if all(self.services_ready.values()):
                logger.info("✅ Week 3 Single Broker System Ready!")
                self._print_success_summary()
                return True
            else:
                logger.error("❌ Some services failed to initialize")
                self._print_failure_summary()
                return False

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

    async def _wait_for_kafka(self, max_retries: int = 20):
        """Wait for single Kafka broker to be ready"""
        logger.info("⏳ Waiting for Kafka broker to be ready...")

        for attempt in range(max_retries):
            try:
                await kafka_admin.start()

                # Test basic connectivity
                topics_info = await kafka_admin.list_topics()
                if topics_info:
                    logger.info("✅ Kafka broker is ready")
                    self.services_ready['kafka_ready'] = True
                    await kafka_admin.stop()
                    return True

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.info(f"   Attempt {attempt + 1}/{max_retries} - waiting...")
                    await asyncio.sleep(3)
                else:
                    raise Exception(f"Kafka not ready after {max_retries} attempts: {e}")
            finally:
                try:
                    await kafka_admin.stop()
                except:
                    pass

    async def _create_topics(self):
        """Create all required Kafka topics for single broker"""
        logger.info("📋 Creating Kafka topics...")

        try:
            await kafka_admin.start()
            results = await kafka_admin.create_topics()

            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)

            logger.info(f"📊 Topic creation results: {success_count}/{total_count} successful")

            for topic, success in results.items():
                status = "✅" if success else "❌"
                logger.info(f"   {status} {topic}")

            self.services_ready['topics_created'] = success_count > 0

        except Exception as e:
            logger.error(f"❌ Failed to create topics: {e}")
            # Don't raise - topics might auto-create
            logger.info("ℹ️  Topics may be auto-created when first used")
        finally:
            await kafka_admin.stop()

    async def _initialize_producer(self):
        """Initialize Kafka producer"""
        logger.info("📤 Initializing Kafka producer...")

        try:
            await event_producer.start()
            self.services_ready['kafka_producer'] = True
            logger.info("✅ Kafka producer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize producer: {e}")
            raise

    async def _test_connectivity(self):
        """Test Kafka connectivity with test events"""
        logger.info("🧪 Testing Kafka connectivity...")

        try:
            # Create test event
            test_event = create_execution_event(
                event_type=EventType.EXECUTION_STARTED,
                execution_id="startup-test",
                workflow_id="system-test",
                execution_status="running",
                metadata={"test": True, "source": "startup_script", "broker": "single"}
            )

            # Publish test event
            success = await event_producer.publish_event(test_event)

            if success:
                logger.info("✅ Connectivity test successful")
                self.services_ready['connectivity_tested'] = True
            else:
                logger.error("❌ Connectivity test failed")

        except Exception as e:
            logger.error(f"❌ Connectivity test error: {e}")

    async def _run_health_checks(self) -> Dict[str, Any]:
        """Run basic health checks for single broker"""
        logger.info("🏥 Running health checks...")

        try:
            await kafka_admin.start()
            topics_info = await kafka_admin.list_topics()
            await kafka_admin.stop()

            if topics_info:
                cluster_id = topics_info.get('cluster_id', 'unknown')
                total_topics = topics_info.get('total_topics', 0)

                logger.info(f"📊 Cluster ID: {cluster_id}")
                logger.info(f"📊 Topics: {total_topics}")
                logger.info("✅ Health check passed")

                return topics_info
            else:
                logger.warning("⚠️  Health check returned no data")
                return {}

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {}

    def _print_success_summary(self):
        """Print success summary"""
        print("\n" + "=" * 60)
        print("🎉 WEEK 3 SINGLE BROKER SYSTEM READY!")
        print("=" * 60)
        print("✅ Kafka Broker: Ready")
        print("✅ Kafka Producer: Ready")
        print("✅ Topics: Created")
        print("✅ Connectivity: Tested")
        print()
        print("🚀 You can now:")
        print("   • Start the FastAPI application: uvicorn app.main:app --reload")
        print("   • Run the event consumer: python consumer.py")
        print("   • Access API docs: http://localhost:8000/docs")
        print("   • Monitor Kafka: http://localhost:1660 (Kafka UI)")
        print()
        print("🧪 Test the system:")
        print("   • curl http://localhost:8000/kafka/test")
        print("   • make demo")
        print("   • make test")
        print("=" * 60)

    def _print_failure_summary(self):
        """Print failure summary"""
        print("\n" + "=" * 60)
        print("❌ WEEK 3 SYSTEM INITIALIZATION FAILED")
        print("=" * 60)

        for service, status in self.services_ready.items():
            emoji = "✅" if status else "❌"
            print(f"{emoji} {service.replace('_', ' ').title()}: {'Ready' if status else 'Failed'}")

        print()
        print("🔧 Troubleshooting:")
        print("   • Check if Kafka is running: docker-compose ps")
        print("   • Check logs: docker-compose logs kafka")
        print("   • Restart services: docker-compose down && docker-compose up -d")
        print("   • Wait for Kafka to start (can take 30-60 seconds)")
        print("=" * 60)


async def main():
    """Main entry point"""
    initializer = Week3Initializer()

    try:
        success = await initializer.initialize_all()
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("🛑 Initialization interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        return 1
    finally:
        # Clean up
        try:
            await event_producer.stop()
        except:
            pass


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)