#!/usr/bin/env python3
"""
Kafka Management CLI Tool

This script provides command-line utilities for managing Kafka topics,
monitoring cluster health, and managing dead letter queues.

Usage:
    python kafka_manager.py --help
    python kafka_manager.py create-topics
    python kafka_manager.py health-check
    python kafka_manager.py list-topics
    python kafka_manager.py monitor-dlq
    python kafka_manager.py reset-offsets --group GROUP_NAME --topic TOPIC_NAME
"""

import asyncio
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any

# Add app directory to path
sys.path.append('../app/')

from app.services.kafka.admin import kafka_admin, dlq_manager
from app.services.kafka.producer import event_producer
from app.services.kafka.consumer import WorkflowEventConsumer
from app.core.kafka_config import kafka_settings


class KafkaManager:
    """CLI manager for Kafka operations"""

    def __init__(self):
        self.admin = kafka_admin

    async def create_topics(self) -> Dict[str, Any]:
        """Create all required Kafka topics"""
        print("🔧 Creating Kafka topics...")

        try:
            await self.admin.start()
            results = await self.admin.create_topics()

            print(f"✅ Topic creation results:")
            for topic, success in results.items():
                status = "✅ Created" if success else "❌ Failed"
                print(f"   {topic}: {status}")

            return results

        except Exception as e:
            print(f"❌ Error creating topics: {e}")
            return {}
        finally:
            await self.admin.stop()

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive cluster health check"""
        print("🏥 Performing Kafka cluster health check...")

        try:
            await self.admin.start()
            health = await self.admin.monitor_cluster_health()

            print(f"📊 Cluster Health Report")
            print(f"Overall Status: {health.get('overall_status', 'unknown').upper()}")
            print(f"Timestamp: {datetime.now().isoformat()}")
            print()

            # Cluster info
            cluster_info = health.get('cluster_info', {})
            print(f"🖥️  Cluster Information:")
            print(f"   Cluster ID: {cluster_info.get('cluster_id', 'unknown')}")
            print(f"   Brokers: {cluster_info.get('brokers', 0)}")
            print(f"   Total Topics: {cluster_info.get('total_topics', 0)}")
            print()

            # Topic health
            topic_health = health.get('topic_health', {})
            print(f"📋 Topic Health:")
            for topic, info in topic_health.items():
                status = info.get('status', 'unknown')
                emoji = "✅" if status == 'healthy' else "❌"
                print(f"   {emoji} {topic}: {status}")
                if info.get('exists'):
                    print(f"      Partitions: {info.get('partitions', 0)}")
                    print(f"      Replication Factor: {info.get('replication_factor', 0)}")
                    print(f"      Messages: {info.get('total_messages', 0)}")
            print()

            # Consumer groups
            consumer_groups = health.get('consumer_groups', {})
            print(f"👥 Consumer Groups ({len(consumer_groups)}):")
            for group, info in consumer_groups.items():
                if 'error' not in info:
                    print(f"   📊 {group}: {info.get('state', 'unknown')} ({info.get('members', 0)} members)")
                else:
                    print(f"   ❌ {group}: {info['error']}")

            return health

        except Exception as e:
            print(f"❌ Error performing health check: {e}")
            return {}
        finally:
            await self.admin.stop()

    async def list_topics(self) -> Dict[str, Any]:
        """List all topics with detailed information"""
        print("📋 Listing Kafka topics...")

        try:
            await self.admin.start()
            topics_info = await self.admin.list_topics()

            print(f"📊 Topics Information:")
            print(f"Cluster ID: {topics_info.get('cluster_id', 'unknown')}")
            print(f"Brokers: {topics_info.get('brokers', 0)}")
            print(f"Total Topics: {topics_info.get('total_topics', 0)}")
            print()

            topics = topics_info.get('topics', {})
            for topic_name, topic_info in topics.items():
                print(f"📄 {topic_name}")
                print(f"   Partitions: {topic_info.get('partitions', 0)}")
                print(f"   Replication Factor: {topic_info.get('replication_factor', 0)}")

                # Get detailed offset information
                offsets = await self.admin.get_topic_offsets(topic_name)
                if offsets:
                    print(f"   Total Messages: {offsets.get('total_messages', 0)}")
                print()

            return topics_info

        except Exception as e:
            print(f"❌ Error listing topics: {e}")
            return {}
        finally:
            await self.admin.stop()

    async def monitor_dlq(self, limit: int = 50) -> None:
        """Monitor dead letter queue"""
        print(f"💀 Monitoring Dead Letter Queue (showing last {limit} messages)...")

        try:
            dlq_manager.start()
            messages = dlq_manager.get_failed_messages(limit=limit)

            if not messages:
                print("✅ No messages in dead letter queue")
                return

            print(f"📊 Found {len(messages)} failed messages:")
            print()

            for i, message in enumerate(messages, 1):
                print(f"🔴 Message {i}:")
                print(f"   Partition: {message['partition']}")
                print(f"   Offset: {message['offset']}")
                print(f"   Key: {message['key']}")
                print(f"   Timestamp: {datetime.fromtimestamp(message['timestamp'] / 1000).isoformat()}")

                # Show original event info if available
                original_event = message['value'].get('original_event', {})
                if original_event:
                    print(f"   Event Type: {original_event.get('event_type', 'unknown')}")
                    print(f"   Event ID: {original_event.get('event_id', 'unknown')}")

                # Show error info
                error = message['value'].get('error', 'Unknown error')
                print(f"   Error: {error}")
                print()

        except Exception as e:
            print(f"❌ Error monitoring DLQ: {e}")
        finally:
            dlq_manager.stop()

    async def reset_consumer_offsets(self, group_id: str, topic: str, reset_to: str = 'earliest') -> bool:
        """Reset consumer group offsets"""
        print(f"⚠️  Resetting consumer group '{group_id}' offsets for topic '{topic}' to '{reset_to}'")
        print("This operation will affect message consumption. Continue? (y/N): ", end='')

        confirm = input().lower()
        if confirm != 'y':
            print("❌ Operation cancelled")
            return False

        try:
            await self.admin.start()
            success = await self.admin.reset_consumer_group_offsets(group_id, topic, reset_to)

            if success:
                print(f"✅ Successfully reset offsets for group '{group_id}'")
            else:
                print(f"❌ Failed to reset offsets for group '{group_id}'")

            return success

        except Exception as e:
            print(f"❌ Error resetting offsets: {e}")
            return False
        finally:
            await self.admin.stop()

    async def test_event_flow(self) -> None:
        """Test event publishing and consumption"""
        print("🧪 Testing event flow...")

        try:
            # Start producer
            await event_producer.start()

            # Create test event
            from app.schemas.events import create_execution_event, EventType
            test_event = create_execution_event(
                event_type=EventType.EXECUTION_STARTED,
                execution_id="test-execution-123",
                workflow_id="test-workflow-456",
                execution_status="running",
                metadata={"test": True}
            )

            print("📤 Publishing test event...")
            success = await event_producer.publish_event(test_event)

            if success:
                print("✅ Test event published successfully")
            else:
                print("❌ Failed to publish test event")

        except Exception as e:
            print(f"❌ Error testing event flow: {e}")
        finally:
            await event_producer.stop()

    async def show_config(self) -> None:
        """Show current Kafka configuration"""
        print("⚙️  Current Kafka Configuration:")
        print()

        config_items = [
            ("Bootstrap Servers", kafka_settings.KAFKA_BOOTSTRAP_SERVERS),
            ("Workflow Events Topic", kafka_settings.KAFKA_WORKFLOW_EVENTS_TOPIC),
            ("Execution Events Topic", kafka_settings.KAFKA_EXECUTION_EVENTS_TOPIC),
            ("Node Events Topic", kafka_settings.KAFKA_NODE_EVENTS_TOPIC),
            ("Dead Letter Topic", kafka_settings.KAFKA_DEAD_LETTER_TOPIC),
            ("Consumer Group", kafka_settings.KAFKA_CONSUMER_GROUP_ID),
            ("Default Partitions", kafka_settings.KAFKA_DEFAULT_PARTITIONS),
            ("Replication Factor", kafka_settings.KAFKA_REPLICATION_FACTOR),
            ("Auto Create Topics", kafka_settings.KAFKA_AUTO_CREATE_TOPICS),
        ]

        for name, value in config_items:
            print(f"📝 {name}: {value}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Kafka Management CLI")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create topics command
    subparsers.add_parser('create-topics', help='Create all required Kafka topics')

    # Health check command
    subparsers.add_parser('health-check', help='Perform cluster health check')

    # List topics command
    subparsers.add_parser('list-topics', help='List all topics with details')

    # Monitor DLQ command
    dlq_parser = subparsers.add_parser('monitor-dlq', help='Monitor dead letter queue')
    dlq_parser.add_argument('--limit', type=int, default=50, help='Number of messages to show')

    # Reset offsets command
    reset_parser = subparsers.add_parser('reset-offsets', help='Reset consumer group offsets')
    reset_parser.add_argument('--group', required=True, help='Consumer group ID')
    reset_parser.add_argument('--topic', required=True, help='Topic name')
    reset_parser.add_argument('--reset-to', default='earliest', choices=['earliest', 'latest'], help='Reset position')

    # Test event flow command
    subparsers.add_parser('test-events', help='Test event publishing and consumption')

    # Show config command
    subparsers.add_parser('show-config', help='Show current Kafka configuration')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = KafkaManager()

    try:
        if args.command == 'create-topics':
            await manager.create_topics()
        elif args.command == 'health-check':
            await manager.health_check()
        elif args.command == 'list-topics':
            await manager.list_topics()
        elif args.command == 'monitor-dlq':
            await manager.monitor_dlq(args.limit)
        elif args.command == 'reset-offsets':
            await manager.reset_consumer_offsets(args.group, args.topic, args.reset_to)
        elif args.command == 'test-events':
            await manager.test_event_flow()
        elif args.command == 'show-config':
            await manager.show_config()
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)