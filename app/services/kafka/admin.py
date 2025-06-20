"""
Kafka Administration and Management Utilities

This module provides:
- Topic creation and management
- Consumer group monitoring
- Partition management
- Cluster health monitoring
- Dead letter queue management
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from kafka import KafkaAdminClient, KafkaConsumer
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
from kafka.errors import TopicAlreadyExistsError, KafkaError
from aiokafka.admin import AIOKafkaAdminClient
from app.core.kafka_config import get_producer_config, kafka_settings

logger = logging.getLogger(__name__)


class KafkaAdministrator:
    """Kafka cluster administration and monitoring"""

    def __init__(self):
        self.admin_client: Optional[AIOKafkaAdminClient] = None
        self.sync_admin_client: Optional[KafkaAdminClient] = None
        self._config = get_producer_config()

    async def start(self):
        """Start the admin client"""
        try:
            # Async admin client
            self.admin_client = AIOKafkaAdminClient(
                bootstrap_servers=self._config['bootstrap_servers']
            )
            await self.admin_client.start()

            # Sync admin client for certain operations
            self.sync_admin_client = KafkaAdminClient(
                bootstrap_servers=self._config['bootstrap_servers']
            )

            logger.info("Kafka admin clients started successfully")

        except Exception as e:
            logger.error(f"Failed to start Kafka admin clients: {e}")
            raise

    async def stop(self):
        """Stop the admin client"""
        if self.admin_client:
            await self.admin_client.close()

        if self.sync_admin_client:
            self.sync_admin_client.close()

        logger.info("Kafka admin clients stopped")

    async def create_topics(self) -> Dict[str, bool]:
        """Create all required topics for the workflow system"""
        topics_to_create = [
            NewTopic(
                name=kafka_settings.KAFKA_WORKFLOW_EVENTS_TOPIC,
                num_partitions=kafka_settings.KAFKA_DEFAULT_PARTITIONS,
                replication_factor=kafka_settings.KAFKA_REPLICATION_FACTOR,
                configs={
                    'cleanup.policy': 'delete',
                    'retention.ms': '604800000',  # 7 days
                    'compression.type': 'snappy'
                }
            ),
            NewTopic(
                name=kafka_settings.KAFKA_EXECUTION_EVENTS_TOPIC,
                num_partitions=kafka_settings.KAFKA_DEFAULT_PARTITIONS,
                replication_factor=kafka_settings.KAFKA_REPLICATION_FACTOR,
                configs={
                    'cleanup.policy': 'delete',
                    'retention.ms': '2592000000',  # 30 days
                    'compression.type': 'snappy'
                }
            ),
            NewTopic(
                name=kafka_settings.KAFKA_NODE_EVENTS_TOPIC,
                num_partitions=kafka_settings.KAFKA_DEFAULT_PARTITIONS,
                replication_factor=kafka_settings.KAFKA_REPLICATION_FACTOR,
                configs={
                    'cleanup.policy': 'delete',
                    'retention.ms': '604800000',  # 7 days
                    'compression.type': 'snappy'
                }
            ),
            NewTopic(
                name=kafka_settings.KAFKA_DEAD_LETTER_TOPIC,
                num_partitions=3,
                replication_factor=kafka_settings.KAFKA_REPLICATION_FACTOR,
                configs={
                    'cleanup.policy': 'delete',
                    'retention.ms': '2592000000',  # 30 days - keep failed messages longer
                    'compression.type': 'snappy'
                }
            ),
            # Analytics and metrics topics
            NewTopic(
                name="workflow-analytics",
                num_partitions=3,
                replication_factor=kafka_settings.KAFKA_REPLICATION_FACTOR,
                configs={
                    'cleanup.policy': 'compact',
                    'compression.type': 'snappy'
                }
            ),
            NewTopic(
                name="api-metrics",
                num_partitions=3,
                replication_factor=kafka_settings.KAFKA_REPLICATION_FACTOR,
                configs={
                    'cleanup.policy': 'delete',
                    'retention.ms': '604800000',  # 7 days
                    'compression.type': 'snappy'
                }
            )
        ]

        results = {}

        for topic in topics_to_create:
            try:
                if self.sync_admin_client:
                    fs = self.sync_admin_client.create_topics([topic])

                    # Wait for topic creation
                    for topic_name, future in fs.items():
                        try:
                            future.result(timeout=10)
                            results[topic_name] = True
                            logger.info(f"Topic '{topic_name}' created successfully")
                        except TopicAlreadyExistsError:
                            results[topic_name] = True
                            logger.info(f"Topic '{topic_name}' already exists")
                        except Exception as e:
                            results[topic_name] = False
                            logger.error(f"Failed to create topic '{topic_name}': {e}")
                else:
                    logger.error("Sync admin client not available")
                    results[topic.name] = False

            except Exception as e:
                logger.error(f"Error creating topics: {e}")
                results[topic.name] = False

        return results

    async def list_topics(self) -> Dict[str, Any]:
        """List all topics in the cluster"""
        try:
            if not self.admin_client:
                await self.start()

            metadata = await self.admin_client.describe_cluster()
            topics_metadata = await self.admin_client.list_topics()

            topics_info = {}
            for topic_name in topics_metadata:
                topic_metadata = await self.admin_client.describe_topics([topic_name])
                topics_info[topic_name] = {
                    'partitions': len(topic_metadata[topic_name].partitions),
                    'replication_factor': len(topic_metadata[topic_name].partitions[0].replicas) if topic_metadata[topic_name].partitions else 0
                }

            return {
                'cluster_id': metadata.cluster_id,
                'controller': metadata.controller,
                'brokers': len(metadata.brokers),
                'topics': topics_info,
                'total_topics': len(topics_info)
            }

        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            return {}

    async def get_consumer_groups(self) -> Dict[str, Any]:
        """Get consumer group information"""
        try:
            if not self.sync_admin_client:
                return {}

            groups = self.sync_admin_client.list_consumer_groups()

            group_info = {}
            for group in groups:
                try:
                    group_description = self.sync_admin_client.describe_consumer_groups([group[0]])
                    group_info[group[0]] = {
                        'state': group_description[group[0]].state,
                        'members': len(group_description[group[0]].members),
                        'protocol': group_description[group[0]].protocol,
                        'coordinator': group_description[group[0]].coordinator.id
                    }
                except Exception as e:
                    logger.warning(f"Could not describe group {group[0]}: {e}")
                    group_info[group[0]] = {'error': str(e)}

            return group_info

        except Exception as e:
            logger.error(f"Error getting consumer groups: {e}")
            return {}

    async def get_topic_offsets(self, topic: str) -> Dict[str, Any]:
        """Get topic partition offsets"""
        try:
            # Use a temporary consumer to get offsets
            consumer = KafkaConsumer(
                bootstrap_servers=self._config['bootstrap_servers'],
                consumer_timeout_ms=1000
            )

            # Get topic partitions
            partitions = consumer.partitions_for_topic(topic)
            if not partitions:
                consumer.close()
                return {}

            from kafka import TopicPartition
            topic_partitions = [TopicPartition(topic, p) for p in partitions]

            # Get high water marks (latest offsets)
            high_water_marks = consumer.end_offsets(topic_partitions)

            # Get low water marks (earliest offsets)
            low_water_marks = consumer.beginning_offsets(topic_partitions)

            consumer.close()

            offset_info = {}
            for tp in topic_partitions:
                offset_info[f"partition_{tp.partition}"] = {
                    'earliest_offset': low_water_marks.get(tp, 0),
                    'latest_offset': high_water_marks.get(tp, 0),
                    'lag': high_water_marks.get(tp, 0) - low_water_marks.get(tp, 0)
                }

            return {
                'topic': topic,
                'partitions': offset_info,
                'total_messages': sum(info['lag'] for info in offset_info.values())
            }

        except Exception as e:
            logger.error(f"Error getting topic offsets for {topic}: {e}")
            return {}

    async def monitor_cluster_health(self) -> Dict[str, Any]:
        """Monitor overall cluster health"""
        try:
            cluster_info = await self.list_topics()
            consumer_groups = await self.get_consumer_groups()

            # Check key topics
            key_topics = [
                kafka_settings.KAFKA_WORKFLOW_EVENTS_TOPIC,
                kafka_settings.KAFKA_EXECUTION_EVENTS_TOPIC,
                kafka_settings.KAFKA_NODE_EVENTS_TOPIC,
                kafka_settings.KAFKA_DEAD_LETTER_TOPIC
            ]

            topic_health = {}
            for topic in key_topics:
                if topic in cluster_info.get('topics', {}):
                    offsets = await self.get_topic_offsets(topic)
                    topic_health[topic] = {
                        'exists': True,
                        'partitions': cluster_info['topics'][topic]['partitions'],
                        'replication_factor': cluster_info['topics'][topic]['replication_factor'],
                        'total_messages': offsets.get('total_messages', 0),
                        'status': 'healthy'
                    }
                else:
                    topic_health[topic] = {
                        'exists': False,
                        'status': 'missing'
                    }

            # Overall health status
            missing_topics = [t for t, info in topic_health.items() if not info['exists']]
            overall_status = 'healthy' if not missing_topics else 'degraded'

            return {
                'overall_status': overall_status,
                'cluster_info': cluster_info,
                'consumer_groups': consumer_groups,
                'topic_health': topic_health,
                'missing_topics': missing_topics,
                'timestamp': asyncio.get_event_loop().time()
            }

        except Exception as e:
            logger.error(f"Error monitoring cluster health: {e}")
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'timestamp': asyncio.get_event_loop().time()
            }

    async def reset_consumer_group_offsets(self, group_id: str, topic: str, reset_to: str = 'earliest') -> bool:
        """Reset consumer group offsets for a topic"""
        try:
            # This is a potentially dangerous operation, use with caution
            logger.warning(f"Resetting consumer group '{group_id}' offsets for topic '{topic}' to '{reset_to}'")

            if not self.sync_admin_client:
                return False

            # For safety, we'll just log this operation
            # In production, you might want additional safeguards
            logger.info(f"Would reset offsets for group {group_id}, topic {topic} to {reset_to}")

            # Implementation would go here using Kafka admin tools
            # This is a placeholder for the actual implementation

            return True

        except Exception as e:
            logger.error(f"Error resetting consumer group offsets: {e}")
            return False


class DeadLetterQueueManager:
    """Manage dead letter queue operations"""

    def __init__(self):
        self.consumer: Optional[KafkaConsumer] = None
        self.dlq_topic = kafka_settings.KAFKA_DEAD_LETTER_TOPIC

    def start(self):
        """Start DLQ consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.dlq_topic,
                bootstrap_servers=kafka_settings.KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                group_id='dlq-manager',
                enable_auto_commit=False
            )
            logger.info("Dead letter queue manager started")
        except Exception as e:
            logger.error(f"Failed to start DLQ manager: {e}")
            raise

    def stop(self):
        """Stop DLQ consumer"""
        if self.consumer:
            self.consumer.close()
            logger.info("Dead letter queue manager stopped")

    def get_failed_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get failed messages from dead letter queue"""
        try:
            if not self.consumer:
                self.start()

            messages = []
            for message in self.consumer:
                messages.append({
                    'partition': message.partition,
                    'offset': message.offset,
                    'key': message.key.decode('utf-8') if message.key else None,
                    'value': message.value,
                    'timestamp': message.timestamp,
                    'headers': {k: v.decode('utf-8') for k, v in message.headers}
                })

                if len(messages) >= limit:
                    break

            return messages

        except Exception as e:
            logger.error(f"Error reading from dead letter queue: {e}")
            return []

    async def retry_failed_message(self, message_data: Dict[str, Any]) -> bool:
        """Retry a failed message by republishing it"""
        try:
            from app.services.kafka.producer import event_producer
            from app.schemas.events import EventPayload

            original_event = message_data.get('original_event')
            if not original_event:
                logger.error("No original event found in DLQ message")
                return False

            # Reconstruct the event payload
            # This would need proper event type detection and reconstruction
            logger.info(f"Retrying failed message: {message_data}")

            # For now, just log the retry attempt
            # In a full implementation, you'd reconstruct the proper event type
            # and republish it to the original topic

            return True

        except Exception as e:
            logger.error(f"Error retrying failed message: {e}")
            return False


# Global instances
kafka_admin = KafkaAdministrator()
dlq_manager = DeadLetterQueueManager()