from pydantic_settings import BaseSettings
from typing import List, Optional


class KafkaSettings(BaseSettings):
    """Kafka configuration settings for single broker setup"""

    # Kafka broker configuration (single broker)
    KAFKA_BOOTSTRAP_SERVER: str = "localhost:9092"
    KAFKA_SECURITY_PROTOCOL: str = "PLAINTEXT"
    KAFKA_SASL_MECHANISM: Optional[str] = None
    KAFKA_SASL_USERNAME: Optional[str] = None
    KAFKA_SASL_PASSWORD: Optional[str] = None

    # Producer configuration
    KAFKA_PRODUCER_ACKS: str = "1"  # Changed from "all" for single broker
    KAFKA_PRODUCER_RETRIES: int = 3
    KAFKA_PRODUCER_BATCH_SIZE: int = 16384  # Smaller batch for single broker
    KAFKA_PRODUCER_LINGER_MS: int = 10  # Reduced linger time
    KAFKA_PRODUCER_COMPRESSION_TYPE: str = "snappy"
    KAFKA_PRODUCER_ENABLE_IDEMPOTENCE: bool = True
    KAFKA_PRODUCER_MAX_IN_FLIGHT: int = 5  # Increased for better throughput

    # Consumer configuration
    KAFKA_CONSUMER_GROUP_ID: str = "workflow-processor"
    KAFKA_CONSUMER_AUTO_OFFSET_RESET: str = "earliest"
    KAFKA_CONSUMER_ENABLE_AUTO_COMMIT: bool = False  # Manual commit for reliability
    KAFKA_CONSUMER_FETCH_MIN_BYTES: int = 524288  # 512KB - smaller for single broker
    KAFKA_CONSUMER_FETCH_MAX_WAIT_MS: int = 500
    KAFKA_CONSUMER_SESSION_TIMEOUT_MS: int = 30000
    KAFKA_CONSUMER_HEARTBEAT_INTERVAL_MS: int = 10000

    # Topic configuration
    KAFKA_WORKFLOW_EVENTS_TOPIC: str = "workflow-events"
    KAFKA_NODE_EVENTS_TOPIC: str = "node-events"
    KAFKA_EXECUTION_EVENTS_TOPIC: str = "execution-events"
    KAFKA_DEAD_LETTER_TOPIC: str = "dead-letter-queue"

    # Replication and partitions (adjusted for single broker)
    KAFKA_REPLICATION_FACTOR: int = 1  # Single broker = replication factor 1
    KAFKA_DEFAULT_PARTITIONS: int = 3  # Reduced partitions for single broker

    # Avro Schema Registry
    KAFKA_SCHEMA_REGISTRY_URL: Optional[str] = None
    KAFKA_USE_AVRO: bool = False

    # Monitoring and timeouts
    KAFKA_REQUEST_TIMEOUT_MS: int = 30000
    KAFKA_RETRY_BACKOFF_MS: int = 1000
    KAFKA_RECONNECT_BACKOFF_MS: int = 50
    KAFKA_MAX_POLL_RECORDS: int = 500

    # Development settings
    KAFKA_AUTO_CREATE_TOPICS: bool = True
    KAFKA_LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = ""


# Global Kafka settings instance
kafka_settings = KafkaSettings()


def get_producer_config() -> dict:
    """Get Kafka producer configuration for single broker"""
    config = {
        'bootstrap_servers': kafka_settings.KAFKA_BOOTSTRAP_SERVERS,
        'acks': kafka_settings.KAFKA_PRODUCER_ACKS,
        'retries': kafka_settings.KAFKA_PRODUCER_RETRIES,
        'batch_size': kafka_settings.KAFKA_PRODUCER_BATCH_SIZE,
        'linger_ms': kafka_settings.KAFKA_PRODUCER_LINGER_MS,
        'compression_type': kafka_settings.KAFKA_PRODUCER_COMPRESSION_TYPE,
        'enable_idempotence': kafka_settings.KAFKA_PRODUCER_ENABLE_IDEMPOTENCE,
        'max_in_flight_requests_per_connection': kafka_settings.KAFKA_PRODUCER_MAX_IN_FLIGHT,
        'request_timeout_ms': kafka_settings.KAFKA_REQUEST_TIMEOUT_MS,
        'retry_backoff_ms': kafka_settings.KAFKA_RETRY_BACKOFF_MS,
    }

    # Add security config if needed
    if kafka_settings.KAFKA_SECURITY_PROTOCOL != "PLAINTEXT":
        config.update({
            'security_protocol': kafka_settings.KAFKA_SECURITY_PROTOCOL,
            'sasl_mechanism': kafka_settings.KAFKA_SASL_MECHANISM,
            'sasl_plain_username': kafka_settings.KAFKA_SASL_USERNAME,
            'sasl_plain_password': kafka_settings.KAFKA_SASL_PASSWORD,
        })

    return config


def get_consumer_config() -> dict:
    """Get Kafka consumer configuration for single broker"""
    config = {
        'bootstrap_servers': kafka_settings.KAFKA_BOOTSTRAP_SERVERS,
        'group_id': kafka_settings.KAFKA_CONSUMER_GROUP_ID,
        'auto_offset_reset': kafka_settings.KAFKA_CONSUMER_AUTO_OFFSET_RESET,
        'enable_auto_commit': kafka_settings.KAFKA_CONSUMER_ENABLE_AUTO_COMMIT,
        'fetch_min_bytes': kafka_settings.KAFKA_CONSUMER_FETCH_MIN_BYTES,
        'fetch_max_wait_ms': kafka_settings.KAFKA_CONSUMER_FETCH_MAX_WAIT_MS,
        'session_timeout_ms': kafka_settings.KAFKA_CONSUMER_SESSION_TIMEOUT_MS,
        'heartbeat_interval_ms': kafka_settings.KAFKA_CONSUMER_HEARTBEAT_INTERVAL_MS,
        'max_poll_records': kafka_settings.KAFKA_MAX_POLL_RECORDS,
        'request_timeout_ms': kafka_settings.KAFKA_REQUEST_TIMEOUT_MS,
        'retry_backoff_ms': kafka_settings.KAFKA_RETRY_BACKOFF_MS,
        'reconnect_backoff_ms': kafka_settings.KAFKA_RECONNECT_BACKOFF_MS,
    }

    # Add security config if needed
    if kafka_settings.KAFKA_SECURITY_PROTOCOL != "PLAINTEXT":
        config.update({
            'security_protocol': kafka_settings.KAFKA_SECURITY_PROTOCOL,
            'sasl_mechanism': kafka_settings.KAFKA_SASL_MECHANISM,
            'sasl_plain_username': kafka_settings.KAFKA_SASL_USERNAME,
            'sasl_plain_password': kafka_settings.KAFKA_SASL_PASSWORD,
        })

    return config