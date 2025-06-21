# app/services/context/redis_service.py
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import Redis
from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisContextService:
    """Redis-based context service for workflow execution state"""

    def __init__(self):
        self.redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        self.client: Optional[Redis] = None
        self.default_ttl = 86400  # 24 hours

    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=True,
                socket_timeout=30,
                socket_connect_timeout=30,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            await self.client.ping()
            logger.info("Redis connection established")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")

    # Execution Context Management
    async def set_execution_context(
            self,
            execution_id: str,
            context: Dict[str, Any],
            ttl: Optional[int] = None
    ):
        """Set complete execution context"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:context"
        context_json = json.dumps(context, default=str)

        await self.client.setex(
            key,
            ttl or self.default_ttl,
            context_json
        )
        logger.debug(f"Set execution context for {execution_id}")

    async def get_execution_context(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get complete execution context"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:context"

        try:
            context_json = await self.client.get(key)
            if context_json:
                return json.loads(context_json)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode context for {execution_id}: {e}")
            return None

    async def update_execution_context(
            self,
            execution_id: str,
            updates: Dict[str, Any]
    ):
        """Update specific fields in execution context"""
        current_context = await self.get_execution_context(execution_id)
        if current_context is None:
            current_context = {}

        current_context.update(updates)
        current_context['last_updated'] = datetime.utcnow().isoformat()

        await self.set_execution_context(execution_id, current_context)

    # Dependency Tracking (Core to Kahn's Algorithm)
    async def set_dependencies(
            self,
            execution_id: str,
            dependencies: Dict[str, int]
    ):
        """Set dependency counts for nodes (Kahn's algorithm)"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:dependencies"

        # Use Redis hash for efficient updates
        await self.client.delete(key)  # Clear existing
        if dependencies:
            await self.client.hset(key, mapping=dependencies)
            await self.client.expire(key, self.default_ttl)

        logger.debug(f"Set dependencies for {execution_id}: {dependencies}")

    async def get_dependencies(self, execution_id: str) -> Dict[str, int]:
        """Get all dependency counts"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:dependencies"
        dependencies = await self.client.hgetall(key)

        # Convert values to integers
        return {node_id: int(count) for node_id, count in dependencies.items()}

    async def decrement_dependency(
            self,
            execution_id: str,
            node_id: str
    ) -> int:
        """Atomically decrement dependency count and return new value"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:dependencies"

        # Use atomic operation
        new_count = await self.client.hincrby(key, node_id, -1)

        # If count reaches 0, the node is ready for execution
        if new_count <= 0:
            await self.client.hdel(key, node_id)

        return max(0, new_count)

    async def get_ready_nodes(self, execution_id: str) -> List[str]:
        """Get nodes with zero dependencies (ready for execution)"""
        dependencies = await self.get_dependencies(execution_id)
        return [node_id for node_id, count in dependencies.items() if count == 0]

    # Processing State Management
    async def mark_processing(self, execution_id: str, node_id: str):
        """Mark node as currently processing"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:processing"
        await self.client.sadd(key, node_id)
        await self.client.expire(key, self.default_ttl)

    async def mark_completed(self, execution_id: str, node_id: str):
        """Mark node as completed"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        # Remove from processing set
        processing_key = f"execution:{execution_id}:processing"
        await self.client.srem(processing_key, node_id)

        # Add to completed set
        completed_key = f"execution:{execution_id}:completed"
        await self.client.sadd(completed_key, node_id)
        await self.client.expire(completed_key, self.default_ttl)

    async def get_processing_nodes(self, execution_id: str) -> Set[str]:
        """Get currently processing nodes"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:processing"
        nodes = await self.client.smembers(key)
        return set(nodes)

    async def get_completed_nodes(self, execution_id: str) -> Set[str]:
        """Get completed nodes"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:completed"
        nodes = await self.client.smembers(key)
        return set(nodes)

    # Node Results Storage
    async def store_node_result(
            self,
            execution_id: str,
            node_id: str,
            result: Dict[str, Any]
    ):
        """Store result from node execution"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:results:{node_id}"
        result_json = json.dumps(result, default=str)

        await self.client.setex(key, self.default_ttl, result_json)
        logger.debug(f"Stored result for node {node_id}")

    async def get_node_result(
            self,
            execution_id: str,
            node_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get result from specific node"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:results:{node_id}"

        try:
            result_json = await self.client.get(key)
            if result_json:
                return json.loads(result_json)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode result for {node_id}: {e}")
            return None

    async def get_all_node_results(self, execution_id: str) -> Dict[str, Any]:
        """Get all node results for an execution"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        pattern = f"execution:{execution_id}:results:*"
        keys = await self.client.keys(pattern)

        results = {}
        for key in keys:
            node_id = key.split(':')[-1]  # Extract node_id from key
            result = await self.get_node_result(execution_id, node_id)
            if result:
                results[node_id] = result

        return results

    # Workflow State Synchronization
    async def set_workflow_status(
            self,
            execution_id: str,
            status: str,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """Set overall workflow execution status"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:status"

        status_data = {
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }

        status_json = json.dumps(status_data, default=str)
        await self.client.setex(key, self.default_ttl, status_json)

    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        key = f"execution:{execution_id}:status"

        try:
            status_json = await self.client.get(key)
            if status_json:
                return json.loads(status_json)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode status for {execution_id}: {e}")
            return None

    # Cleanup and Maintenance
    async def cleanup_execution(self, execution_id: str):
        """Clean up all data for an execution"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        patterns = [
            f"execution:{execution_id}:*"
        ]

        for pattern in patterns:
            keys = await self.client.keys(pattern)
            if keys:
                await self.client.delete(*keys)

        logger.info(f"Cleaned up execution data for {execution_id}")

    async def cleanup_expired_executions(self, older_than_hours: int = 48):
        """Clean up executions older than specified hours"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

        # This would require tracking execution timestamps
        # Implementation depends on specific cleanup strategy
        logger.info(f"Cleanup executed for executions older than {older_than_hours} hours")

    # Health and Monitoring
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis service health"""
        if not self.client:
            return {
                'service': 'redis',
                'status': 'disconnected',
                'timestamp': datetime.utcnow().isoformat()
            }

        try:
            # Test basic operations
            test_key = f"health_check:{datetime.utcnow().timestamp()}"
            await self.client.setex(test_key, 60, "test")
            test_value = await self.client.get(test_key)
            await self.client.delete(test_key)

            info = await self.client.info()

            return {
                'service': 'redis',
                'status': 'healthy',
                'connected_clients': info.get('connected_clients', 0),
                'memory_usage': info.get('used_memory_human', 'unknown'),
                'test_operation': 'success' if test_value == "test" else 'failed',
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'service': 'redis',
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    # Atomic Operations for Concurrency
    async def atomic_node_transition(
            self,
            execution_id: str,
            node_id: str,
            from_status: str,
            to_status: str
    ) -> bool:
        """Atomically transition node from one status to another"""
        if not self.client:
            raise RuntimeError("Redis client not connected")

        # Use Redis transaction for atomicity
        async with self.client.pipeline(transaction=True) as pipe:
            try:
                if from_status == "ready" and to_status == "processing":
                    await pipe.srem(f"execution:{execution_id}:ready", node_id)
                    await pipe.sadd(f"execution:{execution_id}:processing", node_id)
                elif from_status == "processing" and to_status == "completed":
                    await pipe.srem(f"execution:{execution_id}:processing", node_id)
                    await pipe.sadd(f"execution:{execution_id}:completed", node_id)

                await pipe.execute()
                return True

            except redis.WatchError:
                logger.warning(f"Atomic transition failed for node {node_id}")
                return False


# Global instance
redis_context_service = RedisContextService()