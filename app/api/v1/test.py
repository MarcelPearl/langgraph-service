from fastapi import APIRouter, HTTPException

router=APIRouter(prefix="/test",tags=["test"])

if settings.DEBUG:
    @router.post("/kafka")
    async def test_kafka():
        """Test Kafka connectivity"""
        try:
            if not kafka_service.running:
                return {
                    "message": "Kafka service not running",
                    "status": "unavailable"
                }

            from app.services.messaging.kafka_service import NodeExecutionMessage
            import uuid

            test_message = NodeExecutionMessage(
                execution_id=str(uuid.uuid4()),
                workflow_id=str(uuid.uuid4()),
                node_id="test_node",
                node_type="ai_decision",
                node_data={"prompt": "Test prompt", "options": ["yes", "no"]},
                context={"test": True},
                dependencies=[],
                timestamp=str(asyncio.get_event_loop().time())
            )

            await kafka_service.publish_node_execution(test_message)
            return {
                "message": "Kafka test message sent successfully",
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Kafka test failed: {e}")
            return {
                "message": f"Kafka test failed: {str(e)}",
                "status": "error"
            }


    @router.post("/redis")
    async def test_redis():
        """Test Redis connectivity"""
        try:
            test_data = {"test": True, "timestamp": str(asyncio.get_event_loop().time())}
            await redis_context_service.set_execution_context("test_execution", test_data, ttl=60)

            retrieved = await redis_context_service.get_execution_context("test_execution")

            return {
                "message": "Redis test successful",
                "stored": test_data,
                "retrieved": retrieved,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Redis test failed: {e}")
            return {
                "message": f"Redis test failed: {str(e)}",
                "status": "error"
            }
