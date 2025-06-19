#!/usr/bin/env python3
"""
AI Services Debug Script
Debug HuggingFace and OpenAI API issues
"""

import asyncio
import aiohttp
import json
import os
import sys

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.config import settings
from app.services.ai.huggingface_service import hf_api
from app.services.ai.model_factory import model_factory


async def test_huggingface_direct():
    """Test HuggingFace API directly"""
    print("🔍 Testing HuggingFace API directly...")

    try:
        # Test 1: Simple API call
        print("   📡 Testing direct API call...")
        result = await hf_api.generate_text(
            model_name="gpt2",
            prompt="Hello world",
            parameters={"max_tokens": 20}
        )
        print(f"   ✅ Direct API call successful: {result}")
        return True

    except Exception as e:
        print(f"   ❌ Direct API call failed: {e}")

        # Test 2: Check API key format
        print("   🔑 Checking API key...")
        if settings.HUGGINGFACE_API_KEY:
            key = settings.HUGGINGFACE_API_KEY
            if key.startswith('hf_'):
                print(f"   ✅ API key format looks correct: {key[:8]}...{key[-4:]}")
            else:
                print(f"   ⚠️  API key format unusual: {key[:8]}...")
        else:
            print("   ❌ No API key found")

        # Test 3: Manual HTTP request
        print("   🌐 Testing manual HTTP request...")
        await test_hf_manual_request()

        return False


async def test_hf_manual_request():
    """Test HuggingFace with manual HTTP request"""

    headers = {
        "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": "Hello, how are you?",
        "parameters": {
            "max_new_tokens": 20,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    url = "https://api-inference.huggingface.co/models/gpt2"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=30)) as response:
                print(f"   📊 Status: {response.status}")

                if response.status == 200:
                    result = await response.json()
                    print(f"   ✅ Manual request successful: {result}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"   ❌ Manual request failed: {error_text}")

                    # Common error analysis
                    if response.status == 401:
                        print("   🔑 Authentication issue - check API key")
                    elif response.status == 429:
                        print("   ⏱️  Rate limited - wait and retry")
                    elif response.status == 503:
                        print("   🔄 Model loading - this is normal, wait a bit")

                    return False

    except Exception as e:
        print(f"   ❌ Manual request error: {e}")
        return False


async def test_openai_alternative():
    """Test OpenAI as alternative"""
    print("🤖 Testing OpenAI alternative...")

    if not settings.OPENAI_API_KEY:
        print("   ⚠️  No OpenAI API key configured")
        return False

    try:
        # Test OpenAI model
        model = model_factory.get_model(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=30
        )

        response = await model.ainvoke("Hello! Please introduce yourself briefly.")
        print(f"   ✅ OpenAI working: {response.content[:60]}...")
        return True

    except Exception as e:
        print(f"   ❌ OpenAI test failed: {e}")
        return False


async def test_model_factory():
    """Test the model factory"""
    print("🏭 Testing Model Factory...")

    try:
        # Test available providers
        providers = model_factory.get_available_providers()
        print(f"   📋 Available providers: {providers}")

        # Test HuggingFace model creation
        print("   🔧 Testing HuggingFace model creation...")
        hf_model = model_factory.get_model(
            provider="huggingface",
            model_name="microsoft/Phi-3-mini-4k-instruct",
            temperature=0.7
        )
        print(f"   ✅ HuggingFace model created: {type(hf_model)}")

        # Test model invocation
        print("   🚀 Testing model invocation...")
        response = await hf_model.ainvoke("Test message")
        print(f"   📝 Response type: {type(response)}")
        print(f"   📄 Response content: {response}...")

        return True

    except Exception as e:
        print(f"   ❌ Model factory test failed: {e}")
        print(f"   🔍 Error details: {str(e)}")
        return False


async def test_workflow_execution_components():
    """Test workflow execution components"""
    print("🔄 Testing Workflow Execution Components...")

    try:
        from app.services.langgraph.workflow import BasicWorkflowEngine
        from app.core.database import async_session_maker

        # Test workflow engine creation
        async with async_session_maker() as db:
            engine = BasicWorkflowEngine(db)
            print("   ✅ Workflow engine created")

            # Test state manager
            state = engine.state_manager.create_initial_state(
                execution_id="test-123",
                workflow_id="workflow-123",
                input_data={"query": "Hello"},
                ai_config={"provider": "openai", "model": "gpt-3.5-turbo"}
            )
            print("   ✅ State manager working")

            return True

    except Exception as e:
        print(f"   ❌ Workflow execution test failed: {e}")
        return False


async def suggest_fixes():
    """Suggest fixes based on test results"""
    print("🔧 SUGGESTED FIXES:")
    print()

    # Test HuggingFace
    hf_works = await test_huggingface_direct()

    if not hf_works:
        print("🔑 HuggingFace Issues:")
        print("   1. Try regenerating your HuggingFace API key")
        print("   2. Ensure 'read' permissions are enabled")
        print("   3. Check if the key is active and not rate-limited")
        print("   4. Try waiting 1-2 minutes (models may be loading)")
        print()

    # Test OpenAI alternative
    # openai_works = await test_openai_alternative()
    #
    # if openai_works:
    #     print("✅ OpenAI Alternative Available:")
    #     print("   • You can use OpenAI models while debugging HuggingFace")
    #     print("   • Update your default provider to 'openai'")
    #     print()

    # Test components
    components_work = await test_model_factory()
    workflow_works = await test_workflow_execution_components()

    if components_work and workflow_works:
        print("✅ Core Components Working:")
        print("   • Model factory is functional")
        print("   • Workflow engine is operational")
        print("   • Issue is likely in the AI service integration")
        print()

    print("📋 RECOMMENDED ACTIONS:")
    if openai_works and not hf_works:
        print("   1. Switch to OpenAI temporarily")
        print("   2. Debug HuggingFace separately")
        print("   3. Update default AI config to use OpenAI")
    elif not hf_works:
        print("   1. Regenerate HuggingFace API key")
        print("   2. Wait for rate limits to reset")
        print("   3. Test with a different model")

    return hf_works, openai_works, components_work, workflow_works


async def main():
    """Run comprehensive AI services debug"""
    print("🔍 AI SERVICES DEBUG")
    print("=" * 50)
    print()

    print("📊 Configuration:")
    print(f"   HuggingFace Key: {'✅ Set' if settings.HUGGINGFACE_API_KEY else '❌ Missing'}")
    print(f"   OpenAI Key: {'✅ Set' if settings.OPENAI_API_KEY else '❌ Missing'}")
    print(f"   Database: {settings.DATABASE_URL[:30]}...")
    print()

    # Run diagnostics
    hf_works, openai_works, components_work, workflow_works = await suggest_fixes()

    print()
    print("=" * 50)
    print("🎯 DEBUG SUMMARY")
    print(f"   HuggingFace API: {'✅' if hf_works else '❌'}")
    print(f"   OpenAI API: {'✅' if openai_works else '❌'}")
    print(f"   Core Components: {'✅' if components_work else '❌'}")
    print(f"   Workflow Engine: {'✅' if workflow_works else '❌'}")

    if openai_works or hf_works:
        print()
        print("🎉 At least one AI provider is working!")
        print("💡 Your platform can be functional with working providers")
    else:
        print()
        print("⚠️  AI providers need attention")
        print("🔧 Focus on API key configuration and network connectivity")


if __name__ == "__main__":
    asyncio.run(main())