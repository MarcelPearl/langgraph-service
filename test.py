import asyncio
import sys

sys.path.append('.')


async def test_all_models():
    """Test all models with fallbacks"""
    print("🧪 Testing HuggingFace Models with Fallbacks")
    print("=" * 60)

    from app.services.ai.huggingface_service import hf_api
    from app.services.ai.local_hugging_face_service import local_hf_service

    # Test order: preferred -> fallback
    models_to_test = [
        {
            "primary": "microsoft/Phi-3-mini-4k-instruct",
            "fallback": "microsoft/DialoGPT-medium",
            "prompt": "Hello! Please introduce yourself briefly."
        },
        {
            "primary": "deepseek/deepseek-v3-0324",
            "fallback": "gpt2",
            "prompt": "Hello! Please introduce yourself briefly."
        },
        {
            "primary": "deepset/roberta-base-squad2",
            "fallback": "distilbert-base-cased-distilled-squad",
            "prompt": "Context: Paris is the capital of France. Question: What is the capital of France?"
        }
    ]

    for test_case in models_to_test:
        primary = test_case["primary"]
        fallback = test_case["fallback"]
        prompt = test_case["prompt"]

        print(f"\n🔍 Testing {primary}...")

        # Test API version
        try:
            result = await hf_api.generate_text(primary, prompt, {"max_tokens": 50})

            if result.get("fallback"):
                print(f"   ⚠️  API fallback for {primary}")
            else:
                print(f"   ✅ API works: {result['text'][:50]}...")
        except Exception as e:
            print(f"   ❌ API error: {e}")

        # Test local version with fallback
        print(f"   🏠 Testing local version...")
        try:
            local_result = await local_hf_service.generate_text_local(primary, prompt)

            if local_result.get("error"):
                print(f"   ⚠️  Primary model failed, trying fallback: {fallback}")

                # Try fallback model
                fallback_result = await local_hf_service.generate_text_local(fallback, prompt)

                if fallback_result.get("error"):
                    print(f"   ❌ Fallback also failed: {fallback_result['text']}")
                else:
                    print(f"   ✅ Fallback works: {fallback_result['text'][:50]}...")
            else:
                print(f"   ✅ Primary works: {local_result['text'][:50]}...")

        except Exception as e:
            print(f"   ❌ Local error: {e}")






if __name__ == "__main__":
    asyncio.run(test_all_models())