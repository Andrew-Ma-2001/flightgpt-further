#!/usr/bin/env python
"""
vLLM Connection and Functionality Test Script
This script tests if vLLM server is working properly with both text and vision inputs.
"""

import os
import sys
import time
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image
import numpy as np

# Configuration
VLLM_BASE_URL = "http://0.0.0.0:8989/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "qwen_2_5_vl_7b"
TEST_TIMEOUT = 60  # seconds

def create_test_image(size=(256, 256), color='red'):
    """Create a simple test image"""
    img = Image.new('RGB', size, color=color)
    # Draw a simple pattern
    pixels = img.load()
    for i in range(size[0]):
        for j in range(size[1]):
            if (i + j) % 40 < 20:
                pixels[i, j] = (255, 255, 255)
    return img

def image_to_data_url(image):
    """Convert PIL Image to data URL"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def test_server_health():
    """Test if server is reachable"""
    print("=" * 60)
    print("TEST 1: Server Health Check")
    print("=" * 60)
    
    try:
        import requests
        response = requests.get(f"{VLLM_BASE_URL.replace('/v1', '')}/health", timeout=5)
        print(f"✓ Server is reachable")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text}")
        return True
    except Exception as e:
        print(f"✗ Server is NOT reachable: {e}")
        return False

def test_text_only():
    """Test basic text-only inference"""
    print("\n" + "=" * 60)
    print("TEST 2: Text-Only Inference (Simple)")
    print("=" * 60)
    
    try:
        client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=TEST_TIMEOUT
        )
        
        print(f"Sending request with {TEST_TIMEOUT}s timeout...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, I am working!' and nothing else."}
            ],
            max_tokens=50,
            temperature=0.0
        )
        
        elapsed = time.time() - start_time
        result = response.choices[0].message.content
        
        print(f"✓ Text inference SUCCESS!")
        print(f"  Time taken: {elapsed:.2f}s")
        print(f"  Response: {result}")
        print(f"  Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        return True
        
    except Exception as e:
        print(f"✗ Text inference FAILED: {e}")
        return False

def test_vision_simple():
    """Test vision inference with simple synthetic image"""
    print("\n" + "=" * 60)
    print("TEST 3: Vision Inference (Synthetic Image)")
    print("=" * 60)
    
    try:
        # Create a simple test image
        print("Creating test image (256x256 red with white pattern)...")
        test_img = create_test_image(size=(256, 256), color='red')
        img_url = image_to_data_url(test_img)
        
        client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=TEST_TIMEOUT
        )
        
        print(f"Sending vision request with {TEST_TIMEOUT}s timeout...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Answer in one word."},
                        {"type": "image_url", "image_url": {"url": img_url}}
                    ]
                }
            ],
            max_tokens=50,
            temperature=0.0
        )
        
        elapsed = time.time() - start_time
        result = response.choices[0].message.content
        
        print(f"✓ Vision inference SUCCESS!")
        print(f"  Time taken: {elapsed:.2f}s")
        print(f"  Response: {result}")
        print(f"  Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        return True
        
    except Exception as e:
        print(f"✗ Vision inference FAILED: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def test_vision_real_image():
    """Test with a real image from your dataset if available"""
    print("\n" + "=" * 60)
    print("TEST 4: Vision Inference (Real Dataset Image)")
    print("=" * 60)
    
    # Try to find a real image from your dataset
    possible_paths = [
        "./data/rgbd-new",
        "./data_examples/images",
        "./data/training_data/images"
    ]
    
    real_image_path = None
    for base_path in possible_paths:
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        real_image_path = os.path.join(root, file)
                        break
                if real_image_path:
                    break
        if real_image_path:
            break
    
    if not real_image_path:
        print("⊘ Skipping: No real images found in dataset")
        return None
    
    try:
        print(f"Found image: {real_image_path}")
        
        # Load and check image
        img = Image.open(real_image_path)
        print(f"  Image size: {img.size}, Mode: {img.mode}")
        
        # Resize if too large
        max_size = 1024
        if max(img.size) > max_size:
            print(f"  Resizing image from {img.size} to max {max_size}px...")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to data URL
        img_url = image_to_data_url(img)
        print(f"  Image data URL size: {len(img_url) / 1024:.1f} KB")
        
        client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=TEST_TIMEOUT
        )
        
        print(f"Sending real image request with {TEST_TIMEOUT}s timeout...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one sentence."},
                        {"type": "image_url", "image_url": {"url": img_url}}
                    ]
                }
            ],
            max_tokens=100,
            temperature=0.0
        )
        
        elapsed = time.time() - start_time
        result = response.choices[0].message.content
        
        print(f"✓ Real image inference SUCCESS!")
        print(f"  Time taken: {elapsed:.2f}s")
        print(f"  Response: {result}")
        print(f"  Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        return True
        
    except Exception as e:
        print(f"✗ Real image inference FAILED: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def test_concurrent_requests():
    """Test if server can handle multiple requests"""
    print("\n" + "=" * 60)
    print("TEST 5: Concurrent Requests (2 parallel)")
    print("=" * 60)
    
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=TEST_TIMEOUT
        )
        
        def make_request(i):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": f"Count to {i+1} and stop."}
                ],
                max_tokens=20,
                temperature=0.0
            )
            return i, response.choices[0].message.content
        
        print("Sending 2 concurrent requests...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(make_request, i) for i in range(2)]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        elapsed = time.time() - start_time
        
        print(f"✓ Concurrent requests SUCCESS!")
        print(f"  Time taken: {elapsed:.2f}s")
        for i, result in sorted(results):
            print(f"  Request {i}: {result}")
        return True
        
    except Exception as e:
        print(f"✗ Concurrent requests FAILED: {e}")
        return False

def main():
    print("\n" + "🚀" * 30)
    print("vLLM Server Diagnostic Test Suite")
    print("🚀" * 30)
    print(f"\nConfiguration:")
    print(f"  Server URL: {VLLM_BASE_URL}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Timeout: {TEST_TIMEOUT}s")
    print()
    
    results = {}
    
    # Run tests
    results['health'] = test_server_health()
    
    if not results['health']:
        print("\n" + "!" * 60)
        print("CRITICAL: Server is not reachable. Please check:")
        print("  1. Is vLLM server running?")
        print("  2. Is it listening on port 8989?")
        print("  3. Run: curl http://0.0.0.0:8989/health")
        print("!" * 60)
        sys.exit(1)
    
    results['text'] = test_text_only()
    results['vision_simple'] = test_vision_simple()
    results['vision_real'] = test_vision_real_image()
    results['concurrent'] = test_concurrent_requests()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {test_name.upper():20s}: {status}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if results['text'] and not results['vision_simple']:
        print("⚠ Text works but vision fails!")
        print("  → Problem is likely in multi-modal processing")
        print("  → Try restarting vLLM with: --enforce-eager --disable-v2-block-manager")
    elif not results['text']:
        print("⚠ Even text inference is failing!")
        print("  → vLLM inference engine is completely stuck")
        print("  → Restart vLLM server and try again")
    elif results['vision_simple'] and not results['vision_real']:
        print("⚠ Simple images work but real images fail!")
        print("  → Your dataset images might be too large or corrupted")
        print("  → Check image sizes in your dataset")
    elif all(v for v in results.values() if v is not None):
        print("✓ All tests passed! vLLM is working correctly.")
        print("  → The issue with eval.py might be:")
        print("     - Request timeout is too short")
        print("     - Too many concurrent requests")
        print("     - Specific images causing issues")
    
    print()

if __name__ == "__main__":
    main()

