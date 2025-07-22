#!/usr/bin/env python3
"""
API Contract Compliance Test Script
Tests all endpoints against the provided API contract specification
"""

import requests
import json
import io
from PIL import Image
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"
API_TOKEN = "COcEJgOmhw0bwjhZdHwxDxWee7ZGGBRj"
HEADERS = {"Authorization": f"{API_TOKEN}"}


def create_test_image():
    """Create a test image for API calls"""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return img_bytes.getvalue()


def test_store_api():
    """Test the /store endpoint according to contract"""
    print("\n=== Testing Store API ===")

    # Prepare test data according to contract
    store_metadata = {
        "transaction_id": "test_cust_001",
        "msisdn": "+1234567890",
        "created_on": "2024-01-15T10:30:00Z",
        "id_type": "national_id",
        "id_number": "ID123456789"
    }

    image_data = create_test_image()

    # Prepare multipart form data
    files = {
        'image': ('test.jpg', image_data, 'image/jpeg'),
        'metadata': (None, json.dumps(store_metadata))
    }

    response = requests.post(
        f"{BASE_URL}/v1/dedup/face/search",
        files=files,
        headers=HEADERS
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    # Validate response structure according to contract
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "total_matches" in data
        assert "metadata" in data
        assert "results" in data

        # Validate metadata echo back
        returned_metadata = data["metadata"]
        assert returned_metadata["transaction_id"] == search_metadata["transaction_id"]
        assert returned_metadata["id_type"] == search_metadata["id_type"]

        # Validate results structure
        if data["total_matches"] > 0:
            result = data["results"][0]
            assert "similarity_score" in result
            assert "metadata" in result
            assert "msisdn" in result["metadata"]

        print("✅ Search API test passed")
    else:
        print("❌ Search API test failed")

    return response.status_code == 200


def test_purge_api():
    """Test the /purge endpoint according to contract"""
    print("\n=== Testing Purge API ===")

    purge_data = {
        "transaction_id": "test_cust_001"
    }

    response = requests.post(
        f"{BASE_URL}/v1/dedup/face/purge",
        json=purge_data,
        headers=HEADERS
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    # Validate response structure
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert data["transaction_id"] == "test_cust_001"
        assert "message" in data
        print("✅ Purge API test passed")
    else:
        print("❌ Purge API test failed")

    return response.status_code == 200


def test_error_handling():
    """Test error responses match contract format"""
    print("\n=== Testing Error Handling ===")

    # Test invalid authentication
    response = requests.post(
        f"{BASE_URL}/v1/dedup/face/store",
        files={'image': ('test.jpg', b'fake', 'image/jpeg')},
        headers={"Authorization": "Bearer invalid_token"}
    )

    print(f"Auth Error Status: {response.status_code}")
    if response.status_code == 401:
        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        print("✅ Error format test passed")


def run_all_tests():
    """Run complete test suite"""
    print("🚀 Starting API Contract Compliance Tests")
    print(f"Base URL: {BASE_URL}")
    print(f"Using Token: {API_TOKEN}")

    try:
        # Test health endpoint first
        health_response = requests.get(f"{BASE_URL}/health")
        print(f"\nHealth Check: {health_response.status_code}")

        if health_response.status_code != 200:
            print("❌ Server not healthy, aborting tests")
            return

        # Run tests in order
        store_success = test_store_api()
        search_success = test_search_api()
        purge_success = test_purge_api()
        test_error_handling()

        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        print(f"Store API: {'✅ PASS' if store_success else '❌ FAIL'}")
        print(f"Search API: {'✅ PASS' if search_success else '❌ FAIL'}")
        print(f"Purge API: {'✅ PASS' if purge_success else '❌ FAIL'}")

        if all([store_success, search_success, purge_success]):
            print("\n🎉 All tests passed! API is contract compliant.")
        else:
            print("\n⚠️  Some tests failed. Check implementation.")

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")


if __name__ == "__main__":
    run_all_tests()