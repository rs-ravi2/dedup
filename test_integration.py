import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app
import tempfile
import os
from PIL import Image
import io
import json


class TestIntegration:
    """Comprehensive integration test for the deduplication API"""

    def setup_method(self):
        """Setup test environment"""
        self.client = TestClient(app)
        self.test_token = "your-api-key-here"
        self.headers = {"Authorization": f"Bearer {self.test_token}"}

        # Create test image
        self.test_image = self.create_test_image()

    def create_test_image(self) -> bytes:
        """Create a test image"""
        image = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        return img_bytes.getvalue()

    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/v1/dedup/face/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_store_customer_success(self):
        """Test successful customer storage"""
        metadata = {
            "transaction_id": "test_customer_001",
            "msisdn": "1234567890",
            "created_on": "2024-01-01T00:00:00Z",
            "id_type": "national_id",
            "id_number": "ID123456789",
        }

        files = {"image": ("test.jpg", self.test_image, "image/jpeg")}
        data = {"metadata": json.dumps(metadata)}

        response = self.client.post(
            "/v1/dedup/face/store", files=files, data=data, headers=self.headers
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert result["transaction_id"] == "test_customer_001"

    def test_store_duplicate_customer(self):
        """Test storing duplicate customer"""
        metadata = {
            "transaction_id": "test_customer_002",
            "msisdn": "1234567891",
            "created_on": "2024-01-01T00:00:00Z",
            "id_type": "national_id",
            "id_number": "ID123456790",
        }

        files = {"image": ("test.jpg", self.test_image, "image/jpeg")}
        data = {"metadata": json.dumps(metadata)}

        # Store first time
        response1 = self.client.post(
            "/v1/dedup/face/store", files=files, data=data, headers=self.headers
        )
        assert response1.status_code == 200

        # Store second time (should fail)
        response2 = self.client.post(
            "/v1/dedup/face/store", files=files, data=data, headers=self.headers
        )
        assert response2.status_code == 409  # Conflict

    def test_search_customers(self):
        """Test customer search functionality"""
        # First, store a customer
        metadata = {
            "transaction_id": "test_customer_003",
            "msisdn": "1234567892",
            "created_on": "2024-01-01T00:00:00Z",
            "id_type": "national_id",
            "id_number": "ID123456791",
        }

        files = {"image": ("test.jpg", self.test_image, "image/jpeg")}
        data = {"metadata": json.dumps(metadata)}

        store_response = self.client.post(
            "/v1/dedup/face/store", files=files, data=data, headers=self.headers
        )
        assert store_response.status_code == 200

        # Now search with similar image
        search_files = {"image": ("search.jpg", self.test_image, "image/jpeg")}
        search_data = {"threshold": 0.5, "limit": 10}

        search_response = self.client.post(
            "/v1/dedup/face/search",
            files=search_files,
            data=search_data,
            headers=self.headers,
        )

        assert search_response.status_code == 200
        result = search_response.json()
        assert result["status"] == "success"
        assert "total_matches" in result
        assert "results" in result

    def test_purge_customer(self):
        """Test customer purge functionality"""
        # Store a customer first
        metadata = {
            "transaction_id": "test_customer_004",
            "msisdn": "1234567893",
            "created_on": "2024-01-01T00:00:00Z",
            "id_type": "national_id",
            "id_number": "ID123456792",
        }

        files = {"image": ("test.jpg", self.test_image, "image/jpeg")}
        data = {"metadata": json.dumps(metadata)}

        store_response = self.client.post(
            "/v1/dedup/face/store", files=files, data=data, headers=self.headers
        )
        assert store_response.status_code == 200

        # Now purge the customer
        purge_data = {"transaction_id": "test_customer_004"}
        purge_response = self.client.post(
            "/v1/dedup/face/purge", json=purge_data, headers=self.headers
        )

        assert purge_response.status_code == 200
        result = purge_response.json()
        assert result["status"] == "success"
        assert result["transaction_id"] == "test_customer_004"

    def test_purge_nonexistent_customer(self):
        """Test purging non-existent customer"""
        purge_data = {"transaction_id": "nonexistent_customer"}
        purge_response = self.client.post(
            "/v1/dedup/face/purge", json=purge_data, headers=self.headers
        )

        assert purge_response.status_code == 404  # Not Found

    def test_invalid_authentication(self):
        """Test invalid authentication"""
        metadata = {
            "transaction_id": "test_customer_005",
            "msisdn": "1234567894",
            "created_on": "2024-01-01T00:00:00Z",
            "id_type": "national_id",
            "id_number": "ID123456793",
        }

        files = {"image": ("test.jpg", self.test_image, "image/jpeg")}
        data = {"metadata": json.dumps(metadata)}

        # No authentication header
        response = self.client.post("/v1/dedup/face/store", files=files, data=data)

        assert response.status_code == 403  # Forbidden

    def test_invalid_image_format(self):
        """Test invalid image format"""
        metadata = {
            "transaction_id": "test_customer_006",
            "msisdn": "1234567895",
            "created_on": "2024-01-01T00:00:00Z",
            "id_type": "national_id",
            "id_number": "ID123456794",
        }

        # Send text file instead of image
        files = {"image": ("test.txt", b"not an image", "text/plain")}
        data = {"metadata": json.dumps(metadata)}

        response = self.client.post(
            "/v1/dedup/face/store", files=files, data=data, headers=self.headers
        )

        assert response.status_code == 400  # Bad Request

    def test_invalid_metadata(self):
        """Test invalid metadata format"""
        files = {"image": ("test.jpg", self.test_image, "image/jpeg")}
        data = {"metadata": "invalid json"}

        response = self.client.post(
            "/v1/dedup/face/store", files=files, data=data, headers=self.headers
        )

        assert response.status_code == 400  # Bad Request
