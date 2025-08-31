"""
Tests for OmniParser API
Tests UI element detection and OCR capabilities
"""

import unittest
import requests
import json
import base64
import os
from pathlib import Path

# Test configuration
OMNIPARSER_URL = "https://teen-alt-clocks-athletes.trycloudflare.com"
TEST_IMAGE_PATH = Path(__file__).parent / "test.png"


class TestOmniParserAPI(unittest.TestCase):
    """Test suite for OmniParser API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Verify test image exists
        if not TEST_IMAGE_PATH.exists():
            raise FileNotFoundError(f"Test image not found: {TEST_IMAGE_PATH}")
        
        # Store base URL
        cls.base_url = OMNIPARSER_URL
        cls.test_image = TEST_IMAGE_PATH
    
    def test_01_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        
        # Check status code
        self.assertEqual(response.status_code, 200, "Health check should return 200")
        
        # Check response content
        data = response.json()
        self.assertIn("status", data, "Response should contain status")
        print(f"Health check passed: {data}")
    
    def test_02_root_endpoint(self):
        """Test root endpoint for service info"""
        response = requests.get(self.base_url, timeout=10)
        
        # Check status code
        self.assertEqual(response.status_code, 200, "Root endpoint should return 200")
        
        # Check response contains service info
        data = response.json()
        self.assertIn("service", data, "Should contain service name")
        self.assertIn("status", data, "Should contain status")
        self.assertEqual(data["service"], "OmniParser API", "Service name should be OmniParser API")
        print(f"Service info: {data}")
    
    def test_03_parse_image(self):
        """Test parsing UI elements from image"""
        # Prepare image file
        with open(self.test_image, "rb") as f:
            files = {"file": ("test.png", f, "image/png")}
            
            # Send parse request
            response = requests.post(
                f"{self.base_url}/parse",
                files=files,
                timeout=30
            )
        
        # Check status code
        self.assertEqual(response.status_code, 200, f"Parse should return 200, got {response.status_code}")
        
        # Check response structure
        data = response.json()
        self.assertIn("status", data, "Response should contain status")
        self.assertEqual(data["status"], "success", "Parse should be successful")
        
        # Check for elements
        self.assertIn("elements", data, "Response should contain elements")
        self.assertIsInstance(data["elements"], list, "Elements should be a list")
        
        # Check element structure if any elements found
        if data["elements"]:
            element = data["elements"][0]
            self.assertIn("type", element, "Element should have type")
            self.assertIn("shape", element, "Element should have shape")
            
            # Check shape structure
            shape = element["shape"]
            required_shape_fields = ["x", "y", "width", "height"]
            for field in required_shape_fields:
                self.assertIn(field, shape, f"Shape should have {field}")
                self.assertIsInstance(shape[field], (int, float), f"{field} should be numeric")
        
        # Check for device info
        self.assertIn("device_used", data, "Response should contain device_used")
        
        # Report findings
        print(f"Found {len(data['elements'])} UI elements")
        print(f"Device used: {data.get('device_used', 'unknown')}")
        
        # Check for specific UI elements we expect
        text_elements = [e for e in data["elements"] if e.get("type") == "text"]
        icon_elements = [e for e in data["elements"] if e.get("type") == "icon"]
        
        print(f"Text elements: {len(text_elements)}")
        print(f"Icon elements: {len(icon_elements)}")
        
        # Look for Play Store element
        play_store_found = False
        for element in text_elements:
            if element.get("text") and "Play Store" in element.get("text", ""):
                play_store_found = True
                print(f"Found Play Store at position: x={element['shape']['x']}, y={element['shape']['y']}")
                break
        
        if not play_store_found:
            print("Play Store text not detected in elements")
    
    def test_04_parse_with_annotated_image(self):
        """Test that annotated image is returned"""
        with open(self.test_image, "rb") as f:
            files = {"file": ("test.png", f, "image/png")}
            response = requests.post(f"{self.base_url}/parse", files=files, timeout=30)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check for annotated image
        if "annotated_image" in data:
            annotated = data["annotated_image"]
            
            # Check if it's a data URL
            if annotated.startswith("data:image"):
                # Extract base64 part
                base64_part = annotated.split(",")[1] if "," in annotated else annotated
                
                # Try to decode to verify it's valid base64
                try:
                    decoded = base64.b64decode(base64_part)
                    self.assertGreater(len(decoded), 0, "Annotated image should have content")
                    print(f"Annotated image size: {len(decoded)} bytes")
                except Exception as e:
                    self.fail(f"Failed to decode annotated image: {e}")
            else:
                print("Annotated image is raw base64")
        else:
            print("No annotated image in response")
    
    def test_05_invalid_file_type(self):
        """Test with invalid file type"""
        # Create a text file
        files = {"file": ("test.txt", b"This is not an image", "text/plain")}
        
        response = requests.post(f"{self.base_url}/parse", files=files, timeout=30)
        
        # Should either reject or handle gracefully
        if response.status_code == 200:
            data = response.json()
            # If it accepts the file, status should indicate failure
            if data.get("status") == "success":
                # Check if it returned empty elements
                self.assertEqual(len(data.get("elements", [])), 0, "Should return no elements for text file")
        else:
            # Error response is also acceptable
            self.assertIn(response.status_code, [400, 422, 500], "Should return error for invalid file")
    
    def test_06_model_info(self):
        """Test model info endpoint if available"""
        response = requests.get(f"{self.base_url}/model_info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Model info: {data}")
            # Check for expected fields
            possible_fields = ["model_name", "version", "device", "status"]
            for field in possible_fields:
                if field in data:
                    print(f"  {field}: {data[field]}")
        else:
            print(f"Model info endpoint not available (status: {response.status_code})")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Testing OmniParser API")
    print(f"URL: {OMNIPARSER_URL}")
    print(f"Test Image: {TEST_IMAGE_PATH}")
    print("=" * 60)
    
    unittest.main(verbosity=2)