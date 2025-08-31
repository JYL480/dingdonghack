"""
Tests for Image Embedding API
Tests feature extraction and similarity matching capabilities
"""

import unittest
import requests
import json
import numpy as np
import os
from pathlib import Path

# Test configuration
EMBEDDING_URL = "https://compaq-royal-elvis-vancouver.trycloudflare.com"
TEST_IMAGE_PATH = Path(__file__).parent / "test.png"


class TestImageEmbeddingAPI(unittest.TestCase):
    """Test suite for Image Embedding API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Verify test image exists
        if not TEST_IMAGE_PATH.exists():
            raise FileNotFoundError(f"Test image not found: {TEST_IMAGE_PATH}")
        
        # Store base URL
        cls.base_url = EMBEDDING_URL
        cls.test_image = TEST_IMAGE_PATH
        
        # Default model settings
        cls.default_model = "resnet50"
        cls.default_image_size = 224
    
    def test_01_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        
        # Check status code
        self.assertEqual(response.status_code, 200, "Health check should return 200")
        
        # Check response content
        data = response.json()
        print(f"Health check response: {data}")
    
    def test_02_root_endpoint(self):
        """Test root endpoint for service info"""
        response = requests.get(self.base_url, timeout=10)
        
        # Check status code
        self.assertEqual(response.status_code, 200, "Root endpoint should return 200")
        
        # Check response
        try:
            data = response.json()
            print(f"Service info: {data}")
        except:
            print(f"Root endpoint returned: {response.text[:200]}")
    
    def test_03_set_model(self):
        """Test setting the model configuration"""
        # Test with resnet50
        payload = {
            "model_name": self.default_model,
            "image_size": self.default_image_size
        }
        
        response = requests.post(
            f"{self.base_url}/set_model",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            print(f"Model set successfully: {data}")
            
            # Verify model was set
            if "model_name" in data:
                self.assertEqual(data["model_name"], self.default_model, "Model name should match")
            if "image_size" in data:
                self.assertEqual(data["image_size"], self.default_image_size, "Image size should match")
        else:
            print(f"Set model returned status {response.status_code}: {response.text[:200]}")
    
    def test_04_extract_single_features(self):
        """Test extracting features from a single image"""
        # First ensure model is set
        self.test_03_set_model()
        
        # Extract features from test image
        with open(self.test_image, "rb") as f:
            files = {"file": ("test.png", f, "image/png")}
            
            response = requests.post(
                f"{self.base_url}/extract_single/",
                files=files,
                timeout=30
            )
        
        # Check status code
        self.assertEqual(response.status_code, 200, f"Extract should return 200, got {response.status_code}")
        
        # Check response structure
        data = response.json()
        
        # Check for features
        self.assertIn("features", data, "Response should contain features")
        self.assertIsInstance(data["features"], list, "Features should be a list")
        
        # Check feature dimensions
        if data["features"]:
            features = data["features"][0] if isinstance(data["features"][0], list) else data["features"]
            feature_dim = len(features)
            print(f"Feature dimension: {feature_dim}")
            
            # ResNet50 typically outputs 2048-dimensional features
            self.assertGreater(feature_dim, 0, "Feature dimension should be positive")
            
            # Check that features are numeric
            for i, val in enumerate(features[:5]):  # Check first 5 values
                self.assertIsInstance(val, (int, float), f"Feature[{i}] should be numeric")
            
            print(f"Sample feature values (first 5): {features[:5]}")
        
        # Check for shape info
        if "shape" in data:
            print(f"Feature shape: {data['shape']}")
        
        # Check for model info
        if "model_name" in data:
            print(f"Model used: {data['model_name']}")
        
        # Check for timing info
        if "time_taken" in data:
            print(f"Processing time: {data['time_taken']} seconds")
    
    def test_05_extract_batch_features(self):
        """Test extracting features from multiple images (if supported)"""
        # Check if batch endpoint exists
        response = requests.get(f"{self.base_url}/docs", timeout=10)
        
        if response.status_code == 200 and "/extract_batch" in response.text:
            print("Batch extraction endpoint found")
            
            # Try batch extraction with same image twice
            with open(self.test_image, "rb") as f:
                image_data = f.read()
                
                files = [
                    ("files", ("test1.png", image_data, "image/png")),
                    ("files", ("test2.png", image_data, "image/png"))
                ]
                
                response = requests.post(
                    f"{self.base_url}/extract_batch/",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Batch extraction successful: extracted {len(data.get('features', []))} feature sets")
            else:
                print(f"Batch extraction returned status {response.status_code}")
        else:
            print("Batch extraction endpoint not available")
    
    def test_06_feature_similarity(self):
        """Test that same image produces similar features"""
        # Extract features twice from same image
        features_list = []
        
        for i in range(2):
            with open(self.test_image, "rb") as f:
                files = {"file": ("test.png", f, "image/png")}
                response = requests.post(
                    f"{self.base_url}/extract_single/",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                if "features" in data:
                    features = data["features"][0] if isinstance(data["features"][0], list) else data["features"]
                    features_list.append(features)
        
        # Compare features if we got both
        if len(features_list) == 2:
            # Convert to numpy arrays for easy comparison
            feat1 = np.array(features_list[0])
            feat2 = np.array(features_list[1])
            
            # Calculate cosine similarity
            dot_product = np.dot(feat1, feat2)
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            
            if norm1 > 0 and norm2 > 0:
                cosine_similarity = dot_product / (norm1 * norm2)
                print(f"Cosine similarity between same image features: {cosine_similarity:.4f}")
                
                # Same image should have very high similarity (close to 1.0)
                self.assertGreater(cosine_similarity, 0.99, "Same image should have high similarity")
            
            # Calculate L2 distance
            l2_distance = np.linalg.norm(feat1 - feat2)
            print(f"L2 distance between same image features: {l2_distance:.4f}")
            
            # Same image should have very low distance (close to 0)
            self.assertLess(l2_distance, 0.1, "Same image should have low distance")
    
    def test_07_different_models(self):
        """Test with different model configurations"""
        models_to_test = [
            {"model_name": "resnet50", "image_size": 224},
            {"model_name": "resnet101", "image_size": 224},
            # Add more models if supported
        ]
        
        for model_config in models_to_test:
            # Try to set model
            response = requests.post(
                f"{self.base_url}/set_model",
                json=model_config,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Testing with model: {model_config['model_name']}")
                
                # Extract features with this model
                with open(self.test_image, "rb") as f:
                    files = {"file": ("test.png", f, "image/png")}
                    response = requests.post(
                        f"{self.base_url}/extract_single/",
                        files=files,
                        timeout=30
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    if "features" in data:
                        features = data["features"][0] if isinstance(data["features"][0], list) else data["features"]
                        print(f"  {model_config['model_name']} feature dim: {len(features)}")
            else:
                print(f"Model {model_config['model_name']} not available")
    
    def test_08_invalid_image(self):
        """Test with invalid image data"""
        # Test with non-image file
        files = {"file": ("test.txt", b"This is not an image", "text/plain")}
        
        response = requests.post(
            f"{self.base_url}/extract_single/",
            files=files,
            timeout=30
        )
        
        # Should either reject or handle gracefully
        if response.status_code != 200:
            print(f"Invalid file correctly rejected with status {response.status_code}")
        else:
            print("Invalid file was processed - checking response")
            data = response.json()
            if "error" in data:
                print(f"Error message: {data['error']}")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Testing Image Embedding API")
    print(f"URL: {EMBEDDING_URL}")
    print(f"Test Image: {TEST_IMAGE_PATH}")
    print("=" * 60)
    
    unittest.main(verbosity=2)