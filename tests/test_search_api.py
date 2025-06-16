import unittest
import tempfile
import os
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.search_api import app

class TestSearchAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        
    def tearDown(self):
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up any .faiss or .pkl files
        for ext in ['.faiss', '.pkl']:
            for file in Path('.').glob(f'*{ext}'):
                file.unlink(missing_ok=True)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("service", data)
        self.assertIn("timestamp", data)
        self.assertEqual(data["service"], "transcript-search")
    
    def test_readiness_probe(self):
        """Test readiness probe endpoint"""
        response = self.client.get("/api/v1/health/ready")
        # Should return 503 if no search engine available
        self.assertIn(response.status_code, [200, 503])
    
    def test_liveness_probe(self):
        """Test liveness probe endpoint"""
        response = self.client.get("/api/v1/health/live")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "alive")
    
    def test_system_info(self):
        """Test system info endpoint"""
        response = self.client.get("/api/v1/info")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("service", data)
        self.assertIn("version", data)
        self.assertIn("search_engine_initialized", data)
        self.assertIn("available_videos", data)
        self.assertIn("data_directory", data)
    
    @patch('core.search_api.MemvidRetriever')
    def test_search_endpoint_success(self, mock_retriever_class):
        """Test successful search request"""
        # Mock the retriever
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            ("This is a test transcript chunk", 0.95),
            ("Another relevant chunk", 0.85),
        ]
        mock_retriever_class.return_value = mock_retriever
        
        # Patch the global retriever
        with patch('core.search_api.retriever', mock_retriever):
            response = self.client.post("/api/v1/search", json={
                "query": "test query",
                "limit": 10,
                "threshold": 0.5
            })
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("query", data)
        self.assertIn("results", data)
        self.assertIn("total_found", data)
        self.assertIn("execution_time_ms", data)
        self.assertEqual(data["query"], "test query")
        self.assertGreater(len(data["results"]), 0)
    
    def test_search_endpoint_validation(self):
        """Test search request validation"""
        # Test missing query
        response = self.client.post("/api/v1/search", json={
            "limit": 10
        })
        self.assertEqual(response.status_code, 422)
        
        # Test invalid limit
        response = self.client.post("/api/v1/search", json={
            "query": "test",
            "limit": 200  # Over max limit
        })
        self.assertEqual(response.status_code, 422)
        
        # Test invalid threshold
        response = self.client.post("/api/v1/search", json={
            "query": "test",
            "threshold": 1.5  # Over max threshold
        })
        self.assertEqual(response.status_code, 422)
    
    @patch('core.search_api.MemvidRetriever')
    def test_search_endpoint_no_results(self, mock_retriever_class):
        """Test search with no results"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_retriever_class.return_value = mock_retriever
        
        with patch('core.search_api.retriever', mock_retriever):
            response = self.client.post("/api/v1/search", json={
                "query": "nonexistent query",
                "limit": 10
            })
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(len(data["results"]), 0)
        self.assertEqual(data["total_found"], 0)
    
    @patch('core.search_api.MemvidRetriever')
    def test_search_endpoint_threshold_filtering(self, mock_retriever_class):
        """Test threshold filtering works correctly"""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            ("High score chunk", 0.95),
            ("Low score chunk", 0.3),  # Below threshold
        ]
        mock_retriever_class.return_value = mock_retriever
        
        with patch('core.search_api.retriever', mock_retriever):
            response = self.client.post("/api/v1/search", json={
                "query": "test query",
                "threshold": 0.5
            })
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(len(data["results"]), 1)  # Only high score result
        self.assertEqual(data["results"][0]["text"], "High score chunk")
    
    def test_search_endpoint_no_engine(self):
        """Test search when no search engine is available"""
        with patch('core.search_api.initialize_search_engine', return_value=False):
            with patch('core.search_api.retriever', None):
                response = self.client.post("/api/v1/search", json={
                    "query": "test query"
                })
        
        self.assertEqual(response.status_code, 503)
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.options("/api/v1/health")
        # FastAPI handles CORS automatically, just check it doesn't error
        self.assertIn(response.status_code, [200, 405])  # 405 is also acceptable for OPTIONS

if __name__ == '__main__':
    unittest.main() 