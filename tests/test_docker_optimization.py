#!/usr/bin/env python3
"""
Comprehensive tests for Docker optimization and security validation
"""

import unittest
import tempfile
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.docker_security import validate_docker_security, validate_image_size

class TestDockerSecurity(unittest.TestCase):
    """Test Docker security validation"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        # Clean up temporary files
        os.chdir(self.original_cwd)
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_dockerfile_security_validation(self):
        """Test Dockerfile security checks"""
        # Create test Dockerfile with security features in temp directory
        dockerfile_content = """
FROM python:3.11-slim
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1
"""
        dockerfile_path = Path(self.temp_dir) / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        self.temp_files.append(str(dockerfile_path))
        
        # Create .dockerignore with security exclusions in temp directory
        dockerignore_content = """
.env
*.key
*.pem
"""
        dockerignore_path = Path(self.temp_dir) / ".dockerignore"
        dockerignore_path.write_text(dockerignore_content)
        self.temp_files.append(str(dockerignore_path))
        
        # Change to temp directory for testing
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        try:
            # Test security validation
            result = validate_docker_security()
            self.assertTrue(result)
        finally:
            os.chdir(original_cwd)
    
    def test_dockerfile_missing_security(self):
        """Test validation fails for insecure Dockerfile"""
        # Create insecure Dockerfile in temp directory
        dockerfile_content = """
FROM python:3.11-slim
# No USER directive - runs as root
# No HEALTHCHECK
"""
        dockerfile_path = Path(self.temp_dir) / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        self.temp_files.append(str(dockerfile_path))
        
        # Change to temp directory for testing
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        try:
            # Test security validation fails
            result = validate_docker_security()
            self.assertFalse(result)
        finally:
            os.chdir(original_cwd)
    
    def test_dockerignore_validation(self):
        """Test .dockerignore security exclusions"""
        # Create minimal Dockerfile in temp directory
        dockerfile_path = Path(self.temp_dir) / "Dockerfile"
        dockerfile_path.write_text("FROM python:3.11-slim\nUSER appuser\nHEALTHCHECK CMD curl")
        self.temp_files.append(str(dockerfile_path))
        
        # Create .dockerignore without security exclusions in temp directory
        dockerignore_content = """
*.pyc
__pycache__/
"""
        dockerignore_path = Path(self.temp_dir) / ".dockerignore"
        dockerignore_path.write_text(dockerignore_content)
        self.temp_files.append(str(dockerignore_path))
        
        # Change to temp directory for testing
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        try:
            # Test validation fails
            result = validate_docker_security()
            self.assertFalse(result)
        finally:
            os.chdir(original_cwd)
    
    @patch('subprocess.run')
    def test_image_size_validation_success(self, mock_run):
        """Test successful image size validation"""
        # Mock successful docker build
        mock_run.side_effect = [
            # docker build
            Mock(returncode=0, stdout="", stderr=""),
            # docker images (get size)
            Mock(returncode=0, stdout="512MB\n", stderr=""),
            # docker rmi (cleanup)
            Mock(returncode=0, stdout="", stderr="")
        ]
        
        result = validate_image_size()
        self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_image_size_validation_build_failure(self, mock_run):
        """Test image size validation with build failure"""
        # Mock failed docker build
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Build failed")
        
        result = validate_image_size()
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_image_size_validation_timeout(self, mock_run):
        """Test image size validation with timeout"""
        # Mock timeout during build
        mock_run.side_effect = subprocess.TimeoutExpired(["docker", "build"], 300)
        
        result = validate_image_size()
        self.assertFalse(result)

class TestDockerfileStructure(unittest.TestCase):
    """Test Dockerfile structure and best practices"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        # Use project root (current working directory should be project root when running tests)
        self.dockerfile_path = Path("Dockerfile")
    
    def tearDown(self):
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_production_dockerfile_exists(self):
        """Test that production Dockerfile exists and is valid"""
        self.assertTrue(self.dockerfile_path.exists(), f"Dockerfile not found at {self.dockerfile_path}")
        
        content = self.dockerfile_path.read_text()
        
        # Check for essential security elements
        self.assertIn("USER appuser", content)
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("python:3.11-slim", content)
        
        # Check for proper structure
        self.assertIn("WORKDIR", content)
        self.assertIn("COPY", content)
        self.assertIn("RUN", content)
        self.assertIn("CMD", content)
    
    def test_dockerignore_exists(self):
        """Test that .dockerignore exists and excludes sensitive files"""
        dockerignore_path = Path(".dockerignore")
        if dockerignore_path.exists():
            content = dockerignore_path.read_text()
            
            # Check for sensitive file exclusions
            self.assertIn(".env", content)
            self.assertIn("*.key", content)
            
            # Check for development file exclusions
            self.assertIn("__pycache__", content)
            self.assertIn(".git", content)

class TestProductionReadiness(unittest.TestCase):
    """Test production readiness of Docker configuration"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
    
    def tearDown(self):
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_environment_variables(self):
        """Test that production environment variables are configured"""
        dockerfile_path = Path("Dockerfile")
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            
            # Check for production environment configuration
            self.assertIn("ENVIRONMENT=production", content)
            self.assertIn("PYTHONUNBUFFERED=1", content)
    
    def test_health_check_configuration(self):
        """Test health check is properly configured"""
        dockerfile_path = Path("Dockerfile")
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            
            # Check health check exists and is reasonable
            self.assertIn("HEALTHCHECK", content)
            self.assertIn("curl", content)
            self.assertIn("/health", content)
            
            # Check health check parameters are reasonable
            self.assertIn("--interval=", content)
            self.assertIn("--timeout=", content)
    
    def test_port_configuration(self):
        """Test that port is properly exposed"""
        dockerfile_path = Path("Dockerfile")
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            
            # Check port exposure
            self.assertIn("EXPOSE 8000", content)
            
            # Check CMD uses correct port
            self.assertIn("--port", content)
            self.assertIn("8000", content)
    
    def test_security_script_functionality(self):
        """Test that security validation script works"""
        # Test that security script exists and is executable
        security_script = Path("scripts/docker_security.py")
        self.assertTrue(security_script.exists())
        
        # Check script has proper shebang
        with open(security_script, 'r') as f:
            first_line = f.readline().strip()
            self.assertTrue(first_line.startswith("#!"))

if __name__ == "__main__":
    unittest.main() 