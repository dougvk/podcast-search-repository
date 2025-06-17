#!/usr/bin/env python3
"""
Tests for deployment infrastructure
"""

import unittest
import os
import tempfile
import shutil
import subprocess
import yaml

class TestDeploymentInfrastructure(unittest.TestCase):
    """Test deployment configuration and scripts"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
    
    def tearDown(self):
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_env_file_exists(self):
        """Test deployment.env file exists and has required variables"""
        self.assertTrue(os.path.exists("deployment.env"))
        
        with open("deployment.env", "r") as f:
            content = f.read()
            
        # Check for required variables
        required_vars = [
            "API_PORT", "API_HOST", "PROMETHEUS_PORT", 
            "GRAFANA_PORT", "GRAFANA_ADMIN_PASSWORD"
        ]
        
        for var in required_vars:
            self.assertIn(var, content, f"Missing required variable: {var}")
    
    def test_docker_compose_file_valid(self):
        """Test docker-compose.prod.yml is valid YAML"""
        self.assertTrue(os.path.exists("docker-compose.prod.yml"))
        
        with open("docker-compose.prod.yml", "r") as f:
            try:
                compose_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                self.fail(f"Invalid YAML in docker-compose.prod.yml: {e}")
        
        # Check required services
        self.assertIn("services", compose_config)
        services = compose_config["services"]
        
        required_services = ["app", "prometheus", "grafana", "nginx"]
        for service in required_services:
            self.assertIn(service, services, f"Missing service: {service}")
    
    def test_nginx_config_exists(self):
        """Test nginx.conf exists and has basic structure"""
        self.assertTrue(os.path.exists("nginx.conf"))
        
        with open("nginx.conf", "r") as f:
            content = f.read()
        
        # Check for basic nginx structure
        self.assertIn("events", content)
        self.assertIn("http", content)
        self.assertIn("server", content)
        self.assertIn("location", content)
    
    def test_deploy_script_executable(self):
        """Test deploy.sh script exists and is executable"""
        self.assertTrue(os.path.exists("deploy.sh"))
        
        # Check if file is executable
        file_stat = os.stat("deploy.sh")
        self.assertTrue(file_stat.st_mode & 0o111, "deploy.sh is not executable")
    
    def test_backup_script_executable(self):
        """Test backup.sh script exists and is executable"""
        self.assertTrue(os.path.exists("backup.sh"))
        
        # Check if file is executable
        file_stat = os.stat("backup.sh")
        self.assertTrue(file_stat.st_mode & 0o111, "backup.sh is not executable")
    
    def test_deployment_documentation_exists(self):
        """Test DEPLOYMENT.md documentation exists"""
        self.assertTrue(os.path.exists("DEPLOYMENT.md"))
        
        with open("DEPLOYMENT.md", "r") as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            "Quick Start", "Services", "Configuration", 
            "Management", "VPS Requirements"
        ]
        
        for section in required_sections:
            self.assertIn(section, content, f"Missing section: {section}")
    
    def test_docker_compose_environment_variables(self):
        """Test docker-compose uses environment variables correctly"""
        with open("docker-compose.prod.yml", "r") as f:
            content = f.read()
        
        # Check for environment variable usage
        env_vars = [
            "${API_PORT:-8000}", "${PROMETHEUS_PORT:-9090}", 
            "${GRAFANA_PORT:-3000}", "${GRAFANA_ADMIN_PASSWORD:-admin123}"
        ]
        
        for var in env_vars:
            self.assertIn(var, content, f"Missing environment variable: {var}")
    
    def test_docker_compose_volumes(self):
        """Test docker-compose has proper volume configuration"""
        with open("docker-compose.prod.yml", "r") as f:
            compose_config = yaml.safe_load(f)
        
        # Check volumes section exists
        self.assertIn("volumes", compose_config)
        volumes = compose_config["volumes"]
        
        required_volumes = ["prometheus_data", "grafana_data"]
        for volume in required_volumes:
            self.assertIn(volume, volumes, f"Missing volume: {volume}")
    
    def test_docker_compose_health_checks(self):
        """Test docker-compose has health checks configured"""
        with open("docker-compose.prod.yml", "r") as f:
            compose_config = yaml.safe_load(f)
        
        # Check app service has health check
        app_service = compose_config["services"]["app"]
        self.assertIn("healthcheck", app_service)
        
        healthcheck = app_service["healthcheck"]
        self.assertIn("test", healthcheck)
        self.assertIn("interval", healthcheck)
        self.assertIn("timeout", healthcheck)
        self.assertIn("retries", healthcheck)
    
    def test_deploy_script_syntax(self):
        """Test deploy.sh script has valid bash syntax"""
        try:
            # Use bash -n to check syntax without executing
            result = subprocess.run(
                ["bash", "-n", "deploy.sh"], 
                capture_output=True, 
                text=True
            )
            self.assertEqual(result.returncode, 0, 
                           f"deploy.sh syntax error: {result.stderr}")
        except FileNotFoundError:
            self.skipTest("bash not available for syntax checking")
    
    def test_backup_script_syntax(self):
        """Test backup.sh script has valid bash syntax"""
        try:
            # Use bash -n to check syntax without executing
            result = subprocess.run(
                ["bash", "-n", "backup.sh"], 
                capture_output=True, 
                text=True
            )
            self.assertEqual(result.returncode, 0, 
                           f"backup.sh syntax error: {result.stderr}")
        except FileNotFoundError:
            self.skipTest("bash not available for syntax checking")

if __name__ == '__main__':
    unittest.main() 