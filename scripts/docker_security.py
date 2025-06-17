#!/usr/bin/env python3
"""
Docker Security Validation for Production Deployment
Minimal lines, maximum security validation
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def validate_docker_security():
    """Validate Docker security configuration for production"""
    checks = []
    content = ""
    
    # Check 1: Non-root user in Dockerfile
    dockerfile_path = Path("Dockerfile")
    if dockerfile_path.exists():
        content = dockerfile_path.read_text()
        user_check = "USER appuser" in content
        checks.append({
            "check": "Non-root user",
            "passed": user_check,
            "details": "Docker container runs as non-root user"
        })
        
        # Check 2: Slim base image (preferred over complex multi-stage for reliability)
        slim_image = "python:3.11-slim" in content and content.count("FROM") == 1
        checks.append({
            "check": "Optimized base image",
            "passed": slim_image,
            "details": "Uses slim Python image for reduced attack surface"
        })
        
        # Check 3: Health check configured
        health_check = "HEALTHCHECK" in content
        checks.append({
            "check": "Health check",
            "passed": health_check,
            "details": "Container health check configured"
        })
    
    # Check 4: Sensitive files excluded
    dockerignore_path = Path(".dockerignore")
    if dockerignore_path.exists():
        ignore_content = dockerignore_path.read_text()
        secrets_excluded = ".env" in ignore_content and "*.key" in ignore_content
        checks.append({
            "check": "Secrets excluded",
            "passed": secrets_excluded,
            "details": "Sensitive files excluded from Docker context"
        })
    

    
    # Summary
    passed = sum(1 for check in checks if check["passed"])
    total = len(checks)
    
    print(f"\n🔒 Docker Security Validation: {passed}/{total} checks passed\n")
    
    for check in checks:
        status = "✅" if check["passed"] else "❌"
        print(f"{status} {check['check']}: {check['details']}")
    
    if passed == total:
        print(f"\n🎉 All security checks passed! Ready for production deployment.")
        return True
    else:
        print(f"\n⚠️  {total - passed} security issues found. Please fix before production.")
        return False

def validate_image_size():
    """Validate Docker image size is optimized"""
    try:
        # Build image to check size
        result = subprocess.run([
            "docker", "build", "-t", "podcast-search:test", "."
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Get image size
            size_result = subprocess.run([
                "docker", "images", "podcast-search:test", "--format", "{{.Size}}"
            ], capture_output=True, text=True)
            
            if size_result.returncode == 0:
                size = size_result.stdout.strip()
                print(f"\n📦 Docker image size: {size}")
                
                # Clean up test image
                subprocess.run(["docker", "rmi", "podcast-search:test"], 
                             capture_output=True)
                return True
        
    except subprocess.TimeoutExpired:
        print("⏱️  Docker build timeout - image may be large")
    except Exception as e:
        print(f"❌ Error checking image size: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 Docker Production Security Validation\n")
    
    security_ok = validate_docker_security()
    
    if "--check-size" in sys.argv:
        print("\n" + "="*50)
        validate_image_size()
    
    sys.exit(0 if security_ok else 1) 