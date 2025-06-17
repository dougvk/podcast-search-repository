#!/bin/bash
set -e

echo "🚀 Transcription API Deployment Script"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Install Docker Compose first."
    exit 1
fi

# Create directories
mkdir -p data logs

# Copy env file if it doesn't exist
if [ ! -f .env ]; then
    cp deployment.env .env
    echo "📝 Created .env file. Edit it for custom configuration."
fi

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.prod.yml up -d --build

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ API is healthy!"
else
    echo "❌ API health check failed"
    docker-compose -f docker-compose.prod.yml logs app
    exit 1
fi

echo "🎉 Deployment complete!"
echo ""
echo "🔗 Services:"
echo "   API: http://localhost:8000"
echo "   Grafana: http://localhost:3000 (admin/admin123)"
echo "   Prometheus: http://localhost:9090"
echo ""
echo "📊 To enable nginx reverse proxy:"
echo "   docker-compose -f docker-compose.prod.yml --profile with-nginx up -d"
echo ""
echo "🔧 To view logs:"
echo "   docker-compose -f docker-compose.prod.yml logs -f" 