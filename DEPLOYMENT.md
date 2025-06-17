# 🚀 Transcription API Deployment

Simple one-command deployment for VPS or local machine.

## Quick Start

```bash
git clone <your-repo>
cd transcription-test
./deploy.sh
```

That's it! Your transcription API with monitoring is now running.

## Services

- **API**: http://localhost:8000
- **Grafana**: http://localhost:3000 (admin/admin123)  
- **Prometheus**: http://localhost:9090

## Configuration

Edit `.env` file for custom ports and settings:

```bash
API_PORT=8000
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=your_password
```

## With Nginx Reverse Proxy

```bash
docker-compose -f docker-compose.prod.yml --profile with-nginx up -d
```

Access everything on port 80:
- API: http://your-domain/
- Grafana: http://your-domain/grafana/

## Management

```bash
# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop services
docker-compose -f docker-compose.prod.yml down

# Backup data
./backup.sh

# Update
git pull && docker-compose -f docker-compose.prod.yml up -d --build
```

## VPS Requirements

- 1GB+ RAM
- Docker & Docker Compose
- Open ports: 80, 8000, 3000, 9090

Works on any $5-20/month VPS (DigitalOcean, Linode, Vultr, etc.). 