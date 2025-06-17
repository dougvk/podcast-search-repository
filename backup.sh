#!/bin/bash
set -e

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup in $BACKUP_DIR"

# Backup data directory
if [ -d "data" ]; then
    tar -czf "$BACKUP_DIR/data.tar.gz" data/
    echo "✅ Data backed up"
fi

# Backup Docker volumes
docker-compose -f docker-compose.prod.yml exec -T prometheus tar -czf - /prometheus 2>/dev/null > "$BACKUP_DIR/prometheus.tar.gz" || echo "⚠️  Prometheus backup skipped"
docker-compose -f docker-compose.prod.yml exec -T grafana tar -czf - /var/lib/grafana 2>/dev/null > "$BACKUP_DIR/grafana.tar.gz" || echo "⚠️  Grafana backup skipped"

# Backup logs
if [ -d "logs" ]; then
    tar -czf "$BACKUP_DIR/logs.tar.gz" logs/
    echo "✅ Logs backed up"
fi

echo "🎉 Backup complete: $BACKUP_DIR"
echo "💾 Total size: $(du -sh $BACKUP_DIR | cut -f1)" 