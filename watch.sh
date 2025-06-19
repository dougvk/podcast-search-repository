# Check every 30 seconds what's happening
while true; do
  echo "=== $(date) ==="
  echo "Process running: $(pgrep -f process_podcasts.py)"
  echo "Files in data/:"
  ls -la data/*.mp4 data/*.json 2>/dev/null || echo "No output files yet"
  echo "Memory usage:"
  ps aux | grep process_podcasts.py | grep -v grep
  echo ""
  sleep 30
done
