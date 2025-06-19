#!/bin/bash
set -e

echo "🚀 Universal Transcription Search System"
echo "========================================"

# Default selector if not provided
SELECTOR_SCRIPT=${SELECTOR_SCRIPT:-"selectors/random_20.py"}

echo "📂 Using selector: $SELECTOR_SCRIPT"

# Check if selector exists
if [ ! -f "$SELECTOR_SCRIPT" ]; then
    echo "❌ Selector script not found: $SELECTOR_SCRIPT"
    exit 1
fi

# Step 1: Run selector and process episodes
echo "🔍 Step 1: Selecting and processing episodes..."
python universal_processor.py "$SELECTOR_SCRIPT"

if [ $? -eq 0 ]; then
    echo "✅ Episode processing completed"
else
    echo "❌ Episode processing failed"
    exit 1
fi

# Step 2: Start API server
echo "🌐 Step 2: Starting API server..."
echo "🔗 API will be available at http://0.0.0.0:8000"
exec python universal_api.py 