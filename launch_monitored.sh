#!/bin/bash

# Launch with memory monitoring

echo "🔍 Launching ChatTTS with memory monitoring..."

# Activate virtual environment
source chattts-venv/bin/activate

# Start memory monitoring in background
(
    while true; do
        MEMORY_USAGE=$(ps aux | grep "python voice_api.py" | grep -v grep | awk '{print $6}')
        if [ ! -z "$MEMORY_USAGE" ]; then
            MEMORY_MB=$((MEMORY_USAGE / 1024))
            echo "📊 Current memory usage: ${MEMORY_MB} MB"
        fi
        sleep 5
    done
) &

MONITOR_PID=$!

# Launch the server
python voice_api.py

# Clean up monitoring
kill $MONITOR_PID 