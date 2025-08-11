#!/bin/bash

# Launch ChatTTS with maximum memory allocation

echo "🚀 Launching ChatTTS with maximum memory allocation..."

# Check system memory
TOTAL_MEM=$(sysctl hw.memsize | awk '{print $2/1024/1024/1024}')
echo "💾 Total system memory: ${TOTAL_MEM} GB"

# Set environment variables for maximum memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)

# Calculate optimal memory allocation
if [ $(echo "$TOTAL_MEM >= 32" | bc -l) -eq 1 ]; then
    echo "🔥 High-end system detected - using maximum memory allocation"
    export CHTTTS_MAX_MEMORY=1
    export CHTTTS_COMPILE=1
    export CHTTTS_EXPERIMENTAL=1
elif [ $(echo "$TOTAL_MEM >= 16" | bc -l) -eq 1 ]; then
    echo "⚡ Mid-range system detected - using optimized memory allocation"
    export CHTTTS_MAX_MEMORY=1
    export CHTTTS_COMPILE=1
    export CHTTTS_EXPERIMENTAL=0
else
    echo "💡 Standard system - using balanced memory allocation"
    export CHTTTS_MAX_MEMORY=0
    export CHTTTS_COMPILE=0
    export CHTTTS_EXPERIMENTAL=0
fi

# Activate virtual environment
source chattts-venv/bin/activate

# Launch with memory optimization
echo "🎯 Starting ChatTTS voice API server..."
python voice_api.py 