# ChatTTS Voice Engine Setup Guide

This guide will help you set up ChatTTS as a voice engine for generating speech from text responses in your services.

## Overview

ChatTTS is a generative speech model designed specifically for dialogue scenarios. It supports:
- Multiple languages (English, Chinese, and more)
- Multi-speaker capabilities
- Control over prosodic elements (laughter, pauses, intonation)
- Natural and expressive speech generation

## Prerequisites

- Python 3.11+ (tested with Python 3.12.8)
- macOS (tested on macOS 14.4.0)
- At least 4GB of RAM (more recommended for better performance)
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/2noise/ChatTTS
cd ChatTTS
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv chattts-venv

# Activate virtual environment
source chattts-venv/bin/activate  # On macOS/Linux
# or
chattts-venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install additional audio codec support
pip install soundfile

# Install FastAPI for the voice API server
pip install fastapi uvicorn requests
```

### 4. Fix Known Issues

The current version has a bug in the GPT model's attention mask handling. Apply the fix:

```bash
# Edit ChatTTS/model/gpt.py and add the safety check
# Around line 230, add: and max_cache_length > 0
```

## Quick Test

### 1. Basic Functionality Test

```bash
# Activate virtual environment
source chattts-venv/bin/activate

# Run the simple test
python test_simple.py
```

This should generate a `test_output.wav` file with the speech "Hello world".

### 2. Command Line Interface

```bash
# Test the command line interface
python examples/cmd/run.py "Hello, this is a test of ChatTTS."
```

### 3. Web Interface

```bash
# Launch the web UI
python examples/web/webui.py

# Open http://localhost:7860 in your browser
```

## Voice API Server

### 1. Start the API Server

```bash
# Activate virtual environment
source chattts-venv/bin/activate

# Start the voice API server
python voice_api.py
```

The server will start on `http://localhost:8000`

### 2. API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /speakers` - Get available speaker IDs
- `POST /tts` - Convert text to speech
- `GET /download/{filename}` - Download generated audio
- `GET /cleanup` - Clean up temporary files

### 3. Test the API

```bash
# In another terminal, test the API
python test_api_client.py
```

## Usage Examples

### 1. Basic Text-to-Speech

```python
import ChatTTS

# Initialize ChatTTS
chat = ChatTTS.Chat()
chat.load(compile=False)

# Generate speech
text = "Hello, welcome to our service!"
wavs = chat.infer(text, skip_refine_text=True)

# Save audio
import soundfile as sf
sf.write("output.wav", wavs[0], 24000)
```

### 2. Using the Voice API

```python
import requests

# Generate speech via API
response = requests.post("http://localhost:8000/tts", json={
    "text": "Hello, this is a test message.",
    "temperature": 0.3,
    "top_p": 0.7,
    "top_k": 20
})

if response.status_code == 200:
    data = response.json()
    print(f"Audio generated: {data['audio_file']}")
    
    # Download the audio
    audio_response = requests.get(f"http://localhost:8000/download/{data['audio_file'].split('/')[-1]}")
    with open("downloaded_audio.wav", "wb") as f:
        f.write(audio_response.content)
```

### 3. Custom Speaker Control

```python
# Sample a random speaker
speaker = chat.sample_random_speaker()

# Use specific speaker for generation
wavs = chat.infer(
    text,
    skip_refine_text=True,
    params_infer_code=ChatTTS.Chat.InferCodeParams(
        spk_emb=speaker,
        temperature=0.3,
        top_P=0.7,
        top_K=20
    )
)
```

## Integration with Services

### 1. As a Microservice

The voice API server can be deployed as a microservice:

```bash
# Run with custom host/port
uvicorn voice_api:app --host 0.0.0.0 --port 8000

# Run with multiple workers
uvicorn voice_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "voice_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Environment Variables

```bash
# Set environment variables for customization
export CHTTTS_VER=v0.2.4
export HF_HOME=/path/to/huggingface/cache
```

## Performance Considerations

### 1. Memory Usage

- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM
- **GPU**: Optional but recommended for faster generation

### 2. Generation Speed

- **CPU**: ~3-5 seconds for 10-15 word sentences
- **GPU**: ~1-2 seconds for 10-15 word sentences
- **Real-time factor**: ~0.3 (30% of real-time)

### 3. Model Loading

- **First load**: 10-30 seconds (depending on hardware)
- **Subsequent loads**: Faster due to caching
- **Memory**: Models stay loaded in memory

## Troubleshooting

### 1. Common Issues

**"narrow(): length must be non-negative"**
- Apply the GPT model fix mentioned above
- Use `skip_refine_text=True` in inference calls

**"Couldn't find appropriate backend to handle uri"**
- Install soundfile: `pip install soundfile`
- Use the provided test scripts

**Models fail to load**
- Check that all model files are present in the `asset/` directory
- Ensure sufficient memory is available
- Check Python version compatibility

### 2. Performance Issues

**Slow generation**
- Use `compile=True` in chat.load() (may require more memory)
- Reduce text length or batch size
- Use GPU if available

**High memory usage**
- Use `compile=False`
- Process shorter text segments
- Restart the service periodically

### 3. Audio Quality Issues

**Poor audio quality**
- Adjust temperature, top_p, and top_k parameters
- Try different speakers
- Use longer, more natural text

**Audio artifacts**
- Check audio sample rate (should be 24000 Hz)
- Ensure proper audio codec installation
- Verify output format compatibility

## Advanced Features

### 1. Text Control Tokens

```python
# Control laughter, pauses, and intonation
text = "What is [uv_break]your favorite food?[laugh][lbreak]"

# Use oral control
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]'
)
```

### 2. Streaming Generation

```python
# Generate audio in streaming mode
wavs = chat.infer(
    text,
    stream=True,
    skip_refine_text=True
)

for wav_segment in wavs:
    # Process each segment as it's generated
    process_audio_segment(wav_segment)
```

### 3. Custom Model Paths

```python
# Load models from custom location
chat.load(
    source="custom",
    custom_path="/path/to/models"
)
```

## Monitoring and Logging

### 1. Log Levels

```python
import logging

# Set logging level
logging.basicConfig(level=logging.INFO)

# ChatTTS logger
logger = logging.getLogger("ChatTTS")
logger.setLevel(logging.DEBUG)
```

### 2. Health Checks

```python
# Check if models are loaded
if chat.has_loaded():
    print("All models loaded successfully")
else:
    print("Some models failed to load")
```

### 3. Performance Metrics

```python
import time

start_time = time.time()
wavs = chat.infer(text)
end_time = time.time()

print(f"Generation time: {end_time - start_time:.2f} seconds")
print(f"Audio length: {len(wavs[0]) / 24000:.2f} seconds")
```

## Security Considerations

### 1. Input Validation

```python
# Validate text input
def validate_text(text):
    if len(text) > 1000:  # Limit text length
        raise ValueError("Text too long")
    if not text.strip():
        raise ValueError("Empty text")
    return text.strip()
```

### 2. Rate Limiting

```python
# Implement rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/tts")
@limiter.limit("10/minute")
async def text_to_speech(request: Request, ...):
    # Your TTS logic here
    pass
```

### 3. File Cleanup

```python
# Regular cleanup of temporary files
@app.on_event("startup")
async def schedule_cleanup():
    # Schedule cleanup every hour
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    while True:
        await asyncio.sleep(3600)  # 1 hour
        cleanup_temp_files()
```

## Support and Resources

- **GitHub**: https://github.com/2noise/ChatTTS
- **Documentation**: Check the `docs/` directory
- **Issues**: Report bugs on GitHub
- **Community**: Join Discord or QQ groups mentioned in the README

## License

ChatTTS is published under the AGPL-3.0 license. Please ensure compliance with the license terms when using this software.

---

This setup guide should get you started with using ChatTTS as a voice engine. The system is now ready to generate high-quality speech from text responses in your services! 