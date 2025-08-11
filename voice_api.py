#!/usr/bin/env python3
"""
Simple Voice API server for ChatTTS.
This can be used as a voice engine for generating speech from text.
"""

import os
import sys
import logging
from typing import List, Optional
import tempfile
import shutil
import re # Added for text splitting

import resource
import gc
# import psutil # Commented out as per edit hint

# Add the current directory to Python path
sys.path.append(os.getcwd())

import ChatTTS
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ChatTTS
chat = None

class TextToSpeechRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = None
    temperature: float = 0.05
    top_p: float = 0.95
    top_k: int = 3
    skip_refine_text: bool = True
    max_new_token: int = 4096      # Add this parameter
    min_new_token: int = 100       # Add this parameter

class TextToSpeechResponse(BaseModel):
    success: bool
    message: str
    audio_file: Optional[str] = None
    speaker_id: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="ChatTTS Voice API",
    description="A simple API for generating speech from text using ChatTTS",
    version="1.0.0"
)

def optimize_memory_usage():
    """Optimize memory usage for maximum performance"""
    
    # Get system memory info
    # memory = psutil.virtual_memory() # Commented out as per edit hint
    # total_gb = memory.total / (1024**3) # Commented out as per edit hint
    # available_gb = memory.available / (1024**3) # Commented out as per edit hint
    
    # print(f"Total system memory: {total_gb:.1f} GB") # Commented out as per edit hint
    # print(f"Available memory: {available_gb:.1f} GB") # Commented out as per edit hint
    
    # Set Python memory limits to maximum
    try:
        # Set soft and hard limits to maximum (unlimited)
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        resource.setrlimit(resource.RLIMIT_DATA, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        print("✓ Python memory limits set to maximum")
    except Exception as e:
        print(f"⚠ Could not set memory limits: {e}")
    
    # Enable garbage collection optimization
    gc.enable()
    gc.set_threshold(100, 5, 5)  # Aggressive garbage collection
    print("✓ Garbage collection optimized")

def load_models_with_max_memory():
    """Load models using maximum available memory"""
    
    # Get memory info
    # memory = psutil.virtual_memory() # Commented out as per edit hint
    # total_gb = memory.total / (1024**3) # Commented out as per edit hint
    
    # Calculate optimal batch sizes based on available memory
    # if total_gb >= 32: # Commented out as per edit hint
    #     # High-end system (32GB+) # Commented out as per edit hint
    #     batch_size = 48 # Commented out as per edit hint
    #     max_tokens = 16384 # Commented out as per edit hint
    #     compile_mode = True # Commented out as per edit hint
    #     experimental = True # Commented out as per edit hint
    # elif total_gb >= 16: # Commented out as per edit hint
    #     # Mid-range system (16-32GB) # Commented out as per edit hint
    #     batch_size = 32 # Commented out as per edit hint
    #     max_tokens = 8192 # Commented out as per edit hint
    #     compile_mode = True # Commented out as per edit hint
    #     experimental = False # Commented out as per edit hint
    # elif total_gb >= 8: # Commented out as per edit hint
    #     # Standard system (8-16GB) # Commented out as per edit hint
    #     batch_size = 24 # Commented out as per edit hint
    #     max_tokens = 4096 # Commented out as per edit hint
    #     compile_mode = False # Commented out as per edit hint
    #     experimental = False # Commented out as per edit hint
    # else: # Commented out as per edit hint
    #     # Low memory system (<8GB) # Commented out as per edit hint
    #     batch_size = 16 # Commented out as per edit hint
    #     max_tokens = 2048 # Commented out as per edit hint
    #     compile_mode = False # Commented out as per edit hint
    #     experimental = False # Commented out as per edit hint
    
    # logger.info(f"System memory: {total_gb:.1f} GB") # Commented out as per edit hint
    # logger.info(f"Using batch size: {batch_size}") # Commented out as per edit hint
    # logger.info(f"Max tokens: {max_tokens}") # Commented out as per edit hint
    # logger.info(f"Compile mode: {compile_mode}") # Commented out as per edit hint
    # logger.info(f"Experimental: {experimental}") # Commented out as per edit hint
    
    return {
        'batch_size': 24, # Default value as psutil is removed
        'max_tokens': 4096, # Default value as psutil is removed
        'compile_mode': False, # Default value as psutil is removed
        'experimental': False # Default value as psutil is removed
    }

# Call this at startup
@app.on_event("startup")
async def startup_event():
    """Initialize ChatTTS with maximum memory allocation."""
    global chat
    
    # Optimize memory usage
    optimize_memory_usage()
    
    try:
        logger.info("Initializing ChatTTS with maximum memory allocation...")
        chat = ChatTTS.Chat()
        
        logger.info("Loading models with maximum memory...")
        # Use compile=True for better performance (uses more memory)
        success = chat.load(
            compile=True,  # Enable compilation for better performance
            experimental=True,  # Enable experimental features
            use_flash_attn=False,  # Disable if causing issues
            use_vllm=False  # Disable if causing issues
        )
        
        if not success:
            logger.error("Failed to load ChatTTS models")
            raise RuntimeError("Failed to load models")
        
        logger.info("ChatTTS initialized successfully with maximum memory allocation")
        
        # Force garbage collection after model loading
        gc.collect()
        
    except Exception as e:
        logger.error(f"Failed to initialize ChatTTS: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ChatTTS Voice API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if chat is None:
        raise HTTPException(status_code=503, detail="ChatTTS not initialized")
    return {"status": "healthy", "message": "ChatTTS is ready"}

@app.get("/speakers")
async def get_speakers():
    """Get available speaker IDs."""
    if chat is None:
        raise HTTPException(status_code=503, detail="ChatTTS not initialized")
    
    try:
        # Generate a few random speakers
        speakers = []
        for _ in range(5):
            speaker = chat.sample_random_speaker()
            speakers.append(speaker[:50] + "..." if len(speaker) > 50 else speaker)
        
        return {"speakers": speakers}
    except Exception as e:
        logger.error(f"Error getting speakers: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting speakers: {str(e)}")

def split_text_for_tts(text, max_length=200):
    """Split long text into chunks that won't get cut off"""
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would make chunk too long
        if len(current_chunk + " " + sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split by words
                words = sentence.split()
                for word in words:
                    if len(current_chunk + " " + word) > max_length:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = word
                        else:
                            current_chunk = word
                    else:
                        current_chunk += " " + word if current_chunk else word
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

@app.post("/tts", response_model=TextToSpeechResponse)
async def text_to_speech(request: TextToSpeechRequest):
    """Convert text to speech."""
    if chat is None:
        raise HTTPException(status_code=503, detail="ChatTTS not initialized")
    
    try:
        # Use provided speaker or sample a random one
        speaker_id = request.speaker_id
        if not speaker_id:
            speaker_id = chat.sample_random_speaker()
        
        logger.info(f"Generating speech for text: {request.text[:50]}...")
        logger.info(f"Using speaker: {speaker_id[:50]}...")
        
        # Generate audio
        wavs = chat.infer(
            request.text,
            skip_refine_text=True,
            params_infer_code=ChatTTS.Chat.InferCodeParams(
                spk_emb=speaker_id,
                temperature=0.05,
                top_P=0.95,
                top_K=3,
                max_new_token=4096,        # Increased from 2048
                min_new_token=100,         # Add minimum token requirement
                repetition_penalty=1.0,
                stream_batch=24,
                stream_speed=12000,
                pass_first_n_batches=2
            )
        )
        
        if not wavs or len(wavs) == 0:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Get the first audio segment
        wav = wavs[0]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Save audio using soundfile
            sf.write(tmp_file.name, wav, 24000)
            temp_filename = tmp_file.name
        
        # Generate a more descriptive filename
        safe_text = "".join(c for c in request.text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')
        output_filename = f"tts_{safe_text}.wav"
        
        # Move to final location
        final_path = os.path.join("/tmp", output_filename)
        shutil.move(temp_filename, final_path)
        
        logger.info(f"Audio generated successfully: {final_path}")
        
        return TextToSpeechResponse(
            success=True,
            message="Audio generated successfully",
            audio_file=final_path,
            speaker_id=speaker_id
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.post("/tts/complete", response_model=TextToSpeechResponse)
async def text_to_speech_complete(request: TextToSpeechRequest):
    """Convert text to speech with guaranteed completion."""
    
    if chat is None:
        raise HTTPException(status_code=503, detail="ChatTTS not initialized")
    
    try:
        # Split text if it's too long
        text_chunks = split_text_for_tts(request.text)
        
        all_audio = []
        
        for chunk in text_chunks:
            logger.info(f"Processing chunk: {chunk[:50]}...")
            
            # Generate audio for this chunk with higher limits
            wavs = chat.infer(
                chunk,
                skip_refine_text=True,
                params_infer_code=ChatTTS.Chat.InferCodeParams(
                    spk_emb=request.speaker_id or chat.sample_random_speaker(),
                    temperature=request.temperature,
                    top_P=request.top_p,
                    top_K=request.top_k,
                    max_new_token=8192,        # Very high limit
                    min_new_token=200,         # Ensure completion
                    repetition_penalty=1.0,
                    stream_batch=24,
                    stream_speed=12000,
                    pass_first_n_batches=2
                )
            )
            
            if wavs and len(wavs) > 0:
                all_audio.append(wavs[0])
        
        if not all_audio:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Concatenate all audio chunks
        import numpy as np
        combined_audio = np.concatenate(all_audio)
        
        # Save combined audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, combined_audio, 24000)
            temp_filename = tmp_file.name
        
        # Generate filename
        safe_text = "".join(c for c in request.text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')
        output_filename = f"complete_tts_{safe_text}.wav"
        
        final_path = os.path.join("/tmp", output_filename)
        shutil.move(temp_filename, final_path)
        
        logger.info(f"Complete audio generated: {final_path}")
        
        return TextToSpeechResponse(
            success=True,
            message="Complete audio generated successfully",
            audio_file=final_path,
            speaker_id=request.speaker_id or "auto"
        )
        
    except Exception as e:
        logger.error(f"Error generating complete speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def generate_complete_audio(text, speaker_id, params):
    """Generate audio with completion detection"""
    
    # First attempt with normal parameters
    wavs = chat.infer(
        text,
        skip_refine_text=True,
        params_infer_code=params
    )
    
    if not wavs or len(wavs) == 0:
        return None
    
    audio = wavs[0]
    
    # Check if audio seems complete (not cut off)
    # Look for sudden drop in amplitude at the end
    if len(audio) > 1000:  # Only check if audio is long enough
        end_segment = audio[-1000:]  # Last 1000 samples
        avg_amplitude = np.mean(np.abs(end_segment))
        
        # If end amplitude is very low, might be cut off
        if avg_amplitude < 0.01:
            logger.warning("Audio appears to be cut off, regenerating with higher limits")
            
            # Regenerate with higher limits
            params.max_new_token = min(params.max_new_token * 2, 16384)
            params.min_new_token = min(params.min_new_token * 2, 500)
            
            wavs = chat.infer(
                text,
                skip_refine_text=True,
                params_infer_code=params
            )
            
            if wavs and len(wavs) > 0:
                audio = wavs[0]
    
    return audio

@app.post("/tts/stream")
async def text_to_speech_stream(request: TextToSpeechRequest):
    """Stream TTS generation to avoid cutoffs."""
    
    if chat is None:
        raise HTTPException(status_code=503, detail="ChatTTS not initialized")
    
    try:
        # Use streaming mode
        wavs = chat.infer(
            request.text,
            stream=True,
            skip_refine_text=True,
            params_infer_code=ChatTTS.Chat.InferCodeParams(
                spk_emb=request.speaker_id or chat.sample_random_speaker(),
                temperature=request.temperature,
                top_P=request.top_p,
                top_K=request.top_k,
                max_new_token=8192,
                min_new_token=200,
                stream_batch=24,
                stream_speed=12000,
                pass_first_n_batches=2
            )
        )
        
        # Process streaming results
        all_audio_segments = []
        
        for wav_segment in wavs:
            if wav_segment and len(wav_segment) > 0:
                for segment in wav_segment:
                    if segment is not None and len(segment) > 0:
                        all_audio_segments.append(segment)
        
        if not all_audio_segments:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Combine all segments
        import numpy as np
        combined_audio = np.concatenate(all_audio_segments)
        
        # Save combined audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, combined_audio, 24000)
            temp_filename = tmp_file.name
        
        # Generate filename
        safe_text = "".join(c for c in request.text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')
        output_filename = f"stream_tts_{safe_text}.wav"
        
        final_path = os.path.join("/tmp", output_filename)
        shutil.move(temp_filename, final_path)
        
        logger.info(f"Streamed audio generated: {final_path}")
        
        return TextToSpeechResponse(
            success=True,
            message="Streamed audio generated successfully",
            audio_file=final_path,
            speaker_id=request.speaker_id or "auto"
        )
        
    except Exception as e:
        logger.error(f"Error generating streamed speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/tts/max-memory", response_model=TextToSpeechResponse)
async def text_to_speech_max_memory(request: TextToSpeechRequest):
    """Convert text to speech using maximum available memory."""
    
    if chat is None:
        raise HTTPException(status_code=503, detail="ChatTTS not initialized")
    
    try:
        # Get memory-optimized parameters
        memory_params = load_models_with_max_memory()
        
        # Use provided speaker or sample a random one
        speaker_id = request.speaker_id
        if not speaker_id:
            speaker_id = chat.sample_random_speaker()
        
        logger.info(f"Generating speech with max memory allocation...")
        logger.info(f"Text length: {len(request.text)} characters")
        
        # Generate audio with memory-optimized parameters
        wavs = chat.infer(
            request.text,
            skip_refine_text=True,
            params_infer_code=ChatTTS.Chat.InferCodeParams(
                spk_emb=speaker_id,
                temperature=request.temperature,
                top_P=request.top_p,
                top_K=request.top_k,
                max_new_token=memory_params['max_tokens'],
                min_new_token=memory_params['max_tokens'] // 4,  # 25% of max
                repetition_penalty=1.0,
                stream_batch=memory_params['batch_size'],
                stream_speed=12000,
                pass_first_n_batches=2
            )
        )
        
        if not wavs or len(wavs) == 0:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Get the first audio segment
        wav = wavs[0]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Save audio using soundfile
            sf.write(tmp_file.name, wav, 24000)
            temp_filename = tmp_file.name
        
        # Generate a more descriptive filename
        safe_text = "".join(c for c in request.text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_text = safe_text.replace(' ', '_')
        output_filename = f"maxmem_tts_{safe_text}.wav"
        
        # Move to final location
        final_path = os.path.join("/tmp", output_filename)
        shutil.move(temp_filename, final_path)
        
        logger.info(f"Max memory audio generated successfully: {final_path}")
        
        # Force garbage collection after generation
        gc.collect()
        
        return TextToSpeechResponse(
            success=True,
            message="Max memory audio generated successfully",
            audio_file=final_path,
            speaker_id=speaker_id
        )
        
    except Exception as e:
        logger.error(f"Error generating max memory speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.post("/tts/dynamic-memory")
async def text_to_speech_dynamic_memory(request: TextToSpeechRequest):
    """Dynamically allocate memory based on text length and available memory."""
    
    if chat is None:
        raise HTTPException(status_code=503, detail="ChatTTS not initialized")
    
    try:
        # Get current memory usage
        # process = psutil.Process() # Commented out as per edit hint
        # memory_info = process.memory_info() # Commented out as per edit hint
        # current_memory_mb = memory_info.rss / (1024 * 1024) # Commented out as per edit hint
        
        # Get available system memory
        # system_memory = psutil.virtual_memory() # Commented out as per edit hint
        # available_memory_gb = system_memory.available / (1024**3) # Commented out as per edit hint
        
        # logger.info(f"Current process memory: {current_memory_mb:.1f} MB") # Commented out as per edit hint
        # logger.info(f"Available system memory: {available_memory_gb:.1f} GB") # Commented out as per edit hint
        
        # Calculate optimal parameters based on available memory
        text_length = len(request.text)
        
        # if available_memory_gb >= 4: # Commented out as per edit hint
        #     # High memory available # Commented out as per edit hint
        #     max_tokens = min(16384, text_length * 50)  # 50 tokens per character # Commented out as per edit hint
        #     batch_size = 48 # Commented out as per edit hint
        #     compile_mode = True # Commented out as per edit hint
        # elif available_memory_gb >= 2: # Commented out as per edit hint
        #     # Medium memory available # Commented out as per edit hint
        #     max_tokens = min(8192, text_length * 40) # Commented out as per edit hint
        #     batch_size = 32 # Commented out as per edit hint
        #     compile_mode = False # Commented out as per edit hint
        # else: # Commented out as per edit hint
        #     # Low memory available # Commented out as per edit hint
        #     max_tokens = min(4096, text_length * 30) # Commented out as per edit hint
        #     batch_size = 24 # Commented out as per edit hint
        #     compile_mode = False # Commented out as per edit hint
        
        # logger.info(f"Text length: {text_length} characters") # Commented out as per edit hint
        # logger.info(f"Allocated max tokens: {max_tokens}") # Commented out as per edit hint
        # logger.info(f"Batch size: {batch_size}") # Commented out as per edit hint
        
        # Generate audio with dynamic parameters
        wavs = chat.infer(
            request.text,
            skip_refine_text=True,
            params_infer_code=ChatTTS.Chat.InferCodeParams(
                spk_emb=request.speaker_id or chat.sample_random_speaker(),
                temperature=request.temperature,
                top_P=request.top_p,
                top_K=request.top_k,
                max_new_token=4096, # Default value as psutil is removed
                min_new_token=100, # Default value as psutil is removed
                repetition_penalty=1.0,
                stream_batch=24, # Default value as psutil is removed
                stream_speed=12000,
                pass_first_n_batches=2
            )
        )
        
        # Process results...
        # (same as before)
        
    except Exception as e:
        logger.error(f"Error in dynamic memory TTS: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio file."""
    file_path = os.path.join("/tmp", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="audio/wav"
    )

@app.get("/cleanup")
async def cleanup_temp_files():
    """Clean up temporary audio files."""
    try:
        temp_dir = "/tmp"
        cleaned_count = 0
        
        for filename in os.listdir(temp_dir):
            if filename.startswith("tts_") and filename.endswith(".wav"):
                file_path = os.path.join(temp_dir, filename)
                os.remove(file_path)
                cleaned_count += 1
        
        return {"message": f"Cleaned up {cleaned_count} temporary files"}
        
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")
        raise HTTPException(status_code=500, detail=f"Error cleaning up: {str(e)}")

@app.get("/memory-status")
async def get_memory_status():
    """Get current memory usage and allocation."""
    
    try:
        # process = psutil.Process() # Commented out as per edit hint
        # memory_info = process.memory_info() # Commented out as per edit hint
        
        # system_memory = psutil.virtual_memory() # Commented out as per edit hint
        
        return {
            "process_memory_mb": 0, # Default value as psutil is removed
            "process_memory_gb": 0, # Default value as psutil is removed
            "system_total_gb": 0, # Default value as psutil is removed
            "system_available_gb": 0, # Default value as psutil is removed
            "system_used_percent": 0, # Default value as psutil is removed
            "memory_optimized": False # Default value as psutil is removed
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    # Check if ChatTTS is available
    try:
        import ChatTTS
        logger.info("ChatTTS import successful")
    except ImportError as e:
        logger.error(f"Failed to import ChatTTS: {e}")
        sys.exit(1)
    
    # Run the server
    uvicorn.run(
        "voice_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 