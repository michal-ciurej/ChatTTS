#!/usr/bin/env python3
"""
Simple client to test the ChatTTS Voice API.
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_speakers():
    """Test getting available speakers."""
    try:
        response = requests.get(f"{API_BASE_URL}/speakers")
        print(f"Speakers endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Available speakers: {data['speakers']}")
            return data['speakers'][0] if data['speakers'] else None
        return None
    except Exception as e:
        print(f"Speakers endpoint failed: {e}")
        return None

def test_tts(text, speaker_id=None):
    """Test text-to-speech conversion."""
    try:
        payload = {
            "text": text,
            "temperature": 0.3,
            "top_p": 0.7,
            "top_k": 20,
            "skip_refine_text": True
        }
        
        if speaker_id:
            payload["speaker_id"] = speaker_id
        
        print(f"Sending TTS request: {payload}")
        
        response = requests.post(f"{API_BASE_URL}/tts", json=payload)
        print(f"TTS endpoint: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"TTS response: {data}")
            
            # Download the audio file
            if data.get("audio_file"):
                filename = data["audio_file"].split("/")[-1]
                download_url = f"{API_BASE_URL}/download/{filename}"
                print(f"Download URL: {download_url}")
                
                # Download the file
                audio_response = requests.get(download_url)
                if audio_response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(audio_response.content)
                    print(f"Audio downloaded to: {filename}")
                else:
                    print(f"Failed to download audio: {audio_response.status_code}")
            
            return data
        else:
            print(f"TTS failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"TTS request failed: {e}")
        return None

def main():
    """Main test function."""
    print("Testing ChatTTS Voice API...")
    
    # Test health
    if not test_health():
        print("API is not healthy, exiting.")
        return
    
    # Wait a bit for models to load
    print("Waiting for models to load...")
    time.sleep(5)
    
    # Test health again
    if not test_health():
        print("API is still not healthy after waiting.")
        return
    
    # Test speakers
    speaker_id = test_speakers()
    
    # Test TTS
    test_texts = [
        "Hello, this is a test of the ChatTTS voice API.",
        "The quick brown fox jumps over the lazy dog.",
        "Welcome to the world of text-to-speech technology."
    ]
    
    for text in test_texts:
        print(f"\n--- Testing TTS with: {text} ---")
        result = test_tts(text, speaker_id)
        if result:
            print("TTS test successful!")
        else:
            print("TTS test failed!")
        
        # Wait between requests
        time.sleep(2)
    
    print("\n--- Testing cleanup ---")
    try:
        response = requests.get(f"{API_BASE_URL}/cleanup")
        print(f"Cleanup: {response.status_code}")
        if response.status_code == 200:
            print(f"Cleanup response: {response.json()}")
    except Exception as e:
        print(f"Cleanup failed: {e}")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main() 