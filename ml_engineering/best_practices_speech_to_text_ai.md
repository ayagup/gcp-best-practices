# Speech-to-Text AI Best Practices

*Last Updated: January 4, 2026*

## Overview

Speech-to-Text AI converts audio to text using Google's neural network models, supporting 125+ languages with features like speaker diarization and automatic punctuation.

---

## 1. Speech Recognition Best Practices

### Audio Quality

**Best Practices:**
- Use high-quality audio (16kHz or 48kHz sample rate)
- Minimize background noise
- Use appropriate audio encoding
- Test with sample audio first

```python
from google.cloud import speech_v1
import io

def transcribe_audio_file(
    audio_file_path,
    language_code='en-US',
    sample_rate_hertz=16000,
    enable_automatic_punctuation=True
):
    """Transcribe audio file with optimal settings."""
    
    client = speech_v1.SpeechClient()
    
    with io.open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()
    
    audio = speech_v1.RecognitionAudio(content=content)
    
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hertz,
        language_code=language_code,
        enable_automatic_punctuation=enable_automatic_punctuation,
        enable_word_time_offsets=True,
        enable_word_confidence=True,
        model='default',  # or 'phone_call', 'video', 'command_and_search'
        use_enhanced=True,  # Enhanced model for better accuracy
        max_alternatives=1,
    )
    
    print(f"Transcribing audio file: {audio_file_path}")
    
    response = client.recognize(config=config, audio=audio)
    
    # Process results
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        
        print(f"\nTranscript {i + 1}:")
        print(f"  Text: {alternative.transcript}")
        print(f"  Confidence: {alternative.confidence:.2%}")
        
        # Word-level details
        if alternative.words:
            print(f"  Words: {len(alternative.words)}")
            for word_info in alternative.words[:5]:  # Show first 5
                word = word_info.word
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()
                confidence = word_info.confidence
                
                print(f"    '{word}' ({start_time:.2f}s - {end_time:.2f}s): {confidence:.2%}")
    
    return response.results

# Example usage
results = transcribe_audio_file(
    'audio.wav',
    language_code='en-US',
    sample_rate_hertz=16000
)
```

### Streaming Recognition

**Best Practices:**
- Use for real-time transcription
- Handle streaming limits (5 minutes per stream)
- Implement proper buffering
- Handle interim results

```python
def transcribe_streaming(stream_file, language_code='en-US'):
    """Transcribe audio stream in real-time."""
    
    client = speech_v1.SpeechClient()
    
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
        interim_results=True,  # Get interim results
    )
    
    streaming_config = speech_v1.StreamingRecognitionConfig(
        config=config,
        single_utterance=False
    )
    
    def request_generator(stream_file):
        """Generate audio chunks for streaming."""
        with io.open(stream_file, 'rb') as audio_file:
            while True:
                chunk = audio_file.read(4096)
                if not chunk:
                    break
                yield speech_v1.StreamingRecognizeRequest(audio_content=chunk)
    
    requests = request_generator(stream_file)
    
    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests
    )
    
    print("Streaming transcription:")
    
    for response in responses:
        for result in response.results:
            alternative = result.alternatives[0]
            
            if result.is_final:
                print(f"\nFinal: {alternative.transcript}")
            else:
                print(f"Interim: {alternative.transcript}", end='\r')

# Microphone streaming
def transcribe_microphone():
    """Transcribe from microphone in real-time."""
    
    import pyaudio
    
    RATE = 16000
    CHUNK = int(RATE / 10)  # 100ms
    
    client = speech_v1.SpeechClient()
    
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='en-US',
        enable_automatic_punctuation=True,
    )
    
    streaming_config = speech_v1.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )
    
    audio_interface = pyaudio.PyAudio()
    audio_stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    
    def audio_generator():
        """Generate audio chunks from microphone."""
        while True:
            chunk = audio_stream.read(CHUNK)
            yield speech_v1.StreamingRecognizeRequest(audio_content=chunk)
    
    requests = audio_generator()
    
    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests
    )
    
    print("Listening... (Ctrl+C to stop)")
    
    try:
        for response in responses:
            for result in response.results:
                if result.is_final:
                    print(f"\n{result.alternatives[0].transcript}")
    except KeyboardInterrupt:
        audio_stream.stop_stream()
        audio_stream.close()
        audio_interface.terminate()
```

### Speaker Diarization

**Best Practices:**
- Enable for multi-speaker audio
- Set speaker count range
- Use with high-quality audio
- Post-process for accuracy

```python
def transcribe_with_diarization(
    audio_file_path,
    min_speaker_count=2,
    max_speaker_count=6,
    language_code='en-US'
):
    """Transcribe audio with speaker diarization."""
    
    client = speech_v1.SpeechClient()
    
    with io.open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()
    
    audio = speech_v1.RecognitionAudio(content=content)
    
    diarization_config = speech_v1.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=min_speaker_count,
        max_speaker_count=max_speaker_count,
    )
    
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
        diarization_config=diarization_config,
    )
    
    print(f"Transcribing with speaker diarization...")
    
    response = client.recognize(config=config, audio=audio)
    
    # Process diarized results
    result = response.results[-1]
    
    words_info = result.alternatives[0].words
    
    # Group by speaker
    current_speaker = None
    speaker_segments = []
    current_segment = {'speaker': None, 'words': []}
    
    for word_info in words_info:
        if word_info.speaker_tag != current_speaker:
            if current_segment['words']:
                speaker_segments.append(current_segment)
            
            current_speaker = word_info.speaker_tag
            current_segment = {
                'speaker': current_speaker,
                'words': [word_info.word]
            }
        else:
            current_segment['words'].append(word_info.word)
    
    if current_segment['words']:
        speaker_segments.append(current_segment)
    
    # Print formatted transcript
    print("\nTranscript with speakers:\n")
    for segment in speaker_segments:
        speaker = f"Speaker {segment['speaker']}"
        text = ' '.join(segment['words'])
        print(f"{speaker}: {text}\n")
    
    return speaker_segments
```

### Long Audio Files

**Best Practices:**
- Use asynchronous recognition for files > 1 minute
- Store audio in Cloud Storage
- Poll for operation completion
- Handle timeouts appropriately

```python
def transcribe_long_audio(
    gcs_uri,
    language_code='en-US',
    sample_rate_hertz=16000
):
    """Transcribe long audio file from Cloud Storage."""
    
    client = speech_v1.SpeechClient()
    
    audio = speech_v1.RecognitionAudio(uri=gcs_uri)
    
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=sample_rate_hertz,
        language_code=language_code,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        enable_word_confidence=True,
    )
    
    print(f"Starting long-running transcription for: {gcs_uri}")
    
    operation = client.long_running_recognize(config=config, audio=audio)
    
    print("Waiting for operation to complete...")
    response = operation.result(timeout=600)  # 10 minutes timeout
    
    # Save results
    full_transcript = []
    
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        
        print(f"\nSegment {i + 1}:")
        print(f"  Transcript: {alternative.transcript}")
        print(f"  Confidence: {alternative.confidence:.2%}")
        
        full_transcript.append(alternative.transcript)
    
    # Save to file
    with open('transcript.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(full_transcript))
    
    print(f"\nFull transcript saved to transcript.txt")
    
    return response.results

# Example
transcribe_long_audio(
    'gs://my-bucket/long-audio.flac',
    language_code='en-US',
    sample_rate_hertz=16000
)
```

---

## 2. Model Selection

### Choose the Right Model

```python
# Available models and their use cases
SPEECH_MODELS = {
    'default': 'General-purpose, balanced accuracy and speed',
    'command_and_search': 'Short queries, voice commands',
    'phone_call': 'Phone audio (8kHz or 16kHz)',
    'video': 'Video audio with multiple speakers',
    'medical_conversation': 'Medical conversations',
    'medical_dictation': 'Medical dictation',
}

def transcribe_with_model(audio_file, model='default', language_code='en-US'):
    """Transcribe with specific model."""
    
    client = speech_v1.SpeechClient()
    
    with io.open(audio_file, 'rb') as f:
        content = f.read()
    
    audio = speech_v1.RecognitionAudio(content=content)
    
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        model=model,
        use_enhanced=True if model in ['phone_call', 'video'] else False,
    )
    
    response = client.recognize(config=config, audio=audio)
    
    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")
    
    return response.results
```

---

## 3. Language Support

### Multi-Language Recognition

```python
def transcribe_multilingual(audio_file, alternative_languages=None):
    """Transcribe with automatic language detection."""
    
    if alternative_languages is None:
        alternative_languages = ['es-ES', 'fr-FR', 'de-DE']
    
    client = speech_v1.SpeechClient()
    
    with io.open(audio_file, 'rb') as f:
        content = f.read()
    
    audio = speech_v1.RecognitionAudio(content=content)
    
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',  # Primary language
        alternative_language_codes=alternative_languages,
        enable_automatic_punctuation=True,
    )
    
    response = client.recognize(config=config, audio=audio)
    
    for result in response.results:
        alternative = result.alternatives[0]
        detected_language = result.language_code
        
        print(f"Detected language: {detected_language}")
        print(f"Transcript: {alternative.transcript}")
    
    return response.results
```

---

## 4. Cost Optimization

```python
# Use appropriate models to optimize costs
def optimize_transcription_cost(audio_file, audio_length_seconds):
    """Choose cost-effective options."""
    
    client = speech_v1.SpeechClient()
    
    # For short audio < 60s, use sync recognition
    if audio_length_seconds < 60:
        # Cheaper, faster
        with io.open(audio_file, 'rb') as f:
            content = f.read()
        audio = speech_v1.RecognitionAudio(content=content)
    else:
        # Upload to GCS for long audio
        # Use async recognition
        audio = speech_v1.RecognitionAudio(uri=f'gs://bucket/{audio_file}')
    
    # Use standard model if enhanced not needed
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
        use_enhanced=False,  # Standard model is cheaper
        enable_automatic_punctuation=True,
    )
    
    if audio_length_seconds < 60:
        response = client.recognize(config=config, audio=audio)
    else:
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result()
    
    return response.results
```

---

## 5. Quick Reference Checklist

### Setup
- [ ] Enable Speech-to-Text API
- [ ] Configure authentication
- [ ] Choose appropriate audio format
- [ ] Test audio quality
- [ ] Set up Cloud Storage for long audio

### Recognition
- [ ] Select appropriate model
- [ ] Configure language settings
- [ ] Enable punctuation
- [ ] Use speaker diarization if needed
- [ ] Handle streaming limits

### Production
- [ ] Implement error handling
- [ ] Monitor API usage
- [ ] Optimize for cost
- [ ] Cache results when possible
- [ ] Set up monitoring

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
