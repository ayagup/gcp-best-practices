# Text-to-Speech AI Best Practices

*Last Updated: January 4, 2026*

## Overview

Text-to-Speech AI converts text to natural-sounding speech using advanced neural networks, supporting 220+ voices across 40+ languages with customizable speech parameters.

---

## 1. Voice Selection Best Practices

### Choosing the Right Voice

**Voice Types:**
- **Standard**: Basic neural voices
- **WaveNet**: High-quality neural voices
- **Neural2**: Latest generation voices
- **Studio**: Premium broadcast-quality voices
- **Custom Voice**: Trained on your audio

```python
from google.cloud import texttospeech

def list_available_voices(language_code=None):
    """List all available voices with their characteristics."""
    
    client = texttospeech.TextToSpeechClient()
    
    voices = client.list_voices(language_code=language_code)
    
    print(f"Available voices: {len(voices.voices)}\n")
    
    # Group by language
    from collections import defaultdict
    voices_by_language = defaultdict(list)
    
    for voice in voices.voices:
        for language in voice.language_codes:
            voices_by_language[language].append({
                'name': voice.name,
                'gender': texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                'sample_rate': voice.natural_sample_rate_hertz
            })
    
    # Display voices
    for lang, voice_list in sorted(voices_by_language.items()):
        print(f"\n{lang} ({len(voice_list)} voices):")
        for voice in voice_list[:5]:  # Show first 5
            print(f"  - {voice['name']} ({voice['gender']}) @ {voice['sample_rate']}Hz")
    
    return voices.voices

def synthesize_text_basic(text, voice_name='en-US-Neural2-C', language_code='en-US'):
    """Synthesize speech from text with specific voice."""
    
    client = texttospeech.TextToSpeechClient()
    
    # Set the text input
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )
    
    # Select the type of audio file
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0,  # Normal speed
        pitch=0.0,  # Normal pitch
        volume_gain_db=0.0,  # Normal volume
    )
    
    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    
    # Save the audio
    output_filename = 'output.mp3'
    with open(output_filename, 'wb') as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "{output_filename}"')
    
    return response.audio_content

# Example usage
text = "Hello! Welcome to Google Cloud Text-to-Speech API."
synthesize_text_basic(text, voice_name='en-US-Neural2-F')
```

### Voice Comparison

```python
def compare_voice_types(text):
    """Compare different voice types for the same text."""
    
    client = texttospeech.TextToSpeechClient()
    
    voice_types = [
        ('en-US-Standard-C', 'Standard'),
        ('en-US-Wavenet-C', 'WaveNet'),
        ('en-US-Neural2-C', 'Neural2'),
        ('en-US-Studio-M', 'Studio'),
    ]
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    for voice_name, voice_type in voice_types:
        voice = texttospeech.VoiceSelectionParams(
            name=voice_name,
            language_code='en-US'
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        try:
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            filename = f'output_{voice_type.lower()}.mp3'
            with open(filename, 'wb') as out:
                out.write(response.audio_content)
            
            print(f"✓ {voice_type}: {filename}")
        except Exception as e:
            print(f"✗ {voice_type}: {e}")
```

---

## 2. SSML (Speech Synthesis Markup Language)

### Basic SSML

```python
def synthesize_ssml(ssml_text, voice_name='en-US-Neural2-C'):
    """Synthesize speech using SSML for advanced control."""
    
    client = texttospeech.TextToSpeechClient()
    
    # Set the SSML input
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        name=voice_name
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    
    with open('ssml_output.mp3', 'wb') as out:
        out.write(response.audio_content)
        print('SSML audio content written to file "ssml_output.mp3"')
    
    return response.audio_content

# Example SSML with various features
ssml_example = """
<speak>
    Hello! <break time="500ms"/>
    
    <prosody rate="slow" pitch="-2st">
        This text is spoken slowly and at a lower pitch.
    </prosody>
    
    <break time="1s"/>
    
    <prosody rate="fast" pitch="+2st">
        This text is spoken quickly and at a higher pitch.
    </prosody>
    
    <break time="1s"/>
    
    <emphasis level="strong">This is very important!</emphasis>
    
    <break time="500ms"/>
    
    The price is <say-as interpret-as="currency" language="en-US">$42.99</say-as>.
    
    <break time="500ms"/>
    
    Call us at <say-as interpret-as="telephone">1-800-555-0123</say-as>.
    
    <break time="500ms"/>
    
    The date is <say-as interpret-as="date" format="mdy">10/15/2024</say-as>.
</speak>
"""

synthesize_ssml(ssml_example)
```

### Advanced SSML Features

```python
def create_advanced_ssml():
    """Create SSML with advanced features."""
    
    ssml = """
    <speak>
        <!-- Paragraphs and breaks -->
        <p>
            <s>This is the first sentence.</s>
            <s>This is the second sentence.</s>
        </p>
        
        <break time="2s"/>
        
        <!-- Prosody control -->
        <p>
            Normal speech. 
            <prosody volume="x-loud">Loud speech.</prosody>
            <prosody volume="x-soft">Soft speech.</prosody>
        </p>
        
        <break time="1s"/>
        
        <!-- Emphasis -->
        <p>
            <emphasis level="moderate">Moderate emphasis.</emphasis>
            <emphasis level="strong">Strong emphasis!</emphasis>
        </p>
        
        <break time="1s"/>
        
        <!-- Say-as for numbers and dates -->
        <p>
            Phone: <say-as interpret-as="telephone">555-1234</say-as>
            <break time="500ms"/>
            Date: <say-as interpret-as="date" format="ymd">2024-10-15</say-as>
            <break time="500ms"/>
            Ordinal: <say-as interpret-as="ordinal">1</say-as>
            <break time="500ms"/>
            Cardinal: <say-as interpret-as="cardinal">12345</say-as>
        </p>
        
        <break time="1s"/>
        
        <!-- Sub (pronunciation) -->
        <p>
            <sub alias="World Wide Web Consortium">W3C</sub>
        </p>
        
        <break time="1s"/>
        
        <!-- Phoneme for pronunciation -->
        <p>
            You say <phoneme alphabet="ipa" ph="pəˈkɑːn">pecan</phoneme>.
            I say <phoneme alphabet="ipa" ph="ˈpi.kæn">pecan</phoneme>.
        </p>
    </speak>
    """
    
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        name='en-US-Neural2-F'
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    
    with open('advanced_ssml.mp3', 'wb') as out:
        out.write(response.audio_content)
    
    print('Advanced SSML audio saved')
```

---

## 3. Audio Configuration

### Audio Formats and Quality

```python
def synthesize_with_audio_config(text, audio_format='mp3', sample_rate=24000):
    """Synthesize with specific audio configuration."""
    
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        name='en-US-Neural2-C'
    )
    
    # Audio encoding options
    encoding_map = {
        'mp3': texttospeech.AudioEncoding.MP3,
        'wav': texttospeech.AudioEncoding.LINEAR16,
        'ogg': texttospeech.AudioEncoding.OGG_OPUS,
        'mulaw': texttospeech.AudioEncoding.MULAW,
    }
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=encoding_map.get(audio_format, texttospeech.AudioEncoding.MP3),
        speaking_rate=1.0,  # 0.25 to 4.0
        pitch=0.0,  # -20.0 to 20.0
        volume_gain_db=0.0,  # -96.0 to 16.0
        sample_rate_hertz=sample_rate,
        effects_profile_id=['small-bluetooth-speaker-class-device'],  # Audio profile
    )
    
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    
    extension = audio_format if audio_format != 'wav' else 'wav'
    filename = f'output.{extension}'
    
    with open(filename, 'wb') as out:
        out.write(response.audio_content)
        print(f'Audio saved: {filename}')
    
    return response.audio_content

# Audio profiles for different devices
AUDIO_PROFILES = [
    'wearable-class-device',
    'handset-class-device',
    'headphone-class-device',
    'small-bluetooth-speaker-class-device',
    'medium-bluetooth-speaker-class-device',
    'large-home-entertainment-class-device',
    'large-automotive-class-device',
    'telephony-class-application',
]
```

### Speed and Pitch Control

```python
def create_variations(text, base_voice='en-US-Neural2-C'):
    """Create audio variations with different speeds and pitches."""
    
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        name=base_voice
    )
    
    variations = [
        ('normal', 1.0, 0.0),
        ('slow', 0.75, 0.0),
        ('fast', 1.25, 0.0),
        ('low_pitch', 1.0, -5.0),
        ('high_pitch', 1.0, 5.0),
    ]
    
    for name, rate, pitch in variations:
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=rate,
            pitch=pitch
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        filename = f'variation_{name}.mp3'
        with open(filename, 'wb') as out:
            out.write(response.audio_content)
        
        print(f'Created: {filename} (rate={rate}, pitch={pitch})')
```

---

## 4. Batch Processing and Caching

```python
import hashlib
import os
from pathlib import Path

class TTSCache:
    """Cache for Text-to-Speech results."""
    
    def __init__(self, cache_dir='tts_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.client = texttospeech.TextToSpeechClient()
    
    def _get_cache_key(self, text, voice_name, audio_config_params):
        """Generate cache key from parameters."""
        config_str = f"{text}:{voice_name}:{audio_config_params}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def synthesize_cached(
        self,
        text,
        voice_name='en-US-Neural2-C',
        language_code='en-US',
        speaking_rate=1.0,
        pitch=0.0
    ):
        """Synthesize with caching."""
        
        config_params = f"{speaking_rate}:{pitch}"
        cache_key = self._get_cache_key(text, voice_name, config_params)
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        
        # Check cache
        if cache_file.exists():
            print(f"Cache hit: {cache_key}")
            with open(cache_file, 'rb') as f:
                return f.read()
        
        # Synthesize
        print(f"Cache miss, synthesizing: {text[:50]}...")
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch
        )
        
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            f.write(response.audio_content)
        
        return response.audio_content

def batch_synthesize(texts, voice_name='en-US-Neural2-C'):
    """Synthesize multiple texts efficiently."""
    
    cache = TTSCache()
    
    results = []
    
    for i, text in enumerate(texts, 1):
        print(f"\nProcessing {i}/{len(texts)}")
        
        audio_content = cache.synthesize_cached(
            text=text,
            voice_name=voice_name
        )
        
        results.append({
            'text': text,
            'audio_length': len(audio_content),
            'filename': f'batch_{i}.mp3'
        })
        
        # Save individual file
        with open(f'batch_{i}.mp3', 'wb') as f:
            f.write(audio_content)
    
    print(f"\nBatch synthesis complete: {len(results)} files")
    return results

# Example usage
texts_to_synthesize = [
    "Welcome to our service.",
    "Thank you for calling.",
    "Your order has been confirmed.",
    "Please hold while we connect you.",
]

batch_results = batch_synthesize(texts_to_synthesize)
```

---

## 5. Multi-Language Support

```python
def synthesize_multilingual(translations):
    """Synthesize text in multiple languages."""
    
    client = texttospeech.TextToSpeechClient()
    
    # Language to voice mapping
    language_voices = {
        'en-US': 'en-US-Neural2-C',
        'es-ES': 'es-ES-Neural2-A',
        'fr-FR': 'fr-FR-Neural2-A',
        'de-DE': 'de-DE-Neural2-A',
        'ja-JP': 'ja-JP-Neural2-B',
        'zh-CN': 'cmn-CN-Neural2-A',
    }
    
    results = []
    
    for lang_code, text in translations.items():
        voice_name = language_voices.get(lang_code)
        
        if not voice_name:
            print(f"No voice configured for {lang_code}")
            continue
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        filename = f'output_{lang_code}.mp3'
        with open(filename, 'wb') as out:
            out.write(response.audio_content)
        
        results.append({
            'language': lang_code,
            'text': text,
            'filename': filename
        })
        
        print(f"✓ {lang_code}: {filename}")
    
    return results

# Example
translations = {
    'en-US': 'Hello, welcome to our service.',
    'es-ES': 'Hola, bienvenido a nuestro servicio.',
    'fr-FR': 'Bonjour, bienvenue dans notre service.',
    'de-DE': 'Hallo, willkommen bei unserem Service.',
}

synthesize_multilingual(translations)
```

---

## 6. Error Handling and Retry Logic

```python
import time
from google.api_core import retry, exceptions

def synthesize_with_retry(text, max_retries=3):
    """Synthesize with automatic retry on failure."""
    
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        name='en-US-Neural2-C'
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    # Retry configuration
    retry_config = retry.Retry(
        initial=1.0,  # Initial delay
        maximum=60.0,  # Maximum delay
        multiplier=2.0,  # Exponential backoff
        deadline=300.0,  # 5 minutes total
        predicate=retry.if_exception_type(
            exceptions.ServiceUnavailable,
            exceptions.DeadlineExceeded,
            exceptions.ResourceExhausted,
        ),
    )
    
    try:
        response = client.synthesize_speech(
            request={
                'input': synthesis_input,
                'voice': voice,
                'audio_config': audio_config
            },
            retry=retry_config
        )
        
        print("Synthesis successful")
        return response.audio_content
        
    except exceptions.InvalidArgument as e:
        print(f"Invalid request: {e}")
        raise
    except exceptions.ResourceExhausted as e:
        print(f"Quota exceeded: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Enable Text-to-Speech API
- [ ] Configure authentication
- [ ] Choose voice type (Standard/WaveNet/Neural2/Studio)
- [ ] Test voice quality
- [ ] Set up caching if needed

### Synthesis
- [ ] Select appropriate voice
- [ ] Configure audio format
- [ ] Use SSML for advanced control
- [ ] Set speaking rate and pitch
- [ ] Choose audio profile for device

### Production
- [ ] Implement caching
- [ ] Handle errors gracefully
- [ ] Monitor API usage and quotas
- [ ] Optimize for cost
- [ ] Test across languages

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
