# Translation AI Best Practices

*Last Updated: January 4, 2026*

## Overview

Translation AI provides fast, dynamic translation of text and websites into over 135 languages using neural machine translation. This document covers best practices for the Translation API and AutoML Translation.

---

## 1. Translation API Best Practices

### Basic Translation

**Best Practices:**
- Detect language automatically when unknown
- Use appropriate API version (v2 vs v3)
- Handle HTML content correctly
- Batch requests for efficiency

```python
from google.cloud import translate_v3
from google.cloud import translate_v2

def translate_text_v3(
    project_id,
    text,
    target_language,
    source_language=None,
    location='global'
):
    """Translate text using Translation API v3."""
    
    client = translate_v3.TranslationServiceClient()
    
    parent = f"projects/{project_id}/locations/{location}"
    
    # Detect language if not specified
    if not source_language:
        source_language = detect_language(project_id, text)
        print(f"Detected language: {source_language}")
    
    response = client.translate_text(
        request={
            'parent': parent,
            'contents': [text],
            'mime_type': 'text/plain',
            'source_language_code': source_language,
            'target_language_code': target_language,
        }
    )
    
    for translation in response.translations:
        print(f"Original ({source_language}): {text}")
        print(f"Translated ({target_language}): {translation.translated_text}")
    
    return response.translations[0].translated_text

def detect_language(project_id, text, location='global'):
    """Detect language of text."""
    
    client = translate_v3.TranslationServiceClient()
    
    parent = f"projects/{project_id}/locations/{location}"
    
    response = client.detect_language(
        request={
            'parent': parent,
            'content': text,
            'mime_type': 'text/plain',
        }
    )
    
    for language in response.languages:
        print(f"Language: {language.language_code} (confidence: {language.confidence:.2f})")
    
    return response.languages[0].language_code

# Example usage
translated = translate_text_v3(
    'my-project',
    'Hello, how are you?',
    'es',  # Spanish
    'en'   # English
)
```

### Batch Translation

**Best Practices:**
- Use batch API for large documents
- Store in Cloud Storage
- Monitor translation jobs
- Handle long-running operations

```python
def batch_translate_text(
    project_id,
    input_uri,
    output_uri,
    source_language,
    target_languages,
    location='us-central1'
):
    """Batch translate documents from Cloud Storage."""
    
    client = translate_v3.TranslationServiceClient()
    
    parent = f"projects/{project_id}/locations/{location}"
    
    # Input configuration
    input_configs = [{
        'gcs_source': {'input_uri': input_uri},
        'mime_type': 'text/plain'
    }]
    
    # Output configuration
    output_config = {
        'gcs_destination': {'output_uri_prefix': output_uri}
    }
    
    # Start batch translation
    operation = client.batch_translate_text(
        request={
            'parent': parent,
            'source_language_code': source_language,
            'target_language_codes': target_languages,
            'input_configs': input_configs,
            'output_config': output_config,
        }
    )
    
    print(f"Batch translation started: {operation.operation.name}")
    print("Waiting for operation to complete...")
    
    response = operation.result(timeout=300)
    
    print(f"\nTotal Characters: {response.total_characters}")
    print(f"Translated Characters: {response.translated_characters}")
    print(f"Output written to: {output_uri}")
    
    return response

# Example
batch_translate_text(
    'my-project',
    'gs://my-bucket/documents/*.txt',
    'gs://my-bucket/translations/',
    'en',
    ['es', 'fr', 'de']
)
```

### Glossaries for Consistent Translation

**Best Practices:**
- Create glossaries for domain-specific terms
- Use for brand names and technical terms
- Update glossaries regularly
- Version control glossaries

```python
def create_glossary(
    project_id,
    glossary_id,
    glossary_uri,
    source_language,
    target_language,
    location='us-central1'
):
    """Create translation glossary."""
    
    client = translate_v3.TranslationServiceClient()
    
    parent = f"projects/{project_id}/locations/{location}"
    glossary_name = f"{parent}/glossaries/{glossary_id}"
    
    language_pair = {
        'source_language_code': source_language,
        'target_language_code': target_language
    }
    
    glossary = {
        'name': glossary_name,
        'language_pair': language_pair,
        'input_config': {
            'gcs_source': {'input_uri': glossary_uri}
        }
    }
    
    operation = client.create_glossary(
        parent=parent,
        glossary=glossary
    )
    
    print(f"Creating glossary: {glossary_name}")
    result = operation.result(timeout=90)
    
    print(f"Glossary created: {result.name}")
    print(f"Entry count: {result.entry_count}")
    
    return result

def translate_with_glossary(
    project_id,
    text,
    source_language,
    target_language,
    glossary_id,
    location='us-central1'
):
    """Translate text using glossary."""
    
    client = translate_v3.TranslationServiceClient()
    
    parent = f"projects/{project_id}/locations/{location}"
    glossary_path = f"{parent}/glossaries/{glossary_id}"
    
    glossary_config = {
        'glossary': glossary_path
    }
    
    response = client.translate_text(
        request={
            'parent': parent,
            'contents': [text],
            'mime_type': 'text/plain',
            'source_language_code': source_language,
            'target_language_code': target_language,
            'glossary_config': glossary_config,
        }
    )
    
    for translation in response.glossary_translations:
        print(f"Translated with glossary: {translation.translated_text}")
    
    return response.glossary_translations[0].translated_text

# Glossary CSV format:
# source_term,target_term
# Google Cloud,Google Cloud
# Machine Learning,Aprendizaje Automático
```

### HTML Translation

**Best Practices:**
- Preserve HTML structure
- Handle attributes correctly
- Maintain links and formatting
- Use appropriate MIME type

```python
def translate_html(project_id, html_content, target_language, source_language='en'):
    """Translate HTML content while preserving structure."""
    
    client = translate_v3.TranslationServiceClient()
    
    parent = f"projects/{project_id}/locations/global"
    
    response = client.translate_text(
        request={
            'parent': parent,
            'contents': [html_content],
            'mime_type': 'text/html',  # Important: specify HTML
            'source_language_code': source_language,
            'target_language_code': target_language,
        }
    )
    
    translated_html = response.translations[0].translated_text
    
    print(f"Original HTML:\n{html_content}\n")
    print(f"Translated HTML:\n{translated_html}")
    
    return translated_html

# Example
html = '''
<html>
<head><title>Welcome to Our Site</title></head>
<body>
    <h1>Hello World</h1>
    <p>This is a <strong>test</strong> paragraph.</p>
    <a href="https://example.com">Learn more</a>
</body>
</html>
'''

translated_html = translate_html('my-project', html, 'es')
```

---

## 2. AutoML Translation Best Practices

### Custom Model Training

**Best Practices:**
- Provide domain-specific training data
- Use parallel text pairs
- Include context and variations
- Minimum 1000 sentence pairs per language

```python
from google.cloud import aiplatform

def create_translation_dataset(
    project_id,
    location,
    display_name,
    source_language,
    target_language,
    gcs_csv_uri
):
    """Create AutoML Translation dataset."""
    
    aiplatform.init(project=project_id, location=location)
    
    # CSV format: source_text,target_text
    dataset = aiplatform.TextDataset.create(
        display_name=display_name,
        gcs_source=gcs_csv_uri,
        import_schema_uri=aiplatform.schema.dataset.ioformat.text.translation,
        sync=True,
    )
    
    print(f"Dataset created: {dataset.resource_name}")
    
    return dataset

def train_automl_translation_model(
    dataset,
    model_display_name,
    source_language,
    target_language
):
    """Train custom translation model."""
    
    job = aiplatform.AutoMLTextTrainingJob(
        display_name=f"{model_display_name}-training",
        prediction_type="translation",
    )
    
    model = job.run(
        dataset=dataset,
        model_display_name=model_display_name,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        source_language=source_language,
        target_language=target_language,
        sync=True,
    )
    
    print(f"Model trained: {model.resource_name}")
    
    return model
```

---

## 3. Multi-Language Support

### Language Detection

```python
def detect_and_translate(project_id, texts, target_language):
    """Detect language and translate multiple texts."""
    
    client = translate_v3.TranslationServiceClient()
    parent = f"projects/{project_id}/locations/global"
    
    results = []
    
    for text in texts:
        # Detect language
        detect_response = client.detect_language(
            request={
                'parent': parent,
                'content': text,
                'mime_type': 'text/plain',
            }
        )
        
        source_lang = detect_response.languages[0].language_code
        confidence = detect_response.languages[0].confidence
        
        # Skip if already in target language
        if source_lang == target_language:
            results.append({
                'original': text,
                'translated': text,
                'source_language': source_lang,
                'skipped': True
            })
            continue
        
        # Translate
        translate_response = client.translate_text(
            request={
                'parent': parent,
                'contents': [text],
                'mime_type': 'text/plain',
                'source_language_code': source_lang,
                'target_language_code': target_language,
            }
        )
        
        results.append({
            'original': text,
            'translated': translate_response.translations[0].translated_text,
            'source_language': source_lang,
            'confidence': confidence,
            'skipped': False
        })
    
    return results

# Example: Translate multilingual content
texts = [
    "Hello, how are you?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?",
    "こんにちは、お元気ですか？"
]

translations = detect_and_translate('my-project', texts, 'en')

for result in translations:
    print(f"Source ({result['source_language']}): {result['original']}")
    print(f"English: {result['translated']}\n")
```

---

## 4. Cost Optimization

### Caching and Optimization

```python
import hashlib
from datetime import datetime, timedelta

class TranslationCache:
    """Cache for translated content."""
    
    def __init__(self, ttl_hours=24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def get_key(self, text, source_lang, target_lang):
        """Generate cache key."""
        content = f"{text}:{source_lang}:{target_lang}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text, source_lang, target_lang):
        """Get cached translation."""
        key = self.get_key(text, source_lang, target_lang)
        
        if key in self.cache:
            translation, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return translation
            else:
                del self.cache[key]
        
        return None
    
    def set(self, text, source_lang, target_lang, translation):
        """Cache translation."""
        key = self.get_key(text, source_lang, target_lang)
        self.cache[key] = (translation, datetime.now())

# Usage
translation_cache = TranslationCache(ttl_hours=24)

def translate_with_cache(project_id, text, target_lang, source_lang='en'):
    """Translate with caching."""
    
    # Check cache
    cached = translation_cache.get(text, source_lang, target_lang)
    if cached:
        print("Using cached translation")
        return cached
    
    # Translate
    translation = translate_text_v3(project_id, text, target_lang, source_lang)
    
    # Cache result
    translation_cache.set(text, source_lang, target_lang, translation)
    
    return translation
```

---

## 5. Error Handling

```python
from google.api_core import exceptions, retry

@retry.Retry(predicate=retry.if_exception_type(
    exceptions.ResourceExhausted,
    exceptions.ServiceUnavailable,
))
def translate_with_retry(project_id, text, target_language):
    """Translate with automatic retry."""
    
    try:
        return translate_text_v3(project_id, text, target_language)
    
    except exceptions.InvalidArgument as e:
        print(f"Invalid argument: {e}")
        raise
    
    except exceptions.NotFound as e:
        print(f"Resource not found: {e}")
        raise
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

---

## 6. Quick Reference Checklist

### Setup
- [ ] Enable Translation API
- [ ] Configure authentication
- [ ] Set up IAM roles
- [ ] Choose API version (v2/v3)
- [ ] Configure quotas

### Translation
- [ ] Detect language when unknown
- [ ] Use glossaries for consistency
- [ ] Handle HTML correctly
- [ ] Batch large volumes
- [ ] Cache common translations

### Production
- [ ] Implement error handling
- [ ] Monitor translation quality
- [ ] Track API usage
- [ ] Set up alerting
- [ ] Optimize costs

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
