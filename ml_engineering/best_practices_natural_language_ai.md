# Natural Language AI Best Practices

*Last Updated: January 4, 2026*

## Overview

Natural Language AI provides pre-trained and custom machine learning models to extract insights from text and understand natural language. This document covers best practices for using the Natural Language API, AutoML Natural Language, and Healthcare Natural Language API.

---

## 1. Natural Language API Best Practices

### Text Preparation

**Best Practices:**
- Ensure text encoding is UTF-8
- Keep text length under API limits (1MB per request)
- Clean and preprocess text appropriately
- Consider language-specific tokenization

```python
from google.cloud import language_v1
import html

def prepare_text_for_analysis(text, max_length=1000000):
    """Prepare text for Natural Language API."""
    
    # Remove HTML entities
    cleaned_text = html.unescape(text)
    
    # Remove excessive whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Check length (API limit: 1MB ~ 1 million characters)
    if len(cleaned_text.encode('utf-8')) > max_length:
        # Truncate if too long
        while len(cleaned_text.encode('utf-8')) > max_length:
            cleaned_text = cleaned_text[:len(cleaned_text)-1000]
        print(f"⚠️  Text truncated to {len(cleaned_text)} characters")
    
    return cleaned_text

# Example
raw_text = """
    This is a sample text with   excessive    whitespace
    and HTML entities like &amp; &lt; &gt;
"""
clean_text = prepare_text_for_analysis(raw_text)
print(f"Cleaned text: {clean_text}")
```

### Entity Recognition

**Best Practices:**
- Use entity salience to identify key entities
- Leverage entity metadata for additional context
- Handle entity types appropriately
- Combine with entity sentiment for deeper insights

```python
def analyze_entities_comprehensive(text, language='en'):
    """Analyze entities with comprehensive information."""
    
    client = language_v1.LanguageServiceClient()
    
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language=language
    )
    
    # Entity analysis
    response = client.analyze_entities(
        request={'document': document, 'encoding_type': 'UTF8'}
    )
    
    print(f"Found {len(response.entities)} entities\n")
    
    # Sort by salience (importance)
    entities_sorted = sorted(
        response.entities,
        key=lambda x: x.salience,
        reverse=True
    )
    
    entity_info = []
    for entity in entities_sorted:
        print(f"Entity: {entity.name}")
        print(f"  Type: {language_v1.Entity.Type(entity.type_).name}")
        print(f"  Salience: {entity.salience:.3f}")
        
        # Metadata (e.g., Wikipedia URL)
        if entity.metadata:
            print(f"  Metadata: {entity.metadata}")
        
        # Mentions in text
        print(f"  Mentions: {len(entity.mentions)}")
        for mention in entity.mentions[:3]:  # Show first 3
            print(f"    - '{mention.text.content}' (Type: {language_v1.EntityMention.Type(mention.type_).name})")
        
        print()
        
        entity_info.append({
            'name': entity.name,
            'type': language_v1.Entity.Type(entity.type_).name,
            'salience': entity.salience,
            'metadata': dict(entity.metadata),
            'mentions': [m.text.content for m in entity.mentions]
        })
    
    return entity_info

# Example usage
text = """
Google Cloud Platform, often abbreviated as GCP, is a suite of cloud computing 
services that runs on the same infrastructure that Google uses internally for 
its end-user products. Sundar Pichai is the CEO of Google and Alphabet Inc.
"""

entities = analyze_entities_comprehensive(text)

# Filter entities by type
def filter_entities_by_type(entities, entity_type):
    """Filter entities by specific type."""
    return [e for e in entities if e['type'] == entity_type]

# Get only people
people = filter_entities_by_type(entities, 'PERSON')
print(f"People mentioned: {[p['name'] for p in people]}")
```

### Sentiment Analysis

**Best Practices:**
- Analyze document-level and sentence-level sentiment
- Consider magnitude alongside score
- Handle mixed sentiment appropriately
- Use entity sentiment for aspect-based analysis

```python
def analyze_sentiment_detailed(text, language='en'):
    """Analyze sentiment with detailed breakdown."""
    
    client = language_v1.LanguageServiceClient()
    
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language=language
    )
    
    # Sentiment analysis
    response = client.analyze_sentiment(
        request={'document': document, 'encoding_type': 'UTF8'}
    )
    
    sentiment = response.document_sentiment
    
    print("Document Sentiment:")
    print(f"  Score: {sentiment.score:.3f} (range: -1.0 to 1.0)")
    print(f"  Magnitude: {sentiment.magnitude:.3f} (range: 0.0 to +inf)")
    
    # Interpret sentiment
    if sentiment.score > 0.25:
        sentiment_label = "Positive"
    elif sentiment.score < -0.25:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    # Magnitude indicates strength
    if sentiment.magnitude > 3.0:
        strength = "Strong"
    elif sentiment.magnitude > 1.0:
        strength = "Moderate"
    else:
        strength = "Weak"
    
    print(f"  Overall: {strength} {sentiment_label} sentiment\n")
    
    # Sentence-level sentiment
    print("Sentence-level Sentiment:")
    for idx, sentence in enumerate(response.sentences):
        sent = sentence.sentiment
        
        print(f"  Sentence {idx + 1}:")
        print(f"    Text: '{sentence.text.content}'")
        print(f"    Score: {sent.score:.3f}, Magnitude: {sent.magnitude:.3f}")
    
    return {
        'document_score': sentiment.score,
        'document_magnitude': sentiment.magnitude,
        'label': sentiment_label,
        'strength': strength,
        'sentences': [
            {
                'text': s.text.content,
                'score': s.sentiment.score,
                'magnitude': s.sentiment.magnitude
            }
            for s in response.sentences
        ]
    }

# Entity sentiment analysis
def analyze_entity_sentiment(text, language='en'):
    """Analyze sentiment for specific entities."""
    
    client = language_v1.LanguageServiceClient()
    
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language=language
    )
    
    response = client.analyze_entity_sentiment(
        request={'document': document, 'encoding_type': 'UTF8'}
    )
    
    print("Entity Sentiment Analysis:\n")
    
    entity_sentiments = []
    for entity in response.entities:
        print(f"Entity: {entity.name}")
        print(f"  Type: {language_v1.Entity.Type(entity.type_).name}")
        print(f"  Sentiment Score: {entity.sentiment.score:.3f}")
        print(f"  Sentiment Magnitude: {entity.sentiment.magnitude:.3f}")
        print(f"  Salience: {entity.salience:.3f}\n")
        
        entity_sentiments.append({
            'entity': entity.name,
            'type': language_v1.Entity.Type(entity.type_).name,
            'sentiment_score': entity.sentiment.score,
            'sentiment_magnitude': entity.sentiment.magnitude,
            'salience': entity.salience
        })
    
    return entity_sentiments

# Example: Product review analysis
review = """
The Google Pixel 8 Pro has an amazing camera and excellent battery life. 
However, the price is quite high and the design feels a bit generic. 
Overall, I'm very satisfied with the performance.
"""

sentiment_result = analyze_sentiment_detailed(review)
entity_sentiments = analyze_entity_sentiment(review)
```

### Content Classification

**Best Practices:**
- Use for content categorization
- Set confidence thresholds
- Combine with custom labels for domain-specific classification
- Handle multiple categories appropriately

```python
def classify_content(text, confidence_threshold=0.5):
    """Classify text content into categories."""
    
    client = language_v1.LanguageServiceClient()
    
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT
    )
    
    response = client.classify_text(request={'document': document})
    
    print(f"Content Classification:\n")
    
    if not response.categories:
        print("No categories found (text may be too short or ambiguous)")
        return []
    
    categories = []
    for category in response.categories:
        if category.confidence >= confidence_threshold:
            print(f"Category: {category.name}")
            print(f"  Confidence: {category.confidence:.3f}\n")
            
            categories.append({
                'name': category.name,
                'confidence': category.confidence
            })
    
    return categories

# Moderate content using classification
def moderate_content_classification(text):
    """Use classification for content moderation."""
    
    categories = classify_content(text, confidence_threshold=0.6)
    
    # Flag potentially sensitive categories
    sensitive_categories = [
        '/Adult',
        '/Sensitive Subjects',
        '/Law & Government/Legal'
    ]
    
    flagged = []
    for category in categories:
        for sensitive in sensitive_categories:
            if category['name'].startswith(sensitive):
                flagged.append(category)
    
    if flagged:
        print("⚠️  Content flagged for review:")
        for cat in flagged:
            print(f"  - {cat['name']} ({cat['confidence']:.2f})")
        return False
    
    return True

# Example: News article classification
article = """
The Federal Reserve announced today that it will maintain interest rates at current 
levels. Economic analysts suggest this decision reflects confidence in the current 
economic recovery while remaining cautious about inflation risks. Stock markets 
responded positively to the announcement.
"""

categories = classify_content(article)
```

### Syntax Analysis

**Best Practices:**
- Use for advanced NLP tasks
- Extract parts of speech
- Analyze sentence structure
- Build dependency trees for understanding

```python
def analyze_syntax(text, language='en'):
    """Perform detailed syntax analysis."""
    
    client = language_v1.LanguageServiceClient()
    
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language=language
    )
    
    response = client.analyze_syntax(
        request={'document': document, 'encoding_type': 'UTF8'}
    )
    
    print("Syntax Analysis:\n")
    
    # Token analysis
    print(f"Total tokens: {len(response.tokens)}\n")
    
    # Count parts of speech
    pos_counts = {}
    for token in response.tokens:
        pos = language_v1.PartOfSpeech.Tag(token.part_of_speech.tag).name
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    print("Parts of Speech Distribution:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pos}: {count}")
    
    print("\nToken Details (first 10):")
    for idx, token in enumerate(response.tokens[:10]):
        print(f"  {idx + 1}. '{token.text.content}'")
        print(f"     POS: {language_v1.PartOfSpeech.Tag(token.part_of_speech.tag).name}")
        print(f"     Lemma: {token.lemma}")
        print(f"     Dependency: {language_v1.DependencyEdge.Label(token.dependency_edge.label).name}")
    
    return response.tokens

# Extract specific parts of speech
def extract_nouns_and_verbs(text):
    """Extract nouns and verbs from text."""
    
    client = language_v1.LanguageServiceClient()
    
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT
    )
    
    response = client.analyze_syntax(
        request={'document': document, 'encoding_type': 'UTF8'}
    )
    
    nouns = []
    verbs = []
    
    for token in response.tokens:
        pos = language_v1.PartOfSpeech.Tag(token.part_of_speech.tag).name
        
        if pos == 'NOUN':
            nouns.append(token.lemma)
        elif pos == 'VERB':
            verbs.append(token.lemma)
    
    print(f"Nouns: {', '.join(set(nouns))}")
    print(f"Verbs: {', '.join(set(verbs))}")
    
    return {'nouns': list(set(nouns)), 'verbs': list(set(verbs))}

# Example
text = "The quick brown fox jumps over the lazy dog while running through the forest."
syntax_result = analyze_syntax(text)
nouns_verbs = extract_nouns_and_verbs(text)
```

---

## 2. AutoML Natural Language Best Practices

### Dataset Preparation

**Best Practices:**
- Minimum 10 examples per label (100+ recommended)
- Balance classes as much as possible
- Use diverse, representative examples
- Clean and normalize text

```python
from google.cloud import aiplatform
from google.cloud import storage
import csv

def prepare_automl_text_dataset(
    texts_with_labels,
    bucket_name,
    csv_filename
):
    """Prepare CSV dataset for AutoML Natural Language."""
    
    # Validate dataset
    label_counts = {}
    for text, label in texts_with_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Dataset Statistics:")
    print(f"  Total examples: {len(texts_with_labels)}")
    print(f"  Unique labels: {len(label_counts)}")
    print("\nLabel Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")
    
    # Check minimum requirements
    min_examples = min(label_counts.values())
    if min_examples < 10:
        print(f"\n⚠️  Warning: Some labels have fewer than 10 examples")
    
    # Create CSV file
    csv_path = f"/tmp/{csv_filename}"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for text, label in texts_with_labels:
            # Clean text
            cleaned_text = text.replace('\n', ' ').replace('"', '""')
            writer.writerow([cleaned_text, label])
    
    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(csv_filename)
    blob.upload_from_filename(csv_path)
    
    gcs_uri = f"gs://{bucket_name}/{csv_filename}"
    print(f"\nDataset uploaded to: {gcs_uri}")
    
    return gcs_uri

def create_automl_text_dataset(
    project_id,
    location,
    display_name,
    gcs_csv_uri
):
    """Create AutoML Natural Language dataset."""
    
    aiplatform.init(project=project_id, location=location)
    
    dataset = aiplatform.TextDataset.create(
        display_name=display_name,
        gcs_source=gcs_csv_uri,
        import_schema_uri=aiplatform.schema.dataset.ioformat.text.single_label_classification,
        sync=True,
    )
    
    print(f"Dataset created: {dataset.resource_name}")
    
    return dataset

# Example: Sentiment classification dataset
texts_with_labels = [
    ("This product is amazing! I love it.", "positive"),
    ("Terrible quality, waste of money.", "negative"),
    ("It's okay, nothing special.", "neutral"),
    # Add more examples...
]

csv_uri = prepare_automl_text_dataset(
    texts_with_labels,
    'my-ml-bucket',
    'sentiment_dataset.csv'
)
```

### Model Training

**Best Practices:**
- Choose appropriate model type
- Set adequate training budget
- Use validation split for evaluation
- Monitor training progress

```python
def train_automl_text_model(
    dataset,
    model_display_name,
    multi_label=False
):
    """Train AutoML Natural Language classification model."""
    
    job = aiplatform.AutoMLTextTrainingJob(
        display_name=f"{model_display_name}-training",
        prediction_type="classification",
        multi_label=multi_label,
        sentiment_max=None,  # Set to 1-10 for sentiment analysis
        labels={'task': 'text-classification', 'version': 'v1'}
    )
    
    print(f"Starting training...")
    
    model = job.run(
        dataset=dataset,
        model_display_name=model_display_name,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        sync=True,
    )
    
    print(f"Model trained: {model.resource_name}")
    
    # Get evaluation metrics
    evaluations = model.list_model_evaluations()
    
    for evaluation in evaluations:
        print(f"\nEvaluation Metrics:")
        metrics = evaluation.metrics
        
        # Classification metrics
        if 'auPrc' in metrics:
            print(f"  AU-PRC: {metrics['auPrc']:.3f}")
            print(f"  AU-ROC: {metrics.get('auRoc', 0):.3f}")
        
        # Confusion matrix
        if 'confusionMatrix' in metrics:
            print(f"  Confusion Matrix:")
            cm = metrics['confusionMatrix']
            if 'rows' in cm:
                for row in cm['rows']:
                    print(f"    {row}")
    
    return model

# Multi-label classification
def train_multi_label_model(dataset, model_display_name):
    """Train model for multi-label classification."""
    
    job = aiplatform.AutoMLTextTrainingJob(
        display_name=f"{model_display_name}-training",
        prediction_type="classification",
        multi_label=True,  # Enable multi-label
    )
    
    model = job.run(
        dataset=dataset,
        model_display_name=model_display_name,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        sync=True,
    )
    
    return model
```

### Model Deployment and Prediction

```python
def deploy_text_model(
    model,
    endpoint_display_name,
    machine_type='n1-standard-2'
):
    """Deploy AutoML text model."""
    
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        labels={'model_type': 'automl-text'}
    )
    
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{model.display_name}-deployment",
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=5,
        sync=True,
    )
    
    print(f"Model deployed to: {endpoint.resource_name}")
    
    return endpoint

def predict_text_classification(endpoint, texts):
    """Make predictions on text."""
    
    instances = [{'content': text} for text in texts]
    
    predictions = endpoint.predict(instances=instances)
    
    results = []
    for idx, (text, pred) in enumerate(zip(texts, predictions.predictions)):
        print(f"\nText {idx + 1}: '{text[:50]}...'")
        
        # Parse predictions
        if isinstance(pred, dict):
            if 'displayNames' in pred and 'confidences' in pred:
                labels = pred['displayNames']
                confidences = pred['confidences']
                
                # Sort by confidence
                sorted_preds = sorted(
                    zip(labels, confidences),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                print("  Predictions:")
                for label, conf in sorted_preds[:3]:
                    print(f"    {label}: {conf:.3f}")
                
                results.append({
                    'text': text,
                    'top_label': sorted_preds[0][0],
                    'confidence': sorted_preds[0][1],
                    'all_predictions': sorted_preds
                })
    
    return results

# Example usage
texts_to_classify = [
    "This movie was absolutely fantastic! Great acting and plot.",
    "I'm disappointed with this purchase. It broke after one day.",
    "The product is decent, does what it's supposed to do."
]

predictions = predict_text_classification(endpoint, texts_to_classify)
```

---

## 3. Healthcare Natural Language API

### Medical Entity Extraction

**Best Practices:**
- Use FHIR-compliant formats
- Handle medical terminology correctly
- Extract structured medical data
- Ensure HIPAA compliance

```python
from google.cloud import healthcare_v1

def analyze_medical_text(project_id, location, text):
    """Analyze medical text for entities."""
    
    client = healthcare_v1.services.nlp_service.NlpServiceClient()
    
    nlp_service = f"projects/{project_id}/locations/{location}/services/nlp"
    
    request = healthcare_v1.AnalyzeEntitiesRequest(
        nlp_service=nlp_service,
        document_content=text
    )
    
    response = client.analyze_entities(request)
    
    print("Medical Entities Found:\n")
    
    entities_by_type = {}
    for entity in response.entities:
        entity_type = entity.entity_id.split('/')[0]
        
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        
        entities_by_type[entity_type].append({
            'text': entity.mention_text,
            'id': entity.entity_id,
            'confidence': entity.confidence
        })
    
    # Display by category
    for entity_type, entities in entities_by_type.items():
        print(f"{entity_type}:")
        for entity in entities:
            print(f"  - {entity['text']} (confidence: {entity['confidence']:.2f})")
        print()
    
    return entities_by_type

# Example: Medical note analysis
medical_note = """
Patient presents with acute myocardial infarction. 
Administered aspirin 325mg and started on heparin drip.
Blood pressure 150/90, heart rate 95 bpm.
Patient has history of type 2 diabetes mellitus and hypertension.
"""

medical_entities = analyze_medical_text(
    'my-project',
    'us-central1',
    medical_note
)
```

---

## 4. Batch Processing

### Efficient Batch Analysis

```python
import concurrent.futures
from typing import List, Dict

def analyze_texts_in_parallel(
    texts: List[str],
    analysis_function,
    max_workers=5
):
    """Process multiple texts in parallel."""
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_text = {
            executor.submit(analysis_function, text): text
            for text in texts
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_text):
            text = future_to_text[future]
            try:
                result = future.result()
                results.append({
                    'text': text,
                    'result': result,
                    'success': True
                })
            except Exception as exc:
                print(f"Error processing text: {exc}")
                results.append({
                    'text': text,
                    'error': str(exc),
                    'success': False
                })
    
    return results

# Example: Batch sentiment analysis
def analyze_sentiment_simple(text):
    """Simple sentiment analysis wrapper."""
    result = analyze_sentiment_detailed(text)
    return {
        'score': result['document_score'],
        'label': result['label']
    }

# Process 100 customer reviews
customer_reviews = [
    "Great product, highly recommend!",
    "Not worth the money, disappointed.",
    # ... more reviews
]

batch_results = analyze_texts_in_parallel(
    customer_reviews[:100],
    analyze_sentiment_simple,
    max_workers=10
)

# Aggregate results
positive = sum(1 for r in batch_results if r['success'] and r['result']['label'] == 'Positive')
negative = sum(1 for r in batch_results if r['success'] and r['result']['label'] == 'Negative')
neutral = sum(1 for r in batch_results if r['success'] and r['result']['label'] == 'Neutral')

print(f"\nBatch Analysis Summary:")
print(f"  Positive: {positive}")
print(f"  Negative: {negative}")
print(f"  Neutral: {neutral}")
```

---

## 5. Cost Optimization

### Optimize API Usage

```python
import hashlib
import json
from datetime import datetime, timedelta

class NLPCache:
    """Cache for NLP API responses."""
    
    def __init__(self, cache_ttl_minutes=60):
        self.cache = {}
        self.ttl = timedelta(minutes=cache_ttl_minutes)
    
    def get_key(self, text, operation):
        """Generate cache key."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{operation}:{text_hash}"
    
    def get(self, text, operation):
        """Get cached result."""
        key = self.get_key(text, operation)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return result
            else:
                del self.cache[key]
        
        return None
    
    def set(self, text, operation, result):
        """Cache result."""
        key = self.get_key(text, operation)
        self.cache[key] = (result, datetime.now())

# Use cache
nlp_cache = NLPCache(cache_ttl_minutes=30)

def analyze_with_cache(text, operation='sentiment'):
    """Analyze text with caching."""
    
    cached = nlp_cache.get(text, operation)
    if cached:
        print("Using cached result")
        return cached
    
    # Perform analysis
    if operation == 'sentiment':
        result = analyze_sentiment_detailed(text)
    elif operation == 'entities':
        result = analyze_entities_comprehensive(text)
    
    nlp_cache.set(text, operation, result)
    
    return result
```

---

## 6. Error Handling

```python
from google.api_core import retry, exceptions

@retry.Retry(predicate=retry.if_exception_type(
    exceptions.ResourceExhausted,
    exceptions.ServiceUnavailable,
))
def analyze_text_with_retry(text):
    """Analyze text with automatic retry."""
    
    client = language_v1.LanguageServiceClient()
    
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT
    )
    
    try:
        response = client.analyze_sentiment(request={'document': document})
        return response
    
    except exceptions.InvalidArgument as e:
        print(f"Invalid argument: {e}")
        raise
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Enable Natural Language API
- [ ] Set up authentication
- [ ] Configure IAM roles
- [ ] Set API quotas
- [ ] Implement error handling

### Text Analysis
- [ ] Prepare and clean text
- [ ] Choose appropriate analysis type
- [ ] Set confidence thresholds
- [ ] Handle multilingual content
- [ ] Cache results appropriately

### Production
- [ ] Implement batch processing
- [ ] Use parallel processing for scale
- [ ] Monitor API usage and costs
- [ ] Set up alerting
- [ ] Implement retry logic
- [ ] Log analysis results

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
