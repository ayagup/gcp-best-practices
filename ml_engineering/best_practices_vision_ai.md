# Vision AI Best Practices

*Last Updated: January 4, 2026*

## Overview

Vision AI provides pre-trained and custom machine learning models to analyze and understand images. This document covers best practices for using Vision API, AutoML Vision, Product Search, and related services.

---

## 1. Vision API Best Practices

### Image Quality and Preparation

**Best Practices:**
- Use high-resolution images (minimum 640x480 pixels)
- Ensure good lighting and contrast
- Avoid overly compressed images
- Use appropriate image formats (JPEG, PNG, GIF, BMP, WebP, RAW, ICO, PDF, TIFF)

```python
from google.cloud import vision
import io

def analyze_image_with_quality_check(image_path):
    """Analyze image with quality validation."""
    
    client = vision.ImageAnnotatorClient()
    
    # Read and validate image
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    # Check file size (API limit: 20MB for direct content, 10MB for JSON)
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Image too large. Use Cloud Storage URI instead.")
    
    image = vision.Image(content=content)
    
    # Perform image properties detection first to check quality
    properties_response = client.image_properties(image=image)
    
    if properties_response.error.message:
        raise Exception(f"Error analyzing image: {properties_response.error.message}")
    
    # Check image quality metrics
    dominant_colors = properties_response.image_properties_annotation.dominant_colors
    
    print(f"Image quality check passed")
    print(f"Dominant colors: {len(dominant_colors.colors)}")
    
    return image

# Example usage
image = analyze_image_with_quality_check('path/to/image.jpg')
```

### Label Detection

**Best Practices:**
- Set appropriate confidence thresholds
- Limit number of results to reduce costs
- Use batch requests for multiple images
- Cache results for frequently accessed images

```python
from google.cloud import vision

def detect_labels_with_filtering(image_path, max_results=10, min_confidence=0.7):
    """Detect labels with confidence filtering."""
    
    client = vision.ImageAnnotatorClient()
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Request label detection
    response = client.label_detection(
        image=image,
        max_results=max_results
    )
    
    labels = response.label_annotations
    
    # Filter by confidence
    filtered_labels = [
        label for label in labels 
        if label.score >= min_confidence
    ]
    
    print(f"Found {len(filtered_labels)} labels with confidence >= {min_confidence}")
    
    for label in filtered_labels:
        print(f"  - {label.description}: {label.score:.2f}")
    
    return filtered_labels

# Batch processing multiple images
def batch_label_detection(image_paths, max_results=10):
    """Process multiple images in batch."""
    
    client = vision.ImageAnnotatorClient()
    
    requests = []
    for image_path in image_paths:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        requests.append({
            'image': image,
            'features': [{
                'type_': vision.Feature.Type.LABEL_DETECTION,
                'max_results': max_results
            }]
        })
    
    # Batch request (up to 16 images per batch)
    response = client.batch_annotate_images(requests=requests)
    
    results = []
    for idx, image_response in enumerate(response.responses):
        if image_response.error.message:
            print(f"Error on image {idx}: {image_response.error.message}")
            continue
        
        results.append({
            'image_path': image_paths[idx],
            'labels': image_response.label_annotations
        })
    
    return results
```

### Face Detection

**Best Practices:**
- Respect privacy and comply with regulations
- Use face detection (not recognition) for privacy
- Implement proper consent mechanisms
- Blur faces for privacy when needed

```python
def detect_faces_with_privacy(image_path, blur_faces=False):
    """Detect faces with privacy considerations."""
    
    client = vision.ImageAnnotatorClient()
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Detect faces
    response = client.face_detection(image=image)
    faces = response.face_annotations
    
    print(f"Found {len(faces)} faces")
    
    face_info = []
    for idx, face in enumerate(faces):
        print(f"\nFace {idx + 1}:")
        print(f"  Joy likelihood: {face.joy_likelihood.name}")
        print(f"  Sorrow likelihood: {face.sorrow_likelihood.name}")
        print(f"  Anger likelihood: {face.anger_likelihood.name}")
        print(f"  Surprise likelihood: {face.surprise_likelihood.name}")
        print(f"  Detection confidence: {face.detection_confidence:.2f}")
        
        # Get bounding box
        vertices = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        
        face_info.append({
            'index': idx,
            'emotions': {
                'joy': face.joy_likelihood.name,
                'sorrow': face.sorrow_likelihood.name,
                'anger': face.anger_likelihood.name,
                'surprise': face.surprise_likelihood.name
            },
            'bounding_box': vertices,
            'confidence': face.detection_confidence
        })
    
    # Optionally blur faces
    if blur_faces and faces:
        from PIL import Image, ImageFilter
        
        pil_image = Image.open(image_path)
        
        for face in faces:
            vertices = face.bounding_poly.vertices
            box = (
                vertices[0].x, vertices[0].y,
                vertices[2].x, vertices[2].y
            )
            
            # Crop, blur, and paste back
            face_region = pil_image.crop(box)
            blurred = face_region.filter(ImageFilter.GaussianBlur(20))
            pil_image.paste(blurred, box)
        
        pil_image.save('blurred_' + image_path)
        print(f"\nBlurred image saved as 'blurred_{image_path}'")
    
    return face_info
```

### Text Detection (OCR)

**Best Practices:**
- Use TEXT_DETECTION for sparse text
- Use DOCUMENT_TEXT_DETECTION for dense text (documents)
- Consider language hints for better accuracy
- Handle multiple languages in documents

```python
def detect_text_optimized(image_path, language_hints=None, document_mode=False):
    """Detect text with optimization options."""
    
    client = vision.ImageAnnotatorClient()
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Set image context with language hints
    image_context = None
    if language_hints:
        image_context = vision.ImageContext(language_hints=language_hints)
    
    # Choose detection type
    if document_mode:
        # For dense text documents
        response = client.document_text_detection(
            image=image,
            image_context=image_context
        )
        
        # Full text annotation provides structured output
        if response.full_text_annotation:
            text = response.full_text_annotation.text
            
            print(f"Detected text ({len(text)} characters):")
            print(text)
            
            # Access structured data
            print(f"\nPages: {len(response.full_text_annotation.pages)}")
            
            for page in response.full_text_annotation.pages:
                print(f"  Blocks: {len(page.blocks)}")
                print(f"  Language: {page.property.detected_languages[0].language_code if page.property.detected_languages else 'Unknown'}")
            
            return {
                'text': text,
                'pages': response.full_text_annotation.pages,
                'confidence': response.full_text_annotation.pages[0].confidence if response.full_text_annotation.pages else 0
            }
    else:
        # For sparse text
        response = client.text_detection(
            image=image,
            image_context=image_context
        )
        
        texts = response.text_annotations
        
        if texts:
            # First annotation contains all text
            full_text = texts[0].description
            
            print(f"Detected text:")
            print(full_text)
            
            # Individual words/phrases
            print(f"\nIndividual text elements: {len(texts) - 1}")
            
            return {
                'text': full_text,
                'elements': texts[1:],  # Skip first (full text)
            }
    
    return None

# Example with language hints
text_result = detect_text_optimized(
    'document.jpg',
    language_hints=['en', 'es', 'fr'],
    document_mode=True
)

# Extract specific information
def extract_structured_data_from_ocr(image_path):
    """Extract structured data like dates, amounts, etc."""
    
    import re
    
    result = detect_text_optimized(image_path, document_mode=True)
    
    if not result:
        return None
    
    text = result['text']
    
    # Extract dates
    date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
    dates = re.findall(date_pattern, text)
    
    # Extract currency amounts
    currency_pattern = r'\$[\d,]+\.?\d*'
    amounts = re.findall(currency_pattern, text)
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    return {
        'dates': dates,
        'amounts': amounts,
        'emails': emails,
        'full_text': text
    }
```

### Object Localization

**Best Practices:**
- Use for detecting and locating objects in images
- Combine with label detection for context
- Filter by confidence scores
- Consider custom models for specific objects

```python
def localize_objects_with_context(image_path, min_confidence=0.5):
    """Localize objects with contextual information."""
    
    client = vision.ImageAnnotatorClient()
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Detect objects
    objects_response = client.object_localization(image=image)
    objects = objects_response.localized_object_annotations
    
    # Also get labels for context
    labels_response = client.label_detection(image=image, max_results=10)
    labels = labels_response.label_annotations
    
    print(f"Found {len(objects)} objects")
    print(f"Image context labels: {', '.join([l.description for l in labels[:5]])}")
    
    filtered_objects = []
    for obj in objects:
        if obj.score >= min_confidence:
            vertices = [(vertex.x, vertex.y) for vertex in obj.bounding_poly.normalized_vertices]
            
            print(f"\n{obj.name} (confidence: {obj.score:.2f})")
            print(f"  Bounding box: {vertices}")
            
            filtered_objects.append({
                'name': obj.name,
                'confidence': obj.score,
                'bounding_box': vertices
            })
    
    return {
        'objects': filtered_objects,
        'context_labels': [l.description for l in labels]
    }

# Visualize detected objects
def visualize_objects(image_path, objects):
    """Draw bounding boxes on image."""
    
    from PIL import Image, ImageDraw, ImageFont
    
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    img_width, img_height = image.size
    
    for obj in objects:
        # Convert normalized coordinates to pixel coordinates
        vertices = [
            (int(x * img_width), int(y * img_height))
            for x, y in obj['bounding_box']
        ]
        
        # Draw polygon
        draw.polygon(vertices, outline='red')
        
        # Draw label
        label = f"{obj['name']} ({obj['confidence']:.2f})"
        draw.text(vertices[0], label, fill='red')
    
    output_path = 'annotated_' + image_path
    image.save(output_path)
    print(f"\nAnnotated image saved as '{output_path}'")
    
    return output_path
```

### Safe Search Detection

**Best Practices:**
- Use for content moderation
- Set appropriate thresholds for your use case
- Log flagged content for review
- Implement human review for edge cases

```python
def moderate_content(image_path, strict_mode=True):
    """Detect inappropriate content."""
    
    client = vision.ImageAnnotatorClient()
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    
    # Likelihood levels: UNKNOWN, VERY_UNLIKELY, UNLIKELY, POSSIBLE, LIKELY, VERY_LIKELY
    
    print("Safe Search Results:")
    print(f"  Adult: {safe.adult.name}")
    print(f"  Spoof: {safe.spoof.name}")
    print(f"  Medical: {safe.medical.name}")
    print(f"  Violence: {safe.violence.name}")
    print(f"  Racy: {safe.racy.name}")
    
    # Define thresholds
    if strict_mode:
        unsafe_threshold = 'POSSIBLE'
    else:
        unsafe_threshold = 'LIKELY'
    
    # Check if content is safe
    is_safe = all([
        safe.adult.name < unsafe_threshold,
        safe.violence.name < unsafe_threshold,
        safe.racy.name < unsafe_threshold
    ])
    
    if not is_safe:
        print("\n⚠️  Content flagged for review")
        return {
            'safe': False,
            'reasons': {
                'adult': safe.adult.name,
                'violence': safe.violence.name,
                'racy': safe.racy.name
            }
        }
    
    print("\n✅ Content appears safe")
    return {'safe': True}

# Batch content moderation
def batch_moderate_images(image_paths):
    """Moderate multiple images efficiently."""
    
    client = vision.ImageAnnotatorClient()
    
    requests = []
    for image_path in image_paths:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        requests.append({
            'image': image,
            'features': [{'type_': vision.Feature.Type.SAFE_SEARCH_DETECTION}]
        })
    
    response = client.batch_annotate_images(requests=requests)
    
    results = []
    for idx, image_response in enumerate(response.responses):
        safe = image_response.safe_search_annotation
        
        is_safe = all([
            safe.adult.name < 'LIKELY',
            safe.violence.name < 'LIKELY'
        ])
        
        results.append({
            'image_path': image_paths[idx],
            'safe': is_safe,
            'details': {
                'adult': safe.adult.name,
                'violence': safe.violence.name,
                'racy': safe.racy.name
            }
        })
    
    return results
```

---

## 2. AutoML Vision Best Practices

### Dataset Preparation

**Best Practices:**
- Minimum 100 images per label (1000+ recommended)
- Balance dataset across labels
- Use high-quality, diverse images
- Include variations (angles, lighting, backgrounds)

```python
from google.cloud import aiplatform
from google.cloud import storage

def prepare_automl_vision_dataset(
    project_id,
    location,
    display_name,
    gcs_csv_path
):
    """Create AutoML Vision dataset."""
    
    aiplatform.init(project=project_id, location=location)
    
    # CSV format: gs://bucket/image.jpg,label
    # or: gs://bucket/image.jpg,xmin,ymin,,,xmax,ymax,,,label (for object detection)
    
    dataset = aiplatform.ImageDataset.create(
        display_name=display_name,
        gcs_source=gcs_csv_path,
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
        sync=True,
    )
    
    print(f"Dataset created: {dataset.resource_name}")
    print(f"Dataset ID: {dataset.name}")
    
    # Validate dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Items: {dataset.metadata.get('items_count', 'Unknown')}")
    
    return dataset

# Generate CSV from labeled images
def generate_dataset_csv(bucket_name, image_folder, labels_dict, output_csv):
    """Generate CSV file for AutoML Vision dataset."""
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    with open(output_csv, 'w') as f:
        for image_name, label in labels_dict.items():
            gcs_uri = f"gs://{bucket_name}/{image_folder}/{image_name}"
            f.write(f"{gcs_uri},{label}\n")
    
    # Upload CSV to GCS
    csv_blob = bucket.blob(f"{image_folder}/dataset.csv")
    csv_blob.upload_from_filename(output_csv)
    
    gcs_csv_uri = f"gs://{bucket_name}/{image_folder}/dataset.csv"
    print(f"Dataset CSV created: {gcs_csv_uri}")
    
    return gcs_csv_uri

# Validate dataset quality
def validate_dataset_quality(dataset):
    """Check dataset quality metrics."""
    
    issues = []
    
    # Get dataset stats
    stats = dataset.metadata
    
    # Check minimum images
    if stats.get('items_count', 0) < 100:
        issues.append("Dataset has fewer than 100 images (recommended minimum)")
    
    # Check label distribution
    label_stats = stats.get('label_stats', {})
    if label_stats:
        max_count = max(label_stats.values())
        min_count = min(label_stats.values())
        
        # Check imbalance (max should not be > 10x min)
        if max_count > min_count * 10:
            issues.append(f"Dataset is imbalanced (ratio: {max_count/min_count:.1f}:1)")
    
    if issues:
        print("Dataset Quality Issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✅ Dataset quality checks passed")
    
    return len(issues) == 0
```

### Model Training

**Best Practices:**
- Choose appropriate model type (cloud, edge)
- Set adequate training budget
- Use validation split for evaluation
- Enable early stopping

```python
def train_automl_vision_model(
    dataset,
    model_display_name,
    training_budget_hours=1,
    model_type='CLOUD'
):
    """Train AutoML Vision model."""
    
    # Model types:
    # - CLOUD: Best accuracy, cloud deployment
    # - CLOUD_HIGH_ACCURACY_1: Higher accuracy, longer training
    # - CLOUD_LOW_LATENCY_1: Lower latency, smaller model
    # - MOBILE_TF_LOW_LATENCY_1: Edge deployment, TensorFlow Lite
    
    job = aiplatform.AutoMLImageTrainingJob(
        display_name=f"{model_display_name}-training",
        prediction_type="classification",
        multi_label=False,
        model_type=model_type,
        base_model=None,  # Optional: use transfer learning
        labels={'env': 'production', 'version': 'v1'}
    )
    
    print(f"Starting training with budget: {training_budget_hours} hours")
    
    model = job.run(
        dataset=dataset,
        model_display_name=model_display_name,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        budget_milli_node_hours=training_budget_hours * 1000,
        disable_early_stopping=False,
        sync=True,
    )
    
    print(f"Model trained: {model.resource_name}")
    
    # Get model evaluation metrics
    evaluations = model.list_model_evaluations()
    
    for evaluation in evaluations:
        print(f"\nEvaluation Metrics:")
        metrics = evaluation.metrics
        
        if 'confidenceMetrics' in metrics:
            # Classification metrics
            print(f"  Confusion Matrix: {metrics.get('confusionMatrix')}")
            
            # Get metrics at 0.5 threshold
            for cm in metrics.get('confidenceMetrics', []):
                if abs(cm.get('confidenceThreshold', 0) - 0.5) < 0.01:
                    print(f"  Precision (0.5): {cm.get('precision', 0):.3f}")
                    print(f"  Recall (0.5): {cm.get('recall', 0):.3f}")
                    print(f"  F1 Score (0.5): {cm.get('f1Score', 0):.3f}")
    
    return model
```

### Model Deployment

**Best Practices:**
- Use appropriate machine types
- Enable auto-scaling
- Monitor prediction latency
- A/B test new models

```python
def deploy_automl_vision_model(
    model,
    endpoint_display_name,
    machine_type='n1-standard-4',
    min_replicas=1,
    max_replicas=5
):
    """Deploy AutoML Vision model to endpoint."""
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        labels={'model_type': 'automl-vision'}
    )
    
    print(f"Endpoint created: {endpoint.resource_name}")
    
    # Deploy model
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{model.display_name}-deployment",
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,
        sync=True,
    )
    
    print(f"Model deployed successfully")
    
    return endpoint

# Make predictions
def predict_with_automl_vision(endpoint, image_path):
    """Make prediction using deployed AutoML Vision model."""
    
    import base64
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    encoded_content = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare instance
    instance = {'content': encoded_content}
    
    # Make prediction
    prediction = endpoint.predict(instances=[instance])
    
    print(f"Predictions for {image_path}:")
    
    # Parse results
    for idx, pred in enumerate(prediction.predictions):
        print(f"\nPrediction {idx + 1}:")
        
        # Get class labels and confidences
        if 'confidences' in pred and 'displayNames' in pred:
            confidences = pred['confidences']
            labels = pred['displayNames']
            
            # Sort by confidence
            results = sorted(
                zip(labels, confidences),
                key=lambda x: x[1],
                reverse=True
            )
            
            for label, confidence in results[:5]:
                print(f"  {label}: {confidence:.3f}")
    
    return prediction.predictions
```

---

## 3. Product Search Best Practices

### Product Catalog Setup

**Best Practices:**
- Organize products in product sets
- Use high-quality product images
- Include multiple views per product
- Add rich metadata

```python
from google.cloud import vision

def create_product_set(
    project_id,
    location,
    product_set_id,
    product_set_display_name
):
    """Create a product set for visual search."""
    
    client = vision.ProductSearchClient()
    
    location_path = f"projects/{project_id}/locations/{location}"
    
    product_set = vision.ProductSet(display_name=product_set_display_name)
    
    response = client.create_product_set(
        parent=location_path,
        product_set=product_set,
        product_set_id=product_set_id
    )
    
    print(f"Product set created: {response.name}")
    
    return response

def create_product_with_images(
    project_id,
    location,
    product_id,
    product_display_name,
    product_category,
    image_uris,
    product_labels=None
):
    """Create product with multiple reference images."""
    
    client = vision.ProductSearchClient()
    
    location_path = f"projects/{project_id}/locations/{location}"
    
    # Create product
    product = vision.Product(
        display_name=product_display_name,
        product_category=product_category,  # e.g., 'apparel-v2', 'homegoods-v2'
        product_labels=product_labels or []
    )
    
    response = client.create_product(
        parent=location_path,
        product=product,
        product_id=product_id
    )
    
    product_name = response.name
    print(f"Product created: {product_name}")
    
    # Add reference images
    for idx, image_uri in enumerate(image_uris):
        reference_image = vision.ReferenceImage(uri=image_uri)
        
        image_response = client.create_reference_image(
            parent=product_name,
            reference_image=reference_image,
            reference_image_id=f"{product_id}-{idx}"
        )
        
        print(f"  Reference image added: {image_response.name}")
    
    return response

# Add product to product set
def add_product_to_set(project_id, location, product_id, product_set_id):
    """Add product to product set."""
    
    client = vision.ProductSearchClient()
    
    product_path = client.product_path(project_id, location, product_id)
    product_set_path = client.product_set_path(project_id, location, product_set_id)
    
    client.add_product_to_product_set(
        name=product_set_path,
        product=product_path
    )
    
    print(f"Product {product_id} added to set {product_set_id}")
```

### Visual Search

**Best Practices:**
- Use appropriate bounding boxes
- Filter by product category
- Set confidence thresholds
- Handle similar products

```python
def search_similar_products(
    project_id,
    location,
    product_set_id,
    image_path,
    filter_expression=None,
    max_results=10
):
    """Search for similar products using an image."""
    
    client = vision.ProductSearchClient()
    
    # Read query image
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Product set path
    product_set_path = client.product_set_path(
        project_id, location, product_set_id
    )
    
    # Image context
    image_context = vision.ImageContext(
        product_search_params=vision.ProductSearchParams(
            product_set=product_set_path,
            product_categories=['apparel-v2'],
            filter=filter_expression  # e.g., "color=red AND style=dress"
        )
    )
    
    # Search
    response = client.product_search(
        image=image,
        image_context=image_context,
        max_results=max_results
    )
    
    print(f"Found {len(response.results)} similar products:")
    
    results = []
    for result in response.results:
        product = result.product
        
        print(f"\n  Product: {product.display_name}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Image: {result.image}")
        
        results.append({
            'product_name': product.display_name,
            'product_id': product.name.split('/')[-1],
            'score': result.score,
            'labels': {label.key: label.value for label in product.product_labels}
        })
    
    return results

# Search with bounding box (partial image search)
def search_products_in_region(
    project_id,
    location,
    product_set_id,
    image_path,
    bounding_box
):
    """Search for products in a specific region of image."""
    
    client = vision.ProductSearchClient()
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    product_set_path = client.product_set_path(
        project_id, location, product_set_id
    )
    
    # Bounding box: [xmin, ymin, xmax, ymax] (normalized 0-1)
    bounding_poly = vision.BoundingPoly(
        normalized_vertices=[
            vision.NormalizedVertex(x=bounding_box[0], y=bounding_box[1]),
            vision.NormalizedVertex(x=bounding_box[2], y=bounding_box[1]),
            vision.NormalizedVertex(x=bounding_box[2], y=bounding_box[3]),
            vision.NormalizedVertex(x=bounding_box[0], y=bounding_box[3]),
        ]
    )
    
    image_context = vision.ImageContext(
        product_search_params=vision.ProductSearchParams(
            product_set=product_set_path,
            product_categories=['apparel-v2'],
            bounding_poly=bounding_poly
        )
    )
    
    response = client.product_search(
        image=image,
        image_context=image_context
    )
    
    return response.results
```

---

## 4. Cost Optimization

### Best Practices

**Optimize API Usage:**
```python
import functools
import hashlib
import json
from datetime import datetime, timedelta

class VisionAPICache:
    """Simple cache for Vision API responses."""
    
    def __init__(self, cache_duration_minutes=60):
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
    
    def get_cache_key(self, image_content, feature_type):
        """Generate cache key from image content."""
        content_hash = hashlib.md5(image_content).hexdigest()
        return f"{feature_type}:{content_hash}"
    
    def get(self, image_content, feature_type):
        """Get cached response."""
        key = self.get_cache_key(image_content, feature_type)
        
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            
            if datetime.now() - timestamp < self.cache_duration:
                print(f"Cache hit for {feature_type}")
                return cached_data
            else:
                del self.cache[key]
        
        return None
    
    def set(self, image_content, feature_type, response):
        """Cache response."""
        key = self.get_cache_key(image_content, feature_type)
        self.cache[key] = (response, datetime.now())

# Use caching
cache = VisionAPICache(cache_duration_minutes=30)

def detect_labels_with_cache(image_path):
    """Detect labels with caching."""
    
    client = vision.ImageAnnotatorClient()
    
    with open(image_path, 'rb') as f:
        content = f.read()
    
    # Check cache
    cached_result = cache.get(content, 'LABEL_DETECTION')
    if cached_result:
        return cached_result
    
    # API call
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    
    # Cache result
    cache.set(content, 'LABEL_DETECTION', response.label_annotations)
    
    return response.label_annotations

# Batch processing to reduce costs
def process_images_in_optimal_batches(image_paths, batch_size=16):
    """Process images in optimal batch sizes."""
    
    client = vision.ImageAnnotatorClient()
    
    all_results = []
    
    # Process in batches of 16 (API limit)
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        
        requests = []
        for image_path in batch:
            with open(image_path, 'rb') as f:
                content = f.read()
            
            requests.append({
                'image': vision.Image(content=content),
                'features': [
                    {'type_': vision.Feature.Type.LABEL_DETECTION, 'max_results': 10}
                ]
            })
        
        response = client.batch_annotate_images(requests=requests)
        all_results.extend(response.responses)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
    
    return all_results
```

---

## 5. Error Handling and Monitoring

### Robust Error Handling

```python
from google.api_core import retry
from google.api_core import exceptions
import time

def analyze_image_with_retry(image_path, max_retries=3):
    """Analyze image with automatic retry logic."""
    
    client = vision.ImageAnnotatorClient()
    
    with open(image_path, 'rb') as f:
        content = f.read()
    
    image = vision.Image(content=content)
    
    for attempt in range(max_retries):
        try:
            response = client.label_detection(image=image)
            
            if response.error.message:
                raise Exception(f"API Error: {response.error.message}")
            
            return response.label_annotations
            
        except exceptions.ResourceExhausted:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        
        except exceptions.InvalidArgument as e:
            print(f"Invalid request: {e}")
            raise
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    return None

# Monitor API usage
def log_api_usage(operation, image_size, response_time):
    """Log API usage for monitoring."""
    
    from google.cloud import logging
    
    logging_client = logging.Client()
    logger = logging_client.logger('vision-api-usage')
    
    logger.log_struct({
        'operation': operation,
        'image_size_bytes': image_size,
        'response_time_ms': response_time,
        'timestamp': datetime.now().isoformat()
    })
```

---

## 6. Security Best Practices

### API Key Management

```bash
# Use service account authentication
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Set up IAM roles
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
    --role="roles/cloudvision.user"

# For product search
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
    --role="roles/cloudvision.productSearchAdmin"
```

### Data Privacy

```python
def process_image_with_privacy(image_path):
    """Process image with privacy considerations."""
    
    # 1. Blur faces before uploading
    # 2. Don't log sensitive image content
    # 3. Use temporary storage
    # 4. Delete images after processing
    
    import tempfile
    import os
    
    # Create temporary copy
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        # Process image
        result = detect_labels_with_filtering(image_path)
        
        # Clean up
        tmp_path = tmp.name
    
    # Delete temporary file
    os.unlink(tmp_path)
    
    return result
```

---

## 7. Quick Reference Checklist

### Pre-deployment
- [ ] Test image quality and formats
- [ ] Validate API quota and limits
- [ ] Set up authentication and IAM roles
- [ ] Implement error handling
- [ ] Configure caching strategy
- [ ] Set up monitoring and logging

### Production
- [ ] Use batch requests for multiple images
- [ ] Implement retry logic with exponential backoff
- [ ] Cache frequently accessed results
- [ ] Monitor API usage and costs
- [ ] Set up alerts for errors and rate limits
- [ ] Implement content moderation
- [ ] Respect privacy regulations

### Optimization
- [ ] Optimize image sizes before upload
- [ ] Use appropriate feature types
- [ ] Set confidence thresholds
- [ ] Filter unnecessary results
- [ ] Use AutoML for custom requirements
- [ ] Implement proper caching
- [ ] Monitor and optimize costs

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
