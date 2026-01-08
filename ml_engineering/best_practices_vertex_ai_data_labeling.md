# Vertex AI Data Labeling Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Data Labeling provides managed human labeling services for creating high-quality training datasets with support for image, video, and text annotation, access to Google's labeling workforce, active learning, and quality assurance workflows.

---

## 1. Labeling Job Setup

### Create and Configure Labeling Jobs

```python
from google.cloud import aiplatform

class DataLabelingManager:
    """Manage Vertex AI Data Labeling jobs."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        
        aiplatform.init(project=project_id, location=location)
    
    def create_dataset_for_labeling(
        self,
        display_name,
        dataset_type='image',
        gcs_source=None,
        bigquery_source=None
    ):
        """Create dataset for labeling.
        
        Args:
            dataset_type: 'image', 'video', or 'text'
        """
        
        if dataset_type == 'image':
            dataset = aiplatform.ImageDataset.create(
                display_name=display_name,
                gcs_source=gcs_source
            )
        elif dataset_type == 'video':
            dataset = aiplatform.VideoDataset.create(
                display_name=display_name,
                gcs_source=gcs_source
            )
        elif dataset_type == 'text':
            dataset = aiplatform.TextDataset.create(
                display_name=display_name,
                gcs_source=gcs_source,
                import_schema_uri=aiplatform.schema.dataset.ioformat.text.single_label_classification
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        print(f"✓ Created {dataset_type} dataset: {display_name}")
        print(f"  Resource name: {dataset.resource_name}")
        
        return dataset
    
    def import_data_to_dataset(
        self,
        dataset_id,
        gcs_source_uris,
        import_schema_uri=None
    ):
        """Import data into dataset."""
        
        dataset = aiplatform.ImageDataset(dataset_id)
        
        dataset.import_data(
            gcs_source=gcs_source_uris,
            import_schema_uri=import_schema_uri
        )
        
        print(f"✓ Imported data to dataset")
        print(f"  Sources: {len(gcs_source_uris)} URIs")
        
        return dataset
    
    def get_dataset_statistics(self, dataset_id):
        """Get dataset statistics."""
        
        dataset = aiplatform.ImageDataset(dataset_id)
        
        print(f"\nDataset Statistics: {dataset.display_name}\n")
        print(f"Resource name: {dataset.resource_name}")
        print(f"Created: {dataset.create_time}")
        print(f"Updated: {dataset.update_time}")
        
        # Note: Additional statistics would be retrieved from dataset metadata
        
        return dataset

# Example usage
labeling_manager = DataLabelingManager(project_id='my-project')

# Create image dataset
# dataset = labeling_manager.create_dataset_for_labeling(
#     display_name='product-images-labeling',
#     dataset_type='image',
#     gcs_source=['gs://my-bucket/images/*.jpg']
# )
```

---

## 2. Image Labeling

### Configure Image Annotation Tasks

```python
class ImageLabelingManager:
    """Manage image labeling tasks."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_image_classification_job(
        self,
        dataset_id,
        instruction_uri,
        annotation_labels,
        specialist_pool=None
    ):
        """Create image classification labeling job.
        
        Args:
            annotation_labels: List of classification labels
            instruction_uri: GCS URI to labeling instructions
        """
        
        from google.cloud.aiplatform_v1 import DataLabelingJob
        from google.cloud.aiplatform_v1.types import data_labeling_job
        
        dataset = aiplatform.ImageDataset(dataset_id)
        
        # Configure classification inputs
        inputs = {
            'annotation_specs': [
                {'display_name': label} for label in annotation_labels
            ]
        }
        
        print(f"✓ Creating image classification labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Labels: {len(annotation_labels)}")
        print(f"  Instruction URI: {instruction_uri}")
        
        # Note: Actual API call would create the labeling job
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'classification',
            'labels': annotation_labels
        }
    
    def create_bounding_box_job(
        self,
        dataset_id,
        instruction_uri,
        annotation_labels,
        allow_multi_label=False
    ):
        """Create bounding box labeling job."""
        
        dataset = aiplatform.ImageDataset(dataset_id)
        
        inputs = {
            'annotation_specs': [
                {'display_name': label} for label in annotation_labels
            ],
            'allow_multi_label': allow_multi_label
        }
        
        print(f"✓ Creating bounding box labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Object classes: {len(annotation_labels)}")
        print(f"  Multi-label: {allow_multi_label}")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'bounding_box',
            'labels': annotation_labels
        }
    
    def create_polygon_segmentation_job(
        self,
        dataset_id,
        instruction_uri,
        annotation_labels
    ):
        """Create polygon segmentation labeling job."""
        
        dataset = aiplatform.ImageDataset(dataset_id)
        
        inputs = {
            'annotation_specs': [
                {'display_name': label} for label in annotation_labels
            ]
        }
        
        print(f"✓ Creating polygon segmentation labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Segmentation classes: {len(annotation_labels)}")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'polygon',
            'labels': annotation_labels
        }
    
    def create_polyline_job(
        self,
        dataset_id,
        instruction_uri,
        annotation_labels
    ):
        """Create polyline labeling job (for lane detection, etc.)."""
        
        dataset = aiplatform.ImageDataset(dataset_id)
        
        print(f"✓ Creating polyline labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Polyline classes: {len(annotation_labels)}")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'polyline',
            'labels': annotation_labels
        }

# Example usage
image_labeling = ImageLabelingManager(project_id='my-project')

# Classification job
# job = image_labeling.create_image_classification_job(
#     dataset_id='projects/.../datasets/123',
#     instruction_uri='gs://my-bucket/instructions.pdf',
#     annotation_labels=['dog', 'cat', 'bird', 'other']
# )

# Bounding box job
# job = image_labeling.create_bounding_box_job(
#     dataset_id='projects/.../datasets/123',
#     instruction_uri='gs://my-bucket/instructions.pdf',
#     annotation_labels=['person', 'car', 'bicycle', 'traffic_sign'],
#     allow_multi_label=True
# )
```

---

## 3. Video Labeling

### Configure Video Annotation Tasks

```python
class VideoLabelingManager:
    """Manage video labeling tasks."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_video_classification_job(
        self,
        dataset_id,
        instruction_uri,
        annotation_labels
    ):
        """Create video classification labeling job."""
        
        dataset = aiplatform.VideoDataset(dataset_id)
        
        print(f"✓ Creating video classification labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Labels: {annotation_labels}")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'video_classification',
            'labels': annotation_labels
        }
    
    def create_object_tracking_job(
        self,
        dataset_id,
        instruction_uri,
        annotation_labels,
        frame_sampling_fps=1
    ):
        """Create object tracking labeling job.
        
        Args:
            frame_sampling_fps: Frames per second to sample
        """
        
        dataset = aiplatform.VideoDataset(dataset_id)
        
        print(f"✓ Creating object tracking labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Object classes: {annotation_labels}")
        print(f"  Sampling rate: {frame_sampling_fps} FPS")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'object_tracking',
            'labels': annotation_labels,
            'fps': frame_sampling_fps
        }
    
    def create_action_recognition_job(
        self,
        dataset_id,
        instruction_uri,
        action_labels
    ):
        """Create action recognition labeling job."""
        
        dataset = aiplatform.VideoDataset(dataset_id)
        
        print(f"✓ Creating action recognition labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Actions: {action_labels}")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'action_recognition',
            'labels': action_labels
        }
    
    def create_event_detection_job(
        self,
        dataset_id,
        instruction_uri,
        event_labels
    ):
        """Create event detection labeling job."""
        
        dataset = aiplatform.VideoDataset(dataset_id)
        
        print(f"✓ Creating event detection labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Events: {event_labels}")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'event_detection',
            'labels': event_labels
        }

# Example usage
video_labeling = VideoLabelingManager(project_id='my-project')

# Object tracking job
# job = video_labeling.create_object_tracking_job(
#     dataset_id='projects/.../datasets/456',
#     instruction_uri='gs://my-bucket/video-instructions.pdf',
#     annotation_labels=['person', 'vehicle', 'animal'],
#     frame_sampling_fps=2
# )

# Action recognition job
# job = video_labeling.create_action_recognition_job(
#     dataset_id='projects/.../datasets/456',
#     instruction_uri='gs://my-bucket/action-instructions.pdf',
#     action_labels=['walking', 'running', 'sitting', 'standing']
# )
```

---

## 4. Text Labeling

### Configure Text Annotation Tasks

```python
class TextLabelingManager:
    """Manage text labeling tasks."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_text_classification_job(
        self,
        dataset_id,
        instruction_uri,
        annotation_labels,
        multi_label=False
    ):
        """Create text classification labeling job."""
        
        dataset = aiplatform.TextDataset(dataset_id)
        
        print(f"✓ Creating text classification labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Labels: {annotation_labels}")
        print(f"  Multi-label: {multi_label}")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'text_classification',
            'labels': annotation_labels,
            'multi_label': multi_label
        }
    
    def create_entity_extraction_job(
        self,
        dataset_id,
        instruction_uri,
        entity_types
    ):
        """Create entity extraction labeling job.
        
        Args:
            entity_types: List of entity types to label (e.g., PERSON, ORG, LOCATION)
        """
        
        dataset = aiplatform.TextDataset(dataset_id)
        
        print(f"✓ Creating entity extraction labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Entity types: {entity_types}")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'entity_extraction',
            'entity_types': entity_types
        }
    
    def create_sentiment_analysis_job(
        self,
        dataset_id,
        instruction_uri,
        sentiment_labels=None
    ):
        """Create sentiment analysis labeling job."""
        
        if sentiment_labels is None:
            sentiment_labels = ['positive', 'negative', 'neutral']
        
        dataset = aiplatform.TextDataset(dataset_id)
        
        print(f"✓ Creating sentiment analysis labeling job")
        print(f"  Dataset: {dataset.display_name}")
        print(f"  Sentiment labels: {sentiment_labels}")
        
        return {
            'dataset_id': dataset_id,
            'task_type': 'sentiment_analysis',
            'labels': sentiment_labels
        }

# Example usage
text_labeling = TextLabelingManager(project_id='my-project')

# Text classification job
# job = text_labeling.create_text_classification_job(
#     dataset_id='projects/.../datasets/789',
#     instruction_uri='gs://my-bucket/text-instructions.pdf',
#     annotation_labels=['spam', 'not_spam'],
#     multi_label=False
# )

# Entity extraction job
# job = text_labeling.create_entity_extraction_job(
#     dataset_id='projects/.../datasets/789',
#     instruction_uri='gs://my-bucket/entity-instructions.pdf',
#     entity_types=['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'PRODUCT']
# )
```

---

## 5. Quality Assurance

### Implement Quality Control

```python
class QualityAssuranceManager:
    """Manage labeling quality assurance."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
    
    def configure_multi_labeler_consensus(
        self,
        num_labelers=3,
        consensus_threshold=0.7
    ):
        """Configure multiple labelers for consensus.
        
        Args:
            num_labelers: Number of labelers per item
            consensus_threshold: Agreement threshold (0-1)
        """
        
        config = {
            'replica_count': num_labelers,
            'consensus_config': {
                'agreement_threshold': consensus_threshold
            }
        }
        
        print(f"✓ Configured multi-labeler consensus")
        print(f"  Labelers per item: {num_labelers}")
        print(f"  Consensus threshold: {consensus_threshold}")
        
        return config
    
    def configure_quality_evaluation(
        self,
        evaluation_sample_percentage=10,
        evaluation_job_id=None
    ):
        """Configure quality evaluation with gold standard data.
        
        Args:
            evaluation_sample_percentage: Percentage of items for evaluation
        """
        
        config = {
            'evaluation_sample_percentage': evaluation_sample_percentage,
            'evaluation_job_id': evaluation_job_id
        }
        
        print(f"✓ Configured quality evaluation")
        print(f"  Evaluation sample: {evaluation_sample_percentage}%")
        
        return config
    
    def create_gold_standard_dataset(
        self,
        dataset_id,
        labeled_examples_uri
    ):
        """Create gold standard dataset for quality checks."""
        
        print(f"✓ Creating gold standard dataset")
        print(f"  Source dataset: {dataset_id}")
        print(f"  Gold standard URI: {labeled_examples_uri}")
        
        # Load pre-labeled examples
        # These will be used to evaluate labeler accuracy
        
        return {
            'dataset_id': dataset_id,
            'gold_standard_uri': labeled_examples_uri
        }
    
    def calculate_labeler_accuracy(
        self,
        labeling_job_id,
        gold_standard_labels
    ):
        """Calculate individual labeler accuracy."""
        
        # Mock calculation - actual implementation would query job results
        
        labeler_stats = {
            'labeler_001': {'accuracy': 0.95, 'items_labeled': 1000},
            'labeler_002': {'accuracy': 0.92, 'items_labeled': 950},
            'labeler_003': {'accuracy': 0.88, 'items_labeled': 1100}
        }
        
        print(f"\n=== Labeler Accuracy Report ===\n")
        
        for labeler_id, stats in labeler_stats.items():
            print(f"{labeler_id}:")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Items labeled: {stats['items_labeled']}")
            print()
        
        return labeler_stats
    
    def review_low_confidence_items(
        self,
        labeling_job_id,
        confidence_threshold=0.5
    ):
        """Identify items needing review."""
        
        print(f"Reviewing items with confidence < {confidence_threshold}")
        
        # Mock low-confidence items
        low_confidence_items = [
            {'item_id': 'item_001', 'confidence': 0.45, 'labels': ['cat', 'dog']},
            {'item_id': 'item_002', 'confidence': 0.40, 'labels': ['bicycle', 'motorcycle']},
        ]
        
        print(f"\n⚠ Found {len(low_confidence_items)} low-confidence items:")
        for item in low_confidence_items:
            print(f"  {item['item_id']}: confidence={item['confidence']}, labels={item['labels']}")
        
        return low_confidence_items

# Example usage
qa_manager = QualityAssuranceManager(project_id='my-project')

# Configure consensus
# config = qa_manager.configure_multi_labeler_consensus(
#     num_labelers=3,
#     consensus_threshold=0.8
# )

# Calculate accuracy
# accuracy = qa_manager.calculate_labeler_accuracy(
#     labeling_job_id='job-123',
#     gold_standard_labels={'item_001': 'cat', 'item_002': 'dog'}
# )
```

---

## 6. Active Learning

### Optimize Labeling with Active Learning

```python
class ActiveLearningManager:
    """Implement active learning strategies."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
    
    def select_uncertain_samples(
        self,
        model_predictions,
        uncertainty_threshold=0.5,
        sample_size=100
    ):
        """Select most uncertain samples for labeling.
        
        Args:
            model_predictions: Dict of {item_id: {label: confidence}}
            uncertainty_threshold: Maximum confidence for uncertainty
        """
        
        import numpy as np
        
        uncertain_items = []
        
        for item_id, predictions in model_predictions.items():
            # Calculate uncertainty (1 - max confidence)
            max_confidence = max(predictions.values())
            uncertainty = 1 - max_confidence
            
            if max_confidence < uncertainty_threshold:
                uncertain_items.append({
                    'item_id': item_id,
                    'uncertainty': uncertainty,
                    'predictions': predictions
                })
        
        # Sort by uncertainty (descending)
        uncertain_items.sort(key=lambda x: x['uncertainty'], reverse=True)
        
        # Select top samples
        selected = uncertain_items[:sample_size]
        
        print(f"✓ Selected {len(selected)} uncertain samples for labeling")
        print(f"  Uncertainty threshold: {uncertainty_threshold}")
        print(f"  Average uncertainty: {np.mean([x['uncertainty'] for x in selected]):.3f}")
        
        return selected
    
    def select_diverse_samples(
        self,
        embeddings,
        sample_size=100,
        method='kmeans'
    ):
        """Select diverse samples using clustering.
        
        Args:
            embeddings: Dict of {item_id: embedding_vector}
            method: 'kmeans' or 'random'
        """
        
        from sklearn.cluster import KMeans
        import numpy as np
        
        item_ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[id] for id in item_ids])
        
        if method == 'kmeans':
            # Cluster and select samples from each cluster
            kmeans = KMeans(n_clusters=sample_size)
            kmeans.fit(embedding_matrix)
            
            # Select closest sample to each cluster center
            selected_indices = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(embedding_matrix - center, axis=1)
                selected_indices.append(np.argmin(distances))
            
            selected_ids = [item_ids[i] for i in selected_indices]
        else:
            # Random sampling
            selected_ids = np.random.choice(item_ids, size=sample_size, replace=False).tolist()
        
        print(f"✓ Selected {len(selected_ids)} diverse samples")
        print(f"  Method: {method}")
        
        return selected_ids
    
    def implement_iterative_labeling(
        self,
        dataset_id,
        initial_sample_size=100,
        iteration_sample_size=50,
        max_iterations=10
    ):
        """Implement iterative active learning workflow."""
        
        print(f"=== Active Learning Workflow ===\n")
        print(f"Initial sample size: {initial_sample_size}")
        print(f"Iteration sample size: {iteration_sample_size}")
        print(f"Max iterations: {max_iterations}")
        
        workflow = {
            'dataset_id': dataset_id,
            'iterations': []
        }
        
        for i in range(max_iterations):
            iteration = {
                'iteration': i + 1,
                'sample_size': initial_sample_size if i == 0 else iteration_sample_size,
                'steps': [
                    'Select samples (uncertainty/diversity)',
                    'Send for labeling',
                    'Receive labels',
                    'Retrain model',
                    'Evaluate performance',
                    'Check stopping criteria'
                ]
            }
            
            workflow['iterations'].append(iteration)
            
            print(f"\nIteration {i + 1}:")
            print(f"  Sample size: {iteration['sample_size']}")
        
        print(f"\n✓ Active learning workflow configured")
        
        return workflow

# Example usage
active_learning = ActiveLearningManager(project_id='my-project')

# Select uncertain samples
# model_predictions = {
#     'item_001': {'cat': 0.45, 'dog': 0.55},
#     'item_002': {'cat': 0.95, 'dog': 0.05},
#     'item_003': {'cat': 0.48, 'dog': 0.52}
# }
# uncertain = active_learning.select_uncertain_samples(
#     model_predictions=model_predictions,
#     uncertainty_threshold=0.6,
#     sample_size=100
# )
```

---

## 7. Labeling Instructions

### Create Effective Labeling Guidelines

```python
class LabelingInstructionsManager:
    """Create and manage labeling instructions."""
    
    def __init__(self, project_id):
        self.project_id = project_id
    
    def create_instruction_document(
        self,
        task_type,
        annotation_labels,
        examples_per_label=3
    ):
        """Generate labeling instruction template."""
        
        instruction_template = f"""
# Labeling Instructions - {task_type}

## Task Overview
Annotate items with the appropriate labels from the provided list.

## Annotation Labels
{self._format_labels(annotation_labels)}

## Guidelines

### General Rules
1. Review each item carefully before labeling
2. If unsure, use the "unclear" option or skip
3. Be consistent with label application
4. Follow the examples provided

### Quality Standards
- Accuracy: Ensure correct label selection
- Consistency: Apply same criteria across all items
- Completeness: Label all required elements

## Examples
{self._generate_examples(annotation_labels, examples_per_label)}

## Edge Cases
- Multiple valid labels: Choose the primary/most prominent
- Ambiguous items: Mark as "unclear"
- Poor quality: Flag for review

## Support
Contact: labeling-support@company.com
"""
        
        print(f"✓ Created instruction document")
        print(f"  Task type: {task_type}")
        print(f"  Labels: {len(annotation_labels)}")
        
        return instruction_template
    
    def _format_labels(self, labels):
        """Format label list."""
        return '\n'.join([f"- **{label}**: Description here" for label in labels])
    
    def _generate_examples(self, labels, count):
        """Generate example section."""
        return f"[{count} example(s) for each of {len(labels)} labels would be provided here]"
    
    def upload_instructions_to_gcs(
        self,
        instruction_content,
        gcs_bucket,
        filename='labeling_instructions.pdf'
    ):
        """Upload instructions to Cloud Storage."""
        
        from google.cloud import storage
        
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(filename)
        
        # In practice, would convert to PDF first
        blob.upload_from_string(instruction_content)
        
        gcs_uri = f'gs://{gcs_bucket}/{filename}'
        
        print(f"✓ Uploaded instructions to GCS")
        print(f"  URI: {gcs_uri}")
        
        return gcs_uri

# Example usage
instructions_manager = LabelingInstructionsManager(project_id='my-project')

# Create instructions
# instructions = instructions_manager.create_instruction_document(
#     task_type='Image Classification',
#     annotation_labels=['dog', 'cat', 'bird', 'other'],
#     examples_per_label=3
# )

# Upload to GCS
# uri = instructions_manager.upload_instructions_to_gcs(
#     instruction_content=instructions,
#     gcs_bucket='my-labeling-bucket',
#     filename='pet_classification_instructions.pdf'
# )
```

---

## 8. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Create datasets for labeling
- [ ] Import data from GCS or BigQuery
- [ ] Prepare labeling instructions
- [ ] Configure IAM permissions

### Job Configuration
- [ ] Define annotation labels clearly
- [ ] Create detailed labeling instructions with examples
- [ ] Upload instructions to Cloud Storage
- [ ] Configure specialist pools if needed
- [ ] Set quality assurance parameters

### Quality Control
- [ ] Use multiple labelers for consensus
- [ ] Create gold standard dataset
- [ ] Configure evaluation metrics
- [ ] Monitor labeler accuracy
- [ ] Review low-confidence items

### Active Learning
- [ ] Start with diverse initial sample
- [ ] Implement uncertainty sampling
- [ ] Use iterative labeling approach
- [ ] Monitor model improvement
- [ ] Adjust sampling strategy

### Best Practices
- [ ] Provide clear, unambiguous instructions
- [ ] Include positive and negative examples
- [ ] Test instructions with pilot batch
- [ ] Monitor labeling quality continuously
- [ ] Regular feedback to labelers
- [ ] Document edge cases and resolutions

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
