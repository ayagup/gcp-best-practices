# Video Intelligence AI Best Practices

*Last Updated: January 4, 2026*

## Overview

Video Intelligence AI provides video analysis capabilities including label detection, shot change detection, explicit content detection, object tracking, and speech transcription using advanced machine learning models.

---

## 1. Video Analysis Best Practices

### Label Detection

**Best Practices:**
- Use appropriate detection modes
- Filter by confidence thresholds
- Leverage shot and frame-level labels
- Handle large videos with Cloud Storage

```python
from google.cloud import videointelligence_v1 as videointelligence
import io

def analyze_labels(video_uri, mode='SHOT_MODE'):
    """Detect labels in video at shot, frame, or segment level."""
    
    client = videointelligence.VideoIntelligenceServiceClient()
    
    features = [videointelligence.Feature.LABEL_DETECTION]
    
    # Detection modes: SHOT_MODE, FRAME_MODE, SHOT_AND_FRAME_MODE
    mode_map = {
        'SHOT_MODE': videointelligence.LabelDetectionMode.SHOT_MODE,
        'FRAME_MODE': videointelligence.LabelDetectionMode.FRAME_MODE,
        'SHOT_AND_FRAME_MODE': videointelligence.LabelDetectionMode.SHOT_AND_FRAME_MODE,
    }
    
    config = videointelligence.LabelDetectionConfig(
        label_detection_mode=mode_map.get(mode, videointelligence.LabelDetectionMode.SHOT_MODE),
        stationary_camera=False,
        model='builtin/latest',
    )
    
    context = videointelligence.VideoContext(
        label_detection_config=config
    )
    
    print(f"Processing video: {video_uri}")
    
    operation = client.annotate_video(
        request={
            'features': features,
            'input_uri': video_uri,
            'video_context': context,
        }
    )
    
    print('Waiting for operation to complete...')
    result = operation.result(timeout=600)
    
    # Process segment labels
    print("\n=== Segment Labels ===")
    segment_labels = result.annotation_results[0].segment_label_annotations
    
    for label in segment_labels[:10]:  # Top 10
        print(f"\nLabel: {label.entity.description}")
        
        for category in label.category_entities:
            print(f"  Category: {category.description}")
        
        for segment in label.segments[:3]:  # First 3 segments
            confidence = segment.confidence
            start = segment.segment.start_time_offset.total_seconds()
            end = segment.segment.end_time_offset.total_seconds()
            
            print(f"  Segment: {start:.1f}s - {end:.1f}s (confidence: {confidence:.2%})")
    
    # Process shot labels if available
    if mode in ['SHOT_MODE', 'SHOT_AND_FRAME_MODE']:
        print("\n=== Shot Labels ===")
        shot_labels = result.annotation_results[0].shot_label_annotations
        
        for label in shot_labels[:5]:
            print(f"\nLabel: {label.entity.description}")
            
            for segment in label.segments[:2]:
                confidence = segment.confidence
                start = segment.segment.start_time_offset.total_seconds()
                end = segment.segment.end_time_offset.total_seconds()
                
                print(f"  Shot: {start:.1f}s - {end:.1f}s (confidence: {confidence:.2%})")
    
    return result

# Example usage
video_gcs_uri = 'gs://my-bucket/video.mp4'
analyze_labels(video_gcs_uri, mode='SHOT_MODE')
```

### Shot Change Detection

```python
def detect_shots(video_uri):
    """Detect shot changes in video."""
    
    client = videointelligence.VideoIntelligenceServiceClient()
    
    features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
    
    print(f"Detecting shots in: {video_uri}")
    
    operation = client.annotate_video(
        request={
            'features': features,
            'input_uri': video_uri,
        }
    )
    
    result = operation.result(timeout=600)
    
    shots = result.annotation_results[0].shot_annotations
    
    print(f"\nDetected {len(shots)} shots:\n")
    
    for i, shot in enumerate(shots, 1):
        start = shot.start_time_offset.total_seconds()
        end = shot.end_time_offset.total_seconds()
        duration = end - start
        
        print(f"Shot {i}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
    
    return shots
```

---

## 2. Object Tracking

```python
def track_objects(video_uri):
    """Track objects throughout video."""
    
    client = videointelligence.VideoIntelligenceServiceClient()
    
    features = [videointelligence.Feature.OBJECT_TRACKING]
    
    print(f"Tracking objects in: {video_uri}")
    
    operation = client.annotate_video(
        request={
            'features': features,
            'input_uri': video_uri,
        }
    )
    
    result = operation.result(timeout=600)
    
    object_annotations = result.annotation_results[0].object_annotations
    
    print(f"\nDetected {len(object_annotations)} tracked objects:\n")
    
    for obj in object_annotations[:10]:  # Top 10 objects
        entity = obj.entity.description
        confidence = obj.confidence
        
        print(f"\nObject: {entity} (confidence: {confidence:.2%})")
        print(f"  Frames: {len(obj.frames)}")
        
        # First and last frame positions
        if obj.frames:
            first_frame = obj.frames[0]
            last_frame = obj.frames[-1]
            
            first_time = first_frame.time_offset.total_seconds()
            last_time = last_frame.time_offset.total_seconds()
            
            print(f"  Duration: {first_time:.2f}s - {last_time:.2f}s")
            
            # Bounding box of first frame
            box = first_frame.normalized_bounding_box
            print(f"  First frame position:")
            print(f"    Left: {box.left:.3f}, Top: {box.top:.3f}")
            print(f"    Right: {box.right:.3f}, Bottom: {box.bottom:.3f}")
    
    return object_annotations

def track_objects_with_filter(video_uri, entity_filter=None):
    """Track specific objects with filtering."""
    
    client = videointelligence.VideoIntelligenceServiceClient()
    
    features = [videointelligence.Feature.OBJECT_TRACKING]
    
    # Configure object tracking
    config = videointelligence.ObjectTrackingConfig(
        model='builtin/latest',
    )
    
    context = videointelligence.VideoContext(
        object_tracking_config=config
    )
    
    operation = client.annotate_video(
        request={
            'features': features,
            'input_uri': video_uri,
            'video_context': context,
        }
    )
    
    result = operation.result(timeout=600)
    
    object_annotations = result.annotation_results[0].object_annotations
    
    # Filter objects
    if entity_filter:
        filtered = [
            obj for obj in object_annotations
            if any(keyword.lower() in obj.entity.description.lower()
                   for keyword in entity_filter)
        ]
    else:
        filtered = object_annotations
    
    print(f"Found {len(filtered)} objects matching filter")
    
    return filtered

# Example
track_objects_with_filter(
    'gs://my-bucket/video.mp4',
    entity_filter=['person', 'car', 'dog']
)
```

---

## 3. Explicit Content Detection

```python
def detect_explicit_content(video_uri):
    """Detect explicit or sensitive content in video."""
    
    client = videointelligence.VideoIntelligenceServiceClient()
    
    features = [videointelligence.Feature.EXPLICIT_CONTENT_DETECTION]
    
    print(f"Analyzing explicit content in: {video_uri}")
    
    operation = client.annotate_video(
        request={
            'features': features,
            'input_uri': video_uri,
        }
    )
    
    result = operation.result(timeout=600)
    
    explicit_annotation = result.annotation_results[0].explicit_annotation
    
    # Likelihood levels
    likelihood_names = [
        'UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
        'LIKELY', 'VERY_LIKELY'
    ]
    
    print("\nExplicit content analysis:\n")
    
    frames_by_likelihood = {name: [] for name in likelihood_names}
    
    for frame in explicit_annotation.frames:
        time = frame.time_offset.total_seconds()
        likelihood = videointelligence.Likelihood(frame.pornography_likelihood).name
        
        frames_by_likelihood[likelihood].append(time)
    
    # Summary
    print("Summary by likelihood:")
    for likelihood, times in frames_by_likelihood.items():
        if times:
            print(f"  {likelihood}: {len(times)} frames")
    
    # Flag high-risk frames
    high_risk = (
        frames_by_likelihood.get('LIKELY', []) +
        frames_by_likelihood.get('VERY_LIKELY', [])
    )
    
    if high_risk:
        print(f"\n⚠️  High-risk frames detected at:")
        for time in high_risk[:10]:  # First 10
            print(f"    {time:.2f}s")
    else:
        print("\n✓ No high-risk content detected")
    
    return explicit_annotation
```

---

## 4. Speech Transcription from Video

```python
def transcribe_video_speech(video_uri, language_code='en-US'):
    """Transcribe speech from video audio."""
    
    client = videointelligence.VideoIntelligenceServiceClient()
    
    features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]
    
    # Speech transcription config
    config = videointelligence.SpeechTranscriptionConfig(
        language_code=language_code,
        enable_automatic_punctuation=True,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
    )
    
    context = videointelligence.VideoContext(
        speech_transcription_config=config
    )
    
    print(f"Transcribing speech from: {video_uri}")
    
    operation = client.annotate_video(
        request={
            'features': features,
            'input_uri': video_uri,
            'video_context': context,
        }
    )
    
    result = operation.result(timeout=600)
    
    annotation_results = result.annotation_results[0]
    
    print("\nTranscription:\n")
    
    for speech_transcription in annotation_results.speech_transcriptions:
        # Best alternative
        alternative = speech_transcription.alternatives[0]
        
        print(f"Transcript: {alternative.transcript}")
        print(f"Confidence: {alternative.confidence:.2%}\n")
        
        # Word-level details
        if alternative.words:
            print("Word timings (first 10):")
            for word_info in alternative.words[:10]:
                word = word_info.word
                start = word_info.start_time.total_seconds()
                end = word_info.end_time.total_seconds()
                confidence = word_info.confidence
                
                print(f"  '{word}' ({start:.2f}s - {end:.2f}s): {confidence:.2%}")
    
    return annotation_results.speech_transcriptions
```

---

## 5. Text Detection (OCR in Video)

```python
def detect_text_in_video(video_uri):
    """Detect and track text in video frames."""
    
    client = videointelligence.VideoIntelligenceServiceClient()
    
    features = [videointelligence.Feature.TEXT_DETECTION]
    
    print(f"Detecting text in: {video_uri}")
    
    operation = client.annotate_video(
        request={
            'features': features,
            'input_uri': video_uri,
        }
    )
    
    result = operation.result(timeout=600)
    
    text_annotations = result.annotation_results[0].text_annotations
    
    print(f"\nDetected {len(text_annotations)} text segments:\n")
    
    for text_annotation in text_annotations[:10]:  # First 10
        text = text_annotation.text
        
        print(f"\nText: '{text}'")
        
        # Show where text appears
        for segment in text_annotation.segments[:3]:  # First 3 appearances
            confidence = segment.confidence
            
            if segment.frames:
                frame = segment.frames[0]
                time = frame.time_offset.total_seconds()
                
                print(f"  Appears at: {time:.2f}s (confidence: {confidence:.2%})")
    
    return text_annotations
```

---

## 6. Batch Video Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class VideoAnalysisPipeline:
    """Pipeline for batch video analysis."""
    
    def __init__(self):
        self.client = videointelligence.VideoIntelligenceServiceClient()
    
    def analyze_single_video(self, video_uri, features):
        """Analyze single video with specified features."""
        
        print(f"Starting analysis: {video_uri}")
        
        operation = self.client.annotate_video(
            request={
                'features': features,
                'input_uri': video_uri,
            }
        )
        
        result = operation.result(timeout=600)
        
        print(f"✓ Completed: {video_uri}")
        
        return {
            'video_uri': video_uri,
            'result': result
        }
    
    def analyze_batch(self, video_uris, features, max_workers=3):
        """Analyze multiple videos in parallel."""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.analyze_single_video, uri, features): uri
                for uri in video_uris
            }
            
            for future in as_completed(futures):
                uri = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"✗ Error processing {uri}: {e}")
        
        return results
    
    def comprehensive_analysis(self, video_uri):
        """Run comprehensive analysis with multiple features."""
        
        features = [
            videointelligence.Feature.LABEL_DETECTION,
            videointelligence.Feature.SHOT_CHANGE_DETECTION,
            videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
            videointelligence.Feature.OBJECT_TRACKING,
            videointelligence.Feature.TEXT_DETECTION,
            videointelligence.Feature.SPEECH_TRANSCRIPTION,
        ]
        
        config = videointelligence.SpeechTranscriptionConfig(
            language_code='en-US',
            enable_automatic_punctuation=True,
        )
        
        context = videointelligence.VideoContext(
            speech_transcription_config=config
        )
        
        print(f"Running comprehensive analysis on: {video_uri}")
        
        operation = self.client.annotate_video(
            request={
                'features': features,
                'input_uri': video_uri,
                'video_context': context,
            }
        )
        
        result = operation.result(timeout=1200)  # 20 minutes
        
        annotation = result.annotation_results[0]
        
        summary = {
            'labels': len(annotation.segment_label_annotations),
            'shots': len(annotation.shot_annotations),
            'objects': len(annotation.object_annotations),
            'text_detections': len(annotation.text_annotations),
            'transcriptions': len(annotation.speech_transcriptions),
        }
        
        print("\nAnalysis Summary:")
        for key, count in summary.items():
            print(f"  {key}: {count}")
        
        return result

# Example usage
pipeline = VideoAnalysisPipeline()

# Batch processing
video_uris = [
    'gs://my-bucket/video1.mp4',
    'gs://my-bucket/video2.mp4',
    'gs://my-bucket/video3.mp4',
]

features = [
    videointelligence.Feature.LABEL_DETECTION,
    videointelligence.Feature.SHOT_CHANGE_DETECTION,
]

batch_results = pipeline.analyze_batch(video_uris, features, max_workers=3)
```

---

## 7. Cost Optimization

```python
def optimize_video_analysis(video_uri, video_length_seconds):
    """Choose cost-effective analysis options."""
    
    client = videointelligence.VideoIntelligenceServiceClient()
    
    # For short videos < 60s, use more features
    if video_length_seconds < 60:
        features = [
            videointelligence.Feature.LABEL_DETECTION,
            videointelligence.Feature.SHOT_CHANGE_DETECTION,
            videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
        ]
    else:
        # For longer videos, be selective
        features = [
            videointelligence.Feature.LABEL_DETECTION,
            videointelligence.Feature.SHOT_CHANGE_DETECTION,
        ]
    
    # Use SHOT_MODE instead of FRAME_MODE for labels
    config = videointelligence.LabelDetectionConfig(
        label_detection_mode=videointelligence.LabelDetectionMode.SHOT_MODE,
    )
    
    context = videointelligence.VideoContext(
        label_detection_config=config
    )
    
    operation = client.annotate_video(
        request={
            'features': features,
            'input_uri': video_uri,
            'video_context': context,
        }
    )
    
    result = operation.result(timeout=600)
    
    return result
```

---

## 8. Quick Reference Checklist

### Setup
- [ ] Enable Video Intelligence API
- [ ] Configure authentication
- [ ] Upload videos to Cloud Storage
- [ ] Verify video format compatibility
- [ ] Test with sample videos

### Analysis
- [ ] Choose appropriate features
- [ ] Configure detection modes
- [ ] Set confidence thresholds
- [ ] Handle long-running operations
- [ ] Process results efficiently

### Production
- [ ] Implement error handling
- [ ] Monitor API usage
- [ ] Optimize for cost
- [ ] Cache results when possible
- [ ] Handle quota limits

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
