# Best Practices for Media and Entertainment AI on Google Cloud

## Overview

Media and Entertainment AI on Google Cloud provides specialized services for content analysis, moderation, video intelligence, and media processing. These services enable media companies to automate content workflows, ensure brand safety, personalize user experiences, and extract insights from video content at scale.

## 1. Video Intelligence API

### 1.1 Video Analysis and Annotation

```python
from google.cloud import videointelligence_v1 as videointelligence
from google.cloud.videointelligence_v1 import enums
from typing import List, Dict, Any
import os

class VideoIntelligenceManager:
    """Manager for Video Intelligence API operations."""
    
    def __init__(self, project_id: str):
        """
        Initialize Video Intelligence Manager.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.client = videointelligence.VideoIntelligenceServiceClient()
    
    def analyze_video_labels(
        self,
        gcs_uri: str,
        mode: str = 'SHOT_MODE'
    ) -> Dict[str, Any]:
        """
        Detect labels in video.
        
        Args:
            gcs_uri: GCS URI of video file
            mode: Detection mode (SHOT_MODE, FRAME_MODE, SHOT_AND_FRAME_MODE)
            
        Returns:
            Dictionary with detected labels
        """
        # Configure label detection
        features = [videointelligence.Feature.LABEL_DETECTION]
        
        # Set mode
        if mode == 'SHOT_MODE':
            label_mode = videointelligence.LabelDetectionMode.SHOT_MODE
        elif mode == 'FRAME_MODE':
            label_mode = videointelligence.LabelDetectionMode.FRAME_MODE
        else:
            label_mode = videointelligence.LabelDetectionMode.SHOT_AND_FRAME_MODE
        
        config = videointelligence.LabelDetectionConfig(
            label_detection_mode=label_mode
        )
        
        context = videointelligence.VideoContext(
            label_detection_config=config
        )
        
        # Start annotation
        operation = self.client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri,
                "video_context": context
            }
        )
        
        print(f"Processing video: {gcs_uri}")
        result = operation.result(timeout=600)
        
        # Extract results
        annotation_results = result.annotation_results[0]
        
        labels = {
            'segment_labels': [],
            'shot_labels': [],
            'frame_labels': []
        }
        
        # Process segment labels
        for label in annotation_results.segment_label_annotations:
            labels['segment_labels'].append({
                'entity': label.entity.description,
                'confidence': label.segments[0].confidence,
                'categories': [cat.description for cat in label.category_entities]
            })
        
        # Process shot labels
        for label in annotation_results.shot_label_annotations:
            for segment in label.segments:
                labels['shot_labels'].append({
                    'entity': label.entity.description,
                    'confidence': segment.confidence,
                    'start_time': segment.segment.start_time_offset.seconds,
                    'end_time': segment.segment.end_time_offset.seconds
                })
        
        return labels
    
    def detect_shot_changes(
        self,
        gcs_uri: str
    ) -> List[Dict[str, float]]:
        """
        Detect shot changes in video.
        
        Args:
            gcs_uri: GCS URI of video file
            
        Returns:
            List of shot change timestamps
        """
        features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
        
        operation = self.client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri
            }
        )
        
        result = operation.result(timeout=600)
        annotation_results = result.annotation_results[0]
        
        shots = []
        for shot in annotation_results.shot_annotations:
            shots.append({
                'start_time': shot.start_time_offset.seconds + 
                             shot.start_time_offset.microseconds / 1e6,
                'end_time': shot.end_time_offset.seconds + 
                           shot.end_time_offset.microseconds / 1e6
            })
        
        return shots
    
    def transcribe_video_speech(
        self,
        gcs_uri: str,
        language_code: str = 'en-US'
    ) -> Dict[str, Any]:
        """
        Transcribe speech in video.
        
        Args:
            gcs_uri: GCS URI of video file
            language_code: Language code
            
        Returns:
            Dictionary with transcription results
        """
        features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]
        
        config = videointelligence.SpeechTranscriptionConfig(
            language_code=language_code,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True
        )
        
        context = videointelligence.VideoContext(
            speech_transcription_config=config
        )
        
        operation = self.client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri,
                "video_context": context
            }
        )
        
        result = operation.result(timeout=600)
        annotation_results = result.annotation_results[0]
        
        transcriptions = []
        for transcription in annotation_results.speech_transcriptions:
            for alternative in transcription.alternatives:
                transcriptions.append({
                    'transcript': alternative.transcript,
                    'confidence': alternative.confidence,
                    'words': [
                        {
                            'word': word.word,
                            'start_time': word.start_time.seconds + word.start_time.microseconds / 1e6,
                            'end_time': word.end_time.seconds + word.end_time.microseconds / 1e6
                        }
                        for word in alternative.words
                    ]
                })
        
        return {'transcriptions': transcriptions}
    
    def detect_text_in_video(
        self,
        gcs_uri: str
    ) -> List[Dict[str, Any]]:
        """
        Detect text in video frames.
        
        Args:
            gcs_uri: GCS URI of video file
            
        Returns:
            List of detected text annotations
        """
        features = [videointelligence.Feature.TEXT_DETECTION]
        
        operation = self.client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri
            }
        )
        
        result = operation.result(timeout=600)
        annotation_results = result.annotation_results[0]
        
        text_annotations = []
        for annotation in annotation_results.text_annotations:
            text_annotations.append({
                'text': annotation.text,
                'segments': [
                    {
                        'start_time': segment.segment.start_time_offset.seconds,
                        'end_time': segment.segment.end_time_offset.seconds,
                        'confidence': segment.confidence
                    }
                    for segment in annotation.segments
                ]
            })
        
        return text_annotations
    
    def detect_logos(
        self,
        gcs_uri: str
    ) -> List[Dict[str, Any]]:
        """
        Detect logos in video.
        
        Args:
            gcs_uri: GCS URI of video file
            
        Returns:
            List of detected logos
        """
        features = [videointelligence.Feature.LOGO_RECOGNITION]
        
        operation = self.client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri
            }
        )
        
        result = operation.result(timeout=600)
        annotation_results = result.annotation_results[0]
        
        logos = []
        for annotation in annotation_results.logo_recognition_annotations:
            logos.append({
                'entity': annotation.entity.description,
                'tracks': [
                    {
                        'start_time': track.segment.start_time_offset.seconds,
                        'end_time': track.segment.end_time_offset.seconds,
                        'confidence': track.confidence
                    }
                    for track in annotation.tracks
                ]
            })
        
        return logos
    
    def detect_people(
        self,
        gcs_uri: str,
        include_pose_landmarks: bool = True,
        include_attributes: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect people in video.
        
        Args:
            gcs_uri: GCS URI of video file
            include_pose_landmarks: Include pose landmarks
            include_attributes: Include detected attributes
            
        Returns:
            List of detected people
        """
        features = [videointelligence.Feature.PERSON_DETECTION]
        
        config = videointelligence.PersonDetectionConfig(
            include_bounding_boxes=True,
            include_pose_landmarks=include_pose_landmarks,
            include_attributes=include_attributes
        )
        
        context = videointelligence.VideoContext(
            person_detection_config=config
        )
        
        operation = self.client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri,
                "video_context": context
            }
        )
        
        result = operation.result(timeout=600)
        annotation_results = result.annotation_results[0]
        
        people = []
        for annotation in annotation_results.person_detection_annotations:
            people.append({
                'tracks': [
                    {
                        'start_time': track.segment.start_time_offset.seconds,
                        'end_time': track.segment.end_time_offset.seconds,
                        'confidence': track.confidence,
                        'attributes': [
                            {
                                'name': attr.name,
                                'value': attr.value,
                                'confidence': attr.confidence
                            }
                            for attr in track.attributes
                        ]
                    }
                    for track in annotation.tracks
                ]
            })
        
        return people


# Example usage
manager = VideoIntelligenceManager(
    project_id='my-project'
)

# Detect labels
labels = manager.analyze_video_labels(
    gcs_uri='gs://my-bucket/video.mp4',
    mode='SHOT_MODE'
)

# Detect shot changes
shots = manager.detect_shot_changes(
    gcs_uri='gs://my-bucket/video.mp4'
)

# Transcribe speech
transcription = manager.transcribe_video_speech(
    gcs_uri='gs://my-bucket/video.mp4',
    language_code='en-US'
)
```

### 1.2 Content Moderation

```python
class ContentModerationManager:
    """Manager for content moderation operations."""
    
    def __init__(self, project_id: str):
        """
        Initialize Content Moderation Manager.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.client = videointelligence.VideoIntelligenceServiceClient()
    
    def detect_explicit_content(
        self,
        gcs_uri: str,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect explicit content in video.
        
        Args:
            gcs_uri: GCS URI of video file
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with explicit content detections
        """
        features = [videointelligence.Feature.EXPLICIT_CONTENT_DETECTION]
        
        operation = self.client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri
            }
        )
        
        result = operation.result(timeout=600)
        annotation_results = result.annotation_results[0]
        
        explicit_frames = []
        for frame in annotation_results.explicit_annotation.frames:
            pornography_likelihood = frame.pornography_likelihood
            
            # Map likelihood to confidence score
            likelihood_map = {
                1: 0.0,  # VERY_UNLIKELY
                2: 0.25,  # UNLIKELY
                3: 0.5,  # POSSIBLE
                4: 0.75,  # LIKELY
                5: 1.0   # VERY_LIKELY
            }
            
            confidence = likelihood_map.get(pornography_likelihood, 0.0)
            
            if confidence >= confidence_threshold:
                explicit_frames.append({
                    'time_offset': frame.time_offset.seconds + 
                                  frame.time_offset.microseconds / 1e6,
                    'pornography_likelihood': pornography_likelihood,
                    'confidence': confidence
                })
        
        return {
            'explicit_frames': explicit_frames,
            'has_explicit_content': len(explicit_frames) > 0
        }
    
    def moderate_content(
        self,
        gcs_uri: str,
        policies: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Apply content moderation policies.
        
        Args:
            gcs_uri: GCS URI of video file
            policies: Dictionary of policy thresholds
                     (e.g., {'explicit': 0.5, 'violence': 0.7})
            
        Returns:
            Dictionary with moderation results
        """
        # Detect explicit content
        explicit_results = self.detect_explicit_content(
            gcs_uri=gcs_uri,
            confidence_threshold=policies.get('explicit', 0.5)
        )
        
        moderation_results = {
            'video_uri': gcs_uri,
            'violations': [],
            'approved': True
        }
        
        # Check explicit content policy
        if explicit_results['has_explicit_content']:
            moderation_results['violations'].append({
                'type': 'explicit_content',
                'severity': 'high',
                'frames': explicit_results['explicit_frames']
            })
            moderation_results['approved'] = False
        
        return moderation_results
    
    def create_moderation_report(
        self,
        moderation_results: Dict[str, Any]
    ) -> str:
        """
        Create moderation report.
        
        Args:
            moderation_results: Results from moderate_content
            
        Returns:
            Formatted report string
        """
        report = f"Content Moderation Report\n"
        report += f"Video: {moderation_results['video_uri']}\n"
        report += f"Status: {'APPROVED' if moderation_results['approved'] else 'REJECTED'}\n\n"
        
        if moderation_results['violations']:
            report += "Violations:\n"
            for violation in moderation_results['violations']:
                report += f"  - Type: {violation['type']}\n"
                report += f"    Severity: {violation['severity']}\n"
                report += f"    Frame Count: {len(violation['frames'])}\n"
        else:
            report += "No violations detected.\n"
        
        return report


# Example usage
moderator = ContentModerationManager(
    project_id='my-project'
)

# Moderate content
results = moderator.moderate_content(
    gcs_uri='gs://my-bucket/video.mp4',
    policies={
        'explicit': 0.5
    }
)

# Create report
report = moderator.create_moderation_report(results)
```

## 2. Content Recommendation Systems

### 2.1 Video Recommendation Engine

```python
from google.cloud import bigquery
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
import pandas as pd

class VideoRecommendationEngine:
    """Engine for video content recommendations."""
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str
    ):
        """
        Initialize Recommendation Engine.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bq_client = bigquery.Client(project=project_id)
    
    def track_user_interaction(
        self,
        user_id: str,
        video_id: str,
        interaction_type: str,
        watch_duration: float = None,
        rating: float = None
    ):
        """
        Track user interaction with video.
        
        Args:
            user_id: User identifier
            video_id: Video identifier
            interaction_type: Type of interaction (view, like, share, etc.)
            watch_duration: Duration watched in seconds
            rating: User rating (1-5)
        """
        table_id = f"{self.project_id}.{self.dataset_id}.user_interactions"
        
        row = {
            'user_id': user_id,
            'video_id': video_id,
            'interaction_type': interaction_type,
            'watch_duration': watch_duration,
            'rating': rating,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        errors = self.bq_client.insert_rows_json(table_id, [row])
        if errors:
            print(f"Errors: {errors}")
    
    def get_collaborative_recommendations(
        self,
        user_id: str,
        num_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations using collaborative filtering.
        
        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations
            
        Returns:
            List of recommended videos
        """
        # Query user-item interaction matrix
        query = f"""
        SELECT
            user_id,
            video_id,
            COUNT(*) as interaction_count,
            AVG(CASE WHEN rating IS NOT NULL THEN rating ELSE 3 END) as avg_rating
        FROM
            `{self.project_id}.{self.dataset_id}.user_interactions`
        WHERE
            interaction_type IN ('view', 'like', 'share')
        GROUP BY
            user_id, video_id
        """
        
        df = self.bq_client.query(query).to_dataframe()
        
        # Create user-item matrix
        user_item_matrix = df.pivot_table(
            index='user_id',
            columns='video_id',
            values='avg_rating',
            fill_value=0
        )
        
        if user_id not in user_item_matrix.index:
            return []
        
        # Calculate user similarity
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )
        
        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]
        
        # Get videos watched by similar users
        similar_user_videos = user_item_matrix.loc[similar_users.index]
        user_videos = user_item_matrix.loc[user_id]
        
        # Calculate recommendation scores
        recommendations = {}
        for video_id in user_item_matrix.columns:
            if user_videos[video_id] == 0:  # User hasn't watched
                score = (similar_user_videos[video_id] * similar_users).sum() / similar_users.sum()
                recommendations[video_id] = score
        
        # Sort by score
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_recommendations]
        
        return [
            {'video_id': video_id, 'score': float(score)}
            for video_id, score in sorted_recommendations
        ]
    
    def get_content_based_recommendations(
        self,
        video_id: str,
        num_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on video content similarity.
        
        Args:
            video_id: Video identifier
            num_recommendations: Number of recommendations
            
        Returns:
            List of similar videos
        """
        # Query video features
        query = f"""
        SELECT
            video_id,
            genre,
            tags,
            duration_minutes,
            language
        FROM
            `{self.project_id}.{self.dataset_id}.video_metadata`
        WHERE
            video_id = '{video_id}'
        """
        
        target_video = self.bq_client.query(query).to_dataframe().iloc[0]
        
        # Find similar videos
        query = f"""
        SELECT
            video_id,
            genre,
            tags,
            duration_minutes,
            language,
            avg_rating,
            view_count
        FROM
            `{self.project_id}.{self.dataset_id}.video_metadata`
        WHERE
            video_id != '{video_id}'
            AND genre = '{target_video['genre']}'
        ORDER BY
            avg_rating DESC,
            view_count DESC
        LIMIT {num_recommendations}
        """
        
        recommendations = self.bq_client.query(query).to_dataframe()
        
        return recommendations.to_dict('records')
    
    def get_trending_videos(
        self,
        time_window_hours: int = 24,
        num_videos: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trending videos based on recent activity.
        
        Args:
            time_window_hours: Time window in hours
            num_videos: Number of videos to return
            
        Returns:
            List of trending videos
        """
        query = f"""
        SELECT
            video_id,
            COUNT(DISTINCT user_id) as unique_viewers,
            COUNT(*) as total_interactions,
            AVG(watch_duration) as avg_watch_duration
        FROM
            `{self.project_id}.{self.dataset_id}.user_interactions`
        WHERE
            timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {time_window_hours} HOUR)
        GROUP BY
            video_id
        ORDER BY
            unique_viewers DESC,
            total_interactions DESC
        LIMIT {num_videos}
        """
        
        trending = self.bq_client.query(query).to_dataframe()
        return trending.to_dict('records')


# Example usage
rec_engine = VideoRecommendationEngine(
    project_id='my-project',
    dataset_id='media_analytics'
)

# Track interaction
rec_engine.track_user_interaction(
    user_id='user123',
    video_id='video456',
    interaction_type='view',
    watch_duration=1800.0,
    rating=4.5
)

# Get recommendations
collaborative_recs = rec_engine.get_collaborative_recommendations(
    user_id='user123',
    num_recommendations=10
)

content_recs = rec_engine.get_content_based_recommendations(
    video_id='video456',
    num_recommendations=10
)
```

## 3. Media Asset Management

### 3.1 Automated Video Processing

```python
from google.cloud import storage
from google.cloud import dataproc_v1
import json

class MediaAssetManager:
    """Manager for media asset processing."""
    
    def __init__(
        self,
        project_id: str,
        region: str = 'us-central1'
    ):
        """
        Initialize Media Asset Manager.
        
        Args:
            project_id: GCP project ID
            region: GCP region
        """
        self.project_id = project_id
        self.region = region
        self.storage_client = storage.Client()
        self.video_client = videointelligence.VideoIntelligenceServiceClient()
    
    def create_video_highlights(
        self,
        gcs_uri: str,
        output_uri: str,
        highlight_keywords: List[str]
    ) -> str:
        """
        Create video highlights based on keywords.
        
        Args:
            gcs_uri: Input video GCS URI
            output_uri: Output GCS URI for highlights
            highlight_keywords: Keywords to identify highlights
            
        Returns:
            Output URI
        """
        # Detect labels and speech
        features = [
            videointelligence.Feature.LABEL_DETECTION,
            videointelligence.Feature.SPEECH_TRANSCRIPTION
        ]
        
        config = videointelligence.SpeechTranscriptionConfig(
            language_code='en-US',
            enable_word_time_offsets=True
        )
        
        context = videointelligence.VideoContext(
            speech_transcription_config=config
        )
        
        operation = self.video_client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri,
                "video_context": context
            }
        )
        
        result = operation.result(timeout=600)
        annotation_results = result.annotation_results[0]
        
        # Find highlight segments
        highlight_segments = []
        
        # Check labels
        for label in annotation_results.segment_label_annotations:
            if label.entity.description.lower() in [kw.lower() for kw in highlight_keywords]:
                for segment in label.segments:
                    highlight_segments.append({
                        'start_time': segment.segment.start_time_offset.seconds,
                        'end_time': segment.segment.end_time_offset.seconds,
                        'reason': f"Label: {label.entity.description}"
                    })
        
        # Check speech transcription
        for transcription in annotation_results.speech_transcriptions:
            for alternative in transcription.alternatives:
                transcript_lower = alternative.transcript.lower()
                for keyword in highlight_keywords:
                    if keyword.lower() in transcript_lower:
                        # Find word timing
                        for word in alternative.words:
                            if keyword.lower() in word.word.lower():
                                highlight_segments.append({
                                    'start_time': word.start_time.seconds,
                                    'end_time': word.end_time.seconds + 5,  # Add 5s context
                                    'reason': f"Keyword: {keyword}"
                                })
        
        # Save highlight metadata
        output_bucket = output_uri.split('/')[2]
        output_blob = '/'.join(output_uri.split('/')[3:]) + '_highlights.json'
        
        bucket = self.storage_client.bucket(output_bucket)
        blob = bucket.blob(output_blob)
        blob.upload_from_string(json.dumps(highlight_segments, indent=2))
        
        return f"gs://{output_bucket}/{output_blob}"
    
    def detect_copyright_content(
        self,
        gcs_uri: str,
        reference_database: str
    ) -> Dict[str, Any]:
        """
        Detect potential copyright violations.
        
        Args:
            gcs_uri: Video GCS URI
            reference_database: Reference content database
            
        Returns:
            Dictionary with copyright detection results
        """
        # This would integrate with a copyright detection service
        # For demonstration, using Video Intelligence API
        
        features = [
            videointelligence.Feature.LOGO_RECOGNITION,
            videointelligence.Feature.TEXT_DETECTION
        ]
        
        operation = self.video_client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri
            }
        )
        
        result = operation.result(timeout=600)
        annotation_results = result.annotation_results[0]
        
        copyright_indicators = {
            'detected_logos': [],
            'detected_text': [],
            'potential_violations': []
        }
        
        # Check for known brand logos
        for logo in annotation_results.logo_recognition_annotations:
            copyright_indicators['detected_logos'].append({
                'entity': logo.entity.description,
                'tracks': len(logo.tracks)
            })
        
        # Check for copyright text
        for text in annotation_results.text_annotations:
            if any(term in text.text.lower() for term in ['copyright', '©', 'trademark', '™']):
                copyright_indicators['detected_text'].append({
                    'text': text.text,
                    'segments': len(text.segments)
                })
        
        return copyright_indicators
    
    def generate_video_thumbnails(
        self,
        gcs_uri: str,
        output_prefix: str,
        num_thumbnails: int = 5
    ) -> List[str]:
        """
        Generate video thumbnails.
        
        Args:
            gcs_uri: Video GCS URI
            output_prefix: Output GCS prefix for thumbnails
            num_thumbnails: Number of thumbnails to generate
            
        Returns:
            List of thumbnail URIs
        """
        # Detect shot changes
        features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
        
        operation = self.video_client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri
            }
        )
        
        result = operation.result(timeout=600)
        annotation_results = result.annotation_results[0]
        
        # Select representative frames
        shots = annotation_results.shot_annotations
        if len(shots) == 0:
            return []
        
        # Select evenly distributed shots
        indices = np.linspace(0, len(shots) - 1, num_thumbnails, dtype=int)
        selected_shots = [shots[i] for i in indices]
        
        thumbnail_uris = []
        for i, shot in enumerate(selected_shots):
            # Calculate middle of shot
            start = shot.start_time_offset.seconds
            end = shot.end_time_offset.seconds
            middle_time = (start + end) / 2
            
            # In production, you would extract frame at this timestamp
            # and save to GCS
            thumbnail_uri = f"{output_prefix}/thumbnail_{i}_{int(middle_time)}s.jpg"
            thumbnail_uris.append(thumbnail_uri)
        
        return thumbnail_uris


# Example usage
asset_manager = MediaAssetManager(
    project_id='my-project',
    region='us-central1'
)

# Create highlights
highlights = asset_manager.create_video_highlights(
    gcs_uri='gs://my-bucket/video.mp4',
    output_uri='gs://my-bucket/processed/',
    highlight_keywords=['goal', 'touchdown', 'winning', 'amazing']
)

# Detect copyright
copyright_results = asset_manager.detect_copyright_content(
    gcs_uri='gs://my-bucket/video.mp4',
    reference_database='my-copyright-db'
)
```

## 4. Live Streaming Analytics

### 4.1 Real-time Stream Processing

```python
from google.cloud import pubsub_v1
from google.cloud import firestore
import json
from datetime import datetime
from typing import Dict, Any

class LiveStreamAnalytics:
    """Analytics for live streaming content."""
    
    def __init__(
        self,
        project_id: str,
        topic_id: str
    ):
        """
        Initialize Live Stream Analytics.
        
        Args:
            project_id: GCP project ID
            topic_id: Pub/Sub topic ID
        """
        self.project_id = project_id
        self.topic_id = topic_id
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_id)
        self.db = firestore.Client()
    
    def track_viewer_event(
        self,
        stream_id: str,
        user_id: str,
        event_type: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Track viewer event in live stream.
        
        Args:
            stream_id: Stream identifier
            user_id: User identifier
            event_type: Event type (join, leave, chat, reaction)
            metadata: Additional event metadata
        """
        event = {
            'stream_id': stream_id,
            'user_id': user_id,
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        # Publish to Pub/Sub for real-time processing
        message_data = json.dumps(event).encode('utf-8')
        future = self.publisher.publish(self.topic_path, message_data)
        future.result()
        
        # Store in Firestore for queries
        doc_ref = self.db.collection('stream_events').document()
        doc_ref.set(event)
    
    def get_current_viewers(
        self,
        stream_id: str
    ) -> int:
        """
        Get current viewer count.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Current viewer count
        """
        # Query recent join/leave events
        stream_ref = self.db.collection('stream_events')
        query = stream_ref.where('stream_id', '==', stream_id)
        
        # Calculate active viewers (joined but not left)
        viewers = {}
        for doc in query.stream():
            event = doc.to_dict()
            user_id = event['user_id']
            event_type = event['event_type']
            
            if event_type == 'join':
                viewers[user_id] = True
            elif event_type == 'leave':
                viewers.pop(user_id, None)
        
        return len(viewers)
    
    def get_engagement_metrics(
        self,
        stream_id: str
    ) -> Dict[str, Any]:
        """
        Calculate stream engagement metrics.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Dictionary with engagement metrics
        """
        stream_ref = self.db.collection('stream_events')
        query = stream_ref.where('stream_id', '==', stream_id)
        
        metrics = {
            'total_viewers': set(),
            'peak_viewers': 0,
            'chat_messages': 0,
            'reactions': 0,
            'avg_watch_time': 0
        }
        
        current_viewers = 0
        watch_times = {}
        
        for doc in query.stream():
            event = doc.to_dict()
            user_id = event['user_id']
            event_type = event['event_type']
            
            metrics['total_viewers'].add(user_id)
            
            if event_type == 'join':
                current_viewers += 1
                watch_times[user_id] = event['timestamp']
                metrics['peak_viewers'] = max(metrics['peak_viewers'], current_viewers)
            elif event_type == 'leave':
                current_viewers -= 1
                # Calculate watch time
                if user_id in watch_times:
                    join_time = datetime.fromisoformat(watch_times[user_id])
                    leave_time = datetime.fromisoformat(event['timestamp'])
                    watch_time = (leave_time - join_time).total_seconds()
                    watch_times[user_id] = watch_time
            elif event_type == 'chat':
                metrics['chat_messages'] += 1
            elif event_type == 'reaction':
                metrics['reactions'] += 1
        
        # Calculate average watch time
        completed_watches = [t for t in watch_times.values() if isinstance(t, float)]
        if completed_watches:
            metrics['avg_watch_time'] = sum(completed_watches) / len(completed_watches)
        
        metrics['total_viewers'] = len(metrics['total_viewers'])
        
        return metrics


# Example usage
stream_analytics = LiveStreamAnalytics(
    project_id='my-project',
    topic_id='stream-events'
)

# Track events
stream_analytics.track_viewer_event(
    stream_id='live123',
    user_id='user456',
    event_type='join'
)

# Get metrics
current_viewers = stream_analytics.get_current_viewers('live123')
engagement = stream_analytics.get_engagement_metrics('live123')
```

## 5. Quick Reference Checklist

### Video Intelligence Setup
- [ ] Enable Video Intelligence API
- [ ] Set up GCS bucket for video storage
- [ ] Configure IAM permissions for video access
- [ ] Implement video annotation pipeline
- [ ] Set up result storage (BigQuery/Firestore)

### Content Moderation
- [ ] Define content policies and thresholds
- [ ] Implement explicit content detection
- [ ] Set up automated moderation workflows
- [ ] Create review queue for flagged content
- [ ] Implement appeal process

### Recommendation System
- [ ] Set up user interaction tracking
- [ ] Implement collaborative filtering
- [ ] Build content-based recommendations
- [ ] Create hybrid recommendation models
- [ ] Monitor recommendation performance

### Media Processing
- [ ] Implement video transcoding pipeline
- [ ] Set up thumbnail generation
- [ ] Configure copyright detection
- [ ] Create highlight generation workflow
- [ ] Optimize media delivery (CDN)

### Live Streaming
- [ ] Set up real-time event tracking
- [ ] Implement viewer analytics
- [ ] Configure engagement metrics
- [ ] Set up alerting for stream health
- [ ] Monitor latency and quality

### Security & Compliance
- [ ] Implement content encryption (at rest/in transit)
- [ ] Set up DRM for premium content
- [ ] Configure access controls (IAM)
- [ ] Enable audit logging
- [ ] Ensure COPPA/GDPR compliance

### Performance Optimization
- [ ] Use batch processing for large video libraries
- [ ] Implement caching for recommendations
- [ ] Optimize video encoding settings
- [ ] Use parallel processing for annotations
- [ ] Monitor API quota usage

### Cost Management
- [ ] Use lifecycle policies for old content
- [ ] Implement tiered storage (Standard/Nearline/Coldline)
- [ ] Optimize video resolution/bitrate
- [ ] Monitor API usage and costs
- [ ] Use committed use discounts

### Monitoring & Alerting
- [ ] Set up Cloud Monitoring dashboards
- [ ] Configure alerting for failures
- [ ] Monitor processing latency
- [ ] Track content moderation queue
- [ ] Monitor recommendation accuracy
