# Vertex AI Model Monitoring Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Model Monitoring provides automated detection of training-serving skew, prediction drift, feature attribution drift, and anomalies to maintain ML model quality in production.

---

## 1. Model Monitoring Setup

### Configure Monitoring Job

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring

class ModelMonitoringManager:
    """Manage Vertex AI Model Monitoring."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        
        aiplatform.init(project=project_id, location=location)
    
    def create_monitoring_job(
        self,
        endpoint_id,
        model_id,
        training_dataset_uri,
        monitoring_frequency_hours=24,
        sample_rate=0.8,
        alert_emails=None
    ):
        """Create model monitoring job."""
        
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        # Configure skew detection
        skew_config = model_monitoring.SkewDetectionConfig(
            data_source=training_dataset_uri,
            skew_thresholds={
                'feature1': 0.3,
                'feature2': 0.3
            },
            attribute_skew_thresholds={
                'feature1': 0.3
            }
        )
        
        # Configure drift detection
        drift_config = model_monitoring.DriftDetectionConfig(
            drift_thresholds={
                'feature1': 0.3,
                'feature2': 0.3
            }
        )
        
        # Configure monitoring
        objective_config = model_monitoring.ObjectiveConfig(
            skew_detection_config=skew_config,
            drift_detection_config=drift_config,
            explanation_config=model_monitoring.ExplanationConfig()
        )
        
        # Email alerts
        if alert_emails is None:
            alert_emails = []
        
        email_alert_config = model_monitoring.EmailAlertConfig(
            user_emails=alert_emails
        )
        
        # Create monitoring job
        monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
            display_name=f'monitoring-{model_id}',
            endpoint=endpoint,
            logging_sampling_strategy=model_monitoring.RandomSampleConfig(
                sample_rate=sample_rate
            ),
            schedule_config=model_monitoring.ScheduleConfig(
                monitor_interval=monitoring_frequency_hours * 3600
            ),
            alert_config=email_alert_config,
            objective_configs=[objective_config]
        )
        
        print(f"✓ Created monitoring job: {monitoring_job.display_name}")
        print(f"  Resource name: {monitoring_job.resource_name}")
        print(f"  Frequency: every {monitoring_frequency_hours} hours")
        print(f"  Sample rate: {sample_rate * 100}%")
        
        return monitoring_job
    
    def list_monitoring_jobs(self):
        """List all monitoring jobs."""
        
        jobs = aiplatform.ModelDeploymentMonitoringJob.list()
        
        print(f"\nMonitoring Jobs ({len(jobs)}):\n")
        
        for job in jobs:
            print(f"{job.display_name}")
            print(f"  State: {job.state}")
            print(f"  Created: {job.create_time}")
            print(f"  Endpoint: {job.endpoint}")
            print()
        
        return jobs

# Example usage
monitoring_manager = ModelMonitoringManager(project_id='my-project')

# Create monitoring job
# monitoring_job = monitoring_manager.create_monitoring_job(
#     endpoint_id='projects/.../endpoints/123',
#     model_id='fraud-detection-model',
#     training_dataset_uri='bq://my-project.datasets.training_data',
#     monitoring_frequency_hours=24,
#     sample_rate=0.8,
#     alert_emails=['ml-team@company.com']
# )
```

---

## 2. Skew Detection

### Detect Training-Serving Skew

```python
class SkewDetector:
    """Detect training-serving skew."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def configure_skew_detection(
        self,
        training_data_source,
        feature_thresholds,
        attribution_score_threshold=0.3
    ):
        """Configure skew detection thresholds."""
        
        skew_config = model_monitoring.SkewDetectionConfig(
            data_source=training_data_source,
            skew_thresholds=feature_thresholds,
            attribute_skew_thresholds={
                feature: attribution_score_threshold
                for feature in feature_thresholds.keys()
            },
            target_field='label'
        )
        
        print(f"✓ Configured skew detection")
        print(f"  Training data: {training_data_source}")
        print(f"  Features monitored: {len(feature_thresholds)}")
        
        return skew_config
    
    def analyze_skew_results(self, monitoring_job_id):
        """Analyze skew detection results."""
        
        from google.cloud import bigquery
        
        # Query monitoring logs
        bq_client = bigquery.Client(project=self.project_id)
        
        query = f"""
        SELECT
            feature_name,
            skew_score,
            threshold,
            training_distribution,
            serving_distribution,
            timestamp
        FROM
            `{self.project_id}.model_monitoring.skew_results`
        WHERE
            monitoring_job_id = '{monitoring_job_id}'
            AND skew_score > threshold
        ORDER BY
            skew_score DESC,
            timestamp DESC
        LIMIT 100
        """
        
        results = bq_client.query(query).to_dataframe()
        
        if len(results) > 0:
            print(f"\n⚠ Skew Detected ({len(results)} features):\n")
            
            for _, row in results.iterrows():
                print(f"Feature: {row['feature_name']}")
                print(f"  Skew score: {row['skew_score']:.4f} (threshold: {row['threshold']:.4f})")
                print(f"  Training dist: {row['training_distribution']}")
                print(f"  Serving dist: {row['serving_distribution']}")
                print(f"  Timestamp: {row['timestamp']}")
                print()
        else:
            print("✓ No skew detected")
        
        return results
    
    def get_skew_summary(self, monitoring_job_id, days=7):
        """Get skew summary over time."""
        
        from google.cloud import bigquery
        import pandas as pd
        
        bq_client = bigquery.Client(project=self.project_id)
        
        query = f"""
        SELECT
            DATE(timestamp) as date,
            feature_name,
            AVG(skew_score) as avg_skew,
            MAX(skew_score) as max_skew,
            COUNT(*) as measurement_count
        FROM
            `{self.project_id}.model_monitoring.skew_results`
        WHERE
            monitoring_job_id = '{monitoring_job_id}'
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        GROUP BY
            date, feature_name
        ORDER BY
            date DESC, avg_skew DESC
        """
        
        summary = bq_client.query(query).to_dataframe()
        
        print(f"\n=== Skew Summary (Last {days} days) ===\n")
        print(summary.to_string(index=False))
        
        return summary

# Example usage
skew_detector = SkewDetector(project_id='my-project')

# Configure skew detection
# skew_config = skew_detector.configure_skew_detection(
#     training_data_source='bq://my-project.datasets.training_data',
#     feature_thresholds={
#         'transaction_amount': 0.3,
#         'user_age': 0.25,
#         'product_category': 0.35
#     },
#     attribution_score_threshold=0.3
# )

# Analyze skew results
# skew_results = skew_detector.analyze_skew_results('monitoring-job-123')
```

---

## 3. Drift Detection

### Detect Prediction Drift

```python
class DriftDetector:
    """Detect prediction and feature drift."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def configure_drift_detection(
        self,
        feature_thresholds,
        prediction_drift_threshold=0.3
    ):
        """Configure drift detection thresholds."""
        
        drift_config = model_monitoring.DriftDetectionConfig(
            drift_thresholds=feature_thresholds,
            attribute_drift_thresholds={
                feature: prediction_drift_threshold
                for feature in feature_thresholds.keys()
            }
        )
        
        print(f"✓ Configured drift detection")
        print(f"  Features monitored: {len(feature_thresholds)}")
        print(f"  Prediction drift threshold: {prediction_drift_threshold}")
        
        return drift_config
    
    def analyze_drift_results(self, monitoring_job_id, window_hours=24):
        """Analyze drift detection results."""
        
        from google.cloud import bigquery
        
        bq_client = bigquery.Client(project=self.project_id)
        
        query = f"""
        SELECT
            feature_name,
            drift_score,
            threshold,
            baseline_distribution,
            current_distribution,
            timestamp
        FROM
            `{self.project_id}.model_monitoring.drift_results`
        WHERE
            monitoring_job_id = '{monitoring_job_id}'
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {window_hours} HOUR)
            AND drift_score > threshold
        ORDER BY
            drift_score DESC,
            timestamp DESC
        """
        
        results = bq_client.query(query).to_dataframe()
        
        if len(results) > 0:
            print(f"\n⚠ Drift Detected ({len(results)} features):\n")
            
            for _, row in results.iterrows():
                print(f"Feature: {row['feature_name']}")
                print(f"  Drift score: {row['drift_score']:.4f} (threshold: {row['threshold']:.4f})")
                print(f"  Baseline dist: {row['baseline_distribution']}")
                print(f"  Current dist: {row['current_distribution']}")
                print(f"  Timestamp: {row['timestamp']}")
                print()
        else:
            print(f"✓ No drift detected in last {window_hours} hours")
        
        return results
    
    def track_drift_over_time(self, monitoring_job_id, days=30):
        """Track drift trends over time."""
        
        from google.cloud import bigquery
        import pandas as pd
        
        bq_client = bigquery.Client(project=self.project_id)
        
        query = f"""
        SELECT
            DATE(timestamp) as date,
            feature_name,
            AVG(drift_score) as avg_drift,
            MAX(drift_score) as max_drift,
            COUNTIF(drift_score > threshold) as alerts
        FROM
            `{self.project_id}.model_monitoring.drift_results`
        WHERE
            monitoring_job_id = '{monitoring_job_id}'
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        GROUP BY
            date, feature_name
        HAVING
            max_drift > 0.2  -- Show features with significant drift
        ORDER BY
            date DESC, avg_drift DESC
        """
        
        trends = bq_client.query(query).to_dataframe()
        
        print(f"\n=== Drift Trends (Last {days} days) ===\n")
        print(trends.to_string(index=False))
        
        return trends

# Example usage
drift_detector = DriftDetector(project_id='my-project')

# Configure drift detection
# drift_config = drift_detector.configure_drift_detection(
#     feature_thresholds={
#         'transaction_amount': 0.3,
#         'user_age': 0.25,
#         'product_category': 0.35,
#         'prediction_score': 0.3
#     },
#     prediction_drift_threshold=0.3
# )

# Analyze drift
# drift_results = drift_detector.analyze_drift_results('monitoring-job-123', window_hours=24)
```

---

## 4. Anomaly Detection

### Detect Anomalies in Predictions

```python
class AnomalyDetector:
    """Detect anomalies in model predictions."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def detect_prediction_anomalies(
        self,
        endpoint_id,
        threshold_stddevs=3
    ):
        """Detect anomalies in prediction distribution."""
        
        from google.cloud import bigquery
        import numpy as np
        
        bq_client = bigquery.Client(project=self.project_id)
        
        # Get recent predictions
        query = f"""
        SELECT
            prediction_value,
            prediction_timestamp
        FROM
            `{self.project_id}.predictions.endpoint_{endpoint_id}`
        WHERE
            prediction_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        ORDER BY
            prediction_timestamp DESC
        """
        
        predictions = bq_client.query(query).to_dataframe()
        
        if len(predictions) == 0:
            print("No predictions found")
            return []
        
        # Calculate statistics
        mean = predictions['prediction_value'].mean()
        std = predictions['prediction_value'].std()
        
        # Detect anomalies
        predictions['z_score'] = np.abs(
            (predictions['prediction_value'] - mean) / std
        )
        
        anomalies = predictions[
            predictions['z_score'] > threshold_stddevs
        ]
        
        if len(anomalies) > 0:
            print(f"\n⚠ Detected {len(anomalies)} anomalous predictions:\n")
            
            for _, row in anomalies.head(10).iterrows():
                print(f"Value: {row['prediction_value']:.4f}")
                print(f"  Z-score: {row['z_score']:.2f}")
                print(f"  Timestamp: {row['prediction_timestamp']}")
                print()
        else:
            print(f"✓ No anomalies detected (threshold: {threshold_stddevs} std devs)")
        
        return anomalies
    
    def detect_feature_anomalies(
        self,
        endpoint_id,
        feature_name,
        window_hours=24
    ):
        """Detect anomalies in feature values."""
        
        from google.cloud import bigquery
        from scipy import stats
        
        bq_client = bigquery.Client(project=self.project_id)
        
        query = f"""
        SELECT
            JSON_EXTRACT_SCALAR(instances, '$.{feature_name}') as feature_value,
            prediction_timestamp
        FROM
            `{self.project_id}.predictions.endpoint_{endpoint_id}`
        WHERE
            prediction_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {window_hours} HOUR)
        ORDER BY
            prediction_timestamp DESC
        """
        
        feature_data = bq_client.query(query).to_dataframe()
        
        # Convert to numeric
        feature_data['feature_value'] = pd.to_numeric(
            feature_data['feature_value'],
            errors='coerce'
        )
        
        # Remove nulls
        feature_data = feature_data.dropna()
        
        if len(feature_data) == 0:
            print(f"No data found for feature: {feature_name}")
            return []
        
        # Use IQR method for anomaly detection
        Q1 = feature_data['feature_value'].quantile(0.25)
        Q3 = feature_data['feature_value'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = feature_data[
            (feature_data['feature_value'] < lower_bound) |
            (feature_data['feature_value'] > upper_bound)
        ]
        
        if len(anomalies) > 0:
            print(f"\n⚠ Feature '{feature_name}' anomalies: {len(anomalies)}")
            print(f"  Normal range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Anomalous values: {anomalies['feature_value'].tolist()[:10]}")
        else:
            print(f"✓ No anomalies in feature '{feature_name}'")
        
        return anomalies

# Example usage
import pandas as pd

anomaly_detector = AnomalyDetector(project_id='my-project')

# Detect prediction anomalies
# anomalies = anomaly_detector.detect_prediction_anomalies(
#     endpoint_id='123456',
#     threshold_stddevs=3
# )

# Detect feature anomalies
# feature_anomalies = anomaly_detector.detect_feature_anomalies(
#     endpoint_id='123456',
#     feature_name='transaction_amount',
#     window_hours=24
# )
```

---

## 5. Alert Configuration

### Set Up Monitoring Alerts

```python
class MonitoringAlerts:
    """Configure and manage monitoring alerts."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
    
    def create_alert_policy(
        self,
        display_name,
        condition_threshold,
        notification_channels,
        metric_type='custom.googleapis.com/vertex_ai/prediction/drift_score'
    ):
        """Create Cloud Monitoring alert policy."""
        
        from google.cloud import monitoring_v3
        
        client = monitoring_v3.AlertPolicyServiceClient()
        project_name = f"projects/{self.project_id}"
        
        # Define alert condition
        condition = monitoring_v3.AlertPolicy.Condition(
            display_name=display_name,
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter=f'resource.type="aiplatform.googleapis.com/Endpoint" AND metric.type="{metric_type}"',
                comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                threshold_value=condition_threshold,
                duration={'seconds': 300},  # 5 minutes
                aggregations=[
                    monitoring_v3.Aggregation(
                        alignment_period={'seconds': 60},
                        per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                    )
                ]
            )
        )
        
        # Create alert policy
        policy = monitoring_v3.AlertPolicy(
            display_name=display_name,
            conditions=[condition],
            notification_channels=notification_channels,
            combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND,
            enabled=True
        )
        
        created_policy = client.create_alert_policy(
            name=project_name,
            alert_policy=policy
        )
        
        print(f"✓ Created alert policy: {display_name}")
        print(f"  Threshold: {condition_threshold}")
        print(f"  Channels: {len(notification_channels)}")
        
        return created_policy
    
    def create_notification_channel(
        self,
        display_name,
        channel_type='email',
        email_address=None,
        slack_webhook=None
    ):
        """Create notification channel."""
        
        from google.cloud import monitoring_v3
        
        client = monitoring_v3.NotificationChannelServiceClient()
        project_name = f"projects/{self.project_id}"
        
        if channel_type == 'email' and email_address:
            channel = monitoring_v3.NotificationChannel(
                type='email',
                display_name=display_name,
                labels={'email_address': email_address}
            )
        elif channel_type == 'slack' and slack_webhook:
            channel = monitoring_v3.NotificationChannel(
                type='slack',
                display_name=display_name,
                labels={'url': slack_webhook}
            )
        else:
            raise ValueError("Invalid channel configuration")
        
        created_channel = client.create_notification_channel(
            name=project_name,
            notification_channel=channel
        )
        
        print(f"✓ Created notification channel: {display_name}")
        
        return created_channel
    
    def setup_drift_alert(
        self,
        drift_threshold=0.3,
        notification_emails=None
    ):
        """Set up drift monitoring alert."""
        
        if notification_emails is None:
            notification_emails = []
        
        # Create notification channels
        channels = []
        for email in notification_emails:
            channel = self.create_notification_channel(
                display_name=f'Email - {email}',
                channel_type='email',
                email_address=email
            )
            channels.append(channel.name)
        
        # Create alert policy
        policy = self.create_alert_policy(
            display_name='Model Drift Alert',
            condition_threshold=drift_threshold,
            notification_channels=channels,
            metric_type='custom.googleapis.com/vertex_ai/prediction/drift_score'
        )
        
        return policy

# Example usage
alerts = MonitoringAlerts(project_id='my-project')

# Set up drift alert
# policy = alerts.setup_drift_alert(
#     drift_threshold=0.3,
#     notification_emails=['ml-team@company.com', 'on-call@company.com']
# )
```

---

## 6. Performance Tracking

```python
class PerformanceTracker:
    """Track model performance metrics over time."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
    
    def track_model_metrics(
        self,
        endpoint_id,
        metrics_table,
        window_days=30
    ):
        """Track performance metrics from BigQuery."""
        
        from google.cloud import bigquery
        
        bq_client = bigquery.Client(project=self.project_id)
        
        query = f"""
        SELECT
            DATE(prediction_timestamp) as date,
            COUNT(*) as prediction_count,
            AVG(prediction_confidence) as avg_confidence,
            AVG(prediction_latency_ms) as avg_latency_ms,
            COUNTIF(prediction_error IS NOT NULL) as error_count
        FROM
            `{metrics_table}`
        WHERE
            endpoint_id = '{endpoint_id}'
            AND prediction_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {window_days} DAY)
        GROUP BY
            date
        ORDER BY
            date DESC
        """
        
        metrics = bq_client.query(query).to_dataframe()
        
        print(f"\n=== Performance Metrics (Last {window_days} days) ===\n")
        print(metrics.to_string(index=False))
        
        return metrics
    
    def calculate_model_accuracy(
        self,
        predictions_table,
        ground_truth_table,
        window_days=7
    ):
        """Calculate model accuracy from ground truth."""
        
        from google.cloud import bigquery
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        bq_client = bigquery.Client(project=self.project_id)
        
        query = f"""
        SELECT
            p.prediction_value,
            g.actual_value
        FROM
            `{predictions_table}` p
        INNER JOIN
            `{ground_truth_table}` g
        ON
            p.prediction_id = g.prediction_id
        WHERE
            p.prediction_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {window_days} DAY)
        """
        
        data = bq_client.query(query).to_dataframe()
        
        if len(data) == 0:
            print("No ground truth data available")
            return {}
        
        accuracy = accuracy_score(data['actual_value'], data['prediction_value'])
        precision = precision_score(data['actual_value'], data['prediction_value'], average='weighted')
        recall = recall_score(data['actual_value'], data['prediction_value'], average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sample_size': len(data)
        }
        
        print(f"\n=== Model Performance (Last {window_days} days) ===\n")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Samples:   {len(data)}")
        
        return metrics

# Example usage
tracker = PerformanceTracker(project_id='my-project')

# Track metrics
# metrics = tracker.track_model_metrics(
#     endpoint_id='123456',
#     metrics_table='my-project.monitoring.prediction_metrics',
#     window_days=30
# )
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Enable Model Monitoring on endpoint
- [ ] Configure skew detection thresholds
- [ ] Configure drift detection thresholds
- [ ] Set up training dataset baseline
- [ ] Define sampling strategy

### Monitoring
- [ ] Monitor training-serving skew
- [ ] Track prediction drift
- [ ] Detect feature drift
- [ ] Identify anomalies
- [ ] Review monitoring dashboards regularly

### Alerts
- [ ] Create notification channels
- [ ] Set up drift alert policies
- [ ] Configure skew alerts
- [ ] Define escalation procedures
- [ ] Test alert notifications

### Maintenance
- [ ] Review monitoring results weekly
- [ ] Update thresholds based on findings
- [ ] Retrain models when needed
- [ ] Document monitoring incidents
- [ ] Continuous improvement of thresholds

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
