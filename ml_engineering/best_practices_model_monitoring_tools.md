# Best Practices for Model Monitoring Tools on Google Cloud

## Overview

Model Monitoring Tools on Google Cloud help detect model drift, data quality issues, and performance degradation in production ML systems. Vertex AI Model Monitoring provides automated detection and alerting for feature skew, prediction drift, and training-serving skew.

## 1. Model Monitoring Setup

### 1.1 Monitoring Configuration

```python
from google.cloud import aiplatform
from typing import Dict, Any, List

class ModelMonitoringManager:
    """Manager for model monitoring operations."""
    
    def __init__(self, project_id: str, location: str = 'us-central1'):
        """
        Initialize Model Monitoring Manager.
        
        Args:
            project_id: GCP project ID
            location: GCP region
        """
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_monitoring_job(
        self,
        endpoint_name: str,
        display_name: str,
        training_dataset: str,
        monitoring_config: Dict[str, Any]
    ) -> str:
        """
        Create model monitoring job.
        
        Args:
            endpoint_name: Endpoint resource name
            display_name: Monitoring job display name
            training_dataset: Training dataset URI
            monitoring_config: Monitoring configuration
            
        Returns:
            Monitoring job resource name
        """
        # Get endpoint
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        # Create monitoring job
        monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
            display_name=display_name,
            endpoint=endpoint,
            logging_sampling_strategy={
                'random_sample_config': {
                    'sample_rate': monitoring_config.get('sample_rate', 0.8)
                }
            },
            schedule_config={
                'monitor_interval': {
                    'seconds': monitoring_config.get('interval_seconds', 3600)
                }
            },
            model_deployment_monitoring_objective_configs=[
                {
                    'deployed_model_id': model.id,
                    'objective_config': {
                        'training_dataset': {
                            'target_field': monitoring_config.get('target_field'),
                            'bigquery_source': {
                                'input_uri': training_dataset
                            },
                            'data_format': 'bigquery'
                        },
                        'training_prediction_skew_detection_config': {
                            'skew_thresholds': monitoring_config.get('skew_thresholds', {}),
                            'attribution_score_skew_thresholds': monitoring_config.get('attribution_thresholds', {})
                        },
                        'prediction_drift_detection_config': {
                            'drift_thresholds': monitoring_config.get('drift_thresholds', {}),
                            'attribution_score_drift_thresholds': monitoring_config.get('attribution_thresholds', {})
                        }
                    }
                }
                for model in endpoint.list_models()
            ],
            alert_config={
                'email_alert_config': {
                    'user_emails': monitoring_config.get('alert_emails', [])
                }
            }
        )
        
        print(f"Created monitoring job: {monitoring_job.resource_name}")
        return monitoring_job.resource_name
    
    def configure_drift_detection(
        self,
        features: List[str],
        drift_threshold: float = 0.3,
        attribution_threshold: float = 0.3
    ) -> Dict[str, Dict[str, float]]:
        """
        Configure drift detection thresholds.
        
        Args:
            features: List of feature names
            drift_threshold: Default drift threshold
            attribution_threshold: Default attribution threshold
            
        Returns:
            Dictionary of thresholds by feature
        """
        skew_thresholds = {}
        drift_thresholds = {}
        attribution_thresholds = {}
        
        for feature in features:
            skew_thresholds[feature] = {'value': drift_threshold}
            drift_thresholds[feature] = {'value': drift_threshold}
            attribution_thresholds[feature] = {'value': attribution_threshold}
        
        return {
            'skew_thresholds': skew_thresholds,
            'drift_thresholds': drift_thresholds,
            'attribution_thresholds': attribution_thresholds
        }
    
    def get_monitoring_stats(
        self,
        monitoring_job_name: str
    ) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Args:
            monitoring_job_name: Monitoring job resource name
            
        Returns:
            Dictionary with monitoring stats
        """
        monitoring_job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        
        # Get stats from the monitoring job
        stats = {
            'display_name': monitoring_job.display_name,
            'state': monitoring_job.state,
            'create_time': monitoring_job.create_time,
            'update_time': monitoring_job.update_time,
            'schedule_state': monitoring_job.schedule_state,
            'latest_monitoring_pipeline_metadata': monitoring_job.latest_monitoring_pipeline_metadata
        }
        
        return stats


# Example usage
monitoring_manager = ModelMonitoringManager(
    project_id='my-project',
    location='us-central1'
)

# Configure drift detection
features = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
thresholds = monitoring_manager.configure_drift_detection(
    features=features,
    drift_threshold=0.3,
    attribution_threshold=0.3
)

# Create monitoring job
monitoring_config = {
    'sample_rate': 0.8,
    'interval_seconds': 3600,  # 1 hour
    'target_field': 'label',
    'alert_emails': ['ml-team@example.com'],
    **thresholds
}

monitoring_job = monitoring_manager.create_monitoring_job(
    endpoint_name='projects/123/locations/us-central1/endpoints/456',
    display_name='production-model-monitoring',
    training_dataset='bq://my-project.ml_dataset.training_data',
    monitoring_config=monitoring_config
)
```

## 2. Drift Detection

### 2.1 Feature and Prediction Drift

```python
import numpy as np
from scipy import stats
from typing import Tuple

class DriftDetector:
    """Detector for feature and prediction drift."""
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize Drift Detector.
        
        Args:
            threshold: P-value threshold for drift detection
        """
        self.threshold = threshold
    
    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            expected: Expected distribution (training data)
            actual: Actual distribution (production data)
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate distributions
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return psi
    
    def ks_test(
        self,
        expected: np.ndarray,
        actual: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.
        
        Args:
            expected: Expected distribution
            actual: Actual distribution
            
        Returns:
            Tuple of (statistic, p-value)
        """
        statistic, p_value = stats.ks_2samp(expected, actual)
        return statistic, p_value
    
    def chi_square_test(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> Tuple[float, float]:
        """
        Perform Chi-Square test for categorical data.
        
        Args:
            expected: Expected distribution
            actual: Actual distribution
            bins: Number of bins
            
        Returns:
            Tuple of (statistic, p-value)
        """
        # Create bins
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate frequencies
        expected_freq = np.histogram(expected, bins=breakpoints)[0]
        actual_freq = np.histogram(actual, bins=breakpoints)[0]
        
        # Perform test
        statistic, p_value = stats.chisquare(actual_freq, expected_freq)
        return statistic, p_value
    
    def detect_drift(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        method: str = 'psi'
    ) -> Dict[str, Any]:
        """
        Detect drift using specified method.
        
        Args:
            expected: Expected distribution
            actual: Actual distribution
            method: Detection method ('psi', 'ks', 'chi_square')
            
        Returns:
            Dictionary with drift detection results
        """
        if method == 'psi':
            psi_value = self.calculate_psi(expected, actual)
            drift_detected = psi_value > 0.2  # PSI > 0.2 indicates significant drift
            
            return {
                'method': 'PSI',
                'value': psi_value,
                'drift_detected': drift_detected,
                'severity': 'high' if psi_value > 0.25 else 'medium' if psi_value > 0.1 else 'low'
            }
        
        elif method == 'ks':
            statistic, p_value = self.ks_test(expected, actual)
            drift_detected = p_value < self.threshold
            
            return {
                'method': 'KS Test',
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': drift_detected
            }
        
        elif method == 'chi_square':
            statistic, p_value = self.chi_square_test(expected, actual)
            drift_detected = p_value < self.threshold
            
            return {
                'method': 'Chi-Square',
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': drift_detected
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")


# Example usage
detector = DriftDetector(threshold=0.05)

# Sample data
training_data = np.random.normal(0, 1, 1000)
production_data = np.random.normal(0.5, 1.2, 1000)  # Shifted distribution

# Detect drift using PSI
psi_result = detector.detect_drift(training_data, production_data, method='psi')
print(f"PSI Drift Detection:")
print(f"  PSI Value: {psi_result['value']:.4f}")
print(f"  Drift Detected: {psi_result['drift_detected']}")
print(f"  Severity: {psi_result['severity']}")

# Detect drift using KS test
ks_result = detector.detect_drift(training_data, production_data, method='ks')
print(f"\nKS Test Drift Detection:")
print(f"  Statistic: {ks_result['statistic']:.4f}")
print(f"  P-value: {ks_result['p_value']:.4f}")
print(f"  Drift Detected: {ks_result['drift_detected']}")
```

## 3. Performance Monitoring

### 3.1 Model Performance Tracking

```python
from google.cloud import bigquery, monitoring_v3
import time
from datetime import datetime

class PerformanceMonitor:
    """Monitor for model performance metrics."""
    
    def __init__(self, project_id: str):
        """
        Initialize Performance Monitor.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        self.metrics_client = monitoring_v3.MetricServiceClient()
    
    def log_prediction_metrics(
        self,
        model_id: str,
        prediction_id: str,
        latency_ms: float,
        error: bool = False
    ) -> None:
        """
        Log prediction metrics to Cloud Monitoring.
        
        Args:
            model_id: Model identifier
            prediction_id: Prediction identifier
            latency_ms: Prediction latency in milliseconds
            error: Whether prediction had error
        """
        # Create time series for latency
        series = monitoring_v3.TimeSeries()
        series.metric.type = 'custom.googleapis.com/ml/prediction_latency'
        series.metric.labels['model_id'] = model_id
        series.resource.type = 'global'
        
        # Add data point
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        interval = monitoring_v3.TimeInterval(
            {'end_time': {'seconds': seconds, 'nanos': nanos}}
        )
        point = monitoring_v3.Point({
            'interval': interval,
            'value': {'double_value': latency_ms}
        })
        series.points = [point]
        
        # Write time series
        project_name = f"projects/{self.project_id}"
        self.metrics_client.create_time_series(
            name=project_name,
            time_series=[series]
        )
        
        # Log error if present
        if error:
            error_series = monitoring_v3.TimeSeries()
            error_series.metric.type = 'custom.googleapis.com/ml/prediction_errors'
            error_series.metric.labels['model_id'] = model_id
            error_series.resource.type = 'global'
            
            error_point = monitoring_v3.Point({
                'interval': interval,
                'value': {'int64_value': 1}
            })
            error_series.points = [error_point]
            
            self.metrics_client.create_time_series(
                name=project_name,
                time_series=[error_series]
            )
    
    def calculate_model_accuracy(
        self,
        predictions_table: str,
        ground_truth_table: str,
        time_window_hours: int = 24
    ) -> float:
        """
        Calculate model accuracy from BigQuery tables.
        
        Args:
            predictions_table: Predictions table ID
            ground_truth_table: Ground truth table ID
            time_window_hours: Time window in hours
            
        Returns:
            Accuracy score
        """
        query = f"""
        SELECT
            COUNT(CASE WHEN p.prediction = g.actual THEN 1 END) / COUNT(*) as accuracy
        FROM
            `{predictions_table}` p
        JOIN
            `{ground_truth_table}` g
        ON
            p.request_id = g.request_id
        WHERE
            p.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {time_window_hours} HOUR)
        """
        
        query_job = self.bq_client.query(query)
        results = query_job.result()
        
        for row in results:
            return row.accuracy
        
        return 0.0
    
    def get_performance_metrics(
        self,
        model_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, float]:
        """
        Get comprehensive performance metrics.
        
        Args:
            model_id: Model identifier
            time_window_hours: Time window in hours
            
        Returns:
            Dictionary of performance metrics
        """
        # Query Cloud Monitoring for metrics
        project_name = f"projects/{self.project_id}"
        
        # Get average latency
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        interval = monitoring_v3.TimeInterval({
            'end_time': {'seconds': seconds, 'nanos': nanos},
            'start_time': {'seconds': seconds - (time_window_hours * 3600), 'nanos': nanos}
        })
        
        # Build filter for latency metrics
        filter_str = f'metric.type="custom.googleapis.com/ml/prediction_latency" AND metric.labels.model_id="{model_id}"'
        
        results = self.metrics_client.list_time_series(
            request={
                'name': project_name,
                'filter': filter_str,
                'interval': interval,
                'view': monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
            }
        )
        
        latencies = []
        for result in results:
            for point in result.points:
                latencies.append(point.value.double_value)
        
        # Calculate metrics
        metrics = {
            'avg_latency_ms': np.mean(latencies) if latencies else 0.0,
            'p50_latency_ms': np.percentile(latencies, 50) if latencies else 0.0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0.0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0.0,
            'request_count': len(latencies)
        }
        
        return metrics


# Example usage
monitor = PerformanceMonitor(project_id='my-project')

# Log prediction metrics
monitor.log_prediction_metrics(
    model_id='sentiment-classifier-v1',
    prediction_id='pred-12345',
    latency_ms=45.2,
    error=False
)

# Get performance metrics
metrics = monitor.get_performance_metrics(
    model_id='sentiment-classifier-v1',
    time_window_hours=24
)

print("Performance Metrics (24h):")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.2f}")
```

## 4. Alerting and Notifications

### 4.1 Alert Configuration

```python
from google.cloud import monitoring_v3
from google.api_core import protobuf_helpers

class AlertManager:
    """Manager for monitoring alerts."""
    
    def __init__(self, project_id: str):
        """
        Initialize Alert Manager.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.client = monitoring_v3.AlertPolicyServiceClient()
    
    def create_drift_alert(
        self,
        display_name: str,
        metric_type: str,
        threshold: float,
        notification_channels: List[str]
    ) -> str:
        """
        Create alert policy for drift detection.
        
        Args:
            display_name: Alert policy display name
            metric_type: Metric type to monitor
            threshold: Alert threshold
            notification_channels: List of notification channel IDs
            
        Returns:
            Alert policy name
        """
        project_name = f"projects/{self.project_id}"
        
        # Create alert policy
        alert_policy = monitoring_v3.AlertPolicy({
            'display_name': display_name,
            'conditions': [{
                'display_name': f'{display_name} condition',
                'condition_threshold': {
                    'filter': f'metric.type="{metric_type}"',
                    'comparison': monitoring_v3.ComparisonType.COMPARISON_GT,
                    'threshold_value': threshold,
                    'duration': {'seconds': 300},  # 5 minutes
                    'aggregations': [{
                        'alignment_period': {'seconds': 60},
                        'per_series_aligner': monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
                    }]
                }
            }],
            'notification_channels': notification_channels,
            'alert_strategy': {
                'auto_close': {'seconds': 86400}  # 24 hours
            },
            'combiner': monitoring_v3.AlertPolicy.ConditionCombinerType.AND
        })
        
        policy = self.client.create_alert_policy(
            name=project_name,
            alert_policy=alert_policy
        )
        
        print(f"Created alert policy: {policy.name}")
        return policy.name
    
    def create_notification_channel(
        self,
        display_name: str,
        email: str
    ) -> str:
        """
        Create email notification channel.
        
        Args:
            display_name: Channel display name
            email: Email address
            
        Returns:
            Notification channel name
        """
        project_name = f"projects/{self.project_id}"
        
        channel = monitoring_v3.NotificationChannel({
            'display_name': display_name,
            'type': 'email',
            'labels': {'email_address': email},
            'enabled': True
        })
        
        channel = self.client.create_notification_channel(
            name=project_name,
            notification_channel=channel
        )
        
        print(f"Created notification channel: {channel.name}")
        return channel.name


# Example usage
alert_manager = AlertManager(project_id='my-project')

# Create notification channel
channel = alert_manager.create_notification_channel(
    display_name='ML Team Alerts',
    email='ml-team@example.com'
)

# Create drift alert
alert = alert_manager.create_drift_alert(
    display_name='Feature Drift Alert',
    metric_type='custom.googleapis.com/ml/feature_drift',
    threshold=0.3,
    notification_channels=[channel]
)
```

## 5. Quick Reference Checklist

### Monitoring Setup
- [ ] Enable Vertex AI Model Monitoring
- [ ] Configure sampling rate (0.1-1.0)
- [ ] Set monitoring interval (hourly/daily)
- [ ] Define training dataset reference
- [ ] Configure drift thresholds
- [ ] Set up alert notifications

### Drift Detection
- [ ] Monitor feature drift (PSI, KS test)
- [ ] Track prediction drift
- [ ] Check training-serving skew
- [ ] Monitor data quality
- [ ] Set appropriate thresholds
- [ ] Review drift reports regularly

### Performance Tracking
- [ ] Log prediction latency
- [ ] Track error rates
- [ ] Monitor throughput
- [ ] Calculate accuracy metrics
- [ ] Track resource utilization
- [ ] Analyze performance trends

### Alerting
- [ ] Create alert policies
- [ ] Configure notification channels
- [ ] Set alert thresholds
- [ ] Define escalation procedures
- [ ] Test alert delivery
- [ ] Document alert responses

### Best Practices
- [ ] Monitor all production models
- [ ] Use appropriate sampling rates
- [ ] Set realistic thresholds
- [ ] Automate drift detection
- [ ] Integrate with CI/CD
- [ ] Maintain monitoring dashboards
- [ ] Regular model retraining
- [ ] Document monitoring procedures
