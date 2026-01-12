# Comprehensive Guide to Model Monitoring

## Overview

Model monitoring is critical for maintaining model performance in production. This guide covers monitoring strategies, implementation in GCP Vertex AI, and best practices for detecting and responding to model degradation.

## Table of Contents

1. [Why Model Monitoring is Critical](#1-why-model-monitoring-is-critical)
2. [Types of Model Monitoring](#2-types-of-model-monitoring)
3. [Vertex AI Model Monitoring](#3-vertex-ai-model-monitoring)
4. [Setting Up Model Monitoring](#4-setting-up-model-monitoring)
5. [Monitoring Metrics and Thresholds](#5-monitoring-metrics-and-thresholds)
6. [Alert Configuration](#6-alert-configuration)
7. [BigQuery ML Monitoring](#7-bigquery-ml-monitoring)
8. [Custom Monitoring Solutions](#8-custom-monitoring-solutions)
9. [Best Practices](#9-best-practices)
10. [Troubleshooting and Response](#10-troubleshooting-and-response)

---

## 1. Why Model Monitoring is Critical

### 1.1 Common Problems in Production Models

```python
"""
MODEL DEGRADATION CAUSES:

1. DATA DRIFT
   - Input feature distributions change over time
   - Example: Customer behavior shifts during pandemic
   - Impact: Model trained on old patterns becomes less accurate

2. CONCEPT DRIFT
   - Relationship between features and target changes
   - Example: What constitutes "spam" evolves over time
   - Impact: Model's learned patterns become obsolete

3. UPSTREAM DATA ISSUES
   - Data pipeline failures or bugs
   - Example: Feature engineering logic breaks
   - Impact: Wrong inputs → wrong predictions

4. MODEL STALENESS
   - Model not retrained with recent data
   - Example: Seasonal patterns not captured
   - Impact: Predictions based on outdated patterns

5. SERVING INFRASTRUCTURE ISSUES
   - Latency problems
   - Example: Model serving timeout
   - Impact: Slow or failed predictions

6. LABEL SHIFT
   - Distribution of target variable changes
   - Example: Fraud rate increases suddenly
   - Impact: Class imbalance issues
"""

# Example: Data Drift Detection
import pandas as pd
import numpy as np
from scipy import stats

def detect_data_drift(training_data, production_data, feature_name, threshold=0.05):
    """
    Detect data drift using Kolmogorov-Smirnov test
    
    Args:
        training_data: Training dataset
        production_data: Recent production data
        feature_name: Feature to check
        threshold: P-value threshold (default 0.05)
    
    Returns:
        Dictionary with drift detection results
    """
    
    train_values = training_data[feature_name].dropna()
    prod_values = production_data[feature_name].dropna()
    
    # Perform KS test
    statistic, p_value = stats.ks_2samp(train_values, prod_values)
    
    # Check if drift detected
    drift_detected = p_value < threshold
    
    result = {
        'feature': feature_name,
        'ks_statistic': statistic,
        'p_value': p_value,
        'drift_detected': drift_detected,
        'severity': 'HIGH' if statistic > 0.2 else 'MEDIUM' if statistic > 0.1 else 'LOW',
        'train_mean': train_values.mean(),
        'prod_mean': prod_values.mean(),
        'train_std': train_values.std(),
        'prod_std': prod_values.std()
    }
    
    return result

# Usage
drift_results = detect_data_drift(
    training_data=train_df,
    production_data=prod_df,
    feature_name='age',
    threshold=0.05
)

if drift_results['drift_detected']:
    print(f"⚠️ DRIFT DETECTED in {drift_results['feature']}")
    print(f"   KS Statistic: {drift_results['ks_statistic']:.4f}")
    print(f"   P-value: {drift_results['p_value']:.4f}")
    print(f"   Severity: {drift_results['severity']}")
```

### 1.2 Monitoring Strategy

```python
"""
COMPREHENSIVE MONITORING STRATEGY:

1. INPUT MONITORING
   - Feature distributions
   - Missing values
   - Data types
   - Value ranges
   
2. PREDICTION MONITORING
   - Prediction distributions
   - Confidence scores
   - Prediction patterns
   
3. PERFORMANCE MONITORING
   - Model accuracy (when labels available)
   - Precision/Recall
   - F1-score
   - AUC-ROC
   
4. OPERATIONAL MONITORING
   - Latency (p50, p95, p99)
   - Throughput (QPS)
   - Error rates
   - Resource utilization
   
5. BUSINESS METRICS
   - Revenue impact
   - User satisfaction
   - Conversion rates
   - Cost per prediction
"""
```

---

## 2. Types of Model Monitoring

### 2.1 Data Drift Detection

```python
"""
DATA DRIFT: Changes in input feature distributions

Detection Methods:
1. Statistical Tests (KS test, Chi-square)
2. Distribution Distance (KL divergence, Wasserstein)
3. Population Stability Index (PSI)
"""

# Population Stability Index (PSI)
def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index
    
    PSI < 0.1: No significant change
    0.1 ≤ PSI < 0.25: Moderate change
    PSI ≥ 0.25: Significant change (retrain recommended)
    """
    
    # Create bins
    breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
    
    # Calculate distributions
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Add small value to avoid log(0)
    expected_percents = expected_percents + 0.0001
    actual_percents = actual_percents + 0.0001
    
    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)
    
    return psi

# Example usage
train_feature = train_df['age'].values
prod_feature = prod_df['age'].values

psi_score = calculate_psi(train_feature, prod_feature)

print(f"PSI Score: {psi_score:.4f}")
if psi_score < 0.1:
    print("✅ No significant drift")
elif psi_score < 0.25:
    print("⚠️ Moderate drift detected")
else:
    print("🚨 Significant drift - retrain recommended!")
```

### 2.2 Prediction Drift Detection

```python
"""
PREDICTION DRIFT: Changes in model output distribution
"""

def detect_prediction_drift(train_predictions, prod_predictions):
    """
    Detect drift in prediction distributions
    """
    
    # For classification: Check class distribution
    train_dist = pd.Series(train_predictions).value_counts(normalize=True)
    prod_dist = pd.Series(prod_predictions).value_counts(normalize=True)
    
    # Calculate Total Variation Distance
    tvd = 0.5 * np.sum(np.abs(train_dist - prod_dist))
    
    # Chi-square test
    chi2_stat, p_value = stats.chisquare(
        f_obs=prod_dist.values * len(prod_predictions),
        f_exp=train_dist.values * len(prod_predictions)
    )
    
    return {
        'tvd': tvd,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'drift_detected': p_value < 0.05,
        'train_distribution': train_dist.to_dict(),
        'prod_distribution': prod_dist.to_dict()
    }

# For regression: Check prediction statistics
def check_regression_drift(train_predictions, prod_predictions):
    """
    Check drift in regression predictions
    """
    
    train_mean = np.mean(train_predictions)
    prod_mean = np.mean(prod_predictions)
    train_std = np.std(train_predictions)
    prod_std = np.std(prod_predictions)
    
    # Percentage change
    mean_change = abs((prod_mean - train_mean) / train_mean) * 100
    std_change = abs((prod_std - train_std) / train_std) * 100
    
    # Statistical test
    statistic, p_value = stats.ks_2samp(train_predictions, prod_predictions)
    
    return {
        'mean_change_percent': mean_change,
        'std_change_percent': std_change,
        'ks_statistic': statistic,
        'p_value': p_value,
        'drift_detected': p_value < 0.05 or mean_change > 10 or std_change > 20
    }
```

### 2.3 Performance Monitoring

```python
"""
PERFORMANCE MONITORING: Track actual model performance

Requires ground truth labels (delayed feedback)
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class PerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, baseline_metrics):
        """
        Args:
            baseline_metrics: Dict of baseline metrics from validation
        """
        self.baseline_metrics = baseline_metrics
        self.performance_history = []
        
    def evaluate_batch(self, y_true, y_pred, y_pred_proba=None, timestamp=None):
        """
        Evaluate a batch of predictions
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            timestamp: Batch timestamp (optional)
        """
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'timestamp': timestamp or pd.Timestamp.now(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_samples': len(y_true)
        }
        
        # Add AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                metrics['auc_roc'] = auc
            except:
                pass
        
        # Check for degradation
        metrics['degradation'] = self._check_degradation(metrics)
        
        self.performance_history.append(metrics)
        
        return metrics
    
    def _check_degradation(self, current_metrics):
        """Check if performance degraded from baseline"""
        
        degradation = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in self.baseline_metrics:
                baseline = self.baseline_metrics[metric]
                current = current_metrics[metric]
                
                # Calculate percentage drop
                drop = ((baseline - current) / baseline) * 100
                
                degradation[metric] = {
                    'baseline': baseline,
                    'current': current,
                    'drop_percent': drop,
                    'degraded': drop > 5  # 5% threshold
                }
        
        return degradation
    
    def get_summary(self):
        """Get summary of performance over time"""
        
        if not self.performance_history:
            return None
        
        df = pd.DataFrame(self.performance_history)
        
        summary = {
            'mean_accuracy': df['accuracy'].mean(),
            'std_accuracy': df['accuracy'].std(),
            'min_accuracy': df['accuracy'].min(),
            'max_accuracy': df['accuracy'].max(),
            'trend': 'decreasing' if df['accuracy'].iloc[-1] < df['accuracy'].iloc[0] else 'stable/increasing',
            'total_samples': df['n_samples'].sum()
        }
        
        return summary

# Usage
baseline_metrics = {
    'accuracy': 0.92,
    'precision': 0.91,
    'recall': 0.93,
    'f1_score': 0.92
}

monitor = PerformanceMonitor(baseline_metrics)

# Evaluate daily batches
batch_metrics = monitor.evaluate_batch(
    y_true=batch_labels,
    y_pred=batch_predictions,
    y_pred_proba=batch_probabilities,
    timestamp=pd.Timestamp.now()
)

# Check for degradation
if any(d['degraded'] for d in batch_metrics['degradation'].values()):
    print("🚨 PERFORMANCE DEGRADATION DETECTED!")
    print(batch_metrics['degradation'])
```

---

## 3. Vertex AI Model Monitoring

### 3.1 Overview

```python
"""
VERTEX AI MODEL MONITORING FEATURES:

1. Automatic Monitoring
   - Training-serving skew detection
   - Prediction drift detection
   - Feature attribution drift
   
2. Alert Configuration
   - Email notifications
   - Cloud Logging integration
   - Custom alert thresholds
   
3. Built-in Metrics
   - Feature distribution comparison
   - Prediction distribution analysis
   - Anomaly detection
   
4. Monitoring Jobs
   - Scheduled monitoring runs
   - Sampling of prediction data
   - Historical trend analysis
"""
```

### 3.2 Model Monitoring Job Setup

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring

# Initialize Vertex AI
aiplatform.init(
    project='my-project-id',
    location='us-central1'
)

def create_model_monitoring_job(
    endpoint_name,
    model_name,
    training_dataset_uri,
    alert_emails,
    monitoring_frequency='HOURLY',
    skew_thresholds=None,
    drift_thresholds=None
):
    """
    Create comprehensive model monitoring job
    
    Args:
        endpoint_name: Name of deployed endpoint
        model_name: Name of the model
        training_dataset_uri: GCS URI to training dataset
        alert_emails: List of emails for alerts
        monitoring_frequency: How often to run monitoring
        skew_thresholds: Dict of skew thresholds per feature
        drift_thresholds: Dict of drift thresholds per feature
    """
    
    # Default thresholds if not provided
    if skew_thresholds is None:
        skew_thresholds = {
            'age': 0.1,
            'income': 0.1,
            'credit_score': 0.15,
            'default': 0.1  # Default threshold for all features
        }
    
    if drift_thresholds is None:
        drift_thresholds = {
            'age': 0.1,
            'income': 0.1,
            'credit_score': 0.15,
            'default': 0.1
        }
    
    # Get endpoint
    endpoint = aiplatform.Endpoint(endpoint_name)
    
    # Create skew detection config
    skew_config = model_monitoring.SkewDetectionConfig(
        data_source=training_dataset_uri,
        skew_thresholds=skew_thresholds,
        target_field='label'  # Name of label column in training data
    )
    
    # Create drift detection config
    drift_config = model_monitoring.DriftDetectionConfig(
        drift_thresholds=drift_thresholds
    )
    
    # Create explanation config (for feature attribution drift)
    explanation_config = model_monitoring.ExplanationConfig(
        enable_feature_attributes=True,
        explanation_baseline=None  # Will use default baseline
    )
    
    # Create alert config
    alert_config = model_monitoring.EmailAlertConfig(
        user_emails=alert_emails,
        enable_logging=True
    )
    
    # Create monitoring objective config
    objective_config = model_monitoring.ObjectiveConfig(
        skew_detection_config=skew_config,
        drift_detection_config=drift_config,
        explanation_config=explanation_config
    )
    
    # Create monitoring schedule
    monitoring_job = endpoint.create_model_deployment_monitoring_job(
        display_name=f'{model_name}_monitoring',
        logging_sampling_strategy=model_monitoring.RandomSampleConfig(
            sample_rate=0.5  # Sample 50% of predictions
        ),
        schedule_config=model_monitoring.ScheduleConfig(
            monitor_interval=monitoring_frequency  # 'HOURLY', 'DAILY', 'WEEKLY'
        ),
        alert_config=alert_config,
        objective_configs=[objective_config],
        # Advanced options
        enable_monitoring_pipeline_logs=True,
        model_monitoring_job_id=f'{model_name}_monitoring_job',
        stats_anomalies_base_directory=f'gs://my-bucket/monitoring/{model_name}/'
    )
    
    print(f"✅ Monitoring job created: {monitoring_job.resource_name}")
    
    return monitoring_job

# Usage
monitoring_job = create_model_monitoring_job(
    endpoint_name='projects/123/locations/us-central1/endpoints/456',
    model_name='fraud_detection_model',
    training_dataset_uri='gs://my-bucket/training_data.csv',
    alert_emails=['ml-team@company.com', 'data-eng@company.com'],
    monitoring_frequency='HOURLY',
    skew_thresholds={
        'transaction_amount': 0.15,
        'merchant_category': 0.2,
        'time_of_day': 0.1,
        'default': 0.1
    },
    drift_thresholds={
        'transaction_amount': 0.15,
        'merchant_category': 0.2,
        'time_of_day': 0.1,
        'default': 0.1
    }
)
```

### 3.3 Complete Vertex AI Monitoring Setup

```python
"""
COMPLETE VERTEX AI MODEL MONITORING SETUP
"""

from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring
import json

class VertexAIModelMonitor:
    """
    Complete model monitoring setup for Vertex AI
    """
    
    def __init__(self, project_id, location, bucket_name):
        """
        Initialize Vertex AI Model Monitor
        
        Args:
            project_id: GCP project ID
            location: GCP region
            bucket_name: GCS bucket for monitoring artifacts
        """
        aiplatform.init(project=project_id, location=location)
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        
    def setup_monitoring(
        self,
        endpoint_id,
        model_id,
        feature_config,
        alert_config,
        monitoring_frequency='HOURLY'
    ):
        """
        Set up comprehensive model monitoring
        
        Args:
            endpoint_id: Vertex AI endpoint ID
            model_id: Model ID to monitor
            feature_config: Dict with feature monitoring configuration
            alert_config: Dict with alert configuration
            monitoring_frequency: Monitoring interval
        """
        
        # Get endpoint
        endpoint = aiplatform.Endpoint(
            endpoint_name=f'projects/{self.project_id}/locations/{self.location}/endpoints/{endpoint_id}'
        )
        
        # Build objective configs
        objective_configs = []
        
        for feature_name, config in feature_config.items():
            # Training-serving skew detection
            if 'training_dataset' in config:
                skew_config = model_monitoring.SkewDetectionConfig(
                    data_source=config['training_dataset'],
                    skew_thresholds={
                        feature_name: config.get('skew_threshold', 0.1)
                    },
                    target_field=config.get('target_field', 'label')
                )
                
                objective = model_monitoring.ObjectiveConfig(
                    skew_detection_config=skew_config
                )
                objective_configs.append(objective)
            
            # Prediction drift detection
            if config.get('monitor_drift', True):
                drift_config = model_monitoring.DriftDetectionConfig(
                    drift_thresholds={
                        feature_name: config.get('drift_threshold', 0.1)
                    }
                )
                
                objective = model_monitoring.ObjectiveConfig(
                    drift_detection_config=drift_config
                )
                objective_configs.append(objective)
        
        # Create email alert config
        email_alert = model_monitoring.EmailAlertConfig(
            user_emails=alert_config['emails'],
            enable_logging=True
        )
        
        # Create monitoring job
        monitoring_job = endpoint.create_model_deployment_monitoring_job(
            display_name=alert_config.get('job_name', 'model_monitoring'),
            
            # Sampling strategy
            logging_sampling_strategy=model_monitoring.RandomSampleConfig(
                sample_rate=alert_config.get('sample_rate', 0.5)
            ),
            
            # Schedule
            schedule_config=model_monitoring.ScheduleConfig(
                monitor_interval=monitoring_frequency
            ),
            
            # Alerts
            alert_config=email_alert,
            
            # Objectives
            objective_configs=objective_configs,
            
            # Logging
            enable_monitoring_pipeline_logs=True,
            
            # Storage
            stats_anomalies_base_directory=f'gs://{self.bucket_name}/monitoring/{model_id}/',
            
            # Labels for organization
            labels={
                'model_id': model_id,
                'environment': 'production',
                'team': alert_config.get('team', 'ml')
            }
        )
        
        print("=" * 80)
        print("✅ MODEL MONITORING CONFIGURED")
        print("=" * 80)
        print(f"Job Name: {monitoring_job.display_name}")
        print(f"Resource Name: {monitoring_job.resource_name}")
        print(f"Monitoring Frequency: {monitoring_frequency}")
        print(f"Sample Rate: {alert_config.get('sample_rate', 0.5) * 100}%")
        print(f"Alert Emails: {', '.join(alert_config['emails'])}")
        print(f"Monitoring {len(objective_configs)} objectives")
        print("=" * 80)
        
        return monitoring_job
    
    def get_monitoring_stats(self, monitoring_job_name):
        """
        Retrieve monitoring statistics
        
        Args:
            monitoring_job_name: Name of monitoring job
        """
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        
        # Get latest stats
        stats = job.get_model_monitoring_stats()
        
        return stats
    
    def list_anomalies(self, monitoring_job_name, time_window_hours=24):
        """
        List detected anomalies
        
        Args:
            monitoring_job_name: Name of monitoring job
            time_window_hours: Look back window in hours
        """
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        
        # List anomalies
        anomalies = job.list_anomalies(
            time_window_hours=time_window_hours
        )
        
        return anomalies
    
    def pause_monitoring(self, monitoring_job_name):
        """Pause monitoring job"""
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        job.pause()
        print(f"✅ Monitoring job paused: {monitoring_job_name}")
    
    def resume_monitoring(self, monitoring_job_name):
        """Resume monitoring job"""
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        job.resume()
        print(f"✅ Monitoring job resumed: {monitoring_job_name}")
    
    def delete_monitoring(self, monitoring_job_name):
        """Delete monitoring job"""
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        job.delete()
        print(f"✅ Monitoring job deleted: {monitoring_job_name}")

# Usage Example
monitor = VertexAIModelMonitor(
    project_id='my-project',
    location='us-central1',
    bucket_name='my-model-monitoring-bucket'
)

# Feature configuration
feature_config = {
    'transaction_amount': {
        'training_dataset': 'gs://my-bucket/training_data.csv',
        'skew_threshold': 0.15,
        'drift_threshold': 0.15,
        'monitor_drift': True,
        'target_field': 'is_fraud'
    },
    'merchant_category': {
        'training_dataset': 'gs://my-bucket/training_data.csv',
        'skew_threshold': 0.2,
        'drift_threshold': 0.2,
        'monitor_drift': True,
        'target_field': 'is_fraud'
    },
    'customer_age': {
        'skew_threshold': 0.1,
        'drift_threshold': 0.1,
        'monitor_drift': True
    }
}

# Alert configuration
alert_config = {
    'job_name': 'fraud_detection_monitoring',
    'emails': ['ml-team@company.com', 'alerts@company.com'],
    'sample_rate': 0.5,  # Monitor 50% of predictions
    'team': 'fraud-ml'
}

# Set up monitoring
monitoring_job = monitor.setup_monitoring(
    endpoint_id='1234567890',
    model_id='fraud_detection_v1',
    feature_config=feature_config,
    alert_config=alert_config,
    monitoring_frequency='HOURLY'
)

# Later: Get monitoring stats
stats = monitor.get_monitoring_stats(monitoring_job.resource_name)
print(stats)

# List recent anomalies
anomalies = monitor.list_anomalies(
    monitoring_job.resource_name,
    time_window_hours=24
)
for anomaly in anomalies:
    print(f"⚠️ Anomaly detected: {anomaly}")
```

---

## 4. Setting Up Model Monitoring

### 4.1 Monitoring Configuration File

```yaml
# model_monitoring_config.yaml
# Complete monitoring configuration

monitoring_config:
  project_id: "my-gcp-project"
  location: "us-central1"
  
  # Endpoint configuration
  endpoint:
    endpoint_id: "1234567890"
    model_id: "fraud_detection_v1"
    model_version: "v1.2.0"
  
  # Storage configuration
  storage:
    bucket_name: "model-monitoring-artifacts"
    training_data_uri: "gs://my-bucket/training_data/fraud_training.csv"
    monitoring_results_path: "gs://my-bucket/monitoring/results/"
    logs_path: "gs://my-bucket/monitoring/logs/"
  
  # Sampling configuration
  sampling:
    strategy: "random"  # or "stratified"
    sample_rate: 0.5  # Monitor 50% of predictions
    min_samples_per_batch: 1000
    max_samples_per_batch: 10000
  
  # Monitoring schedule
  schedule:
    frequency: "HOURLY"  # HOURLY, DAILY, WEEKLY
    start_time: "00:00"
    timezone: "America/New_York"
  
  # Feature monitoring
  features:
    - name: "transaction_amount"
      type: "numerical"
      skew_threshold: 0.15
      drift_threshold: 0.15
      monitor_skew: true
      monitor_drift: true
      
    - name: "merchant_category"
      type: "categorical"
      skew_threshold: 0.20
      drift_threshold: 0.20
      monitor_skew: true
      monitor_drift: true
      
    - name: "customer_age"
      type: "numerical"
      skew_threshold: 0.10
      drift_threshold: 0.10
      monitor_skew: true
      monitor_drift: true
      
    - name: "transaction_time"
      type: "numerical"
      skew_threshold: 0.12
      drift_threshold: 0.12
      monitor_skew: true
      monitor_drift: true
  
  # Prediction monitoring
  predictions:
    monitor_distribution: true
    drift_threshold: 0.15
    confidence_threshold: 0.5
    monitor_confidence_distribution: true
  
  # Performance monitoring
  performance:
    enabled: true
    ground_truth_delay_hours: 24  # How long until labels available
    metrics:
      - accuracy
      - precision
      - recall
      - f1_score
      - auc_roc
    thresholds:
      accuracy: 0.90  # Alert if below 90%
      precision: 0.88
      recall: 0.92
      f1_score: 0.90
      auc_roc: 0.93
  
  # Alert configuration
  alerts:
    # Email alerts
    email:
      enabled: true
      recipients:
        - ml-team@company.com
        - data-eng@company.com
        - oncall@company.com
      send_daily_summary: true
      summary_time: "09:00"
    
    # Slack alerts
    slack:
      enabled: true
      webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
      channel: "#ml-alerts"
      mention_users:
        - "@ml-lead"
        - "@data-engineer"
    
    # PagerDuty for critical alerts
    pagerduty:
      enabled: true
      integration_key: "YOUR_PAGERDUTY_KEY"
      severity_threshold: "HIGH"  # Only page for HIGH/CRITICAL
    
    # Cloud Logging
    logging:
      enabled: true
      log_name: "model-monitoring"
      severity: "WARNING"
  
  # Alert rules
  alert_rules:
    - name: "high_skew_detection"
      condition: "skew > threshold"
      severity: "MEDIUM"
      action: "email"
      
    - name: "critical_drift"
      condition: "drift > threshold * 1.5"
      severity: "HIGH"
      action: "email,slack,pagerduty"
      
    - name: "performance_degradation"
      condition: "accuracy < threshold"
      severity: "HIGH"
      action: "email,slack"
      
    - name: "serving_errors"
      condition: "error_rate > 5%"
      severity: "CRITICAL"
      action: "email,slack,pagerduty"
  
  # Automated actions
  automated_actions:
    - trigger: "critical_drift"
      action: "pause_endpoint"
      confirmation_required: true
      
    - trigger: "performance_degradation"
      action: "trigger_retraining"
      confirmation_required: false
      
    - trigger: "high_error_rate"
      action: "rollback_model"
      confirmation_required: true
```

### 4.2 Load and Apply Configuration

```python
import yaml
from google.cloud import aiplatform

def load_monitoring_config(config_path):
    """Load monitoring configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['monitoring_config']

def apply_monitoring_config(config):
    """
    Apply monitoring configuration to Vertex AI
    """
    
    # Initialize Vertex AI
    aiplatform.init(
        project=config['project_id'],
        location=config['location']
    )
    
    # Get endpoint
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{config['project_id']}/locations/{config['location']}/endpoints/{config['endpoint']['endpoint_id']}"
    )
    
    # Build skew detection configs
    skew_configs = {}
    drift_configs = {}
    
    for feature in config['features']:
        if feature['monitor_skew']:
            skew_configs[feature['name']] = feature['skew_threshold']
        if feature['monitor_drift']:
            drift_configs[feature['name']] = feature['drift_threshold']
    
    # Create monitoring job
    monitoring_job = endpoint.create_model_deployment_monitoring_job(
        display_name=f"{config['endpoint']['model_id']}_monitoring",
        
        # Sampling
        logging_sampling_strategy=model_monitoring.RandomSampleConfig(
            sample_rate=config['sampling']['sample_rate']
        ),
        
        # Schedule
        schedule_config=model_monitoring.ScheduleConfig(
            monitor_interval=config['schedule']['frequency']
        ),
        
        # Objectives
        objective_configs=[
            model_monitoring.ObjectiveConfig(
                skew_detection_config=model_monitoring.SkewDetectionConfig(
                    data_source=config['storage']['training_data_uri'],
                    skew_thresholds=skew_configs
                ),
                drift_detection_config=model_monitoring.DriftDetectionConfig(
                    drift_thresholds=drift_configs
                )
            )
        ],
        
        # Alerts
        alert_config=model_monitoring.EmailAlertConfig(
            user_emails=config['alerts']['email']['recipients'],
            enable_logging=config['alerts']['logging']['enabled']
        ),
        
        # Storage
        stats_anomalies_base_directory=config['storage']['monitoring_results_path']
    )
    
    print(f"✅ Monitoring job created from config: {monitoring_job.resource_name}")
    
    return monitoring_job

# Usage
config = load_monitoring_config('model_monitoring_config.yaml')
monitoring_job = apply_monitoring_config(config)
```

---

## 5. Monitoring Metrics and Thresholds

### 5.1 Recommended Thresholds

```python
"""
RECOMMENDED MONITORING THRESHOLDS

1. TRAINING-SERVING SKEW
   - Low Threshold: 0.05-0.10 (sensitive, frequent alerts)
   - Medium Threshold: 0.10-0.15 (balanced)
   - High Threshold: 0.15-0.25 (conservative, fewer alerts)
   
   Recommendation: Start with 0.10, adjust based on false positives

2. PREDICTION DRIFT
   - Low: 0.05-0.10
   - Medium: 0.10-0.15
   - High: 0.15-0.25
   
   Recommendation: 0.15 for stable features, 0.10 for critical features

3. PERFORMANCE DEGRADATION
   - Critical: >10% drop from baseline
   - Warning: 5-10% drop from baseline
   - Monitor: <5% drop
   
   Recommendation: 5% threshold for alerts

4. LATENCY
   - P50: <50ms (median)
   - P95: <100ms
   - P99: <200ms
   
   Recommendation: Set based on SLA requirements

5. ERROR RATE
   - Green: <1%
   - Yellow: 1-5%
   - Red: >5%
   
   Recommendation: Alert at 2%, escalate at 5%
"""

# Threshold configuration
MONITORING_THRESHOLDS = {
    # Feature-specific skew thresholds
    'skew': {
        'default': 0.10,
        'critical_features': {
            'transaction_amount': 0.08,
            'customer_risk_score': 0.08
        },
        'stable_features': {
            'customer_age': 0.15,
            'account_tenure': 0.15
        }
    },
    
    # Feature-specific drift thresholds
    'drift': {
        'default': 0.10,
        'critical_features': {
            'transaction_amount': 0.08,
            'merchant_category': 0.10
        },
        'stable_features': {
            'customer_age': 0.15,
            'country': 0.20
        }
    },
    
    # Performance thresholds
    'performance': {
        'accuracy': {
            'baseline': 0.92,
            'warning': 0.87,  # 5% drop
            'critical': 0.83   # 10% drop
        },
        'precision': {
            'baseline': 0.90,
            'warning': 0.85,
            'critical': 0.81
        },
        'recall': {
            'baseline': 0.94,
            'warning': 0.89,
            'critical': 0.85
        },
        'f1_score': {
            'baseline': 0.92,
            'warning': 0.87,
            'critical': 0.83
        }
    },
    
    # Operational thresholds
    'latency_ms': {
        'p50': 50,
        'p95': 100,
        'p99': 200
    },
    
    'error_rate': {
        'warning': 0.02,  # 2%
        'critical': 0.05   # 5%
    },
    
    # Prediction distribution
    'prediction_drift': {
        'default': 0.15,
        'class_imbalance_threshold': 0.20  # Allow more drift for imbalanced classes
    }
}

def get_threshold(feature_name, threshold_type, feature_category='default'):
    """
    Get threshold for a feature
    
    Args:
        feature_name: Name of the feature
        threshold_type: 'skew' or 'drift'
        feature_category: 'default', 'critical_features', or 'stable_features'
    
    Returns:
        Threshold value
    """
    
    thresholds = MONITORING_THRESHOLDS[threshold_type]
    
    # Check if specific threshold exists for this feature
    if feature_category in thresholds and feature_name in thresholds[feature_category]:
        return thresholds[feature_category][feature_name]
    
    # Return default
    return thresholds['default']

# Usage
skew_threshold = get_threshold('transaction_amount', 'skew', 'critical_features')
print(f"Skew threshold for transaction_amount: {skew_threshold}")
```

### 5.2 Adaptive Thresholds

```python
"""
ADAPTIVE THRESHOLDS: Adjust thresholds based on historical data
"""

import numpy as np
from scipy import stats

class AdaptiveThresholdCalculator:
    """
    Calculate adaptive monitoring thresholds based on historical data
    """
    
    def __init__(self, historical_data, confidence_level=0.95):
        """
        Args:
            historical_data: DataFrame with historical monitoring metrics
            confidence_level: Confidence level for threshold calculation
        """
        self.historical_data = historical_data
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
    def calculate_drift_threshold(self, feature_name):
        """
        Calculate adaptive drift threshold
        
        Uses historical drift scores to set threshold
        at mean + (z_score * std)
        """
        
        drift_scores = self.historical_data[f'{feature_name}_drift']
        
        mean_drift = drift_scores.mean()
        std_drift = drift_scores.std()
        
        # Threshold = mean + z * std
        threshold = mean_drift + (self.z_score * std_drift)
        
        # Cap at reasonable values
        threshold = min(threshold, 0.25)  # Max 0.25
        threshold = max(threshold, 0.05)  # Min 0.05
        
        return {
            'feature': feature_name,
            'threshold': threshold,
            'mean_historical_drift': mean_drift,
            'std_historical_drift': std_drift,
            'confidence_level': self.confidence_level
        }
    
    def calculate_performance_threshold(self, metric_name, percentile=5):
        """
        Calculate adaptive performance threshold
        
        Uses historical performance to set threshold
        at the percentile (e.g., 5th percentile = bottom 5%)
        """
        
        metric_values = self.historical_data[metric_name]
        
        # Calculate percentile threshold
        threshold = np.percentile(metric_values, percentile)
        
        return {
            'metric': metric_name,
            'threshold': threshold,
            'percentile': percentile,
            'mean_historical': metric_values.mean(),
            'min_historical': metric_values.min(),
            'max_historical': metric_values.max()
        }
    
    def get_all_adaptive_thresholds(self, features, metrics):
        """
        Get all adaptive thresholds
        
        Args:
            features: List of feature names
            metrics: List of metric names
        """
        
        thresholds = {
            'drift': {},
            'performance': {}
        }
        
        # Calculate drift thresholds
        for feature in features:
            result = self.calculate_drift_threshold(feature)
            thresholds['drift'][feature] = result['threshold']
        
        # Calculate performance thresholds
        for metric in metrics:
            result = self.calculate_performance_threshold(metric)
            thresholds['performance'][metric] = result['threshold']
        
        return thresholds

# Usage
# Load historical monitoring data
historical_data = pd.read_csv('monitoring_history.csv')

calculator = AdaptiveThresholdCalculator(
    historical_data=historical_data,
    confidence_level=0.95
)

# Calculate adaptive thresholds
adaptive_thresholds = calculator.get_all_adaptive_thresholds(
    features=['transaction_amount', 'merchant_category', 'customer_age'],
    metrics=['accuracy', 'precision', 'recall', 'f1_score']
)

print("Adaptive Thresholds:")
print(json.dumps(adaptive_thresholds, indent=2))
```

---

## 6. Alert Configuration

### 6.1 Multi-Channel Alerting

```python
"""
MULTI-CHANNEL ALERTING SYSTEM
"""

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.cloud import logging as cloud_logging

class AlertManager:
    """
    Manage alerts across multiple channels
    """
    
    def __init__(self, config):
        """
        Args:
            config: Alert configuration dict
        """
        self.config = config
        self.cloud_logging_client = cloud_logging.Client()
        self.logger = self.cloud_logging_client.logger('model-monitoring-alerts')
        
    def send_email_alert(self, subject, body, severity='MEDIUM'):
        """
        Send email alert
        
        Args:
            subject: Email subject
            body: Email body (HTML supported)
            severity: Alert severity
        """
        
        if not self.config.get('email', {}).get('enabled', False):
            return
        
        # Email configuration
        smtp_server = self.config['email']['smtp_server']
        smtp_port = self.config['email']['smtp_port']
        sender = self.config['email']['sender']
        password = self.config['email']['password']
        recipients = self.config['email']['recipients']
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{severity}] {subject}"
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        
        # Add body
        html_part = MIMEText(body, 'html')
        msg.attach(html_part)
        
        # Send email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
            
            print(f"✅ Email alert sent: {subject}")
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
    
    def send_slack_alert(self, message, severity='MEDIUM'):
        """
        Send Slack alert
        
        Args:
            message: Alert message
            severity: Alert severity
        """
        
        if not self.config.get('slack', {}).get('enabled', False):
            return
        
        webhook_url = self.config['slack']['webhook_url']
        channel = self.config['slack'].get('channel', '#ml-alerts')
        
        # Color based on severity
        colors = {
            'LOW': '#36a64f',      # Green
            'MEDIUM': '#ff9900',   # Orange
            'HIGH': '#ff0000',     # Red
            'CRITICAL': '#8b0000'  # Dark red
        }
        
        # Emoji based on severity
        emojis = {
            'LOW': ':information_source:',
            'MEDIUM': ':warning:',
            'HIGH': ':rotating_light:',
            'CRITICAL': ':fire:'
        }
        
        payload = {
            'channel': channel,
            'attachments': [{
                'color': colors.get(severity, '#ff9900'),
                'title': f"{emojis.get(severity, ':warning:')} Model Monitoring Alert",
                'text': message,
                'footer': 'Model Monitoring System',
                'ts': int(pd.Timestamp.now().timestamp())
            }]
        }
        
        # Mention users for HIGH/CRITICAL alerts
        if severity in ['HIGH', 'CRITICAL']:
            mentions = self.config['slack'].get('mention_users', [])
            if mentions:
                payload['text'] = ' '.join(mentions)
        
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            print(f"✅ Slack alert sent: {severity}")
            
        except Exception as e:
            print(f"❌ Failed to send Slack alert: {e}")
    
    def send_pagerduty_alert(self, title, description, severity='HIGH'):
        """
        Send PagerDuty alert
        
        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity
        """
        
        if not self.config.get('pagerduty', {}).get('enabled', False):
            return
        
        # Only page for HIGH/CRITICAL
        severity_threshold = self.config['pagerduty'].get('severity_threshold', 'HIGH')
        if severity not in ['HIGH', 'CRITICAL'] and severity_threshold == 'HIGH':
            return
        
        integration_key = self.config['pagerduty']['integration_key']
        
        payload = {
            'routing_key': integration_key,
            'event_action': 'trigger',
            'payload': {
                'summary': title,
                'severity': severity.lower(),
                'source': 'model-monitoring',
                'custom_details': {
                    'description': description
                }
            }
        }
        
        try:
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload
            )
            response.raise_for_status()
            print(f"✅ PagerDuty alert sent: {severity}")
            
        except Exception as e:
            print(f"❌ Failed to send PagerDuty alert: {e}")
    
    def log_to_cloud_logging(self, message, severity='WARNING'):
        """
        Log alert to Cloud Logging
        
        Args:
            message: Log message
            severity: Log severity
        """
        
        if not self.config.get('logging', {}).get('enabled', False):
            return
        
        self.logger.log_text(
            message,
            severity=severity
        )
        
        print(f"✅ Logged to Cloud Logging: {severity}")
    
    def send_alert(self, title, message, severity='MEDIUM', channels=None):
        """
        Send alert to multiple channels
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            channels: List of channels (default: all enabled)
        """
        
        if channels is None:
            channels = ['email', 'slack', 'pagerduty', 'logging']
        
        print(f"\n{'='*80}")
        print(f"🚨 SENDING ALERT: {title}")
        print(f"Severity: {severity}")
        print(f"Channels: {', '.join(channels)}")
        print(f"{'='*80}\n")
        
        if 'email' in channels:
            self.send_email_alert(title, message, severity)
        
        if 'slack' in channels:
            self.send_slack_alert(message, severity)
        
        if 'pagerduty' in channels:
            self.send_pagerduty_alert(title, message, severity)
        
        if 'logging' in channels:
            self.log_to_cloud_logging(message, severity)

# Usage
alert_config = {
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender': 'alerts@company.com',
        'password': 'your-password',
        'recipients': ['ml-team@company.com', 'oncall@company.com']
    },
    'slack': {
        'enabled': True,
        'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        'channel': '#ml-alerts',
        'mention_users': ['@ml-lead', '@data-engineer']
    },
    'pagerduty': {
        'enabled': True,
        'integration_key': 'YOUR_PAGERDUTY_KEY',
        'severity_threshold': 'HIGH'
    },
    'logging': {
        'enabled': True
    }
}

alert_manager = AlertManager(alert_config)

# Send alert
alert_manager.send_alert(
    title='High Drift Detected in transaction_amount Feature',
    message="""
    <h2>Model Monitoring Alert</h2>
    <p><strong>Feature:</strong> transaction_amount</p>
    <p><strong>Drift Score:</strong> 0.35 (threshold: 0.15)</p>
    <p><strong>Recommendation:</strong> Investigate data pipeline and consider retraining</p>
    """,
    severity='HIGH',
    channels=['email', 'slack', 'logging']
)
```

---

*[Continued in next message due to length...]*

Would you like me to continue with:
- Section 7: BigQuery ML Monitoring
- Section 8: Custom Monitoring Solutions
- Section 9: Best Practices
- Section 10: Troubleshooting and Response

Or would you prefer a specific focus on any particular monitoring aspect?<!-- filepath: c:\Users\Lenovo\Documents\docs\examdumps\certification\google_cloud\data_engineer\documentation\ml_notes\model_monitoring.md -->
# Comprehensive Guide to Model Monitoring

## Overview

Model monitoring is critical for maintaining model performance in production. This guide covers monitoring strategies, implementation in GCP Vertex AI, and best practices for detecting and responding to model degradation.

## Table of Contents

1. [Why Model Monitoring is Critical](#1-why-model-monitoring-is-critical)
2. [Types of Model Monitoring](#2-types-of-model-monitoring)
3. [Vertex AI Model Monitoring](#3-vertex-ai-model-monitoring)
4. [Setting Up Model Monitoring](#4-setting-up-model-monitoring)
5. [Monitoring Metrics and Thresholds](#5-monitoring-metrics-and-thresholds)
6. [Alert Configuration](#6-alert-configuration)
7. [BigQuery ML Monitoring](#7-bigquery-ml-monitoring)
8. [Custom Monitoring Solutions](#8-custom-monitoring-solutions)
9. [Best Practices](#9-best-practices)
10. [Troubleshooting and Response](#10-troubleshooting-and-response)

---

## 1. Why Model Monitoring is Critical

### 1.1 Common Problems in Production Models

```python
"""
MODEL DEGRADATION CAUSES:

1. DATA DRIFT
   - Input feature distributions change over time
   - Example: Customer behavior shifts during pandemic
   - Impact: Model trained on old patterns becomes less accurate

2. CONCEPT DRIFT
   - Relationship between features and target changes
   - Example: What constitutes "spam" evolves over time
   - Impact: Model's learned patterns become obsolete

3. UPSTREAM DATA ISSUES
   - Data pipeline failures or bugs
   - Example: Feature engineering logic breaks
   - Impact: Wrong inputs → wrong predictions

4. MODEL STALENESS
   - Model not retrained with recent data
   - Example: Seasonal patterns not captured
   - Impact: Predictions based on outdated patterns

5. SERVING INFRASTRUCTURE ISSUES
   - Latency problems
   - Example: Model serving timeout
   - Impact: Slow or failed predictions

6. LABEL SHIFT
   - Distribution of target variable changes
   - Example: Fraud rate increases suddenly
   - Impact: Class imbalance issues
"""

# Example: Data Drift Detection
import pandas as pd
import numpy as np
from scipy import stats

def detect_data_drift(training_data, production_data, feature_name, threshold=0.05):
    """
    Detect data drift using Kolmogorov-Smirnov test
    
    Args:
        training_data: Training dataset
        production_data: Recent production data
        feature_name: Feature to check
        threshold: P-value threshold (default 0.05)
    
    Returns:
        Dictionary with drift detection results
    """
    
    train_values = training_data[feature_name].dropna()
    prod_values = production_data[feature_name].dropna()
    
    # Perform KS test
    statistic, p_value = stats.ks_2samp(train_values, prod_values)
    
    # Check if drift detected
    drift_detected = p_value < threshold
    
    result = {
        'feature': feature_name,
        'ks_statistic': statistic,
        'p_value': p_value,
        'drift_detected': drift_detected,
        'severity': 'HIGH' if statistic > 0.2 else 'MEDIUM' if statistic > 0.1 else 'LOW',
        'train_mean': train_values.mean(),
        'prod_mean': prod_values.mean(),
        'train_std': train_values.std(),
        'prod_std': prod_values.std()
    }
    
    return result

# Usage
drift_results = detect_data_drift(
    training_data=train_df,
    production_data=prod_df,
    feature_name='age',
    threshold=0.05
)

if drift_results['drift_detected']:
    print(f"⚠️ DRIFT DETECTED in {drift_results['feature']}")
    print(f"   KS Statistic: {drift_results['ks_statistic']:.4f}")
    print(f"   P-value: {drift_results['p_value']:.4f}")
    print(f"   Severity: {drift_results['severity']}")
```

### 1.2 Monitoring Strategy

```python
"""
COMPREHENSIVE MONITORING STRATEGY:

1. INPUT MONITORING
   - Feature distributions
   - Missing values
   - Data types
   - Value ranges
   
2. PREDICTION MONITORING
   - Prediction distributions
   - Confidence scores
   - Prediction patterns
   
3. PERFORMANCE MONITORING
   - Model accuracy (when labels available)
   - Precision/Recall
   - F1-score
   - AUC-ROC
   
4. OPERATIONAL MONITORING
   - Latency (p50, p95, p99)
   - Throughput (QPS)
   - Error rates
   - Resource utilization
   
5. BUSINESS METRICS
   - Revenue impact
   - User satisfaction
   - Conversion rates
   - Cost per prediction
"""
```

---

## 2. Types of Model Monitoring

### 2.1 Data Drift Detection

```python
"""
DATA DRIFT: Changes in input feature distributions

Detection Methods:
1. Statistical Tests (KS test, Chi-square)
2. Distribution Distance (KL divergence, Wasserstein)
3. Population Stability Index (PSI)
"""

# Population Stability Index (PSI)
def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index
    
    PSI < 0.1: No significant change
    0.1 ≤ PSI < 0.25: Moderate change
    PSI ≥ 0.25: Significant change (retrain recommended)
    """
    
    # Create bins
    breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
    
    # Calculate distributions
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Add small value to avoid log(0)
    expected_percents = expected_percents + 0.0001
    actual_percents = actual_percents + 0.0001
    
    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)
    
    return psi

# Example usage
train_feature = train_df['age'].values
prod_feature = prod_df['age'].values

psi_score = calculate_psi(train_feature, prod_feature)

print(f"PSI Score: {psi_score:.4f}")
if psi_score < 0.1:
    print("✅ No significant drift")
elif psi_score < 0.25:
    print("⚠️ Moderate drift detected")
else:
    print("🚨 Significant drift - retrain recommended!")
```

### 2.2 Prediction Drift Detection

```python
"""
PREDICTION DRIFT: Changes in model output distribution
"""

def detect_prediction_drift(train_predictions, prod_predictions):
    """
    Detect drift in prediction distributions
    """
    
    # For classification: Check class distribution
    train_dist = pd.Series(train_predictions).value_counts(normalize=True)
    prod_dist = pd.Series(prod_predictions).value_counts(normalize=True)
    
    # Calculate Total Variation Distance
    tvd = 0.5 * np.sum(np.abs(train_dist - prod_dist))
    
    # Chi-square test
    chi2_stat, p_value = stats.chisquare(
        f_obs=prod_dist.values * len(prod_predictions),
        f_exp=train_dist.values * len(prod_predictions)
    )
    
    return {
        'tvd': tvd,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'drift_detected': p_value < 0.05,
        'train_distribution': train_dist.to_dict(),
        'prod_distribution': prod_dist.to_dict()
    }

# For regression: Check prediction statistics
def check_regression_drift(train_predictions, prod_predictions):
    """
    Check drift in regression predictions
    """
    
    train_mean = np.mean(train_predictions)
    prod_mean = np.mean(prod_predictions)
    train_std = np.std(train_predictions)
    prod_std = np.std(prod_predictions)
    
    # Percentage change
    mean_change = abs((prod_mean - train_mean) / train_mean) * 100
    std_change = abs((prod_std - train_std) / train_std) * 100
    
    # Statistical test
    statistic, p_value = stats.ks_2samp(train_predictions, prod_predictions)
    
    return {
        'mean_change_percent': mean_change,
        'std_change_percent': std_change,
        'ks_statistic': statistic,
        'p_value': p_value,
        'drift_detected': p_value < 0.05 or mean_change > 10 or std_change > 20
    }
```

### 2.3 Performance Monitoring

```python
"""
PERFORMANCE MONITORING: Track actual model performance

Requires ground truth labels (delayed feedback)
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class PerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, baseline_metrics):
        """
        Args:
            baseline_metrics: Dict of baseline metrics from validation
        """
        self.baseline_metrics = baseline_metrics
        self.performance_history = []
        
    def evaluate_batch(self, y_true, y_pred, y_pred_proba=None, timestamp=None):
        """
        Evaluate a batch of predictions
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            timestamp: Batch timestamp (optional)
        """
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'timestamp': timestamp or pd.Timestamp.now(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_samples': len(y_true)
        }
        
        # Add AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                metrics['auc_roc'] = auc
            except:
                pass
        
        # Check for degradation
        metrics['degradation'] = self._check_degradation(metrics)
        
        self.performance_history.append(metrics)
        
        return metrics
    
    def _check_degradation(self, current_metrics):
        """Check if performance degraded from baseline"""
        
        degradation = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in self.baseline_metrics:
                baseline = self.baseline_metrics[metric]
                current = current_metrics[metric]
                
                # Calculate percentage drop
                drop = ((baseline - current) / baseline) * 100
                
                degradation[metric] = {
                    'baseline': baseline,
                    'current': current,
                    'drop_percent': drop,
                    'degraded': drop > 5  # 5% threshold
                }
        
        return degradation
    
    def get_summary(self):
        """Get summary of performance over time"""
        
        if not self.performance_history:
            return None
        
        df = pd.DataFrame(self.performance_history)
        
        summary = {
            'mean_accuracy': df['accuracy'].mean(),
            'std_accuracy': df['accuracy'].std(),
            'min_accuracy': df['accuracy'].min(),
            'max_accuracy': df['accuracy'].max(),
            'trend': 'decreasing' if df['accuracy'].iloc[-1] < df['accuracy'].iloc[0] else 'stable/increasing',
            'total_samples': df['n_samples'].sum()
        }
        
        return summary

# Usage
baseline_metrics = {
    'accuracy': 0.92,
    'precision': 0.91,
    'recall': 0.93,
    'f1_score': 0.92
}

monitor = PerformanceMonitor(baseline_metrics)

# Evaluate daily batches
batch_metrics = monitor.evaluate_batch(
    y_true=batch_labels,
    y_pred=batch_predictions,
    y_pred_proba=batch_probabilities,
    timestamp=pd.Timestamp.now()
)

# Check for degradation
if any(d['degraded'] for d in batch_metrics['degradation'].values()):
    print("🚨 PERFORMANCE DEGRADATION DETECTED!")
    print(batch_metrics['degradation'])
```

---

## 3. Vertex AI Model Monitoring

### 3.1 Overview

```python
"""
VERTEX AI MODEL MONITORING FEATURES:

1. Automatic Monitoring
   - Training-serving skew detection
   - Prediction drift detection
   - Feature attribution drift
   
2. Alert Configuration
   - Email notifications
   - Cloud Logging integration
   - Custom alert thresholds
   
3. Built-in Metrics
   - Feature distribution comparison
   - Prediction distribution analysis
   - Anomaly detection
   
4. Monitoring Jobs
   - Scheduled monitoring runs
   - Sampling of prediction data
   - Historical trend analysis
"""
```

### 3.2 Model Monitoring Job Setup

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring

# Initialize Vertex AI
aiplatform.init(
    project='my-project-id',
    location='us-central1'
)

def create_model_monitoring_job(
    endpoint_name,
    model_name,
    training_dataset_uri,
    alert_emails,
    monitoring_frequency='HOURLY',
    skew_thresholds=None,
    drift_thresholds=None
):
    """
    Create comprehensive model monitoring job
    
    Args:
        endpoint_name: Name of deployed endpoint
        model_name: Name of the model
        training_dataset_uri: GCS URI to training dataset
        alert_emails: List of emails for alerts
        monitoring_frequency: How often to run monitoring
        skew_thresholds: Dict of skew thresholds per feature
        drift_thresholds: Dict of drift thresholds per feature
    """
    
    # Default thresholds if not provided
    if skew_thresholds is None:
        skew_thresholds = {
            'age': 0.1,
            'income': 0.1,
            'credit_score': 0.15,
            'default': 0.1  # Default threshold for all features
        }
    
    if drift_thresholds is None:
        drift_thresholds = {
            'age': 0.1,
            'income': 0.1,
            'credit_score': 0.15,
            'default': 0.1
        }
    
    # Get endpoint
    endpoint = aiplatform.Endpoint(endpoint_name)
    
    # Create skew detection config
    skew_config = model_monitoring.SkewDetectionConfig(
        data_source=training_dataset_uri,
        skew_thresholds=skew_thresholds,
        target_field='label'  # Name of label column in training data
    )
    
    # Create drift detection config
    drift_config = model_monitoring.DriftDetectionConfig(
        drift_thresholds=drift_thresholds
    )
    
    # Create explanation config (for feature attribution drift)
    explanation_config = model_monitoring.ExplanationConfig(
        enable_feature_attributes=True,
        explanation_baseline=None  # Will use default baseline
    )
    
    # Create alert config
    alert_config = model_monitoring.EmailAlertConfig(
        user_emails=alert_emails,
        enable_logging=True
    )
    
    # Create monitoring objective config
    objective_config = model_monitoring.ObjectiveConfig(
        skew_detection_config=skew_config,
        drift_detection_config=drift_config,
        explanation_config=explanation_config
    )
    
    # Create monitoring schedule
    monitoring_job = endpoint.create_model_deployment_monitoring_job(
        display_name=f'{model_name}_monitoring',
        logging_sampling_strategy=model_monitoring.RandomSampleConfig(
            sample_rate=0.5  # Sample 50% of predictions
        ),
        schedule_config=model_monitoring.ScheduleConfig(
            monitor_interval=monitoring_frequency  # 'HOURLY', 'DAILY', 'WEEKLY'
        ),
        alert_config=alert_config,
        objective_configs=[objective_config],
        # Advanced options
        enable_monitoring_pipeline_logs=True,
        model_monitoring_job_id=f'{model_name}_monitoring_job',
        stats_anomalies_base_directory=f'gs://my-bucket/monitoring/{model_name}/'
    )
    
    print(f"✅ Monitoring job created: {monitoring_job.resource_name}")
    
    return monitoring_job

# Usage
monitoring_job = create_model_monitoring_job(
    endpoint_name='projects/123/locations/us-central1/endpoints/456',
    model_name='fraud_detection_model',
    training_dataset_uri='gs://my-bucket/training_data.csv',
    alert_emails=['ml-team@company.com', 'data-eng@company.com'],
    monitoring_frequency='HOURLY',
    skew_thresholds={
        'transaction_amount': 0.15,
        'merchant_category': 0.2,
        'time_of_day': 0.1,
        'default': 0.1
    },
    drift_thresholds={
        'transaction_amount': 0.15,
        'merchant_category': 0.2,
        'time_of_day': 0.1,
        'default': 0.1
    }
)
```

### 3.3 Complete Vertex AI Monitoring Setup

```python
"""
COMPLETE VERTEX AI MODEL MONITORING SETUP
"""

from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring
import json

class VertexAIModelMonitor:
    """
    Complete model monitoring setup for Vertex AI
    """
    
    def __init__(self, project_id, location, bucket_name):
        """
        Initialize Vertex AI Model Monitor
        
        Args:
            project_id: GCP project ID
            location: GCP region
            bucket_name: GCS bucket for monitoring artifacts
        """
        aiplatform.init(project=project_id, location=location)
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        
    def setup_monitoring(
        self,
        endpoint_id,
        model_id,
        feature_config,
        alert_config,
        monitoring_frequency='HOURLY'
    ):
        """
        Set up comprehensive model monitoring
        
        Args:
            endpoint_id: Vertex AI endpoint ID
            model_id: Model ID to monitor
            feature_config: Dict with feature monitoring configuration
            alert_config: Dict with alert configuration
            monitoring_frequency: Monitoring interval
        """
        
        # Get endpoint
        endpoint = aiplatform.Endpoint(
            endpoint_name=f'projects/{self.project_id}/locations/{self.location}/endpoints/{endpoint_id}'
        )
        
        # Build objective configs
        objective_configs = []
        
        for feature_name, config in feature_config.items():
            # Training-serving skew detection
            if 'training_dataset' in config:
                skew_config = model_monitoring.SkewDetectionConfig(
                    data_source=config['training_dataset'],
                    skew_thresholds={
                        feature_name: config.get('skew_threshold', 0.1)
                    },
                    target_field=config.get('target_field', 'label')
                )
                
                objective = model_monitoring.ObjectiveConfig(
                    skew_detection_config=skew_config
                )
                objective_configs.append(objective)
            
            # Prediction drift detection
            if config.get('monitor_drift', True):
                drift_config = model_monitoring.DriftDetectionConfig(
                    drift_thresholds={
                        feature_name: config.get('drift_threshold', 0.1)
                    }
                )
                
                objective = model_monitoring.ObjectiveConfig(
                    drift_detection_config=drift_config
                )
                objective_configs.append(objective)
        
        # Create email alert config
        email_alert = model_monitoring.EmailAlertConfig(
            user_emails=alert_config['emails'],
            enable_logging=True
        )
        
        # Create monitoring job
        monitoring_job = endpoint.create_model_deployment_monitoring_job(
            display_name=alert_config.get('job_name', 'model_monitoring'),
            
            # Sampling strategy
            logging_sampling_strategy=model_monitoring.RandomSampleConfig(
                sample_rate=alert_config.get('sample_rate', 0.5)
            ),
            
            # Schedule
            schedule_config=model_monitoring.ScheduleConfig(
                monitor_interval=monitoring_frequency
            ),
            
            # Alerts
            alert_config=email_alert,
            
            # Objectives
            objective_configs=objective_configs,
            
            # Logging
            enable_monitoring_pipeline_logs=True,
            
            # Storage
            stats_anomalies_base_directory=f'gs://{self.bucket_name}/monitoring/{model_id}/',
            
            # Labels for organization
            labels={
                'model_id': model_id,
                'environment': 'production',
                'team': alert_config.get('team', 'ml')
            }
        )
        
        print("=" * 80)
        print("✅ MODEL MONITORING CONFIGURED")
        print("=" * 80)
        print(f"Job Name: {monitoring_job.display_name}")
        print(f"Resource Name: {monitoring_job.resource_name}")
        print(f"Monitoring Frequency: {monitoring_frequency}")
        print(f"Sample Rate: {alert_config.get('sample_rate', 0.5) * 100}%")
        print(f"Alert Emails: {', '.join(alert_config['emails'])}")
        print(f"Monitoring {len(objective_configs)} objectives")
        print("=" * 80)
        
        return monitoring_job
    
    def get_monitoring_stats(self, monitoring_job_name):
        """
        Retrieve monitoring statistics
        
        Args:
            monitoring_job_name: Name of monitoring job
        """
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        
        # Get latest stats
        stats = job.get_model_monitoring_stats()
        
        return stats
    
    def list_anomalies(self, monitoring_job_name, time_window_hours=24):
        """
        List detected anomalies
        
        Args:
            monitoring_job_name: Name of monitoring job
            time_window_hours: Look back window in hours
        """
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        
        # List anomalies
        anomalies = job.list_anomalies(
            time_window_hours=time_window_hours
        )
        
        return anomalies
    
    def pause_monitoring(self, monitoring_job_name):
        """Pause monitoring job"""
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        job.pause()
        print(f"✅ Monitoring job paused: {monitoring_job_name}")
    
    def resume_monitoring(self, monitoring_job_name):
        """Resume monitoring job"""
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        job.resume()
        print(f"✅ Monitoring job resumed: {monitoring_job_name}")
    
    def delete_monitoring(self, monitoring_job_name):
        """Delete monitoring job"""
        
        job = aiplatform.ModelDeploymentMonitoringJob(monitoring_job_name)
        job.delete()
        print(f"✅ Monitoring job deleted: {monitoring_job_name}")

# Usage Example
monitor = VertexAIModelMonitor(
    project_id='my-project',
    location='us-central1',
    bucket_name='my-model-monitoring-bucket'
)

# Feature configuration
feature_config = {
    'transaction_amount': {
        'training_dataset': 'gs://my-bucket/training_data.csv',
        'skew_threshold': 0.15,
        'drift_threshold': 0.15,
        'monitor_drift': True,
        'target_field': 'is_fraud'
    },
    'merchant_category': {
        'training_dataset': 'gs://my-bucket/training_data.csv',
        'skew_threshold': 0.2,
        'drift_threshold': 0.2,
        'monitor_drift': True,
        'target_field': 'is_fraud'
    },
    'customer_age': {
        'skew_threshold': 0.1,
        'drift_threshold': 0.1,
        'monitor_drift': True
    }
}

# Alert configuration
alert_config = {
    'job_name': 'fraud_detection_monitoring',
    'emails': ['ml-team@company.com', 'alerts@company.com'],
    'sample_rate': 0.5,  # Monitor 50% of predictions
    'team': 'fraud-ml'
}

# Set up monitoring
monitoring_job = monitor.setup_monitoring(
    endpoint_id='1234567890',
    model_id='fraud_detection_v1',
    feature_config=feature_config,
    alert_config=alert_config,
    monitoring_frequency='HOURLY'
)

# Later: Get monitoring stats
stats = monitor.get_monitoring_stats(monitoring_job.resource_name)
print(stats)

# List recent anomalies
anomalies = monitor.list_anomalies(
    monitoring_job.resource_name,
    time_window_hours=24
)
for anomaly in anomalies:
    print(f"⚠️ Anomaly detected: {anomaly}")
```

---

## 4. Setting Up Model Monitoring

### 4.1 Monitoring Configuration File

```yaml
# model_monitoring_config.yaml
# Complete monitoring configuration

monitoring_config:
  project_id: "my-gcp-project"
  location: "us-central1"
  
  # Endpoint configuration
  endpoint:
    endpoint_id: "1234567890"
    model_id: "fraud_detection_v1"
    model_version: "v1.2.0"
  
  # Storage configuration
  storage:
    bucket_name: "model-monitoring-artifacts"
    training_data_uri: "gs://my-bucket/training_data/fraud_training.csv"
    monitoring_results_path: "gs://my-bucket/monitoring/results/"
    logs_path: "gs://my-bucket/monitoring/logs/"
  
  # Sampling configuration
  sampling:
    strategy: "random"  # or "stratified"
    sample_rate: 0.5  # Monitor 50% of predictions
    min_samples_per_batch: 1000
    max_samples_per_batch: 10000
  
  # Monitoring schedule
  schedule:
    frequency: "HOURLY"  # HOURLY, DAILY, WEEKLY
    start_time: "00:00"
    timezone: "America/New_York"
  
  # Feature monitoring
  features:
    - name: "transaction_amount"
      type: "numerical"
      skew_threshold: 0.15
      drift_threshold: 0.15
      monitor_skew: true
      monitor_drift: true
      
    - name: "merchant_category"
      type: "categorical"
      skew_threshold: 0.20
      drift_threshold: 0.20
      monitor_skew: true
      monitor_drift: true
      
    - name: "customer_age"
      type: "numerical"
      skew_threshold: 0.10
      drift_threshold: 0.10
      monitor_skew: true
      monitor_drift: true
      
    - name: "transaction_time"
      type: "numerical"
      skew_threshold: 0.12
      drift_threshold: 0.12
      monitor_skew: true
      monitor_drift: true
  
  # Prediction monitoring
  predictions:
    monitor_distribution: true
    drift_threshold: 0.15
    confidence_threshold: 0.5
    monitor_confidence_distribution: true
  
  # Performance monitoring
  performance:
    enabled: true
    ground_truth_delay_hours: 24  # How long until labels available
    metrics:
      - accuracy
      - precision
      - recall
      - f1_score
      - auc_roc
    thresholds:
      accuracy: 0.90  # Alert if below 90%
      precision: 0.88
      recall: 0.92
      f1_score: 0.90
      auc_roc: 0.93
  
  # Alert configuration
  alerts:
    # Email alerts
    email:
      enabled: true
      recipients:
        - ml-team@company.com
        - data-eng@company.com
        - oncall@company.com
      send_daily_summary: true
      summary_time: "09:00"
    
    # Slack alerts
    slack:
      enabled: true
      webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
      channel: "#ml-alerts"
      mention_users:
        - "@ml-lead"
        - "@data-engineer"
    
    # PagerDuty for critical alerts
    pagerduty:
      enabled: true
      integration_key: "YOUR_PAGERDUTY_KEY"
      severity_threshold: "HIGH"  # Only page for HIGH/CRITICAL
    
    # Cloud Logging
    logging:
      enabled: true
      log_name: "model-monitoring"
      severity: "WARNING"
  
  # Alert rules
  alert_rules:
    - name: "high_skew_detection"
      condition: "skew > threshold"
      severity: "MEDIUM"
      action: "email"
      
    - name: "critical_drift"
      condition: "drift > threshold * 1.5"
      severity: "HIGH"
      action: "email,slack,pagerduty"
      
    - name: "performance_degradation"
      condition: "accuracy < threshold"
      severity: "HIGH"
      action: "email,slack"
      
    - name: "serving_errors"
      condition: "error_rate > 5%"
      severity: "CRITICAL"
      action: "email,slack,pagerduty"
  
  # Automated actions
  automated_actions:
    - trigger: "critical_drift"
      action: "pause_endpoint"
      confirmation_required: true
      
    - trigger: "performance_degradation"
      action: "trigger_retraining"
      confirmation_required: false
      
    - trigger: "high_error_rate"
      action: "rollback_model"
      confirmation_required: true
```

### 4.2 Load and Apply Configuration

```python
import yaml
from google.cloud import aiplatform

def load_monitoring_config(config_path):
    """Load monitoring configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['monitoring_config']

def apply_monitoring_config(config):
    """
    Apply monitoring configuration to Vertex AI
    """
    
    # Initialize Vertex AI
    aiplatform.init(
        project=config['project_id'],
        location=config['location']
    )
    
    # Get endpoint
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{config['project_id']}/locations/{config['location']}/endpoints/{config['endpoint']['endpoint_id']}"
    )
    
    # Build skew detection configs
    skew_configs = {}
    drift_configs = {}
    
    for feature in config['features']:
        if feature['monitor_skew']:
            skew_configs[feature['name']] = feature['skew_threshold']
        if feature['monitor_drift']:
            drift_configs[feature['name']] = feature['drift_threshold']
    
    # Create monitoring job
    monitoring_job = endpoint.create_model_deployment_monitoring_job(
        display_name=f"{config['endpoint']['model_id']}_monitoring",
        
        # Sampling
        logging_sampling_strategy=model_monitoring.RandomSampleConfig(
            sample_rate=config['sampling']['sample_rate']
        ),
        
        # Schedule
        schedule_config=model_monitoring.ScheduleConfig(
            monitor_interval=config['schedule']['frequency']
        ),
        
        # Objectives
        objective_configs=[
            model_monitoring.ObjectiveConfig(
                skew_detection_config=model_monitoring.SkewDetectionConfig(
                    data_source=config['storage']['training_data_uri'],
                    skew_thresholds=skew_configs
                ),
                drift_detection_config=model_monitoring.DriftDetectionConfig(
                    drift_thresholds=drift_configs
                )
            )
        ],
        
        # Alerts
        alert_config=model_monitoring.EmailAlertConfig(
            user_emails=config['alerts']['email']['recipients'],
            enable_logging=config['alerts']['logging']['enabled']
        ),
        
        # Storage
        stats_anomalies_base_directory=config['storage']['monitoring_results_path']
    )
    
    print(f"✅ Monitoring job created from config: {monitoring_job.resource_name}")
    
    return monitoring_job

# Usage
config = load_monitoring_config('model_monitoring_config.yaml')
monitoring_job = apply_monitoring_config(config)
```

---

## 5. Monitoring Metrics and Thresholds

### 5.1 Recommended Thresholds

```python
"""
RECOMMENDED MONITORING THRESHOLDS

1. TRAINING-SERVING SKEW
   - Low Threshold: 0.05-0.10 (sensitive, frequent alerts)
   - Medium Threshold: 0.10-0.15 (balanced)
   - High Threshold: 0.15-0.25 (conservative, fewer alerts)
   
   Recommendation: Start with 0.10, adjust based on false positives

2. PREDICTION DRIFT
   - Low: 0.05-0.10
   - Medium: 0.10-0.15
   - High: 0.15-0.25
   
   Recommendation: 0.15 for stable features, 0.10 for critical features

3. PERFORMANCE DEGRADATION
   - Critical: >10% drop from baseline
   - Warning: 5-10% drop from baseline
   - Monitor: <5% drop
   
   Recommendation: 5% threshold for alerts

4. LATENCY
   - P50: <50ms (median)
   - P95: <100ms
   - P99: <200ms
   
   Recommendation: Set based on SLA requirements

5. ERROR RATE
   - Green: <1%
   - Yellow: 1-5%
   - Red: >5%
   
   Recommendation: Alert at 2%, escalate at 5%
"""

# Threshold configuration
MONITORING_THRESHOLDS = {
    # Feature-specific skew thresholds
    'skew': {
        'default': 0.10,
        'critical_features': {
            'transaction_amount': 0.08,
            'customer_risk_score': 0.08
        },
        'stable_features': {
            'customer_age': 0.15,
            'account_tenure': 0.15
        }
    },
    
    # Feature-specific drift thresholds
    'drift': {
        'default': 0.10,
        'critical_features': {
            'transaction_amount': 0.08,
            'merchant_category': 0.10
        },
        'stable_features': {
            'customer_age': 0.15,
            'country': 0.20
        }
    },
    
    # Performance thresholds
    'performance': {
        'accuracy': {
            'baseline': 0.92,
            'warning': 0.87,  # 5% drop
            'critical': 0.83   # 10% drop
        },
        'precision': {
            'baseline': 0.90,
            'warning': 0.85,
            'critical': 0.81
        },
        'recall': {
            'baseline': 0.94,
            'warning': 0.89,
            'critical': 0.85
        },
        'f1_score': {
            'baseline': 0.92,
            'warning': 0.87,
            'critical': 0.83
        }
    },
    
    # Operational thresholds
    'latency_ms': {
        'p50': 50,
        'p95': 100,
        'p99': 200
    },
    
    'error_rate': {
        'warning': 0.02,  # 2%
        'critical': 0.05   # 5%
    },
    
    # Prediction distribution
    'prediction_drift': {
        'default': 0.15,
        'class_imbalance_threshold': 0.20  # Allow more drift for imbalanced classes
    }
}

def get_threshold(feature_name, threshold_type, feature_category='default'):
    """
    Get threshold for a feature
    
    Args:
        feature_name: Name of the feature
        threshold_type: 'skew' or 'drift'
        feature_category: 'default', 'critical_features', or 'stable_features'
    
    Returns:
        Threshold value
    """
    
    thresholds = MONITORING_THRESHOLDS[threshold_type]
    
    # Check if specific threshold exists for this feature
    if feature_category in thresholds and feature_name in thresholds[feature_category]:
        return thresholds[feature_category][feature_name]
    
    # Return default
    return thresholds['default']

# Usage
skew_threshold = get_threshold('transaction_amount', 'skew', 'critical_features')
print(f"Skew threshold for transaction_amount: {skew_threshold}")
```

### 5.2 Adaptive Thresholds

```python
"""
ADAPTIVE THRESHOLDS: Adjust thresholds based on historical data
"""

import numpy as np
from scipy import stats

class AdaptiveThresholdCalculator:
    """
    Calculate adaptive monitoring thresholds based on historical data
    """
    
    def __init__(self, historical_data, confidence_level=0.95):
        """
        Args:
            historical_data: DataFrame with historical monitoring metrics
            confidence_level: Confidence level for threshold calculation
        """
        self.historical_data = historical_data
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
    def calculate_drift_threshold(self, feature_name):
        """
        Calculate adaptive drift threshold
        
        Uses historical drift scores to set threshold
        at mean + (z_score * std)
        """
        
        drift_scores = self.historical_data[f'{feature_name}_drift']
        
        mean_drift = drift_scores.mean()
        std_drift = drift_scores.std()
        
        # Threshold = mean + z * std
        threshold = mean_drift + (self.z_score * std_drift)
        
        # Cap at reasonable values
        threshold = min(threshold, 0.25)  # Max 0.25
        threshold = max(threshold, 0.05)  # Min 0.05
        
        return {
            'feature': feature_name,
            'threshold': threshold,
            'mean_historical_drift': mean_drift,
            'std_historical_drift': std_drift,
            'confidence_level': self.confidence_level
        }
    
    def calculate_performance_threshold(self, metric_name, percentile=5):
        """
        Calculate adaptive performance threshold
        
        Uses historical performance to set threshold
        at the percentile (e.g., 5th percentile = bottom 5%)
        """
        
        metric_values = self.historical_data[metric_name]
        
        # Calculate percentile threshold
        threshold = np.percentile(metric_values, percentile)
        
        return {
            'metric': metric_name,
            'threshold': threshold,
            'percentile': percentile,
            'mean_historical': metric_values.mean(),
            'min_historical': metric_values.min(),
            'max_historical': metric_values.max()
        }
    
    def get_all_adaptive_thresholds(self, features, metrics):
        """
        Get all adaptive thresholds
        
        Args:
            features: List of feature names
            metrics: List of metric names
        """
        
        thresholds = {
            'drift': {},
            'performance': {}
        }
        
        # Calculate drift thresholds
        for feature in features:
            result = self.calculate_drift_threshold(feature)
            thresholds['drift'][feature] = result['threshold']
        
        # Calculate performance thresholds
        for metric in metrics:
            result = self.calculate_performance_threshold(metric)
            thresholds['performance'][metric] = result['threshold']
        
        return thresholds

# Usage
# Load historical monitoring data
historical_data = pd.read_csv('monitoring_history.csv')

calculator = AdaptiveThresholdCalculator(
    historical_data=historical_data,
    confidence_level=0.95
)

# Calculate adaptive thresholds
adaptive_thresholds = calculator.get_all_adaptive_thresholds(
    features=['transaction_amount', 'merchant_category', 'customer_age'],
    metrics=['accuracy', 'precision', 'recall', 'f1_score']
)

print("Adaptive Thresholds:")
print(json.dumps(adaptive_thresholds, indent=2))
```

---

## 6. Alert Configuration

### 6.1 Multi-Channel Alerting

```python
"""
MULTI-CHANNEL ALERTING SYSTEM
"""

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.cloud import logging as cloud_logging

class AlertManager:
    """
    Manage alerts across multiple channels
    """
    
    def __init__(self, config):
        """
        Args:
            config: Alert configuration dict
        """
        self.config = config
        self.cloud_logging_client = cloud_logging.Client()
        self.logger = self.cloud_logging_client.logger('model-monitoring-alerts')
        
    def send_email_alert(self, subject, body, severity='MEDIUM'):
        """
        Send email alert
        
        Args:
            subject: Email subject
            body: Email body (HTML supported)
            severity: Alert severity
        """
        
        if not self.config.get('email', {}).get('enabled', False):
            return
        
        # Email configuration
        smtp_server = self.config['email']['smtp_server']
        smtp_port = self.config['email']['smtp_port']
        sender = self.config['email']['sender']
        password = self.config['email']['password']
        recipients = self.config['email']['recipients']
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{severity}] {subject}"
        msg['From'] = sender
        msg['To'] = ', '.join(recipients)
        
        # Add body
        html_part = MIMEText(body, 'html')
        msg.attach(html_part)
        
        # Send email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
            
            print(f"✅ Email alert sent: {subject}")
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
    
    def send_slack_alert(self, message, severity='MEDIUM'):
        """
        Send Slack alert
        
        Args:
            message: Alert message
            severity: Alert severity
        """
        
        if not self.config.get('slack', {}).get('enabled', False):
            return
        
        webhook_url = self.config['slack']['webhook_url']
        channel = self.config['slack'].get('channel', '#ml-alerts')
        
        # Color based on severity
        colors = {
            'LOW': '#36a64f',      # Green
            'MEDIUM': '#ff9900',   # Orange
            'HIGH': '#ff0000',     # Red
            'CRITICAL': '#8b0000'  # Dark red
        }
        
        # Emoji based on severity
        emojis = {
            'LOW': ':information_source:',
            'MEDIUM': ':warning:',
            'HIGH': ':rotating_light:',
            'CRITICAL': ':fire:'
        }
        
        payload = {
            'channel': channel,
            'attachments': [{
                'color': colors.get(severity, '#ff9900'),
                'title': f"{emojis.get(severity, ':warning:')} Model Monitoring Alert",
                'text': message,
                'footer': 'Model Monitoring System',
                'ts': int(pd.Timestamp.now().timestamp())
            }]
        }
        
        # Mention users for HIGH/CRITICAL alerts
        if severity in ['HIGH', 'CRITICAL']:
            mentions = self.config['slack'].get('mention_users', [])
            if mentions:
                payload['text'] = ' '.join(mentions)
        
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            print(f"✅ Slack alert sent: {severity}")
            
        except Exception as e:
            print(f"❌ Failed to send Slack alert: {e}")
    
    def send_pagerduty_alert(self, title, description, severity='HIGH'):
        """
        Send PagerDuty alert
        
        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity
        """
        
        if not self.config.get('pagerduty', {}).get('enabled', False):
            return
        
        # Only page for HIGH/CRITICAL
        severity_threshold = self.config['pagerduty'].get('severity_threshold', 'HIGH')
        if severity not in ['HIGH', 'CRITICAL'] and severity_threshold == 'HIGH':
            return
        
        integration_key = self.config['pagerduty']['integration_key']
        
        payload = {
            'routing_key': integration_key,
            'event_action': 'trigger',
            'payload': {
                'summary': title,
                'severity': severity.lower(),
                'source': 'model-monitoring',
                'custom_details': {
                    'description': description
                }
            }
        }
        
        try:
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload
            )
            response.raise_for_status()
            print(f"✅ PagerDuty alert sent: {severity}")
            
        except Exception as e:
            print(f"❌ Failed to send PagerDuty alert: {e}")
    
    def log_to_cloud_logging(self, message, severity='WARNING'):
        """
        Log alert to Cloud Logging
        
        Args:
            message: Log message
            severity: Log severity
        """
        
        if not self.config.get('logging', {}).get('enabled', False):
            return
        
        self.logger.log_text(
            message,
            severity=severity
        )
        
        print(f"✅ Logged to Cloud Logging: {severity}")
    
    def send_alert(self, title, message, severity='MEDIUM', channels=None):
        """
        Send alert to multiple channels
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            channels: List of channels (default: all enabled)
        """
        
        if channels is None:
            channels = ['email', 'slack', 'pagerduty', 'logging']
        
        print(f"\n{'='*80}")
        print(f"🚨 SENDING ALERT: {title}")
        print(f"Severity: {severity}")
        print(f"Channels: {', '.join(channels)}")
        print(f"{'='*80}\n")
        
        if 'email' in channels:
            self.send_email_alert(title, message, severity)
        
        if 'slack' in channels:
            self.send_slack_alert(message, severity)
        
        if 'pagerduty' in channels:
            self.send_pagerduty_alert(title, message, severity)
        
        if 'logging' in channels:
            self.log_to_cloud_logging(message, severity)

# Usage
alert_config = {
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender': 'alerts@company.com',
        'password': 'your-password',
        'recipients': ['ml-team@company.com', 'oncall@company.com']
    },
    'slack': {
        'enabled': True,
        'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        'channel': '#ml-alerts',
        'mention_users': ['@ml-lead', '@data-engineer']
    },
    'pagerduty': {
        'enabled': True,
        'integration_key': 'YOUR_PAGERDUTY_KEY',
        'severity_threshold': 'HIGH'
    },
    'logging': {
        'enabled': True
    }
}

alert_manager = AlertManager(alert_config)

# Send alert
alert_manager.send_alert(
    title='High Drift Detected in transaction_amount Feature',
    message="""
    <h2>Model Monitoring Alert</h2>
    <p><strong>Feature:</strong> transaction_amount</p>
    <p><strong>Drift Score:</strong> 0.35 (threshold: 0.15)</p>
    <p><strong>Recommendation:</strong> Investigate data pipeline and consider retraining</p>
    """,
    severity='HIGH',
    channels=['email', 'slack', 'logging']
)
```

---

*[Continued in next message due to length...]*

Would you like me to continue with:
- Section 7: BigQuery ML Monitoring
- Section 8: Custom Monitoring Solutions
- Section 9: Best Practices
- Section 10: Troubleshooting and Response

Or would you prefer a specific focus on any particular monitoring aspect?