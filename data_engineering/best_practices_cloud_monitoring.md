# Cloud Monitoring Best Practices for Data Engineering

## Overview
Cloud Monitoring (formerly Stackdriver Monitoring) provides visibility into the performance, availability, and health of your applications and infrastructure. For data engineers, it's crucial for monitoring data pipelines, tracking SLAs, and ensuring reliable data delivery. This document covers best practices for monitoring data engineering workloads on GCP.

---

## 1. Monitoring Strategy

### 1.1 Define Monitoring Objectives
- **Operational health**: Is the pipeline running?
- **Performance**: Is it running efficiently?
- **Data quality**: Is the data correct?
- **Cost optimization**: Are we using resources efficiently?
- **SLA compliance**: Are we meeting business requirements?

### 1.2 Key Monitoring Pillars
- **USE Method** (for resources):
  - **Utilization**: Average resource usage over time
  - **Saturation**: Degree of queuing or waiting
  - **Errors**: Count of error events

- **RED Method** (for services):
  - **Rate**: Requests per second
  - **Errors**: Number of failed requests
  - **Duration**: Time taken to process requests

### 1.3 Monitoring Layers
1. **Infrastructure**: VMs, disks, networks
2. **Platform services**: Dataflow, BigQuery, Pub/Sub
3. **Application**: Custom pipeline code
4. **Business metrics**: Data quality, freshness, completeness

---

## 2. Key Metrics for Data Pipelines

### 2.1 Pipeline Health Metrics
- **Execution metrics**:
  - Pipeline run status (success/failure)
  - Execution duration
  - Start and end times
  - Retry counts

- **Data volume metrics**:
  - Records processed
  - Bytes processed
  - Input vs output record counts
  - Data growth rate

- **Performance metrics**:
  - Processing throughput (records/second)
  - Latency (end-to-end)
  - Resource utilization (CPU, memory, disk)
  - Worker count (for auto-scaling services)

### 2.2 Service-Specific Metrics

**Dataflow:**
```
dataflow.googleapis.com/job/elements_added_count
dataflow.googleapis.com/job/system_lag
dataflow.googleapis.com/job/data_watermark_age
dataflow.googleapis.com/job/current_num_vcpus
dataflow.googleapis.com/job/is_failed
```

**BigQuery:**
```
bigquery.googleapis.com/job/num_in_flight_jobs
bigquery.googleapis.com/job/execution_time
bigquery.googleapis.com/slots/total_allocated
bigquery.googleapis.com/storage/stored_bytes
```

**Pub/Sub:**
```
pubsub.googleapis.com/subscription/num_undelivered_messages
pubsub.googleapis.com/subscription/oldest_unacked_message_age
pubsub.googleapis.com/subscription/pull_request_count
pubsub.googleapis.com/topic/send_message_operation_count
```

**Cloud Functions:**
```
cloudfunctions.googleapis.com/function/execution_count
cloudfunctions.googleapis.com/function/execution_times
cloudfunctions.googleapis.com/function/user_memory_bytes
cloudfunctions.googleapis.com/function/instance_count
```

### 2.3 Data Quality Metrics
- **Freshness**: Time since last update
- **Completeness**: % of expected records received
- **Accuracy**: % of records passing validation
- **Consistency**: Data matches across systems
- **Uniqueness**: Duplicate record rate

---

## 3. Custom Metrics

### 3.1 Creating Custom Metrics
- **Write custom metrics from applications**:
  ```python
  from google.cloud import monitoring_v3
  import time
  
  def write_custom_metric(project_id, metric_type, value, labels=None):
      client = monitoring_v3.MetricServiceClient()
      project_name = f"projects/{project_id}"
      
      # Create time series
      series = monitoring_v3.TimeSeries()
      series.metric.type = f"custom.googleapis.com/{metric_type}"
      
      # Add labels
      if labels:
          for key, val in labels.items():
              series.metric.labels[key] = val
      
      # Add data point
      now = time.time()
      seconds = int(now)
      nanos = int((now - seconds) * 10 ** 9)
      
      interval = monitoring_v3.TimeInterval(
          {"end_time": {"seconds": seconds, "nanos": nanos}}
      )
      
      point = monitoring_v3.Point({
          "interval": interval,
          "value": {"double_value": value}
      })
      
      series.points = [point]
      
      # Write to Cloud Monitoring
      client.create_time_series(
          name=project_name, 
          time_series=[series]
      )
  
  # Usage
  write_custom_metric(
      project_id="my-project",
      metric_type="pipeline/records_processed",
      value=15000,
      labels={
          "pipeline": "customer-etl",
          "environment": "production"
      }
  )
  ```

### 3.2 Custom Metric Examples
- **Data quality metrics**:
  ```python
  # Record validation rate
  write_custom_metric(
      project_id,
      "data_quality/validation_pass_rate",
      validation_pass_rate,
      labels={"dataset": "orders", "table": "daily_orders"}
  )
  
  # Data freshness
  freshness_minutes = (current_time - last_update).total_seconds() / 60
  write_custom_metric(
      project_id,
      "data_quality/freshness_minutes",
      freshness_minutes,
      labels={"dataset": "analytics", "table": "customer_metrics"}
  )
  
  # Record count
  write_custom_metric(
      project_id,
      "pipeline/daily_record_count",
      record_count,
      labels={"pipeline": "ingest", "source": "api"}
  )
  ```

### 3.3 OpenTelemetry Integration
- **Use OpenTelemetry for standardized metrics**:
  ```python
  from opentelemetry import metrics
  from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
  from opentelemetry.sdk.metrics import MeterProvider
  from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
  
  # Setup exporter
  exporter = CloudMonitoringMetricsExporter()
  reader = PeriodicExportingMetricReader(exporter)
  provider = MeterProvider(metric_readers=[reader])
  metrics.set_meter_provider(provider)
  
  # Create meter
  meter = metrics.get_meter(__name__)
  
  # Create instruments
  records_counter = meter.create_counter(
      "pipeline.records.processed",
      description="Number of records processed",
      unit="1"
  )
  
  duration_histogram = meter.create_histogram(
      "pipeline.duration",
      description="Pipeline execution duration",
      unit="ms"
  )
  
  # Use instruments
  records_counter.add(1000, {"pipeline": "etl", "stage": "transform"})
  duration_histogram.record(1234.5, {"pipeline": "etl"})
  ```

---

## 4. Dashboards

### 4.1 Dashboard Design Principles
- **Purpose-driven**: Each dashboard should have a clear purpose
- **Audience-specific**: Tailor to the viewer (ops, dev, business)
- **Actionable**: Show metrics that drive decisions
- **At-a-glance**: Most important info visible immediately
- **Drill-down capable**: Allow deeper investigation

### 4.2 Essential Dashboards

**Pipeline Health Dashboard:**
```yaml
Dashboard Contents:
- Pipeline execution status (last 24h)
- Success/failure rate
- Current active jobs
- System lag (for streaming)
- Error count by pipeline
- Resource utilization
```

**Performance Dashboard:**
```yaml
Dashboard Contents:
- Throughput (records/second)
- Processing duration (p50, p95, p99)
- Worker count over time
- CPU and memory utilization
- Backlog size
- Cost per record processed
```

**Data Quality Dashboard:**
```yaml
Dashboard Contents:
- Data freshness by dataset
- Validation pass rate
- Record count trends
- Schema change alerts
- Duplicate record rate
- Completeness percentage
```

**Cost Optimization Dashboard:**
```yaml
Dashboard Contents:
- BigQuery bytes processed/billed
- Dataflow worker hours
- Storage costs by dataset
- Pub/Sub message costs
- Cost per pipeline
- Cost trends and forecasts
```

### 4.3 Creating Dashboards
- **Using Cloud Console**:
  1. Navigate to Cloud Monitoring
  2. Select Dashboards
  3. Create Dashboard
  4. Add charts (line, stacked area, table, etc.)
  5. Configure filters and grouping
  6. Save and share

- **Using Terraform**:
  ```hcl
  resource "google_monitoring_dashboard" "pipeline_health" {
    dashboard_json = jsonencode({
      displayName = "Pipeline Health Dashboard"
      
      mosaicLayout = {
        columns = 12
        tiles = [
          {
            width  = 6
            height = 4
            widget = {
              title = "Pipeline Success Rate"
              xyChart = {
                dataSets = [{
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"custom.googleapis.com/pipeline/status\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_RATE"
                      }
                    }
                  }
                }]
              }
            }
          }
        ]
      }
    })
  }
  ```

### 4.4 Dashboard Best Practices
- **Use consistent time windows** (last 1h, 24h, 7d, 30d)
- **Color code by severity** (green=good, yellow=warning, red=critical)
- **Show trends** (not just current values)
- **Include context** (thresholds, SLAs, historical baselines)
- **Keep it simple**: 6-8 charts per dashboard maximum
- **Version control**: Store dashboard configs in Git

---

## 5. Alerting

### 5.1 Alert Policy Design
- **What to alert on**:
  - Pipeline failures
  - SLA violations
  - Resource exhaustion
  - Data quality issues
  - Anomalies and outliers
  - Cost overruns

### 5.2 Creating Alert Policies
- **Pipeline failure alert**:
  ```bash
  gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="Dataflow Job Failures" \
    --condition-display-name="Job Failed" \
    --condition-threshold-value=1 \
    --condition-threshold-duration=60s \
    --condition-filter='metric.type="dataflow.googleapis.com/job/is_failed" AND resource.type="dataflow_job"'
  ```

- **Using Terraform**:
  ```hcl
  resource "google_monitoring_alert_policy" "pipeline_failure" {
    display_name = "Pipeline Failure Alert"
    combiner     = "OR"
    
    conditions {
      display_name = "Pipeline Failed"
      
      condition_threshold {
        filter          = "metric.type=\"custom.googleapis.com/pipeline/status\" AND metric.label.status=\"failed\""
        duration        = "60s"
        comparison      = "COMPARISON_GT"
        threshold_value = 0
        
        aggregations {
          alignment_period   = "60s"
          per_series_aligner = "ALIGN_RATE"
        }
      }
    }
    
    notification_channels = [
      google_monitoring_notification_channel.email.id,
      google_monitoring_notification_channel.pagerduty.id
    ]
    
    alert_strategy {
      auto_close = "1800s"
    }
  }
  ```

### 5.3 Alert Threshold Best Practices
- **Use dynamic thresholds**: Adapt to normal patterns
- **Consider seasonality**: Different thresholds for different times
- **Account for noise**: Require multiple violations
- **Set appropriate duration**: Avoid flapping alerts
- **Test thresholds**: Validate before production

### 5.4 Alert Fatigue Prevention
- **Actionable alerts only**: Can someone fix it?
- **Appropriate severity**: Not everything is critical
- **Consolidate similar alerts**: Group related issues
- **Use alert policies**: Don't create redundant alerts
- **Regular review**: Disable or adjust noisy alerts
- **Documentation**: Include runbooks in alerts

### 5.5 Notification Channels
- **Email**:
  ```bash
  gcloud alpha monitoring channels create \
    --display-name="Data Team Email" \
    --type=email \
    --channel-labels=email_address=data-team@example.com
  ```

- **Slack**:
  ```bash
  gcloud alpha monitoring channels create \
    --display-name="Slack Data Alerts" \
    --type=slack \
    --channel-labels=url=https://hooks.slack.com/services/...
  ```

- **PagerDuty**:
  ```bash
  gcloud alpha monitoring channels create \
    --display-name="PagerDuty On-Call" \
    --type=pagerduty \
    --channel-labels=service_key=YOUR_SERVICE_KEY
  ```

---

## 6. Service Level Objectives (SLOs)

### 6.1 Defining SLOs
- **Availability SLO**: 99.9% of pipeline runs succeed
- **Latency SLO**: 95% of jobs complete within 1 hour
- **Freshness SLO**: Data updated within 30 minutes
- **Throughput SLO**: Process 1M records per hour

### 6.2 Implementing SLOs
- **Create SLO in Cloud Monitoring**:
  ```yaml
  service:
    name: customer-etl-pipeline
  
  slo:
    - display_name: "Pipeline Availability"
      goal: 0.999  # 99.9%
      rolling_period: 30d
      
      request_based_sli:
        good_total_ratio:
          good_service_filter: |
            metric.type="custom.googleapis.com/pipeline/status"
            metric.label.status="success"
          total_service_filter: |
            metric.type="custom.googleapis.com/pipeline/status"
    
    - display_name: "Data Freshness"
      goal: 0.95  # 95%
      rolling_period: 7d
      
      windows_based_sli:
        window_period: 300s  # 5 minutes
        good_bad_metric_filter: |
          metric.type="custom.googleapis.com/data/freshness_minutes"
          metric.value < 30
  ```

### 6.3 Error Budget
- **Calculate error budget**:
  - SLO: 99.9% availability
  - Error budget: 0.1% = 43 minutes per month
  - Track budget consumption
  - Alert when budget exhausted

### 6.4 SLO-Based Alerting
- **Alert on SLO violation**:
  - Warn at 80% error budget consumed
  - Critical at 100% error budget consumed
  - Include burn rate in alerts

---

## 7. Performance Monitoring

### 7.1 Resource Utilization
- **Monitor critical resources**:
  ```
  # CPU utilization
  compute.googleapis.com/instance/cpu/utilization
  
  # Memory utilization
  compute.googleapis.com/instance/memory/utilization
  
  # Disk I/O
  compute.googleapis.com/instance/disk/read_bytes_count
  compute.googleapis.com/instance/disk/write_bytes_count
  
  # Network
  compute.googleapis.com/instance/network/received_bytes_count
  compute.googleapis.com/instance/network/sent_bytes_count
  ```

### 7.2 Query Performance
- **Monitor BigQuery performance**:
  ```python
  from google.cloud import monitoring_v3
  
  def log_query_performance(job):
      # Extract job metrics
      metrics = {
          "bytes_processed": job.total_bytes_processed,
          "bytes_billed": job.total_bytes_billed,
          "slot_millis": job.slot_millis,
          "duration_ms": (job.ended - job.started).total_seconds() * 1000
      }
      
      # Write to custom metrics
      for metric_name, value in metrics.items():
          write_custom_metric(
              project_id,
              f"bigquery/{metric_name}",
              value,
              labels={"query_type": "analytics"}
          )
  ```

### 7.3 Latency Monitoring
- **Track end-to-end latency**:
  ```python
  import time
  
  class LatencyTracker:
      def __init__(self, pipeline_name):
          self.pipeline_name = pipeline_name
          self.start_time = None
      
      def start(self):
          self.start_time = time.time()
      
      def end(self):
          if self.start_time:
              duration_ms = (time.time() - self.start_time) * 1000
              
              write_custom_metric(
                  project_id,
                  "pipeline/latency_ms",
                  duration_ms,
                  labels={"pipeline": self.pipeline_name}
              )
  
  # Usage
  tracker = LatencyTracker("customer-etl")
  tracker.start()
  # ... pipeline execution ...
  tracker.end()
  ```

---

## 8. Cost Monitoring

### 8.1 Cost Visibility
- **Track costs by service**:
  - BigQuery: bytes processed and stored
  - Dataflow: vCPU hours
  - Pub/Sub: message volume
  - Cloud Storage: storage and operations

### 8.2 Cost Metrics
- **Custom cost metrics**:
  ```python
  def calculate_pipeline_cost(job_metrics):
      # BigQuery costs
      bytes_billed = job_metrics['bytes_billed']
      bq_cost = (bytes_billed / (1024**4)) * 5  # $5 per TB
      
      # Dataflow costs
      vcpu_hours = job_metrics['vcpu_hours']
      dataflow_cost = vcpu_hours * 0.056  # $0.056 per vCPU hour
      
      # Total cost
      total_cost = bq_cost + dataflow_cost
      
      # Write to monitoring
      write_custom_metric(
          project_id,
          "pipeline/cost_usd",
          total_cost,
          labels={"pipeline": "etl"}
      )
      
      return total_cost
  ```

### 8.3 Cost Alerts
- **Alert on cost anomalies**:
  ```hcl
  resource "google_monitoring_alert_policy" "cost_spike" {
    display_name = "Pipeline Cost Spike"
    
    conditions {
      display_name = "Daily Cost > $1000"
      
      condition_threshold {
        filter          = "metric.type=\"custom.googleapis.com/pipeline/cost_usd\""
        duration        = "3600s"
        comparison      = "COMPARISON_GT"
        threshold_value = 1000
        
        aggregations {
          alignment_period     = "3600s"
          per_series_aligner  = "ALIGN_SUM"
        }
      }
    }
  }
  ```

---

## 9. Integration with Data Pipelines

### 9.1 Dataflow Monitoring
- **Monitor Dataflow jobs**:
  ```python
  from google.cloud import monitoring_v3
  from googleapiclient.discovery import build
  
  def monitor_dataflow_job(project, job_id):
      # Get job metrics
      dataflow = build('dataflow', 'v1b3')
      job = dataflow.projects().jobs().get(
          projectId=project,
          jobId=job_id
      ).execute()
      
      # Extract metrics
      metrics = job.get('jobMetadata', {}).get('systemLag', 0)
      
      # Write to monitoring
      write_custom_metric(
          project,
          "dataflow/system_lag_seconds",
          metrics,
          labels={"job_id": job_id}
      )
  ```

### 9.2 Airflow/Composer Monitoring
- **Track DAG performance**:
  ```python
  from airflow.models import DagRun
  from airflow.utils.state import State
  
  def track_dag_metrics(dag_id, execution_date):
      dag_run = DagRun.find(
          dag_id=dag_id,
          execution_date=execution_date
      )[0]
      
      # Calculate duration
      if dag_run.end_date and dag_run.start_date:
          duration = (dag_run.end_date - dag_run.start_date).total_seconds()
          
          write_custom_metric(
              project_id,
              "airflow/dag_duration_seconds",
              duration,
              labels={
                  "dag_id": dag_id,
                  "state": dag_run.state
              }
          )
      
      # Track success/failure
      success = 1 if dag_run.state == State.SUCCESS else 0
      write_custom_metric(
          project_id,
          "airflow/dag_success",
          success,
          labels={"dag_id": dag_id}
      )
  ```

---

## 10. Troubleshooting and Debugging

### 10.1 Using Metrics Explorer
- **Explore metrics interactively**
- **Build custom queries**
- **Visualize trends**
- **Export to dashboards**

### 10.2 Correlating Metrics and Logs
- **Link metrics to logs**:
  ```python
  # Log with metric context
  logging.info(
      "High memory usage detected",
      extra={
          "metric_value": memory_usage,
          "threshold": threshold,
          "resource_id": instance_id
      }
  )
  ```

### 10.3 Debugging Performance Issues
- **Identify bottlenecks**:
  1. Check resource utilization metrics
  2. Review processing durations
  3. Analyze backlog growth
  4. Examine error rates
  5. Correlate with logs

---

## Quick Reference Checklist

### Monitoring Setup
- [ ] Define key metrics to track
- [ ] Create custom metrics for business logic
- [ ] Set up service-specific monitoring
- [ ] Configure dashboards for different audiences
- [ ] Establish SLOs and error budgets
- [ ] Create alert policies
- [ ] Set up notification channels
- [ ] Document monitoring strategy
- [ ] Test alerts and notifications
- [ ] Review and optimize regularly

### Essential Dashboards
- [ ] Pipeline health dashboard
- [ ] Performance dashboard
- [ ] Data quality dashboard
- [ ] Cost optimization dashboard
- [ ] SLO compliance dashboard

### Critical Alerts
- [ ] Pipeline failures
- [ ] SLA violations
- [ ] Resource saturation
- [ ] Data freshness violations
- [ ] Cost anomalies
- [ ] Error rate spikes

---

## Resources

### Official Documentation
- [Cloud Monitoring Documentation](https://cloud.google.com/monitoring/docs)
- [Metrics List](https://cloud.google.com/monitoring/api/metrics_gcp)
- [MQL Reference](https://cloud.google.com/monitoring/mql)

### Tools
- Cloud Monitoring Console
- Metrics Explorer
- gcloud CLI
- Terraform Provider

---

*Last Updated: December 26, 2025*
*Version: 1.0*
