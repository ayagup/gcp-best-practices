# Cloud Logging Best Practices for Data Engineering

## Overview
Cloud Logging is a fully managed service that allows you to store, search, analyze, monitor, and alert on logging data and events from Google Cloud and AWS. For data engineers, it's essential for debugging pipelines, monitoring data quality, and maintaining observability. This document covers best practices for effective logging in data engineering workflows.

---

## 1. Logging Strategy

### 1.1 Define Logging Objectives
- **Operational monitoring**: Track pipeline health and performance
- **Debugging**: Troubleshoot issues quickly
- **Audit and compliance**: Maintain audit trails
- **Data quality**: Log validation results
- **Cost optimization**: Track resource usage

### 1.2 Log Levels
- **Use appropriate severity levels**:
  - `DEBUG`: Detailed debugging information
  - `INFO`: General informational messages
  - `WARNING`: Potentially harmful situations
  - `ERROR`: Error events that might still allow continued execution
  - `CRITICAL`: Severe errors causing termination

- **Production guidelines**:
  - Use INFO for normal operations
  - Use WARNING for degraded performance
  - Use ERROR for failures
  - Avoid DEBUG in production (use selectively)

### 1.3 What to Log
- **Essential log data**:
  - Pipeline start/end times
  - Record counts processed
  - Data quality check results
  - Error messages and stack traces
  - External API calls and responses
  - Configuration changes
  - Performance metrics

- **What NOT to log**:
  - Sensitive data (PII, credentials, tokens)
  - Large data payloads
  - Redundant information
  - High-frequency debug messages in production

---

## 2. Structured Logging

### 2.1 JSON Structured Logs
- **Use structured logging format**:
  ```python
  import json
  import logging
  from datetime import datetime
  
  def log_structured(severity, message, **kwargs):
      log_entry = {
          "severity": severity,
          "message": message,
          "timestamp": datetime.utcnow().isoformat(),
          "pipeline": "customer-etl",
          "version": "1.2.3",
          **kwargs
      }
      print(json.dumps(log_entry))
  
  # Usage
  log_structured("INFO", "Pipeline started", 
                 run_id="12345", 
                 source="bigquery")
  ```

### 2.2 Standard Fields
- **Include standard fields**:
  - `timestamp`: ISO 8601 format
  - `severity`: Log level
  - `message`: Human-readable message
  - `pipeline_name`: Name of the pipeline
  - `run_id`: Unique execution identifier
  - `component`: Specific component or stage
  - `user_id`: User who triggered (if applicable)

### 2.3 Context Fields
- **Add contextual information**:
  ```python
  log_entry = {
      "severity": "INFO",
      "message": "Processing batch completed",
      
      # Execution context
      "run_id": execution_id,
      "pipeline": "order-processing",
      "stage": "transformation",
      
      # Metrics
      "records_processed": 15000,
      "duration_seconds": 45.2,
      "bytes_processed": 2048576,
      
      # Resource info
      "project_id": "my-project",
      "dataset": "orders",
      "table": "daily_orders",
      
      # Error context (if applicable)
      "error_code": None,
      "error_details": None
  }
  ```

---

## 3. Logging in Data Pipelines

### 3.1 Dataflow Logging
- **Dataflow-specific logging**:
  ```python
  import apache_beam as beam
  import logging
  
  class ProcessWithLogging(beam.DoFn):
      def __init__(self):
          self.records_processed = beam.metrics.Metrics.counter(
              'pipeline', 'records_processed'
          )
      
      def process(self, element):
          try:
              # Process element
              result = transform_data(element)
              
              # Increment counter
              self.records_processed.inc()
              
              # Log periodically (every 1000 records)
              if self.records_processed.value % 1000 == 0:
                  logging.info(f"Processed {self.records_processed.value} records")
              
              yield result
              
          except Exception as e:
              logging.error(f"Error processing record: {element}",
                          exc_info=True,
                          extra={'element_id': element.get('id')})
              # Optionally yield to dead letter queue
  ```

### 3.2 Cloud Functions Logging
- **Structured logging in functions**:
  ```python
  import json
  from datetime import datetime
  
  def cloud_function_handler(event, context):
      # Log function invocation
      log_entry = {
          "severity": "INFO",
          "message": "Function invoked",
          "function_name": context.function_name,
          "event_id": context.event_id,
          "event_type": context.event_type,
          "timestamp": datetime.utcnow().isoformat()
      }
      print(json.dumps(log_entry))
      
      try:
          # Process data
          result = process_data(event)
          
          # Log success
          log_entry["message"] = "Processing completed"
          log_entry["records_processed"] = len(result)
          print(json.dumps(log_entry))
          
      except Exception as e:
          # Log error
          log_entry["severity"] = "ERROR"
          log_entry["message"] = f"Processing failed: {str(e)}"
          log_entry["error_type"] = type(e).__name__
          print(json.dumps(log_entry))
          raise
  ```

### 3.3 BigQuery Job Logging
- **Log BigQuery operations**:
  ```python
  from google.cloud import bigquery
  import logging
  
  client = bigquery.Client()
  
  def run_query_with_logging(query, job_config=None):
      job_id = f"job_{int(time.time())}"
      
      logging.info("Starting BigQuery job",
                  extra={
                      "job_id": job_id,
                      "query_length": len(query),
                      "destination": job_config.destination if job_config else None
                  })
      
      try:
          query_job = client.query(query, job_config=job_config, job_id=job_id)
          result = query_job.result()
          
          # Log success with metrics
          logging.info("BigQuery job completed",
                      extra={
                          "job_id": job_id,
                          "bytes_processed": query_job.total_bytes_processed,
                          "bytes_billed": query_job.total_bytes_billed,
                          "slot_millis": query_job.slot_millis,
                          "rows": query_job.result().total_rows,
                          "duration_seconds": query_job.ended - query_job.started
                      })
          
          return result
          
      except Exception as e:
          logging.error("BigQuery job failed",
                       extra={
                           "job_id": job_id,
                           "error": str(e),
                           "error_type": type(e).__name__
                       })
          raise
  ```

### 3.4 Airflow/Composer Logging
- **Logging in Airflow DAGs**:
  ```python
  from airflow import DAG
  from airflow.operators.python import PythonOperator
  import logging
  
  def process_data(**context):
      logger = logging.getLogger(__name__)
      
      # Access Airflow context
      dag_run = context['dag_run']
      task_instance = context['task_instance']
      
      logger.info("Task started",
                 extra={
                     "dag_id": dag_run.dag_id,
                     "run_id": dag_run.run_id,
                     "task_id": task_instance.task_id,
                     "execution_date": str(context['execution_date'])
                 })
      
      try:
          # Process data
          result = perform_processing()
          
          logger.info("Task completed",
                     extra={
                         "records_processed": result.count,
                         "status": "success"
                     })
          
          # Push to XCom for downstream tasks
          return result.count
          
      except Exception as e:
          logger.error("Task failed",
                      extra={
                          "error": str(e),
                          "error_type": type(e).__name__
                      },
                      exc_info=True)
          raise
  ```

---

## 4. Log Export and Analysis

### 4.1 Export to BigQuery
- **Set up log sinks for analysis**:
  ```bash
  # Create BigQuery dataset
  bq mk --dataset \
    --location=US \
    my_project:logs
  
  # Create log sink to BigQuery
  gcloud logging sinks create dataflow-logs-sink \
    bigquery.googleapis.com/projects/MY_PROJECT/datasets/logs \
    --log-filter='resource.type="dataflow_step"'
  ```

- **Best practices for BigQuery exports**:
  - Use partitioned tables by date
  - Set appropriate retention policies
  - Create materialized views for common queries
  - Index frequently filtered fields

### 4.2 Analyzing Logs in BigQuery
- **Query patterns for analysis**:
  ```sql
  -- Error analysis
  SELECT
      timestamp,
      severity,
      jsonPayload.message AS message,
      jsonPayload.pipeline AS pipeline,
      jsonPayload.error_type AS error_type,
      COUNT(*) AS error_count
  FROM `project.logs.dataflow_logs`
  WHERE severity = 'ERROR'
    AND DATE(timestamp) = CURRENT_DATE()
  GROUP BY 1, 2, 3, 4, 5
  ORDER BY error_count DESC
  
  ---
  
  -- Performance analysis
  SELECT
      jsonPayload.pipeline AS pipeline,
      AVG(CAST(jsonPayload.duration_seconds AS FLOAT64)) AS avg_duration,
      MAX(CAST(jsonPayload.duration_seconds AS FLOAT64)) AS max_duration,
      COUNT(*) AS execution_count
  FROM `project.logs.pipeline_logs`
  WHERE jsonPayload.message = 'Pipeline completed'
    AND DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
  GROUP BY pipeline
  ORDER BY avg_duration DESC
  
  ---
  
  -- Data quality metrics
  SELECT
      DATE(timestamp) AS date,
      jsonPayload.pipeline AS pipeline,
      SUM(CAST(jsonPayload.records_processed AS INT64)) AS total_records,
      SUM(CAST(jsonPayload.records_failed AS INT64)) AS failed_records,
      SAFE_DIVIDE(
          SUM(CAST(jsonPayload.records_failed AS INT64)),
          SUM(CAST(jsonPayload.records_processed AS INT64))
      ) * 100 AS failure_rate_pct
  FROM `project.logs.pipeline_logs`
  WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  GROUP BY date, pipeline
  ORDER BY date DESC, failure_rate_pct DESC
  ```

### 4.3 Export to Cloud Storage
- **Archive logs to Cloud Storage**:
  ```bash
  gcloud logging sinks create archive-logs-sink \
    storage.googleapis.com/my-logs-archive \
    --log-filter='resource.type="dataflow_step" AND severity>=WARNING'
  ```

- **Use cases for Cloud Storage export**:
  - Long-term archival
  - Compliance requirements
  - Cost optimization (cheaper than BigQuery for old logs)
  - External analysis tools

### 4.4 Real-time Log Processing
- **Stream logs to Pub/Sub**:
  ```bash
  gcloud logging sinks create pubsub-logs-sink \
    pubsub.googleapis.com/projects/MY_PROJECT/topics/logs-topic \
    --log-filter='severity>=ERROR'
  ```

- **Process logs in real-time**:
  ```python
  from google.cloud import pubsub_v1
  import json
  
  def callback(message):
      log_entry = json.loads(message.data)
      
      # Check for critical errors
      if log_entry.get('severity') == 'CRITICAL':
          send_alert(log_entry)
      
      # Process log
      process_log_entry(log_entry)
      
      message.ack()
  
  subscriber = pubsub_v1.SubscriberClient()
  subscription_path = subscriber.subscription_path(project, subscription)
  
  streaming_pull_future = subscriber.subscribe(
      subscription_path, callback=callback
  )
  ```

---

## 5. Log Filtering and Queries

### 5.1 Effective Log Filters
- **Filter by resource type**:
  ```
  resource.type="dataflow_step"
  resource.type="cloud_function"
  resource.type="bigquery_resource"
  resource.type="k8s_pod"
  ```

- **Filter by severity**:
  ```
  severity>=ERROR
  severity="WARNING" OR severity="ERROR"
  ```

- **Filter by time**:
  ```
  timestamp>="2024-12-26T00:00:00Z"
  timestamp>="2024-12-26T00:00:00Z" AND timestamp<"2024-12-27T00:00:00Z"
  ```

- **Complex filters**:
  ```
  resource.type="dataflow_step"
  AND severity>=ERROR
  AND jsonPayload.pipeline="customer-etl"
  AND timestamp>="2024-12-26T00:00:00Z"
  ```

### 5.2 Text Search
- **Search in log messages**:
  ```
  "OutOfMemoryError"
  "Pipeline failed" OR "Task failed"
  jsonPayload.error_type="ValueError"
  ```

### 5.3 Saved Queries
- **Create reusable queries**:
  - Save frequently used filters
  - Share with team members
  - Use in dashboards and alerts

---

## 6. Monitoring and Alerting

### 6.1 Log-Based Metrics
- **Create metrics from logs**:
  ```bash
  gcloud logging metrics create pipeline_errors \
    --description="Count of pipeline errors" \
    --log-filter='resource.type="dataflow_step" AND severity="ERROR"'
  ```

- **Use cases for log-based metrics**:
  - Error rate tracking
  - Performance metrics
  - Custom business metrics
  - SLA monitoring

### 6.2 Alert Policies
- **Create alerts from log patterns**:
  ```bash
  # Create alert for high error rate
  gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="High Pipeline Error Rate" \
    --condition-display-name="Error rate > 10/min" \
    --condition-threshold-value=10 \
    --condition-threshold-duration=300s
  ```

### 6.3 Alert Best Practices
- **Effective alerting**:
  - Alert on actionable issues only
  - Set appropriate thresholds
  - Include context in alerts
  - Avoid alert fatigue
  - Test alert channels

- **Alert examples**:
  - Pipeline failures
  - High error rates (> threshold)
  - Data quality issues
  - Performance degradation
  - Resource exhaustion

---

## 7. Performance and Cost Optimization

### 7.1 Log Volume Management
- **Reduce log volume**:
  - Use appropriate log levels
  - Sample high-frequency logs
  - Aggregate similar log entries
  - Avoid logging large payloads

- **Sampling pattern**:
  ```python
  import random
  
  def should_log_sample(sample_rate=0.01):
      return random.random() < sample_rate
  
  # Log only 1% of high-frequency events
  if should_log_sample():
      logging.debug("High frequency event", extra={"event": data})
  ```

### 7.2 Log Retention
- **Configure retention policies**:
  ```bash
  # Set bucket retention
  gcloud logging buckets update _Default \
    --location=global \
    --retention-days=30
  ```

- **Retention guidelines**:
  - DEBUG logs: 7 days
  - INFO logs: 30 days
  - WARNING/ERROR logs: 90 days
  - CRITICAL/audit logs: 365+ days or per compliance requirements

### 7.3 Cost Optimization
- **Optimize logging costs**:
  - Use log exclusions for non-essential logs
  - Export to cheaper storage (Cloud Storage)
  - Set appropriate retention periods
  - Use log sampling for high-volume sources

- **Exclude logs**:
  ```bash
  gcloud logging exclusions create exclude-health-checks \
    --log-filter='httpRequest.requestUrl=~"/health"'
  ```

---

## 8. Security and Compliance

### 8.1 Sensitive Data Protection
- **Never log sensitive data**:
  ```python
  # Bad - logging sensitive data
  logging.info(f"Processing payment for card {card_number}")
  
  # Good - redact sensitive data
  logging.info(f"Processing payment for card ****{card_number[-4:]}")
  ```

- **Use Cloud DLP for scanning**:
  - Scan logs for PII
  - Redact sensitive information
  - Set up DLP inspection rules

### 8.2 Access Control
- **Restrict log access**:
  ```bash
  # Grant log viewer role
  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:analyst@example.com" \
    --role="roles/logging.viewer"
  
  # Grant log writer role to service account
  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:pipeline-sa@project.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"
  ```

### 8.3 Audit Logging
- **Enable audit logs**:
  - Admin Activity logs (always enabled)
  - Data Access logs (must enable)
  - System Event logs (always enabled)

- **Monitor audit logs**:
  ```
  protoPayload.methodName="google.cloud.bigquery.v2.JobService.InsertJob"
  protoPayload.authenticationInfo.principalEmail="user@example.com"
  ```

---

## 9. Debugging and Troubleshooting

### 9.1 Correlation IDs
- **Use correlation IDs for tracing**:
  ```python
  import uuid
  
  def process_with_correlation_id(data):
      correlation_id = str(uuid.uuid4())
      
      logging.info("Started processing",
                  extra={"correlation_id": correlation_id})
      
      try:
          result = process_step_1(data, correlation_id)
          result = process_step_2(result, correlation_id)
          result = process_step_3(result, correlation_id)
          
          logging.info("Completed processing",
                      extra={"correlation_id": correlation_id})
          
          return result
          
      except Exception as e:
          logging.error("Processing failed",
                       extra={
                           "correlation_id": correlation_id,
                           "error": str(e)
                       })
          raise
  
  def process_step_1(data, correlation_id):
      logging.info("Step 1",
                  extra={"correlation_id": correlation_id,
                         "step": "transform"})
      # Processing logic
      return transformed_data
  ```

### 9.2 Error Context
- **Include detailed error context**:
  ```python
  try:
      result = process_record(record)
  except Exception as e:
      logging.error(
          "Failed to process record",
          extra={
              "record_id": record.get('id'),
              "record_type": record.get('type'),
              "error_type": type(e).__name__,
              "error_message": str(e),
              "stack_trace": traceback.format_exc(),
              "input_size": len(str(record)),
              "timestamp": datetime.utcnow().isoformat()
          },
          exc_info=True
      )
  ```

### 9.3 Performance Debugging
- **Log performance metrics**:
  ```python
  import time
  
  def log_performance(func):
      def wrapper(*args, **kwargs):
          start_time = time.time()
          
          try:
              result = func(*args, **kwargs)
              duration = time.time() - start_time
              
              logging.info(
                  f"Function {func.__name__} completed",
                  extra={
                      "function": func.__name__,
                      "duration_seconds": duration,
                      "status": "success"
                  }
              )
              
              return result
              
          except Exception as e:
              duration = time.time() - start_time
              
              logging.error(
                  f"Function {func.__name__} failed",
                  extra={
                      "function": func.__name__,
                      "duration_seconds": duration,
                      "status": "error",
                      "error": str(e)
                  }
              )
              raise
      
      return wrapper
  ```

---

## 10. Integration with Other Services

### 10.1 Cloud Monitoring Integration
- **Create dashboards from logs**
- **Set up log-based metrics**
- **Configure alerting policies**

### 10.2 Error Reporting Integration
- **Automatic error grouping**
- **Error tracking and trends**
- **Integration with Cloud Logging**

### 10.3 Cloud Trace Integration
- **Distributed tracing**
- **Log correlation with traces**
- **Performance analysis**

---

## Quick Reference Checklist

### Logging Implementation
- [ ] Use structured (JSON) logging
- [ ] Include correlation IDs
- [ ] Log at appropriate severity levels
- [ ] Include contextual information
- [ ] Never log sensitive data
- [ ] Use consistent field names
- [ ] Add error context and stack traces
- [ ] Log pipeline start/end
- [ ] Log record counts and metrics
- [ ] Implement sampling for high-volume logs

### Log Management
- [ ] Set up log sinks for important logs
- [ ] Export to BigQuery for analysis
- [ ] Configure retention policies
- [ ] Set up log-based metrics
- [ ] Create alert policies
- [ ] Implement log exclusions
- [ ] Control access with IAM
- [ ] Monitor logging costs
- [ ] Create dashboards
- [ ] Document logging strategy

### Troubleshooting
- [ ] Use log filters effectively
- [ ] Search with correlation IDs
- [ ] Analyze error patterns
- [ ] Check log exports
- [ ] Review alert policies
- [ ] Validate IAM permissions
- [ ] Check log retention settings
- [ ] Review cost and volume

---

## Resources

### Official Documentation
- [Cloud Logging Documentation](https://cloud.google.com/logging/docs)
- [Log Query Language](https://cloud.google.com/logging/docs/view/logging-query-language)
- [Logging Best Practices](https://cloud.google.com/logging/docs/best-practices)

### Tools
- Cloud Logging Console
- Logs Explorer
- gcloud CLI
- Cloud Logging API

---

*Last Updated: December 26, 2025*
*Version: 1.0*
