# Cloud Run Best Practices for Data Engineering

## Overview
Cloud Run is a fully managed serverless platform that runs containerized applications. For data engineering, it's ideal for custom data processing logic, APIs, long-running tasks, and services that require specific dependencies or runtimes. This document covers best practices for using Cloud Run in data pipelines.

---

## 1. Container Design Principles

### 1.1 Container Image Optimization
- **Keep images small**:
  - Use minimal base images (Alpine, distroless)
  - Multi-stage builds to reduce final size
  - Remove build tools and dev dependencies
  - Use .dockerignore to exclude unnecessary files

- **Example Dockerfile**:
  ```dockerfile
  # Build stage
  FROM python:3.11-slim AS builder
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --user --no-cache-dir -r requirements.txt
  
  # Runtime stage
  FROM python:3.11-slim
  WORKDIR /app
  COPY --from=builder /root/.local /root/.local
  COPY . .
  
  # Make sure scripts are in PATH
  ENV PATH=/root/.local/bin:$PATH
  
  CMD ["python", "main.py"]
  ```

### 1.2 Stateless Application Design
- **Design for ephemeral containers**:
  - No local file system persistence
  - Use Cloud Storage for file operations
  - Use Memorystore or Firestore for state
  - Assume container can be killed at any time

- **Handle graceful shutdown**:
  ```python
  import signal
  import sys
  
  def graceful_shutdown(signum, frame):
      print("Received shutdown signal, cleaning up...")
      # Close connections, finish processing
      cleanup_resources()
      sys.exit(0)
  
  signal.signal(signal.SIGTERM, graceful_shutdown)
  ```

### 1.3 Environment Configuration
- **Externalize configuration**:
  - Use environment variables
  - Store secrets in Secret Manager
  - Use Cloud Storage for config files
  - Never hardcode credentials or endpoints

- **Example configuration**:
  ```python
  import os
  
  class Config:
      PROJECT_ID = os.getenv('PROJECT_ID')
      DATASET_ID = os.getenv('DATASET_ID')
      BUCKET_NAME = os.getenv('BUCKET_NAME')
      MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
      
      @staticmethod
      def validate():
          required = ['PROJECT_ID', 'DATASET_ID', 'BUCKET_NAME']
          missing = [var for var in required if not os.getenv(var)]
          if missing:
              raise ValueError(f"Missing required env vars: {missing}")
  ```

---

## 2. Service Configuration

### 2.1 CPU and Memory Allocation
- **Right-size resources**:
  - **1 CPU, 512MB**: Light API services, simple transformations
  - **2 CPU, 2GB**: Medium workloads, data validation, enrichment
  - **4 CPU, 8GB**: Heavy processing, complex transformations
  - **8 CPU, 32GB**: Intensive workloads, in-memory analytics

- **Configure in deployment**:
  ```bash
  gcloud run deploy data-processor \
    --image gcr.io/project/image \
    --cpu 2 \
    --memory 2Gi \
    --region us-central1
  ```

### 2.2 Concurrency Settings
- **Configure request concurrency**:
  - Default: 80 concurrent requests per container
  - CPU-bound: Lower concurrency (10-30)
  - I/O-bound: Higher concurrency (50-100)
  - Memory-intensive: Lower to prevent OOM

- **Example configuration**:
  ```bash
  gcloud run deploy my-service \
    --concurrency 50 \
    --cpu 2 \
    --memory 2Gi
  ```

### 2.3 Timeout Configuration
- **Set appropriate timeouts**:
  - Default: 300 seconds (5 minutes)
  - Maximum: 3600 seconds (60 minutes)
  - Set based on expected processing time
  - Add buffer for variability

- **Configure timeout**:
  ```bash
  gcloud run deploy my-service \
    --timeout 1800  # 30 minutes
  ```

### 2.4 Minimum and Maximum Instances
- **Control scaling**:
  ```bash
  gcloud run deploy my-service \
    --min-instances 1 \     # Keep warm for low latency
    --max-instances 100 \   # Limit for cost control
    --cpu 2
  ```

- **When to use min-instances**:
  - Reduce cold start impact
  - Maintain consistent latency
  - Handle traffic spikes quickly
  - Cost: Pay for idle instances

---

## 3. Request Handling

### 3.1 HTTP Server Implementation
- **Use production-ready servers**:
  ```python
  # Using Gunicorn with Flask
  from flask import Flask, request, jsonify
  
  app = Flask(__name__)
  
  @app.route('/process', methods=['POST'])
  def process_data():
      try:
          data = request.get_json()
          result = transform_data(data)
          return jsonify({'status': 'success', 'result': result})
      except Exception as e:
          return jsonify({'status': 'error', 'message': str(e)}), 500
  
  if __name__ == '__main__':
      import os
      port = int(os.getenv('PORT', 8080))
      app.run(host='0.0.0.0', port=port)
  ```

- **Deployment with Gunicorn**:
  ```dockerfile
  CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
  ```

### 3.2 Request Validation
- **Validate all inputs**:
  ```python
  from pydantic import BaseModel, ValidationError
  
  class ProcessRequest(BaseModel):
      data_source: str
      target_table: str
      options: dict = {}
  
  @app.route('/process', methods=['POST'])
  def process():
      try:
          request_data = ProcessRequest(**request.get_json())
          # Process validated data
          result = process_data(request_data)
          return jsonify(result)
      except ValidationError as e:
          return jsonify({'error': 'Invalid request', 'details': e.errors()}), 400
  ```

### 3.3 Authentication and Authorization
- **Implement authentication**:
  ```python
  from google.auth.transport import requests as google_requests
  from google.oauth2 import id_token
  
  def verify_token(token):
      try:
          request = google_requests.Request()
          id_info = id_token.verify_oauth2_token(token, request)
          return id_info
      except ValueError:
          return None
  
  @app.route('/process', methods=['POST'])
  def process():
      auth_header = request.headers.get('Authorization')
      if not auth_header:
          return jsonify({'error': 'No authorization header'}), 401
      
      token = auth_header.split(' ')[1]
      user_info = verify_token(token)
      
      if not user_info:
          return jsonify({'error': 'Invalid token'}), 401
      
      # Process request
      return process_data(request.get_json())
  ```

### 3.4 Async Processing
- **Use async for I/O-bound tasks**:
  ```python
  from fastapi import FastAPI
  from google.cloud import bigquery
  import asyncio
  
  app = FastAPI()
  
  @app.post("/process")
  async def process_data(request: ProcessRequest):
      # Process multiple sources concurrently
      results = await asyncio.gather(
          fetch_from_source1(request.source1),
          fetch_from_source2(request.source2),
          fetch_from_source3(request.source3)
      )
      
      # Combine and transform
      combined = combine_results(results)
      
      # Write to BigQuery
      await write_to_bigquery(combined)
      
      return {"status": "success"}
  ```

---

## 4. Data Processing Patterns

### 4.1 Batch Processing
- **Process files from Cloud Storage**:
  ```python
  from google.cloud import storage, bigquery
  
  def process_batch_file(bucket_name, file_name):
      # Download file
      storage_client = storage.Client()
      bucket = storage_client.bucket(bucket_name)
      blob = bucket.blob(file_name)
      
      # Stream process large files
      bq_client = bigquery.Client()
      rows_to_insert = []
      
      with blob.open('r') as f:
          for line in f:
              row = transform_line(line)
              rows_to_insert.append(row)
              
              # Batch insert every 1000 rows
              if len(rows_to_insert) >= 1000:
                  insert_to_bigquery(bq_client, rows_to_insert)
                  rows_to_insert = []
      
      # Insert remaining rows
      if rows_to_insert:
          insert_to_bigquery(bq_client, rows_to_insert)
  ```

### 4.2 Stream Processing
- **Handle streaming data**:
  ```python
  @app.post("/ingest")
  async def ingest_stream_data(data: StreamData):
      # Validate data
      if not validate_schema(data):
          return {"status": "error", "message": "Invalid schema"}
      
      # Transform
      transformed = transform_data(data)
      
      # Enrich
      enriched = await enrich_data(transformed)
      
      # Write to BigQuery streaming
      await stream_to_bigquery(enriched)
      
      # Publish to Pub/Sub for downstream
      await publish_to_pubsub(enriched)
      
      return {"status": "success"}
  ```

### 4.3 ETL Workflows
- **Implement ETL endpoints**:
  ```python
  @app.post("/etl/extract")
  def extract():
      # Extract from source
      data = extract_from_source()
      # Store in Cloud Storage
      store_raw_data(data)
      return {"status": "extracted", "records": len(data)}
  
  @app.post("/etl/transform")
  def transform():
      # Load raw data
      raw_data = load_raw_data()
      # Transform
      transformed = transform_data(raw_data)
      # Store transformed
      store_transformed_data(transformed)
      return {"status": "transformed", "records": len(transformed)}
  
  @app.post("/etl/load")
  def load():
      # Load transformed data
      data = load_transformed_data()
      # Load to BigQuery
      load_to_bigquery(data)
      return {"status": "loaded", "records": len(data)}
  ```

### 4.4 Data Validation Service
- **Create validation API**:
  ```python
  @app.post("/validate")
  def validate_data(request: ValidationRequest):
      results = {
          'valid': True,
          'errors': [],
          'warnings': []
      }
      
      # Schema validation
      if not validate_schema(request.data):
          results['valid'] = False
          results['errors'].append("Schema validation failed")
      
      # Business rules validation
      rule_results = apply_business_rules(request.data)
      results['errors'].extend(rule_results.errors)
      results['warnings'].extend(rule_results.warnings)
      
      # Data quality checks
      quality_results = check_data_quality(request.data)
      results['warnings'].extend(quality_results.warnings)
      
      return jsonify(results)
  ```

---

## 5. Integration with GCP Services

### 5.1 BigQuery Integration
- **Efficient BigQuery operations**:
  ```python
  from google.cloud import bigquery
  
  # Initialize client once
  bq_client = bigquery.Client()
  
  def query_bigquery(sql):
      query_job = bq_client.query(sql)
      results = query_job.result()
      return [dict(row) for row in results]
  
  def stream_to_bigquery(table_id, rows):
      errors = bq_client.insert_rows_json(table_id, rows)
      if errors:
          raise Exception(f"BigQuery insert errors: {errors}")
  
  def load_from_gcs_to_bigquery(uri, table_id):
      job_config = bigquery.LoadJobConfig(
          source_format=bigquery.SourceFormat.PARQUET,
          write_disposition=bigquery.WriteDisposition.WRITE_APPEND
      )
      
      load_job = bq_client.load_table_from_uri(
          uri, table_id, job_config=job_config
      )
      
      load_job.result()  # Wait for job to complete
  ```

### 5.2 Cloud Storage Integration
- **Efficient file operations**:
  ```python
  from google.cloud import storage
  import io
  
  storage_client = storage.Client()
  
  def upload_to_gcs(bucket_name, blob_name, data):
      bucket = storage_client.bucket(bucket_name)
      blob = bucket.blob(blob_name)
      
      # Upload from memory
      blob.upload_from_string(data, content_type='application/json')
  
  def download_from_gcs(bucket_name, blob_name):
      bucket = storage_client.bucket(bucket_name)
      blob = bucket.blob(blob_name)
      
      # Download to memory
      return blob.download_as_text()
  
  def stream_large_file(bucket_name, blob_name):
      bucket = storage_client.bucket(bucket_name)
      blob = bucket.blob(blob_name)
      
      # Stream download
      with blob.open('r') as f:
          for line in f:
              yield line
  ```

### 5.3 Pub/Sub Integration
- **Publish and subscribe**:
  ```python
  from google.cloud import pubsub_v1
  
  publisher = pubsub_v1.PublisherClient()
  
  def publish_message(topic_path, data, **attributes):
      import json
      
      message = json.dumps(data).encode('utf-8')
      future = publisher.publish(topic_path, message, **attributes)
      
      return future.result()  # Get message ID
  
  @app.post("/pubsub/push")
  def receive_pubsub():
      envelope = request.get_json()
      
      # Verify the request
      if not envelope or 'message' not in envelope:
          return ('Bad Request', 400)
      
      # Decode message
      message = envelope['message']
      data = base64.b64decode(message['data']).decode('utf-8')
      
      # Process message
      process_message(data)
      
      return ('OK', 200)
  ```

### 5.4 Firestore Integration
- **Use Firestore for state management**:
  ```python
  from google.cloud import firestore
  
  db = firestore.Client()
  
  def save_processing_state(job_id, state):
      doc_ref = db.collection('jobs').document(job_id)
      doc_ref.set({
          'state': state,
          'updated_at': firestore.SERVER_TIMESTAMP
      })
  
  def get_processing_state(job_id):
      doc_ref = db.collection('jobs').document(job_id)
      doc = doc_ref.get()
      
      if doc.exists:
          return doc.to_dict()
      return None
  ```

---

## 6. Performance Optimization

### 6.1 Container Startup Optimization
- **Reduce cold start time**:
  - Minimize image size
  - Use startup CPU boost
  - Initialize connections lazily
  - Preload critical dependencies

- **Lazy initialization pattern**:
  ```python
  _bq_client = None
  
  def get_bq_client():
      global _bq_client
      if _bq_client is None:
          _bq_client = bigquery.Client()
      return _bq_client
  ```

### 6.2 Connection Pooling
- **Reuse connections efficiently**:
  ```python
  from sqlalchemy import create_engine
  from sqlalchemy.pool import NullPool
  
  # For Cloud SQL
  def get_db_engine():
      return create_engine(
          f"postgresql+pg8000://{user}:{password}@/{db}",
          pool_size=5,
          max_overflow=2,
          pool_timeout=30,
          pool_recycle=1800
      )
  ```

### 6.3 Caching Strategies
- **Implement caching for repeated data**:
  ```python
  from functools import lru_cache
  from google.cloud import memorystore
  
  # In-memory cache for small data
  @lru_cache(maxsize=1000)
  def get_reference_data(key):
      return fetch_from_source(key)
  
  # Redis cache for larger data
  import redis
  
  redis_client = redis.Redis(host='redis-host', port=6379)
  
  def get_cached_data(key):
      # Try cache first
      cached = redis_client.get(key)
      if cached:
          return json.loads(cached)
      
      # Fetch and cache
      data = fetch_from_source(key)
      redis_client.setex(key, 3600, json.dumps(data))  # 1 hour TTL
      
      return data
  ```

### 6.4 Parallel Processing
- **Process data in parallel**:
  ```python
  from concurrent.futures import ThreadPoolExecutor, as_completed
  
  def process_files_parallel(file_list):
      results = []
      
      with ThreadPoolExecutor(max_workers=10) as executor:
          futures = {
              executor.submit(process_file, file): file 
              for file in file_list
          }
          
          for future in as_completed(futures):
              file = futures[future]
              try:
                  result = future.result()
                  results.append(result)
              except Exception as e:
                  print(f"Error processing {file}: {e}")
      
      return results
  ```

---

## 7. Monitoring and Observability

### 7.1 Structured Logging
- **Implement structured logging**:
  ```python
  import logging
  import json
  import sys
  
  class StructuredLogger:
      def __init__(self):
          self.logger = logging.getLogger(__name__)
          handler = logging.StreamHandler(sys.stdout)
          self.logger.addHandler(handler)
          self.logger.setLevel(logging.INFO)
      
      def log(self, severity, message, **kwargs):
          log_entry = {
              'severity': severity,
              'message': message,
              'timestamp': datetime.utcnow().isoformat(),
              **kwargs
          }
          self.logger.info(json.dumps(log_entry))
  
  logger = StructuredLogger()
  
  @app.route('/process')
  def process():
      logger.log('INFO', 'Processing started', request_id=request.headers.get('X-Request-ID'))
      # Processing logic
      logger.log('INFO', 'Processing completed', records_processed=count)
  ```

### 7.2 Custom Metrics
- **Export custom metrics**:
  ```python
  from google.cloud import monitoring_v3
  import time
  
  client = monitoring_v3.MetricServiceClient()
  project_name = f"projects/{project_id}"
  
  def write_metric(metric_type, value, labels=None):
      series = monitoring_v3.TimeSeries()
      series.metric.type = f"custom.googleapis.com/{metric_type}"
      
      if labels:
          for key, val in labels.items():
              series.metric.labels[key] = val
      
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
      client.create_time_series(name=project_name, time_series=[series])
  ```

### 7.3 Health Checks
- **Implement health endpoints**:
  ```python
  @app.route('/health')
  def health():
      health_status = {
          'status': 'healthy',
          'checks': {}
      }
      
      # Check BigQuery connection
      try:
          bq_client.query("SELECT 1").result()
          health_status['checks']['bigquery'] = 'ok'
      except Exception as e:
          health_status['checks']['bigquery'] = 'error'
          health_status['status'] = 'unhealthy'
      
      # Check Cloud Storage
      try:
          storage_client.list_buckets(max_results=1)
          health_status['checks']['storage'] = 'ok'
      except Exception as e:
          health_status['checks']['storage'] = 'error'
          health_status['status'] = 'unhealthy'
      
      status_code = 200 if health_status['status'] == 'healthy' else 503
      return jsonify(health_status), status_code
  ```

### 7.4 Distributed Tracing
- **Enable Cloud Trace**:
  ```python
  from opencensus.ext.flask.flask_middleware import FlaskMiddleware
  from opencensus.ext.stackdriver.trace_exporter import StackdriverExporter
  from opencensus.trace.samplers import AlwaysOnSampler
  
  app = Flask(__name__)
  
  middleware = FlaskMiddleware(
      app,
      exporter=StackdriverExporter(),
      sampler=AlwaysOnSampler()
  )
  
  # Traces are automatically captured
  ```

---

## 8. Security Best Practices

### 8.1 Service Account Configuration
- **Use dedicated service accounts**:
  ```bash
  # Create service account
  gcloud iam service-accounts create data-processor-sa \
    --display-name="Data Processor Service Account"
  
  # Grant minimal permissions
  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:data-processor-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"
  
  # Deploy with service account
  gcloud run deploy data-processor \
    --service-account=data-processor-sa@PROJECT_ID.iam.gserviceaccount.com
  ```

### 8.2 Secret Management
- **Use Secret Manager**:
  ```python
  from google.cloud import secretmanager
  
  def access_secret(secret_id, version_id="latest"):
      client = secretmanager.SecretManagerServiceClient()
      name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
      response = client.access_secret_version(request={"name": name})
      return response.payload.data.decode('UTF-8')
  
  # Use in application
  api_key = access_secret('api-key')
  db_password = access_secret('db-password')
  ```

### 8.3 Network Security
- **Configure VPC connector**:
  ```bash
  gcloud run deploy my-service \
    --vpc-connector=projects/PROJECT/locations/REGION/connectors/CONNECTOR \
    --vpc-egress=private-ranges-only
  ```

- **Restrict ingress**:
  ```bash
  gcloud run deploy my-service \
    --ingress=internal  # Only accessible from VPC or Cloud Load Balancer
  ```

### 8.4 Input Validation and Sanitization
- **Validate all inputs**:
  ```python
  import re
  
  def sanitize_table_name(table_name):
      # Only allow alphanumeric and underscores
      if not re.match(r'^[a-zA-Z0-9_]+$', table_name):
          raise ValueError("Invalid table name")
      return table_name
  
  def validate_sql_injection(user_input):
      dangerous_patterns = [
          r';\s*DROP',
          r';\s*DELETE',
          r';\s*INSERT',
          r'UNION\s+SELECT'
      ]
      
      for pattern in dangerous_patterns:
          if re.search(pattern, user_input, re.IGNORECASE):
              raise ValueError("Potentially malicious input detected")
  ```

---

## 9. Cost Optimization

### 9.1 Resource Right-Sizing
- **Monitor and optimize**:
  - Start with minimal resources
  - Monitor CPU and memory usage
  - Scale up only when needed
  - Use CPU throttling metric

### 9.2 Minimize Cold Starts
- **Balance cost vs. latency**:
  - Use min-instances for critical services
  - Accept cold starts for infrequent traffic
  - Consider traffic patterns

### 9.3 Request Batching
- **Batch where possible**:
  ```python
  @app.post("/batch-process")
  def batch_process(requests: List[ProcessRequest]):
      # Process multiple requests in single invocation
      results = []
      
      for req in requests:
          result = process_single(req)
          results.append(result)
      
      return {"results": results}
  ```

### 9.4 Efficient Data Transfer
- **Minimize egress costs**:
  - Keep services in same region as data
  - Use regional endpoints
  - Compress large responses
  - Stream large files instead of loading into memory

---

## 10. CI/CD and Deployment

### 10.1 Dockerfile Best Practices
- **Production-ready Dockerfile**:
  ```dockerfile
  FROM python:3.11-slim
  
  # Install dependencies first (better caching)
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  
  # Copy application code
  COPY . .
  
  # Non-root user
  RUN useradd -m -u 1000 appuser && chown -R appuser /app
  USER appuser
  
  # Health check
  HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"
  
  # Run application
  CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
  ```

### 10.2 Automated Deployment
- **GitHub Actions example**:
  ```yaml
  name: Deploy to Cloud Run
  
  on:
    push:
      branches: [main]
  
  jobs:
    deploy:
      runs-on: ubuntu-latest
      
      steps:
        - uses: actions/checkout@v2
        
        - name: Setup Cloud SDK
          uses: google-github-actions/setup-gcloud@v0
          with:
            service_account_key: ${{ secrets.GCP_SA_KEY }}
            project_id: ${{ secrets.GCP_PROJECT }}
        
        - name: Build and Push Container
          run: |
            gcloud builds submit --tag gcr.io/${{ secrets.GCP_PROJECT }}/my-service
        
        - name: Deploy to Cloud Run
          run: |
            gcloud run deploy my-service \
              --image gcr.io/${{ secrets.GCP_PROJECT }}/my-service \
              --platform managed \
              --region us-central1 \
              --allow-unauthenticated
  ```

### 10.3 Blue-Green Deployment
- **Traffic splitting**:
  ```bash
  # Deploy new version
  gcloud run deploy my-service \
    --image gcr.io/project/image:v2 \
    --no-traffic \
    --tag v2
  
  # Test new version
  curl https://v2---my-service-xxxx.run.app
  
  # Gradually shift traffic
  gcloud run services update-traffic my-service \
    --to-revisions v2=50,v1=50
  
  # Full cutover
  gcloud run services update-traffic my-service \
    --to-latest
  ```

---

## 11. Common Anti-Patterns to Avoid

- ❌ **Large container images**: Slow deployments and cold starts
- ❌ **Storing state locally**: Lost when container restarts
- ❌ **No health checks**: Can't detect unhealthy instances
- ❌ **Synchronous long-running tasks**: Use async or job queues
- ❌ **No request timeouts**: Can cause resource exhaustion
- ❌ **Hardcoded configuration**: Use environment variables
- ❌ **No error handling**: Services crash on errors
- ❌ **Single-threaded blocking**: Poor concurrency utilization
- ❌ **No monitoring**: Can't detect or debug issues
- ❌ **Over-provisioning resources**: Wastes money

---

## Quick Reference Checklist

### Before Deployment
- [ ] Container image optimized and small
- [ ] Stateless design implemented
- [ ] Configuration externalized
- [ ] Secrets in Secret Manager
- [ ] Error handling comprehensive
- [ ] Logging structured and meaningful
- [ ] Health check endpoint implemented
- [ ] Resource limits configured appropriately
- [ ] Service account with minimal permissions
- [ ] CI/CD pipeline set up
- [ ] Monitoring and alerting configured
- [ ] Load testing completed

### Regular Operations
- [ ] Monitor CPU and memory usage
- [ ] Review error logs daily
- [ ] Check cost trends weekly
- [ ] Update dependencies monthly
- [ ] Review and optimize performance
- [ ] Test disaster recovery procedures
- [ ] Validate security configurations
- [ ] Update documentation

---

## Resources

### Official Documentation
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Cloud Run Quotas](https://cloud.google.com/run/quotas)

### Tools
- Docker for containerization
- Cloud Build for CI/CD
- Terraform for IaC
- Cloud Monitoring and Logging

---

*Last Updated: December 26, 2025*
*Version: 1.0*
