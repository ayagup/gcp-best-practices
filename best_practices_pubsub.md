# Cloud Pub/Sub Best Practices

*Last Updated: December 25, 2025*

## Overview

Cloud Pub/Sub is a fully managed real-time messaging service that enables asynchronous communication between applications. It provides reliable, many-to-many, asynchronous messaging between decoupled applications.

---

## 1. Topic and Subscription Design

### Topic Organization

**Best Practices:**
- Use descriptive, hierarchical naming conventions
- Organize topics by domain, service, or event type
- Avoid creating too many topics (increases management overhead)
- Use topic labels for organization and cost tracking

```bash
# Good naming examples
projects/myproject/topics/ecommerce-order-created
projects/myproject/topics/inventory-stock-updated
projects/myproject/topics/payment-processed
```

### Subscription Patterns

**Pull Subscriptions:**
- Use when you control message consumption rate
- Batch message retrieval for efficiency
- Ideal for batch processing workloads

```python
from google.cloud import pubsub_v1

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

# Pull with batching
response = subscriber.pull(
    request={
        "subscription": subscription_path,
        "max_messages": 100,
    }
)

# Process messages
ack_ids = []
for received_message in response.received_messages:
    process_message(received_message.message)
    ack_ids.append(received_message.ack_id)

# Acknowledge in batch
subscriber.acknowledge(
    request={
        "subscription": subscription_path,
        "ack_ids": ack_ids,
    }
)
```

**Push Subscriptions:**
- Use for serverless architectures (Cloud Functions, Cloud Run)
- Automatic load balancing and scaling
- Lower latency than pull

```bash
# Create push subscription
gcloud pubsub subscriptions create my-push-sub \
    --topic=my-topic \
    --push-endpoint=https://myservice-abc123.run.app/push-handler \
    --push-auth-service-account=push-sa@myproject.iam.gserviceaccount.com
```

---

## 2. Message Publishing

### Batching Messages

**Best Practices:**
- Enable batching for higher throughput
- Configure appropriate batch size and delay
- Balance between latency and throughput

```python
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.types import BatchSettings

# Configure batching
batch_settings = BatchSettings(
    max_messages=100,  # Maximum messages per batch
    max_bytes=1024 * 1024,  # 1 MB
    max_latency=0.1,  # 100ms
)

publisher = pubsub_v1.PublisherClient(batch_settings=batch_settings)
topic_path = publisher.topic_path(project_id, topic_id)

# Publish with callback
futures = []
for message in messages:
    future = publisher.publish(
        topic_path,
        message.encode("utf-8"),
        origin="data-pipeline",
        timestamp=str(time.time())
    )
    futures.append(future)

# Wait for all messages
for future in futures:
    message_id = future.result()
    print(f"Published: {message_id}")
```

### Message Attributes

**Use Cases:**
- Filtering messages in subscriptions
- Routing messages to different consumers
- Adding metadata without parsing message body

```python
# Publish with attributes
future = publisher.publish(
    topic_path,
    data=json.dumps(order_data).encode("utf-8"),
    event_type="order_created",
    region="us-east1",
    priority="high",
    correlation_id=str(uuid.uuid4())
)
```

### Ordering Keys

**When to Use:**
- Maintain message ordering for specific entities
- Process related events in sequence
- Trade-off: Reduces parallelism

```python
# Publish with ordering key
future = publisher.publish(
    topic_path,
    data=message_data,
    ordering_key=f"user-{user_id}"  # All messages for same user ordered
)
```

---

## 3. Message Consumption

### Acknowledgment Strategies

**Best Practices:**
- Acknowledge only after successful processing
- Use appropriate ack deadline (10-600 seconds)
- Extend deadline for long-running operations

```python
# Streaming pull with manual ack
def callback(message):
    try:
        # Process message
        process_message(message.data)
        
        # Acknowledge on success
        message.ack()
    except Exception as e:
        # Negative acknowledge to retry
        message.nack()
        logger.error(f"Processing failed: {e}")

# Subscribe with callback
streaming_pull_future = subscriber.subscribe(
    subscription_path, 
    callback=callback
)

# Configure flow control
flow_control = pubsub_v1.types.FlowControl(
    max_messages=100,
    max_bytes=1024 * 1024 * 100,  # 100 MB
)
```

### Dead Letter Topics

**Configuration:**
```bash
# Create dead letter topic
gcloud pubsub topics create my-topic-dead-letter

# Create subscription with dead letter policy
gcloud pubsub subscriptions create my-sub \
    --topic=my-topic \
    --dead-letter-topic=my-topic-dead-letter \
    --max-delivery-attempts=5
```

**Python Example:**
```python
from google.cloud.pubsub_v1.types import DeadLetterPolicy

# Create subscription with dead letter
dead_letter_policy = DeadLetterPolicy(
    dead_letter_topic=dead_letter_topic_path,
    max_delivery_attempts=5
)

subscription = subscriber.create_subscription(
    request={
        "name": subscription_path,
        "topic": topic_path,
        "dead_letter_policy": dead_letter_policy,
    }
)
```

---

## 4. Message Filtering

### Subscription Filters

**Best Practices:**
- Filter messages at subscription level
- Reduce processing costs by filtering unwanted messages
- Use message attributes for filtering

```bash
# Create filtered subscription
gcloud pubsub subscriptions create filtered-sub \
    --topic=my-topic \
    --message-filter='attributes.event_type="order_created" AND attributes.region="us-east1"'
```

**Complex Filters:**
```python
# Create subscription with complex filter
filter_string = '''
    attributes.priority="high" AND
    (attributes.event_type="order_created" OR attributes.event_type="payment_processed") AND
    attributes.amount > "1000"
'''

subscription = subscriber.create_subscription(
    request={
        "name": subscription_path,
        "topic": topic_path,
        "filter": filter_string,
    }
)
```

---

## 5. Error Handling and Retry

### Retry Policies

**Best Practices:**
- Implement exponential backoff
- Set maximum retry attempts
- Use dead letter topics for persistent failures

```python
import time
import random

def process_with_retry(message, max_retries=3):
    for attempt in range(max_retries):
        try:
            process_message(message.data)
            message.ack()
            return
        except TransientError as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                message.modify_ack_deadline(
                    seconds=int(wait_time) + 60
                )
            else:
                # Final attempt failed
                message.nack()
                logger.error(f"Max retries exceeded: {e}")
        except PermanentError as e:
            # Don't retry permanent errors
            message.ack()  # Acknowledge to avoid reprocessing
            logger.error(f"Permanent error: {e}")
            send_to_error_queue(message.data, e)
            return
```

### Idempotency

**Implementation:**
```python
import hashlib

def process_message_idempotent(message):
    # Generate message hash as idempotency key
    message_hash = hashlib.sha256(
        f"{message.message_id}{message.data}".encode()
    ).hexdigest()
    
    # Check if already processed
    if is_processed(message_hash):
        logger.info(f"Message already processed: {message_hash}")
        message.ack()
        return
    
    try:
        # Process message
        result = process_message(message.data)
        
        # Store processing record
        mark_as_processed(message_hash, result)
        
        message.ack()
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        message.nack()
```

---

## 6. Performance Optimization

### Flow Control

**Best Practices:**
- Configure max outstanding messages and bytes
- Prevent memory exhaustion
- Balance between throughput and resource usage

```python
from google.cloud.pubsub_v1.types import FlowControl

# Configure flow control
flow_control = FlowControl(
    max_messages=1000,  # Max outstanding messages
    max_bytes=1024 * 1024 * 100,  # 100 MB
    max_lease_duration=600,  # 10 minutes
)

subscriber = pubsub_v1.SubscriberClient()
streaming_pull_future = subscriber.subscribe(
    subscription_path,
    callback=callback,
    flow_control=flow_control
)
```

### Parallel Processing

**Worker Pool Pattern:**
```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def parallel_callback(message):
    def process():
        try:
            process_message(message.data)
            message.ack()
        except Exception as e:
            message.nack()
            logger.error(f"Error: {e}")
    
    return process

# Use thread pool for I/O-bound operations
executor = ThreadPoolExecutor(
    max_workers=multiprocessing.cpu_count() * 2
)

def callback(message):
    executor.submit(parallel_callback(message))

streaming_pull_future = subscriber.subscribe(
    subscription_path,
    callback=callback
)
```

---

## 7. Security

### IAM Permissions

**Principle of Least Privilege:**
```bash
# Publisher permissions
gcloud pubsub topics add-iam-policy-binding my-topic \
    --member=serviceAccount:publisher@myproject.iam.gserviceaccount.com \
    --role=roles/pubsub.publisher

# Subscriber permissions
gcloud pubsub subscriptions add-iam-policy-binding my-sub \
    --member=serviceAccount:subscriber@myproject.iam.gserviceaccount.com \
    --role=roles/pubsub.subscriber
```

### Message Encryption

**Customer-Managed Encryption Keys (CMEK):**
```bash
# Create topic with CMEK
gcloud pubsub topics create my-encrypted-topic \
    --topic-encryption-key=projects/my-project/locations/us-central1/keyRings/my-keyring/cryptoKeys/my-key
```

**Application-Level Encryption:**
```python
from cryptography.fernet import Fernet

# Generate key (store securely in Secret Manager)
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt message before publishing
encrypted_data = cipher.encrypt(message_data.encode())
future = publisher.publish(topic_path, encrypted_data)

# Decrypt on consumption
def callback(message):
    decrypted_data = cipher.decrypt(message.data)
    process_message(decrypted_data)
    message.ack()
```

### VPC Service Controls

**Configuration:**
```bash
# Add Pub/Sub to service perimeter
gcloud access-context-manager perimeters update my-perimeter \
    --add-resources=pubsub.googleapis.com \
    --project=my-project
```

---

## 8. Monitoring and Observability

### Key Metrics to Monitor

**CloudWatch Metrics:**
- `pubsub.googleapis.com/topic/send_request_count`
- `pubsub.googleapis.com/subscription/num_undelivered_messages`
- `pubsub.googleapis.com/subscription/oldest_unacked_message_age`
- `pubsub.googleapis.com/subscription/pull_request_count`

**Monitoring Setup:**
```python
from google.cloud import monitoring_v3

def create_alert_policy(project_id):
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"
    
    alert_policy = monitoring_v3.AlertPolicy(
        display_name="Pub/Sub Undelivered Messages",
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Too many undelivered messages",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='metric.type="pubsub.googleapis.com/subscription/num_undelivered_messages" '
                           'resource.type="pubsub_subscription"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=10000,
                    duration={"seconds": 300},
                ),
            )
        ],
    )
    
    return client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
```

### Logging

**Structured Logging:**
```python
import logging
import json

logger = logging.getLogger(__name__)

def callback(message):
    log_entry = {
        "message_id": message.message_id,
        "publish_time": message.publish_time.isoformat(),
        "attributes": dict(message.attributes),
        "size_bytes": len(message.data),
        "subscription": subscription_path
    }
    
    logger.info(json.dumps(log_entry))
    
    try:
        process_message(message.data)
        message.ack()
        logger.info(f"Processed successfully: {message.message_id}")
    except Exception as e:
        logger.error(f"Processing failed: {message.message_id}", exc_info=True)
        message.nack()
```

---

## 9. Cost Optimization

### Message Size Optimization

**Best Practices:**
- Keep messages under 10 KB when possible
- Use compression for large payloads
- Store large data in Cloud Storage, send references

```python
import gzip
import json

def publish_large_message(publisher, topic_path, data):
    # Compress if data is large
    json_data = json.dumps(data).encode('utf-8')
    
    if len(json_data) > 10240:  # 10 KB
        # Store in Cloud Storage
        blob_path = upload_to_gcs(json_data)
        message = {
            "type": "reference",
            "location": blob_path
        }
    else:
        # Send directly
        message = {
            "type": "inline",
            "data": data
        }
    
    publisher.publish(
        topic_path,
        json.dumps(message).encode('utf-8')
    )
```

### Subscription Management

**Best Practices:**
- Delete unused subscriptions
- Use expiration policies for temporary subscriptions
- Monitor subscription backlog

```bash
# Set subscription expiration
gcloud pubsub subscriptions create temp-sub \
    --topic=my-topic \
    --expiration-period=7d

# List subscriptions with no activity
gcloud pubsub subscriptions list \
    --filter="state:DETACHED"
```

---

## 10. Integration Patterns

### Pub/Sub to BigQuery

**Direct Integration:**
```bash
# Create BigQuery subscription
gcloud pubsub subscriptions create bigquery-sub \
    --topic=my-topic \
    --bigquery-table=myproject:mydataset.mytable \
    --write-metadata
```

### Pub/Sub to Dataflow

**Streaming Pipeline:**
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

def run_pipeline():
    options = PipelineOptions([
        '--project=my-project',
        '--runner=DataflowRunner',
        '--streaming',
        '--region=us-central1'
    ])
    
    with beam.Pipeline(options=options) as p:
        (p
         | 'Read from Pub/Sub' >> beam.io.ReadFromPubSub(
             subscription='projects/my-project/subscriptions/my-sub')
         | 'Parse JSON' >> beam.Map(json.loads)
         | 'Transform' >> beam.Map(transform_message)
         | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
             'my-project:dataset.table',
             schema='field1:STRING,field2:INTEGER',
             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
        )
```

### Pub/Sub to Cloud Functions

**Event-Driven Processing:**
```python
import base64
import json

def process_pubsub(event, context):
    """Cloud Function triggered by Pub/Sub."""
    
    # Decode message
    if 'data' in event:
        message_data = base64.b64decode(event['data']).decode('utf-8')
        message_json = json.loads(message_data)
    
    # Process message
    result = process_message(message_json)
    
    # Attributes
    attributes = event.get('attributes', {})
    
    print(f"Processed message: {context.event_id}")
    return result
```

---

## 11. Schema Evolution

### Schema Validation

**Avro Schema:**
```python
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.types import Encoding

# Create schema
schema_client = pubsub_v1.SchemaServiceClient()
schema_path = schema_client.schema_path(project_id, schema_id)

avro_schema = {
    "type": "record",
    "name": "Order",
    "fields": [
        {"name": "order_id", "type": "string"},
        {"name": "amount", "type": "double"},
        {"name": "timestamp", "type": "long"}
    ]
}

schema = pubsub_v1.types.Schema(
    name=schema_path,
    type_=pubsub_v1.types.Schema.Type.AVRO,
    definition=json.dumps(avro_schema)
)

created_schema = schema_client.create_schema(
    request={"parent": f"projects/{project_id}", "schema": schema, "schema_id": schema_id}
)

# Create topic with schema
topic = publisher.create_topic(
    request={
        "name": topic_path,
        "schema_settings": {
            "schema": schema_path,
            "encoding": Encoding.JSON
        }
    }
)
```

---

## 12. Common Anti-Patterns

### ❌ Anti-Pattern 1: Synchronous Request-Response
**Problem:** Using Pub/Sub for synchronous communication
**Solution:** Use Cloud Tasks or direct API calls

### ❌ Anti-Pattern 2: Not Handling Duplicates
**Problem:** Assuming exactly-once delivery
**Solution:** Implement idempotent processing

### ❌ Anti-Pattern 3: Blocking Operations in Callback
**Problem:** Long-running operations in message callback
**Solution:** Use asynchronous processing or extend ack deadline

### ❌ Anti-Pattern 4: Ignoring Backlog
**Problem:** Not monitoring undelivered messages
**Solution:** Set up alerts for message backlog

### ❌ Anti-Pattern 5: Over-Subscribing
**Problem:** Creating too many subscriptions per topic
**Solution:** Use message filtering or consolidate subscriptions

---

## 13. Quick Reference Checklist

### Publishing
- [ ] Enable message batching for high throughput
- [ ] Add meaningful attributes for filtering
- [ ] Use ordering keys only when necessary
- [ ] Implement proper error handling
- [ ] Monitor publish success rate

### Subscribing
- [ ] Configure appropriate ack deadline
- [ ] Implement idempotent processing
- [ ] Set up dead letter topic
- [ ] Configure flow control
- [ ] Monitor message backlog

### Security
- [ ] Use least privilege IAM roles
- [ ] Enable CMEK if required
- [ ] Implement VPC Service Controls
- [ ] Audit access logs regularly

### Cost Optimization
- [ ] Optimize message size
- [ ] Delete unused subscriptions
- [ ] Use message retention appropriately
- [ ] Monitor egress costs
- [ ] Set subscription expiration policies

### Monitoring
- [ ] Set up alerts for backlog
- [ ] Monitor oldest unacked message age
- [ ] Track publish/pull request rates
- [ ] Log processing errors
- [ ] Monitor subscription health

---

*Best Practices for Google Cloud Data Engineer Certification*
