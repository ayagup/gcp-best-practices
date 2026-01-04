# Dataflow Best Practices

## Overview
Cloud Dataflow is Google Cloud's fully managed service for stream and batch data processing based on Apache Beam. This guide covers best practices for pipeline design, performance optimization, and cost management.

---

## 1. Pipeline Design Best Practices

### Unified Batch and Streaming Model
✅ **Write once, run anywhere**: Same code for batch and streaming
✅ Use **windowing** for time-based aggregations
✅ Use **triggers** for early/late data handling
✅ Design for **scalability** from the start

### Basic Pipeline Structure
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Define pipeline options
options = PipelineOptions(
    project='my-project',
    region='us-central1',
    temp_location='gs://my-bucket/temp',
    staging_location='gs://my-bucket/staging',
    runner='DataflowRunner'
)

# Create pipeline
with beam.Pipeline(options=options) as pipeline:
    (pipeline
     | 'Read' >> beam.io.ReadFromText('gs://input/data.txt')
     | 'Transform' >> beam.Map(lambda x: x.upper())
     | 'Write' >> beam.io.WriteToText('gs://output/result'))
```

### Pipeline Optimization Patterns

**Use Composite Transforms**
```python
# ✅ Good: Reusable composite transform
class ParseAndFilterEvents(beam.PTransform):
    def expand(self, pcoll):
        return (pcoll
                | 'Parse JSON' >> beam.Map(json.loads)
                | 'Filter Valid' >> beam.Filter(lambda x: x.get('valid'))
                | 'Extract Fields' >> beam.Map(extract_fields))

# Use in pipeline
events = (pipeline
          | 'Read' >> beam.io.ReadFromPubSub(subscription=subscription)
          | 'Process' >> ParseAndFilterEvents())
```

**Avoid Excessive Chaining**
```python
# ❌ Bad: Long chain hard to debug
result = (data
          | beam.Map(func1)
          | beam.Map(func2)
          | beam.Map(func3)
          | beam.Map(func4)
          | beam.Map(func5))

# ✅ Good: Break into logical steps
parsed = data | 'Parse' >> beam.Map(func1)
filtered = parsed | 'Filter' >> beam.Map(func2)
enriched = filtered | 'Enrich' >> beam.Map(func3)
result = enriched | 'Format' >> beam.Map(func4)
```

---

## 2. Windowing & Triggers

### Fixed Windows
```python
from apache_beam import window

# 5-minute fixed windows
windowed_data = (events
                 | 'Window' >> beam.WindowInto(
                     window.FixedWindows(5 * 60))  # 5 minutes
                 | 'Aggregate' >> beam.CombinePerKey(sum))
```

### Sliding Windows
```python
# 10-minute sliding window, every 5 minutes
windowed_data = (events
                 | 'Window' >> beam.WindowInto(
                     window.SlidingWindows(
                         size=10 * 60,      # Window size
                         period=5 * 60))    # Slide period
                 | 'Aggregate' >> beam.CombinePerKey(sum))
```

### Session Windows
```python
# Session windows with 10-minute gap
windowed_data = (events
                 | 'Window' >> beam.WindowInto(
                     window.Sessions(10 * 60))
                 | 'Aggregate' >> beam.CombinePerKey(sum))
```

### Triggers for Late Data
```python
from apache_beam.transforms.trigger import AfterWatermark, AfterProcessingTime, AfterCount

# Handle late data with triggers
windowed_data = (events
                 | 'Window' >> beam.WindowInto(
                     window.FixedWindows(60),
                     trigger=AfterWatermark(
                         early=AfterProcessingTime(30),  # Early firing every 30s
                         late=AfterCount(100)),           # Late firing after 100 elements
                     accumulation_mode=beam.transforms.AccumulationMode.ACCUMULATING,
                     allowed_lateness=3600)               # Allow 1 hour late data
                 | 'Aggregate' >> beam.CombinePerKey(sum))
```

### Windowing Best Practices
✅ Use **FixedWindows** for regular time-based aggregations
✅ Use **SessionWindows** for user session analysis
✅ Use **SlidingWindows** for moving averages
✅ Set **allowed_lateness** based on data characteristics
✅ Use **triggers** for early results and late data handling
✅ Choose **accumulation mode** carefully (ACCUMULATING vs DISCARDING)

---

## 3. I/O Connectors & Sources

### BigQuery I/O
```python
# Read from BigQuery
from apache_beam.io.gcp.bigquery import ReadFromBigQuery

events = (pipeline
          | 'Read BQ' >> ReadFromBigQuery(
              query='SELECT * FROM `project.dataset.table` WHERE date = @date',
              use_standard_sql=True,
              query_parameters=[
                  beam.io.BigQueryQueryParameter('date', 'DATE', '2025-12-25')
              ]))

# Write to BigQuery
from apache_beam.io.gcp.bigquery import WriteToBigQuery

(transformed_data
 | 'Write BQ' >> WriteToBigQuery(
     'project:dataset.table',
     schema='field1:STRING,field2:INTEGER,field3:FLOAT',
     write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
     create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED))
```

### Pub/Sub I/O
```python
# Read from Pub/Sub
from apache_beam.io.gcp.pubsub import ReadFromPubSub

messages = (pipeline
            | 'Read Pub/Sub' >> ReadFromPubSub(
                subscription='projects/project/subscriptions/my-sub',
                with_attributes=True))

# Write to Pub/Sub
(processed_data
 | 'Write Pub/Sub' >> beam.io.WriteToPubSub(
     topic='projects/project/topics/output-topic'))
```

### Cloud Storage I/O
```python
# Read text files
lines = pipeline | 'Read GCS' >> beam.io.ReadFromText('gs://bucket/path/*.txt')

# Read Avro files
from apache_beam.io import ReadFromAvro
records = pipeline | 'Read Avro' >> ReadFromAvro('gs://bucket/data/*.avro')

# Read Parquet files
from apache_beam.io import ReadFromParquet
records = pipeline | 'Read Parquet' >> ReadFromParquet('gs://bucket/data/*.parquet')

# Write with compression
(output_data
 | 'Write GCS' >> beam.io.WriteToText(
     'gs://bucket/output/file',
     file_name_suffix='.txt.gz',
     compression_type=beam.io.filesystem.CompressionTypes.GZIP))
```

### Cloud Bigtable I/O
```python
from apache_beam.io.gcp.bigtable import WriteToBigTable

(processed_data
 | 'Write Bigtable' >> WriteToBigTable(
     project_id='my-project',
     instance_id='my-instance',
     table_id='my-table'))
```

---

## 4. Performance Optimization

### Use Efficient Transforms

**Combine Instead of GroupByKey**
```python
# ❌ Bad: GroupByKey + sum (less efficient)
result = (data
          | 'Group' >> beam.GroupByKey()
          | 'Sum' >> beam.Map(lambda kv: (kv[0], sum(kv[1]))))

# ✅ Good: CombinePerKey (more efficient)
result = data | 'Sum' >> beam.CombinePerKey(sum)
```

**Use Stateful Processing**
```python
# For per-key state management
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec

class StatefulDoFn(beam.DoFn):
    STATE_SPEC = ReadModifyWriteStateSpec('state', beam.coders.VarIntCoder())
    
    def process(self, element, state=beam.DoFn.StateParam(STATE_SPEC)):
        key, value = element
        current = state.read() or 0
        new_value = current + value
        state.write(new_value)
        yield (key, new_value)
```

### Fusion Optimization
✅ Dataflow **automatically fuses** compatible transforms
✅ Reduces serialization overhead
✅ Monitor fusion in Dataflow UI

### Avoid Expensive Operations
```python
# ❌ Bad: External API call in Map (blocks worker)
def enrich_with_api(element):
    response = requests.get(f'https://api.example.com/{element}')
    return response.json()

# ✅ Good: Batch API calls
class BatchEnrichFn(beam.DoFn):
    def process_batch(self, elements):
        ids = [e['id'] for e in elements]
        responses = batch_api_call(ids)  # Single API call
        for element, response in zip(elements, responses):
            yield {**element, **response}
```

### Reshuffle to Break Fusion
```python
# Use Reshuffle to break fusion and enable parallelism
result = (data
          | 'Heavy Transform' >> beam.Map(expensive_function)
          | 'Reshuffle' >> beam.Reshuffle()
          | 'Next Transform' >> beam.Map(another_function))
```

---

## 5. Worker Configuration

### Machine Types
✅ **n1-standard-1**: Default, suitable for most workloads
✅ **n1-standard-4**: More CPU/memory for complex transformations
✅ **n1-highmem-2**: Memory-intensive operations
✅ **n2-standard-4**: Better performance than n1

```python
options = PipelineOptions(
    project='my-project',
    region='us-central1',
    machine_type='n1-standard-4',
    num_workers=10,
    max_num_workers=100,
    autoscaling_algorithm='THROUGHPUT_BASED'
)
```

### Autoscaling
✅ **THROUGHPUT_BASED** (default): Based on workload throughput
✅ **NONE**: Fixed number of workers
✅ Set **max_num_workers** to control costs

### Disk Configuration
```python
options = PipelineOptions(
    disk_size_gb=100,              # Worker disk size
    disk_type='pd-ssd',            # Use SSD for better performance
    use_public_ips=False           # Use private IPs (faster, cheaper)
)
```

### Streaming Engine (Recommended for Streaming)
✅ Offloads state and windowing to Dataflow service
✅ Reduces worker memory requirements
✅ Better autoscaling
✅ Lower cost for streaming pipelines

```python
options = PipelineOptions(
    streaming=True,
    enable_streaming_engine=True,
    experiments=['use_runner_v2']  # Required for Streaming Engine
)
```

---

## 6. Error Handling & Monitoring

### Dead Letter Queue Pattern
```python
class SafeParseDoFn(beam.DoFn):
    def process(self, element):
        try:
            parsed = json.loads(element)
            yield beam.pvalue.TaggedOutput('success', parsed)
        except Exception as e:
            yield beam.pvalue.TaggedOutput('error', (element, str(e)))

# Use in pipeline
results = (pipeline
           | 'Read' >> beam.io.ReadFromPubSub(subscription)
           | 'Parse' >> beam.ParDo(SafeParseDoFn()).with_outputs('error', main='success'))

# Process success
success = results.success | 'Write Success' >> WriteToBigQuery(...)

# Process errors (dead letter queue)
errors = results.error | 'Write Errors' >> beam.io.WriteToText('gs://bucket/errors/')
```

### Logging
```python
import logging

class MyDoFn(beam.DoFn):
    def process(self, element):
        logging.info(f'Processing element: {element}')
        try:
            result = process_element(element)
            yield result
        except Exception as e:
            logging.error(f'Error processing {element}: {e}')
            # Don't yield - drop bad element
```

### Metrics & Counters
```python
from apache_beam.metrics import Metrics

class CountingDoFn(beam.DoFn):
    def __init__(self):
        self.success_counter = Metrics.counter('main', 'successful_elements')
        self.error_counter = Metrics.counter('main', 'failed_elements')
        self.processing_time = Metrics.distribution('main', 'processing_time_ms')
    
    def process(self, element):
        start_time = time.time()
        try:
            result = process_element(element)
            self.success_counter.inc()
            self.processing_time.update(int((time.time() - start_time) * 1000))
            yield result
        except Exception as e:
            self.error_counter.inc()
            logging.error(f'Error: {e}')
```

### Monitoring Best Practices
✅ Use **Cloud Monitoring** for job metrics
✅ Set up **alerts** for job failures
✅ Monitor **system lag** (streaming pipelines)
✅ Monitor **data freshness**
✅ Track **custom metrics** with Beam Metrics API
✅ Use **Cloud Logging** for debugging

---

## 7. Testing Pipelines

### Unit Testing
```python
import unittest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

class MyPipelineTest(unittest.TestCase):
    def test_transform(self):
        with TestPipeline() as p:
            input_data = p | beam.Create(['hello', 'world'])
            output = input_data | beam.Map(str.upper)
            
            assert_that(output, equal_to(['HELLO', 'WORLD']))
```

### Integration Testing
```python
# Test with DirectRunner (local)
options = PipelineOptions(
    runner='DirectRunner',
    temp_location='gs://bucket/temp'
)

with beam.Pipeline(options=options) as pipeline:
    # Test pipeline with small dataset
    pass
```

### Testing Best Practices
✅ Write **unit tests** for DoFns
✅ Use **DirectRunner** for local testing
✅ Test with **small datasets** first
✅ Use **test data** in staging environment
✅ Validate **output data** quality
✅ Test **error handling** paths

---

## 8. Cost Optimization

### Batch Processing Optimization
✅ Use **batch mode** when latency allows (cheaper than streaming)
✅ Use **Shuffle Service** for large jobs
✅ **Combine** small files before processing
✅ Use **efficient file formats** (Avro, Parquet)
✅ Set **appropriate worker machine types**

### Streaming Optimization
✅ Use **Streaming Engine** (reduces worker costs)
✅ Set **max_num_workers** to control autoscaling
✅ Use **VPC with Private Google Access** (no egress charges)
✅ Optimize **window sizes** (larger windows = fewer windows to process)
✅ Use **stateless transforms** when possible

### Resource Optimization
```python
options = PipelineOptions(
    project='my-project',
    region='us-central1',
    num_workers=5,                    # Start with fewer workers
    max_num_workers=20,               # Cap autoscaling
    machine_type='n1-standard-1',    # Use smaller machines if possible
    disk_size_gb=50,                  # Minimize disk size
    use_public_ips=False,             # Save on egress costs
    save_main_session=False,          # Faster startup
    experiments=['use_runner_v2']     # Better performance
)
```

### Regional Selection
✅ Choose **region close to data sources** (BigQuery, Cloud Storage)
✅ Use **multi-region** for Cloud Storage sources when possible
✅ Minimize **cross-region data transfer**

---

## 9. Streaming Pipeline Best Practices

### Exactly-Once Processing
✅ Use **WriteToBigQuery** with default settings (ensures exactly-once)
✅ Use **idempotent operations** when writing to external systems
✅ Enable **Streaming Engine** for better guarantees

### Handling Late Data
```python
from apache_beam.transforms.trigger import AfterWatermark, AfterProcessingTime

windowed = (events
            | 'Window' >> beam.WindowInto(
                window.FixedWindows(60),
                trigger=AfterWatermark(late=AfterProcessingTime(60)),
                allowed_lateness=3600,
                accumulation_mode=beam.transforms.AccumulationMode.DISCARDING)
            | 'Aggregate' >> beam.CombinePerKey(sum))
```

### State Management
```python
from apache_beam.transforms.userstate import BagStateSpec, on_timer

class SessionTracker(beam.DoFn):
    BUFFER_STATE = BagStateSpec('buffer', beam.coders.StrUtf8Coder())
    
    def process(self, element, buffer=beam.DoFn.StateParam(BUFFER_STATE)):
        key, value = element
        buffer.add(value)
        
        # Flush every 100 elements
        values = list(buffer.read())
        if len(values) >= 100:
            yield (key, values)
            buffer.clear()
```

### Watermark Management
✅ Understand **watermark propagation**
✅ Use **custom watermark estimators** for custom sources
✅ Monitor **watermark lag** in Dataflow UI

---

## 10. Security Best Practices

### Service Accounts
✅ Use **dedicated service accounts** for Dataflow jobs
✅ Grant **minimum required permissions**
✅ Don't use **default Compute Engine service account**

### Required Permissions
- `roles/dataflow.worker`: For worker VMs
- `roles/storage.objectAdmin`: For temp/staging buckets
- `roles/bigquery.dataEditor`: For BigQuery I/O
- `roles/pubsub.subscriber`: For Pub/Sub reads

### Network Security

#### Standard VPC Configuration
✅ Use **VPC** for worker VMs
✅ Use **Private Google Access** (no public IPs)
✅ Configure **firewall rules** appropriately
✅ Use **VPC Service Controls** for perimeter security

#### Shared VPC Networking Considerations

**Shared VPC** allows organizations to centrally manage networking resources while enabling multiple service projects to use the same VPC network. This is critical for enterprise data engineering deployments with centralized network teams.

##### Shared VPC Architecture Overview

```
Host Project (Network Admin)
├── Shared VPC Network
│   ├── Subnet 1 (us-central1) - Dataflow workers
│   ├── Subnet 2 (us-east1) - Dataflow workers
│   └── Subnet 3 (europe-west1) - Dataflow workers
└── Firewall Rules (centralized)

Service Project A (Data Engineering)
├── Dataflow Job 1 → Uses Host Project subnets
└── Dataflow Job 2 → Uses Host Project subnets

Service Project B (Analytics)
├── Dataflow Job 3 → Uses Host Project subnets
└── BigQuery datasets
```

##### Required IAM Permissions for Shared VPC

**Host Project Permissions:**
```python
# Host Project: Network Admin grants service project access
gcloud compute shared-vpc associated-projects add SERVICE_PROJECT_ID \
    --host-project=HOST_PROJECT_ID

# Grant Dataflow service account network user role in HOST project
gcloud projects add-iam-policy-binding HOST_PROJECT_ID \
    --member="serviceAccount:DATAFLOW_SERVICE_ACCOUNT@SERVICE_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/compute.networkUser"

# Grant subnet-level access (more granular)
gcloud compute networks subnets add-iam-policy-binding SUBNET_NAME \
    --region=REGION \
    --project=HOST_PROJECT_ID \
    --member="serviceAccount:DATAFLOW_SERVICE_ACCOUNT@SERVICE_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/compute.networkUser"
```

**Service Project Permissions:**
```python
# Service Project: Dataflow service account needs these roles
- roles/dataflow.worker
- roles/dataflow.admin (for job submission)
- roles/compute.networkUser (granted in HOST project)
```

##### Dataflow Pipeline Configuration for Shared VPC

```python
from apache_beam.options.pipeline_options import PipelineOptions

# Configure Dataflow to use Shared VPC
options = PipelineOptions(
    project='service-project-id',                          # Service project runs the job
    region='us-central1',
    temp_location='gs://service-project-bucket/temp',
    staging_location='gs://service-project-bucket/staging',
    
    # Shared VPC network configuration
    network='projects/host-project-id/global/networks/shared-vpc-network',
    subnetwork='projects/host-project-id/regions/us-central1/subnetworks/dataflow-subnet',
    
    # Disable public IPs (use Private Google Access)
    use_public_ips=False,
    
    # Service account with network user permissions
    service_account_email='dataflow-sa@service-project-id.iam.gserviceaccount.com',
    
    # Worker configuration
    num_workers=10,
    max_num_workers=50,
    machine_type='n1-standard-4',
    
    runner='DataflowRunner'
)

# Create and run pipeline
with beam.Pipeline(options=options) as pipeline:
    # Pipeline logic
    pass
```

##### Subnet Design Best Practices

**Subnet Sizing for Dataflow:**
```python
# Calculate required IP addresses
# Formula: (max_num_workers + 5) * 2 for safety margin

# Example: 100 max workers
required_ips = (100 + 5) * 2 = 210 IPs
# Use /24 subnet (254 usable IPs) or larger

# Recommended subnet configuration
"""
Subnet Name: dataflow-us-central1
Region: us-central1
IP Range: 10.0.1.0/24 (254 IPs)
Private Google Access: ENABLED
Flow Logs: ENABLED (for monitoring)
"""
```

**Multi-Region Subnet Strategy:**
```python
# Host Project: Create regional subnets for Dataflow
regions = ['us-central1', 'us-east1', 'europe-west1']

for region in regions:
    gcloud compute networks subnets create dataflow-{region} \
        --project=HOST_PROJECT_ID \
        --network=shared-vpc-network \
        --region={region} \
        --range=10.0.{x}.0/24 \
        --enable-private-ip-google-access \
        --enable-flow-logs
```

##### Firewall Rules for Shared VPC

**Required Firewall Rules (Configured in Host Project):**

```python
# 1. Allow internal communication between Dataflow workers
gcloud compute firewall-rules create dataflow-worker-internal \
    --project=HOST_PROJECT_ID \
    --network=shared-vpc-network \
    --direction=INGRESS \
    --priority=1000 \
    --action=ALLOW \
    --rules=tcp:12345-12346,tcp:0-65535,udp:0-65535 \
    --source-tags=dataflow \
    --target-tags=dataflow \
    --description="Allow Dataflow workers to communicate"

# 2. Allow SSH for debugging (optional, restricted to IAP)
gcloud compute firewall-rules create dataflow-ssh-iap \
    --project=HOST_PROJECT_ID \
    --network=shared-vpc-network \
    --direction=INGRESS \
    --priority=1000 \
    --action=ALLOW \
    --rules=tcp:22 \
    --source-ranges=35.235.240.0/20 \
    --target-tags=dataflow \
    --description="Allow SSH via Identity-Aware Proxy"

# 3. Allow health checks from Google
gcloud compute firewall-rules create dataflow-health-checks \
    --project=HOST_PROJECT_ID \
    --network=shared-vpc-network \
    --direction=INGRESS \
    --priority=1000 \
    --action=ALLOW \
    --rules=tcp:0-65535 \
    --source-ranges=130.211.0.0/22,35.191.0.0/16 \
    --target-tags=dataflow \
    --description="Allow Google health checks"

# 4. Deny all other ingress (implicit, but explicit is better)
gcloud compute firewall-rules create dataflow-deny-all \
    --project=HOST_PROJECT_ID \
    --network=shared-vpc-network \
    --direction=INGRESS \
    --priority=65534 \
    --action=DENY \
    --rules=all \
    --target-tags=dataflow \
    --description="Deny all other ingress to Dataflow workers"
```

##### Private Google Access Configuration

**Enable Private Google Access for Dataflow subnets:**

```bash
# Enable Private Google Access on Dataflow subnet
gcloud compute networks subnets update dataflow-us-central1 \
    --project=HOST_PROJECT_ID \
    --region=us-central1 \
    --enable-private-ip-google-access

# Verify Private Google Access is enabled
gcloud compute networks subnets describe dataflow-us-central1 \
    --project=HOST_PROJECT_ID \
    --region=us-central1 \
    --format="get(privateIpGoogleAccess)"
```

**What Private Google Access Enables:**
- ✅ Access to Google APIs (BigQuery, Cloud Storage, Pub/Sub) without public IPs
- ✅ Reduced egress costs (no internet egress charges)
- ✅ Enhanced security (no internet exposure)
- ✅ Access to Private Service Connect endpoints

##### Cloud NAT for External API Access

**If Dataflow workers need to access external APIs (not Google services):**

```bash
# Create Cloud Router in Host Project
gcloud compute routers create dataflow-router \
    --project=HOST_PROJECT_ID \
    --network=shared-vpc-network \
    --region=us-central1

# Create Cloud NAT configuration
gcloud compute routers nats create dataflow-nat \
    --project=HOST_PROJECT_ID \
    --router=dataflow-router \
    --region=us-central1 \
    --nat-custom-subnet-ip-ranges=dataflow-us-central1 \
    --auto-allocate-nat-external-ips \
    --enable-logging
```

##### VPC Service Controls for Data Perimeter

**Protect Dataflow jobs with VPC Service Controls:**

```python
# Create service perimeter in Host Project
gcloud access-context-manager perimeters create dataflow_perimeter \
    --title="Dataflow Data Perimeter" \
    --resources=projects/HOST_PROJECT_ID,projects/SERVICE_PROJECT_ID \
    --restricted-services=bigquery.googleapis.com,storage.googleapis.com,dataflow.googleapis.com \
    --policy=POLICY_ID

# Configure ingress/egress rules
gcloud access-context-manager perimeters update dataflow_perimeter \
    --add-ingress-policies=ingress_policy.yaml \
    --add-egress-policies=egress_policy.yaml \
    --policy=POLICY_ID
```

**Ingress Policy Example (ingress_policy.yaml):**
```yaml
- ingressFrom:
    identities:
      - serviceAccount:dataflow-sa@service-project-id.iam.gserviceaccount.com
    sources:
      - resource: projects/SERVICE_PROJECT_ID
  ingressTo:
    resources:
      - projects/HOST_PROJECT_ID
    operations:
      - serviceName: compute.googleapis.com
        methodSelectors:
          - method: "*"
```

##### Shared VPC Monitoring and Troubleshooting

**Enable VPC Flow Logs:**
```bash
# Enable flow logs for Dataflow subnet
gcloud compute networks subnets update dataflow-us-central1 \
    --project=HOST_PROJECT_ID \
    --region=us-central1 \
    --enable-flow-logs \
    --logging-aggregation-interval=interval-5-sec \
    --logging-flow-sampling=1.0 \
    --logging-metadata=include-all
```

**Query Flow Logs in BigQuery:**
```sql
-- Analyze Dataflow worker network traffic
SELECT
  jsonPayload.connection.src_ip,
  jsonPayload.connection.dest_ip,
  jsonPayload.connection.dest_port,
  jsonPayload.bytes_sent,
  COUNT(*) as connection_count
FROM
  `host-project-id.vpc_flows.compute_googleapis_com_vpc_flows_*`
WHERE
  jsonPayload.src_instance.vm_name LIKE 'dataflow-%'
  AND DATE(_PARTITIONTIME) = CURRENT_DATE()
GROUP BY 1, 2, 3, 4
ORDER BY connection_count DESC
LIMIT 100;
```

**Common Shared VPC Issues and Solutions:**

| Issue | Cause | Solution |
|-------|-------|----------|
| Workers fail to start | Missing `compute.networkUser` role | Grant role in Host Project to service account |
| No internet access | Private Google Access disabled | Enable Private Google Access on subnet |
| Can't access external APIs | No Cloud NAT | Configure Cloud NAT in Host Project |
| Firewall blocks traffic | Restrictive firewall rules | Add allow rules for Dataflow tags |
| Cross-project BigQuery access fails | VPC-SC blocking | Add service project to perimeter or configure egress rules |
| Subnet runs out of IPs | Undersized subnet | Use larger CIDR range (/23 or /22) |

##### Complete Shared VPC Example

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import argparse

def run_shared_vpc_pipeline():
    """
    Complete example of Dataflow pipeline using Shared VPC.
    
    Architecture:
    - Host Project: manages network (shared-vpc-network)
    - Service Project: runs Dataflow job
    - Private Google Access: enabled (no public IPs)
    - VPC Service Controls: protects data perimeter
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--host-project', required=True, help='Host project ID')
    parser.add_argument('--service-project', required=True, help='Service project ID')
    parser.add_argument('--network', required=True, help='Shared VPC network name')
    parser.add_argument('--subnet', required=True, help='Subnet name')
    parser.add_argument('--region', default='us-central1', help='Region')
    
    known_args, pipeline_args = parser.parse_known_args()
    
    # Construct full network and subnet paths
    network_path = f'projects/{known_args.host_project}/global/networks/{known_args.network}'
    subnet_path = f'projects/{known_args.host_project}/regions/{known_args.region}/subnetworks/{known_args.subnet}'
    
    # Configure pipeline options for Shared VPC
    pipeline_options = PipelineOptions(
        pipeline_args,
        project=known_args.service_project,
        region=known_args.region,
        temp_location=f'gs://{known_args.service_project}-dataflow/temp',
        staging_location=f'gs://{known_args.service_project}-dataflow/staging',
        
        # Shared VPC configuration
        network=network_path,
        subnetwork=subnet_path,
        use_public_ips=False,  # Use Private Google Access
        
        # Service account (must have compute.networkUser in host project)
        service_account_email=f'dataflow-sa@{known_args.service_project}.iam.gserviceaccount.com',
        
        # Worker configuration
        num_workers=5,
        max_num_workers=20,
        machine_type='n1-standard-2',
        disk_size_gb=50,
        
        # Additional options
        save_main_session=True,
        runner='DataflowRunner'
    )
    
    # Create pipeline
    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read from BigQuery in service project
        events = (pipeline
                  | 'Read BigQuery' >> beam.io.ReadFromBigQuery(
                      query=f'''
                          SELECT *
                          FROM `{known_args.service_project}.dataset.events`
                          WHERE DATE(timestamp) = CURRENT_DATE()
                      ''',
                      use_standard_sql=True))
        
        # Transform data
        transformed = (events
                       | 'Parse Events' >> beam.Map(lambda x: {
                           'event_id': x['event_id'],
                           'timestamp': x['timestamp'],
                           'value': x['value'] * 1.1
                       }))
        
        # Write back to BigQuery
        (transformed
         | 'Write BigQuery' >> beam.io.WriteToBigQuery(
             f'{known_args.service_project}:dataset.processed_events',
             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
             create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED))

if __name__ == '__main__':
    run_shared_vpc_pipeline()
```

**Running the pipeline:**
```bash
python dataflow_shared_vpc_pipeline.py \
    --host-project=network-host-project \
    --service-project=data-service-project \
    --network=shared-vpc-network \
    --subnet=dataflow-us-central1 \
    --region=us-central1
```

##### Shared VPC Best Practices Summary

✅ **Centralized Network Management**: Use host project for all network resources
✅ **Least Privilege**: Grant `compute.networkUser` at subnet level, not network level
✅ **IP Planning**: Size subnets appropriately (min /24 for production)
✅ **Private Google Access**: Always enable for cost savings and security
✅ **Firewall Rules**: Use network tags for granular control
✅ **Flow Logs**: Enable for troubleshooting and security monitoring
✅ **VPC Service Controls**: Implement for sensitive data workloads
✅ **Cloud NAT**: Only if external API access required
✅ **Documentation**: Maintain clear mapping of service projects to subnets
✅ **Automation**: Use Terraform/Deployment Manager for consistent deployment

### Encryption
✅ **Data in transit**: TLS (automatic)
✅ **Data at rest**: CMEK for temp/staging locations
✅ **CMEK for BigQuery**: Use customer-managed keys

```python
options = PipelineOptions(
    dataflow_kms_key='projects/PROJECT/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY'
)
```

---

## 11. Side Inputs and Side Outputs

### Understanding Side Inputs

**Side Inputs** allow you to provide additional data to a `ParDo` transform that is accessible to all elements being processed. They are commonly used for data enrichment, lookups, and broadcasting reference data to workers.

#### Key Characteristics:
- ✅ **Broadcast to all workers**: Entire side input is available on every worker
- ✅ **Memory-bound**: Must fit in worker memory
- ✅ **Immutable**: Side input data doesn't change during processing
- ✅ **Multiple formats**: AsDict, AsList, AsSingleton, AsIter

#### When to Use Side Inputs:
✅ **Reference data lookups**: Enrich streaming data with dimension tables
✅ **Configuration data**: Apply same rules/mappings to all elements
✅ **Small datasets**: Broadcast tables (<10 GB recommended)
✅ **Join patterns**: Broadcast join for small-to-large joins

### Side Input Views

#### 1. AsDict - Dictionary Lookup

**Use Case:** Key-value lookups for data enrichment

```python
import apache_beam as beam
from apache_beam.pvalue import AsDict

def enrich_with_customer_data(transaction, customer_dict):
    """
    Enrich transaction with customer information.
    
    Args:
        transaction: Transaction record with customer_id
        customer_dict: Dictionary of customer_id -> customer_data
    """
    customer_id = transaction['customer_id']
    customer_info = customer_dict.get(customer_id, {})
    
    return {
        **transaction,
        'customer_name': customer_info.get('name', 'Unknown'),
        'customer_tier': customer_info.get('tier', 'Standard'),
        'customer_region': customer_info.get('region', 'Unknown'),
    }

# Main pipeline
with beam.Pipeline(options=options) as pipeline:
    # Load customer dimension table
    customer_data = (pipeline
                     | 'Read Customers' >> beam.io.ReadFromBigQuery(
                         query='SELECT customer_id, name, tier, region FROM `project.dataset.customers`',
                         use_standard_sql=True)
                     | 'Key by Customer ID' >> beam.Map(
                         lambda x: (x['customer_id'], x)))
    
    # Main transaction stream
    transactions = (pipeline
                    | 'Read Transactions' >> beam.io.ReadFromPubSub(
                        subscription='projects/project/subscriptions/transactions')
                    | 'Parse JSON' >> beam.Map(lambda x: json.loads(x.decode('utf-8'))))
    
    # Enrich with side input
    enriched_transactions = (transactions
                             | 'Enrich' >> beam.Map(
                                 enrich_with_customer_data,
                                 customer_dict=AsDict(customer_data)))
    
    # Write enriched data
    enriched_transactions | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
        'project:dataset.enriched_transactions',
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
```

#### 2. AsList - List of Elements

**Use Case:** Access all elements as a list, useful for validations or filtering

```python
from apache_beam.pvalue import AsList

def filter_by_whitelist(element, allowed_ids_list):
    """Filter elements based on whitelist."""
    if element['id'] in allowed_ids_list:
        return element
    return None

# Create whitelist
whitelist = (pipeline
             | 'Read Whitelist' >> beam.io.ReadFromText('gs://bucket/whitelist.txt')
             | 'Parse IDs' >> beam.Map(str.strip))

# Filter main data
filtered_data = (main_data
                 | 'Filter' >> beam.FlatMap(
                     filter_by_whitelist,
                     allowed_ids_list=AsList(whitelist))
                 | 'Remove None' >> beam.Filter(lambda x: x is not None))
```

#### 3. AsSingleton - Single Value

**Use Case:** Configuration values, thresholds, or aggregated metrics

```python
from apache_beam.pvalue import AsSingleton

def apply_threshold(element, threshold_value):
    """Apply threshold from side input."""
    if element['value'] > threshold_value:
        return element
    return None

# Calculate threshold (e.g., 95th percentile)
threshold = (historical_data
             | 'Extract Values' >> beam.Map(lambda x: x['value'])
             | 'Calculate P95' >> beam.CombineGlobally(
                 lambda values: sorted(values)[int(len(values) * 0.95)]))

# Apply threshold to streaming data
filtered = (streaming_data
            | 'Apply Threshold' >> beam.FlatMap(
                apply_threshold,
                threshold_value=AsSingleton(threshold))
            | 'Remove None' >> beam.Filter(lambda x: x is not None))
```

#### 4. AsIter - Iterable

**Use Case:** Memory-efficient iteration over large side inputs

```python
from apache_beam.pvalue import AsIter

def cross_reference(element, reference_iter):
    """
    Check element against reference data.
    More memory-efficient than AsList for large datasets.
    """
    element_key = element['key']
    for ref_item in reference_iter:
        if ref_item['key'] == element_key:
            return {**element, 'matched': True, 'ref_data': ref_item}
    return {**element, 'matched': False}

# Use AsIter for larger reference data
matched_data = (main_data
                | 'Cross Reference' >> beam.Map(
                    cross_reference,
                    reference_iter=AsIter(reference_data)))
```

### Advanced Side Input Patterns

#### Pattern 1: Multiple Side Inputs

```python
def enrich_with_multiple_sources(
    transaction,
    customer_dict,
    product_dict,
    exchange_rates_dict
):
    """Enrich transaction with multiple reference sources."""
    customer_id = transaction['customer_id']
    product_id = transaction['product_id']
    currency = transaction['currency']
    
    customer = customer_dict.get(customer_id, {})
    product = product_dict.get(product_id, {})
    exchange_rate = exchange_rates_dict.get(currency, 1.0)
    
    return {
        **transaction,
        'customer_name': customer.get('name'),
        'customer_segment': customer.get('segment'),
        'product_name': product.get('name'),
        'product_category': product.get('category'),
        'usd_amount': transaction['amount'] * exchange_rate,
    }

# Create multiple side inputs
customer_side = (pipeline | 'Read Customers' >> ... | 'Key Customers' >> ...)
product_side = (pipeline | 'Read Products' >> ... | 'Key Products' >> ...)
exchange_side = (pipeline | 'Read Rates' >> ... | 'Key Rates' >> ...)

# Apply multiple side inputs
enriched = (transactions
            | 'Enrich' >> beam.Map(
                enrich_with_multiple_sources,
                customer_dict=AsDict(customer_side),
                product_dict=AsDict(product_side),
                exchange_rates_dict=AsDict(exchange_side)))
```

#### Pattern 2: Windowed Side Inputs

**Use Case:** Time-varying reference data (e.g., hourly exchange rates)

```python
from apache_beam import window

def enrich_with_hourly_rates(element, rates_dict):
    """Enrich with time-sensitive exchange rates."""
    currency = element['currency']
    rate = rates_dict.get(currency, 1.0)
    return {**element, 'exchange_rate': rate, 'usd_amount': element['amount'] * rate}

# Window both main and side input data
windowed_transactions = (transactions
                         | 'Window Transactions' >> beam.WindowInto(
                             window.FixedWindows(60 * 60)))  # 1-hour windows

windowed_rates = (exchange_rates
                  | 'Window Rates' >> beam.WindowInto(
                      window.FixedWindows(60 * 60))
                  | 'Key by Currency' >> beam.Map(lambda x: (x['currency'], x['rate'])))

# Enrich within same window
enriched = (windowed_transactions
            | 'Enrich with Rates' >> beam.Map(
                enrich_with_hourly_rates,
                rates_dict=AsDict(windowed_rates)))
```

#### Pattern 3: Side Input with Default Values

```python
from apache_beam.pvalue import AsDict

class EnrichWithDefaults(beam.DoFn):
    """DoFn with default handling for missing side input keys."""
    
    DEFAULT_VALUES = {
        'tier': 'STANDARD',
        'region': 'UNKNOWN',
        'discount': 0.0,
    }
    
    def process(self, element, customer_dict):
        customer_id = element['customer_id']
        customer = customer_dict.get(customer_id, self.DEFAULT_VALUES)
        
        yield {
            **element,
            'customer_tier': customer.get('tier', self.DEFAULT_VALUES['tier']),
            'customer_region': customer.get('region', self.DEFAULT_VALUES['region']),
            'discount_rate': customer.get('discount', self.DEFAULT_VALUES['discount']),
        }

# Use in pipeline
enriched = (transactions
            | 'Enrich' >> beam.ParDo(
                EnrichWithDefaults(),
                customer_dict=AsDict(customer_data)))
```

### Side Input Best Practices

✅ **Size Limits**: Keep side inputs < 10 GB (will be broadcast to all workers)
✅ **Caching**: Side inputs are cached in worker memory
✅ **Windows**: Match window specifications between main and side inputs
✅ **Updates**: For frequently changing data, consider using Dataflow streaming with shorter windows
✅ **Default Values**: Always handle missing keys gracefully
✅ **Multiple Sources**: You can use multiple side inputs in a single transform
❌ **Large Datasets**: Don't use side inputs for large datasets (>10 GB) - use CoGroupByKey instead

### Side Input Memory Optimization

```python
# ❌ Bad: Large side input loaded as dictionary (high memory)
huge_lookup = (pipeline
               | 'Read Huge Table' >> beam.io.ReadFromBigQuery(...)
               | 'To Dict' >> beam.Map(lambda x: (x['key'], x)))

enriched = data | 'Enrich' >> beam.Map(
    enrich_func,
    lookup=AsDict(huge_lookup)  # May cause OOM errors
)

# ✅ Good: Use CoGroupByKey for large joins
keyed_main = data | 'Key Main' >> beam.Map(lambda x: (x['id'], x))
keyed_side = huge_table | 'Key Side' >> beam.Map(lambda x: (x['id'], x))

joined = ((keyed_main, keyed_side)
          | 'CoGroup' >> beam.CoGroupByKey()
          | 'Process Join' >> beam.FlatMap(process_cogrouped_data))
```

---

### Understanding Side Outputs

**Side Outputs** (also called tagged outputs) allow a `ParDo` to emit elements to multiple output PCollections. This is useful for routing, filtering, error handling, and multi-path processing.

#### Key Characteristics:
- ✅ **Multiple outputs**: One transform produces multiple PCollections
- ✅ **Routing logic**: Route elements to different outputs based on conditions
- ✅ **Type safety**: Each output can have different element types
- ✅ **Named tags**: Use descriptive tags for outputs

#### When to Use Side Outputs:
✅ **Error handling**: Separate valid and invalid records (dead letter queue)
✅ **Data routing**: Route to different sinks based on conditions
✅ **Multi-step processing**: Split processing into parallel branches
✅ **A/B testing**: Route traffic to different processing paths

### Basic Side Output Pattern

```python
import apache_beam as beam
from apache_beam import pvalue

class SplitByAmountDoFn(beam.DoFn):
    """Split transactions by amount into high-value and standard."""
    
    # Define output tags
    HIGH_VALUE_TAG = 'high_value'
    STANDARD_TAG = 'standard'
    
    def process(self, element):
        amount = element['amount']
        
        if amount >= 10000:
            # Emit to high-value output
            yield pvalue.TaggedOutput(self.HIGH_VALUE_TAG, element)
        else:
            # Emit to main output (standard)
            yield element

# Apply DoFn with side outputs
with beam.Pipeline(options=options) as pipeline:
    transactions = pipeline | 'Read' >> beam.io.ReadFromPubSub(...)
    
    # Split into multiple outputs
    split_result = (transactions
                    | 'Split by Amount' >> beam.ParDo(SplitByAmountDoFn())
                      .with_outputs(
                          SplitByAmountDoFn.HIGH_VALUE_TAG,
                          main='standard'))
    
    # Access each output
    high_value_transactions = split_result[SplitByAmountDoFn.HIGH_VALUE_TAG]
    standard_transactions = split_result.standard
    
    # Process each stream separately
    (high_value_transactions
     | 'Write High Value' >> beam.io.WriteToBigQuery('project:dataset.high_value'))
    
    (standard_transactions
     | 'Write Standard' >> beam.io.WriteToBigQuery('project:dataset.standard'))
```

### Advanced Side Output Patterns

#### Pattern 1: Error Handling with Dead Letter Queue

```python
import json
import logging

class ParseWithErrorHandling(beam.DoFn):
    """Parse JSON with error handling using side outputs."""
    
    SUCCESS_TAG = 'success'
    ERROR_TAG = 'error'
    
    def process(self, element):
        try:
            # Attempt to parse JSON
            parsed = json.loads(element.decode('utf-8'))
            
            # Validate required fields
            required_fields = ['id', 'timestamp', 'amount']
            if all(field in parsed for field in required_fields):
                yield pvalue.TaggedOutput(self.SUCCESS_TAG, parsed)
            else:
                # Missing fields - send to error output
                error_record = {
                    'raw_data': element.decode('utf-8'),
                    'error': 'Missing required fields',
                    'timestamp': beam.utils.timestamp.Timestamp.now().to_utc_datetime(),
                }
                yield pvalue.TaggedOutput(self.ERROR_TAG, error_record)
                
        except json.JSONDecodeError as e:
            # JSON parse error - send to error output
            error_record = {
                'raw_data': element.decode('utf-8'),
                'error': f'JSON parse error: {str(e)}',
                'timestamp': beam.utils.timestamp.Timestamp.now().to_utc_datetime(),
            }
            yield pvalue.TaggedOutput(self.ERROR_TAG, error_record)
        except Exception as e:
            # Unexpected error
            logging.error(f'Unexpected error processing element: {e}')
            error_record = {
                'raw_data': str(element),
                'error': f'Unexpected error: {str(e)}',
                'timestamp': beam.utils.timestamp.Timestamp.now().to_utc_datetime(),
            }
            yield pvalue.TaggedOutput(self.ERROR_TAG, error_record)

# Use in pipeline
with beam.Pipeline(options=options) as pipeline:
    raw_messages = pipeline | 'Read Pub/Sub' >> beam.io.ReadFromPubSub(...)
    
    # Parse with error handling
    parse_result = (raw_messages
                    | 'Parse JSON' >> beam.ParDo(ParseWithErrorHandling())
                      .with_outputs(
                          ParseWithErrorHandling.ERROR_TAG,
                          main=ParseWithErrorHandling.SUCCESS_TAG))
    
    # Process successful records
    success_data = parse_result[ParseWithErrorHandling.SUCCESS_TAG]
    (success_data
     | 'Transform' >> beam.Map(lambda x: transform_data(x))
     | 'Write Success' >> beam.io.WriteToBigQuery(
         'project:dataset.processed_data',
         write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))
    
    # Process error records (dead letter queue)
    error_data = parse_result[ParseWithErrorHandling.ERROR_TAG]
    (error_data
     | 'Format Errors' >> beam.Map(json.dumps)
     | 'Write Errors' >> beam.io.WriteToText(
         'gs://my-bucket/errors/error',
         file_name_suffix='.json',
         shard_name_template=''))
    
    # Also send error notifications
    (error_data
     | 'Create Alert' >> beam.Map(lambda x: f"Error: {x['error']}")
     | 'Send to Pub/Sub' >> beam.io.WriteToPubSub('projects/project/topics/errors'))
```

#### Pattern 2: Multi-Path Data Routing

```python
class RouteByRegion(beam.DoFn):
    """Route data to different outputs based on region."""
    
    US_TAG = 'us'
    EU_TAG = 'eu'
    ASIA_TAG = 'asia'
    OTHER_TAG = 'other'
    
    def process(self, element):
        region = element.get('region', '').upper()
        
        if region in ['US', 'USA', 'UNITED_STATES']:
            yield pvalue.TaggedOutput(self.US_TAG, element)
        elif region in ['EU', 'UK', 'EUROPE']:
            yield pvalue.TaggedOutput(self.EU_TAG, element)
        elif region in ['ASIA', 'APAC', 'JAPAN', 'CHINA']:
            yield pvalue.TaggedOutput(self.ASIA_TAG, element)
        else:
            yield pvalue.TaggedOutput(self.OTHER_TAG, element)

# Route to different processing pipelines
with beam.Pipeline(options=options) as pipeline:
    transactions = pipeline | 'Read' >> beam.io.ReadFromBigQuery(...)
    
    # Route by region
    regional_data = (transactions
                     | 'Route by Region' >> beam.ParDo(RouteByRegion())
                       .with_outputs(
                           RouteByRegion.US_TAG,
                           RouteByRegion.EU_TAG,
                           RouteByRegion.ASIA_TAG,
                           RouteByRegion.OTHER_TAG))
    
    # Process each region separately (e.g., apply regional tax rules)
    us_processed = (regional_data[RouteByRegion.US_TAG]
                    | 'Apply US Rules' >> beam.Map(apply_us_tax_rules)
                    | 'Write US' >> beam.io.WriteToBigQuery('project:dataset.us_transactions'))
    
    eu_processed = (regional_data[RouteByRegion.EU_TAG]
                    | 'Apply EU Rules' >> beam.Map(apply_eu_vat_rules)
                    | 'Write EU' >> beam.io.WriteToBigQuery('project:dataset.eu_transactions'))
    
    asia_processed = (regional_data[RouteByRegion.ASIA_TAG]
                      | 'Apply ASIA Rules' >> beam.Map(apply_asia_tax_rules)
                      | 'Write ASIA' >> beam.io.WriteToBigQuery('project:dataset.asia_transactions'))
```

#### Pattern 3: Quality Tiers with Side Outputs

```python
class ClassifyDataQuality(beam.DoFn):
    """Classify records by data quality into tiers."""
    
    HIGH_QUALITY = 'high_quality'
    MEDIUM_QUALITY = 'medium_quality'
    LOW_QUALITY = 'low_quality'
    
    def process(self, element):
        # Calculate quality score
        score = self.calculate_quality_score(element)
        
        # Add quality score to element
        enriched = {**element, 'quality_score': score}
        
        # Route based on quality
        if score >= 90:
            yield pvalue.TaggedOutput(self.HIGH_QUALITY, enriched)
        elif score >= 70:
            yield pvalue.TaggedOutput(self.MEDIUM_QUALITY, enriched)
        else:
            yield pvalue.TaggedOutput(self.LOW_QUALITY, enriched)
    
    def calculate_quality_score(self, element):
        """Calculate quality score based on completeness and validity."""
        score = 0
        total_fields = 0
        
        required_fields = ['id', 'timestamp', 'user_id', 'amount']
        optional_fields = ['description', 'category', 'tags']
        
        # Check required fields (50 points)
        for field in required_fields:
            total_fields += 1
            if field in element and element[field]:
                score += 50 / len(required_fields)
        
        # Check optional fields (30 points)
        for field in optional_fields:
            if field in element and element[field]:
                score += 30 / len(optional_fields)
        
        # Check data validity (20 points)
        if self.is_valid_data(element):
            score += 20
        
        return min(score, 100)
    
    def is_valid_data(self, element):
        """Validate data format and ranges."""
        try:
            if 'amount' in element and element['amount'] > 0:
                return True
        except:
            return False
        return False

# Process by quality tier
with beam.Pipeline(options=options) as pipeline:
    raw_data = pipeline | 'Read' >> beam.io.ReadFromPubSub(...)
    
    # Classify by quality
    quality_split = (raw_data
                     | 'Parse' >> beam.Map(json.loads)
                     | 'Classify Quality' >> beam.ParDo(ClassifyDataQuality())
                       .with_outputs(
                           ClassifyDataQuality.HIGH_QUALITY,
                           ClassifyDataQuality.MEDIUM_QUALITY,
                           ClassifyDataQuality.LOW_QUALITY))
    
    # High quality: Direct to production table
    (quality_split[ClassifyDataQuality.HIGH_QUALITY]
     | 'Write High Quality' >> beam.io.WriteToBigQuery(
         'project:dataset.production_data'))
    
    # Medium quality: Additional validation before production
    (quality_split[ClassifyDataQuality.MEDIUM_QUALITY]
     | 'Additional Validation' >> beam.Map(validate_and_clean)
     | 'Write Medium Quality' >> beam.io.WriteToBigQuery(
         'project:dataset.validated_data'))
    
    # Low quality: Send to manual review queue
    (quality_split[ClassifyDataQuality.LOW_QUALITY]
     | 'Format for Review' >> beam.Map(format_for_review)
     | 'Write Low Quality' >> beam.io.WriteToBigQuery(
         'project:dataset.manual_review_queue'))
```

#### Pattern 4: A/B Testing with Side Outputs

```python
import hashlib

class ABTestSplitter(beam.DoFn):
    """Split traffic for A/B testing."""
    
    VARIANT_A = 'variant_a'
    VARIANT_B = 'variant_b'
    
    def __init__(self, split_percentage=50):
        """
        Args:
            split_percentage: Percentage of traffic to variant A (0-100)
        """
        self.split_percentage = split_percentage
    
    def process(self, element):
        # Use consistent hashing for user_id to ensure same user always gets same variant
        user_id = str(element.get('user_id', ''))
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        
        if bucket < self.split_percentage:
            yield pvalue.TaggedOutput(self.VARIANT_A, element)
        else:
            yield pvalue.TaggedOutput(self.VARIANT_B, element)

# A/B test different processing logic
with beam.Pipeline(options=options) as pipeline:
    user_events = pipeline | 'Read Events' >> beam.io.ReadFromPubSub(...)
    
    # Split for A/B test (50/50)
    ab_split = (user_events
                | 'Parse' >> beam.Map(json.loads)
                | 'AB Split' >> beam.ParDo(ABTestSplitter(split_percentage=50))
                  .with_outputs(
                      ABTestSplitter.VARIANT_A,
                      ABTestSplitter.VARIANT_B))
    
    # Variant A: Original algorithm
    variant_a_results = (ab_split[ABTestSplitter.VARIANT_A]
                         | 'Process A' >> beam.Map(process_with_algorithm_a)
                         | 'Tag A' >> beam.Map(lambda x: {**x, 'variant': 'A'}))
    
    # Variant B: New algorithm
    variant_b_results = (ab_split[ABTestSplitter.VARIANT_B]
                         | 'Process B' >> beam.Map(process_with_algorithm_b)
                         | 'Tag B' >> beam.Map(lambda x: {**x, 'variant': 'B'}))
    
    # Combine results for analysis
    all_results = ((variant_a_results, variant_b_results)
                   | 'Flatten' >> beam.Flatten()
                   | 'Write Results' >> beam.io.WriteToBigQuery(
                       'project:dataset.ab_test_results'))
```

### Side Output Best Practices

✅ **Use Descriptive Tags**: Name outputs clearly (e.g., 'high_value', 'errors', 'us_region')
✅ **Handle All Cases**: Ensure every element goes to at least one output
✅ **Main Output**: The unnamed output is accessed via `.main` or without tag
✅ **Error Handling**: Always create an error/dead-letter output for robust pipelines
✅ **Monitoring**: Add metrics/counters for each output to track routing
✅ **Type Consistency**: Each output should have consistent element types

### Common Mistakes to Avoid

❌ **Not declaring outputs**: Must use `.with_outputs()` to declare all tags
❌ **Forgetting main output**: If you yield untagged, must specify main='tag_name'
❌ **Dropping elements**: Make sure every element is emitted to some output
❌ **Wrong tag names**: Tag names in `.with_outputs()` must match `TaggedOutput` tags

### Complete Example: Production Pipeline with Side Inputs and Outputs

```python
import apache_beam as beam
from apache_beam import pvalue
from apache_beam.pvalue import AsDict
import json
import logging

class EnrichAndValidate(beam.DoFn):
    """Enrich with side input and route to side outputs based on validation."""
    
    VALID_TAG = 'valid'
    INVALID_TAG = 'invalid'
    ENRICHMENT_FAILED_TAG = 'enrichment_failed'
    
    def process(self, element, customer_dict, product_dict):
        try:
            # Enrich with customer data (side input)
            customer_id = element.get('customer_id')
            customer = customer_dict.get(customer_id)
            
            if not customer:
                # Customer not found - enrichment failed
                yield pvalue.TaggedOutput(self.ENRICHMENT_FAILED_TAG, {
                    **element,
                    'error': 'Customer not found',
                    'customer_id': customer_id,
                })
                return
            
            # Enrich with product data (side input)
            product_id = element.get('product_id')
            product = product_dict.get(product_id)
            
            if not product:
                # Product not found - enrichment failed
                yield pvalue.TaggedOutput(self.ENRICHMENT_FAILED_TAG, {
                    **element,
                    'error': 'Product not found',
                    'product_id': product_id,
                })
                return
            
            # Create enriched record
            enriched = {
                **element,
                'customer_name': customer['name'],
                'customer_tier': customer['tier'],
                'product_name': product['name'],
                'product_price': product['price'],
                'total_amount': element['quantity'] * product['price'],
            }
            
            # Validate enriched record
            if self.validate_record(enriched):
                yield pvalue.TaggedOutput(self.VALID_TAG, enriched)
            else:
                yield pvalue.TaggedOutput(self.INVALID_TAG, enriched)
                
        except Exception as e:
            logging.error(f'Error processing element: {e}')
            yield pvalue.TaggedOutput(self.ENRICHMENT_FAILED_TAG, {
                **element,
                'error': str(e),
            })
    
    def validate_record(self, record):
        """Validate enriched record."""
        required_fields = ['customer_id', 'product_id', 'quantity', 'total_amount']
        return all(field in record and record[field] is not None for field in required_fields)

# Complete pipeline
with beam.Pipeline(options=options) as pipeline:
    # Side inputs: Load reference data
    customers = (pipeline
                 | 'Read Customers' >> beam.io.ReadFromBigQuery(
                     query='SELECT customer_id, name, tier FROM `project.dataset.customers`',
                     use_standard_sql=True)
                 | 'Key Customers' >> beam.Map(lambda x: (x['customer_id'], x)))
    
    products = (pipeline
                | 'Read Products' >> beam.io.ReadFromBigQuery(
                    query='SELECT product_id, name, price FROM `project.dataset.products`',
                    use_standard_sql=True)
                | 'Key Products' >> beam.Map(lambda x: (x['product_id'], x)))
    
    # Main input: Transaction stream
    transactions = (pipeline
                    | 'Read Transactions' >> beam.io.ReadFromPubSub(
                        subscription='projects/project/subscriptions/transactions')
                    | 'Parse JSON' >> beam.Map(lambda x: json.loads(x.decode('utf-8'))))
    
    # Enrich and validate with side inputs and outputs
    result = (transactions
              | 'Enrich and Validate' >> beam.ParDo(
                  EnrichAndValidate(),
                  customer_dict=AsDict(customers),
                  product_dict=AsDict(products))
                .with_outputs(
                    EnrichAndValidate.INVALID_TAG,
                    EnrichAndValidate.ENRICHMENT_FAILED_TAG,
                    main=EnrichAndValidate.VALID_TAG))
    
    # Process valid records
    (result[EnrichAndValidate.VALID_TAG]
     | 'Write Valid' >> beam.io.WriteToBigQuery(
         'project:dataset.validated_transactions',
         write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))
    
    # Process invalid records
    (result[EnrichAndValidate.INVALID_TAG]
     | 'Write Invalid' >> beam.io.WriteToBigQuery(
         'project:dataset.invalid_transactions',
         write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))
    
    # Process enrichment failures
    (result[EnrichAndValidate.ENRICHMENT_FAILED_TAG]
     | 'Format Errors' >> beam.Map(json.dumps)
     | 'Write Enrichment Errors' >> beam.io.WriteToText(
         'gs://my-bucket/enrichment-errors/error',
         file_name_suffix='.json'))
```

---

## 12. Common Patterns

### Slowly Changing Dimensions (SCD)
```python
# Type 2 SCD - Track history with effective dates
def process_scd_type2(new_record, existing_records):
    # Close existing active record
    # Insert new record with current timestamp
    pass
```

### Data Enrichment

For comprehensive data enrichment patterns using side inputs, see **Section 11: Side Inputs and Side Outputs**.

```python
# Basic enrichment with side input (see Section 11 for more examples)
def enrich_with_lookup(element, lookup_table):
    key = element['id']
    enrichment = lookup_table.get(key, {})
    return {**element, **enrichment}

lookup_data = (pipeline
               | 'Read Lookup' >> ReadFromBigQuery(...)
               | 'Key by ID' >> beam.Map(lambda x: (x['id'], x)))

main_data = pipeline | 'Read Main' >> ReadFromPubSub(...)

enriched = (main_data
            | 'Enrich' >> beam.Map(
                enrich_with_lookup,
                lookup_table=beam.pvalue.AsDict(lookup_data)))
```

### Data Deduplication
```python
# Deduplicate based on key within window
def deduplicate_fn(element_list):
    unique = {}
    for element in element_list:
        key = element['id']
        if key not in unique:
            unique[key] = element
    return unique.values()

deduplicated = (events
                | 'Window' >> beam.WindowInto(window.FixedWindows(60))
                | 'Add Key' >> beam.Map(lambda x: (x['id'], x))
                | 'Group' >> beam.GroupByKey()
                | 'Dedup' >> beam.FlatMap(lambda kv: deduplicate_fn(kv[1])))
```

---

## 13. Common Anti-Patterns to Avoid

### General Pipeline Anti-Patterns
❌ **Using GroupByKey unnecessarily**: Use CombinePerKey when possible
❌ **Excessive logging**: Slows down pipeline
❌ **Large side inputs (>10 GB)**: Broadcast to all workers causes OOM errors - use CoGroupByKey instead
❌ **Unbounded PCollections in batch**: Use windowing
❌ **Not handling late data**: Data loss in streaming
❌ **Synchronous external calls**: Use batch API calls
❌ **Not using Streaming Engine**: Higher costs for streaming
❌ **Too many small files**: Combine before processing
❌ **Not setting max_num_workers**: Runaway costs
❌ **Hot keys**: Causes worker stragglers
❌ **Not testing locally**: Issues found late in production

### Side Inputs/Outputs Anti-Patterns
❌ **Not declaring side outputs**: Must use `.with_outputs()` for all tags
❌ **Missing error outputs**: No dead letter queue for invalid data
❌ **Side input window mismatch**: Side input and main input must have matching windows
❌ **Forgetting AsDict/AsList**: Side inputs need appropriate view transforms

### Shared VPC & Networking Anti-Patterns
❌ **Missing compute.networkUser role**: Service account needs this in host project, not service project
❌ **Using project-level network permissions**: Grant subnet-level access for least privilege
❌ **Undersized subnets**: Calculate IPs needed: `(max_num_workers + 5) × 2`, use /24 minimum
❌ **Forgetting Private Google Access**: Results in expensive egress costs and security issues
❌ **Using public IPs in Shared VPC**: Defeats purpose of centralized network security
❌ **Not enabling flow logs**: Makes troubleshooting network issues nearly impossible
❌ **Wrong firewall priorities**: Ensure allow rules have lower priority than deny rules
❌ **Hardcoding network paths**: Use variables for host project and network names
❌ **No Cloud NAT when needed**: Workers can't reach external APIs without NAT or public IPs
❌ **Mixing host/service project resources**: All network resources must be in host project

---

## Quick Reference Checklist

### Pipeline Configuration
- [ ] Use appropriate runner (DirectRunner for testing, DataflowRunner for production)
- [ ] Configure temp and staging locations
- [ ] Set appropriate machine types and worker counts
- [ ] Enable autoscaling with max_num_workers
- [ ] Use Streaming Engine for streaming pipelines

### Data Processing
- [ ] Implement error handling with dead letter queues using side outputs
- [ ] Add custom metrics and logging
- [ ] Use efficient I/O connectors
- [ ] Optimize with CombinePerKey instead of GroupByKey
- [ ] Handle late data with appropriate triggers
- [ ] Use appropriate windowing strategies
- [ ] Implement exactly-once processing semantics

### Side Inputs
- [ ] Keep side inputs < 10 GB for memory efficiency
- [ ] Use AsDict for key-value lookups
- [ ] Use AsList for small reference lists
- [ ] Use AsSingleton for global configuration values
- [ ] Match window specifications between main and side inputs
- [ ] Handle missing keys with default values

### Side Outputs
- [ ] Declare all output tags with `.with_outputs()`
- [ ] Create error/dead-letter output for invalid records
- [ ] Use descriptive tag names (e.g., 'high_value', 'errors')
- [ ] Ensure every element goes to at least one output
- [ ] Add monitoring metrics for each output path

### Shared VPC & Networking
- [ ] Grant `compute.networkUser` role to service account in host project
- [ ] Use subnet-level permissions (not network-level) for least privilege
- [ ] Specify full network path: `projects/HOST/global/networks/NETWORK`
- [ ] Specify full subnet path: `projects/HOST/regions/REGION/subnetworks/SUBNET`
- [ ] Enable Private Google Access on Dataflow subnets
- [ ] Set `use_public_ips=False` in pipeline options
- [ ] Size subnets appropriately: minimum /24 for production workloads
- [ ] Configure firewall rules with dataflow network tags
- [ ] Enable VPC Flow Logs for troubleshooting
- [ ] Set up Cloud NAT if external API access is required
- [ ] Implement VPC Service Controls for sensitive data
- [ ] Document subnet-to-service-project mappings

### Security & Operations
- [ ] Use VPC with Private Google Access
- [ ] Test pipelines locally before deployment
- [ ] Monitor pipeline metrics in Cloud Monitoring
- [ ] Set up alerts for job failures
- [ ] Use composite transforms for reusability
- [ ] Implement CMEK for data encryption
- [ ] Use dedicated service accounts with least privilege

---

## Additional Resources

- [Dataflow Documentation](https://cloud.google.com/dataflow/docs)
- [Apache Beam Programming Guide](https://beam.apache.org/documentation/programming-guide/)
- [Dataflow Best Practices](https://cloud.google.com/dataflow/docs/guides/best-practices)
- [Pipeline Design Patterns](https://beam.apache.org/documentation/patterns/overview/)
- [Pricing Calculator](https://cloud.google.com/products/calculator)
- [Side Inputs Documentation](https://beam.apache.org/documentation/programming-guide/#side-inputs)
- [Side Outputs Documentation](https://beam.apache.org/documentation/programming-guide/#additional-outputs)
- [Shared VPC Documentation](https://cloud.google.com/vpc/docs/shared-vpc)
- [VPC Service Controls](https://cloud.google.com/vpc-service-controls/docs)
- [Private Google Access](https://cloud.google.com/vpc/docs/private-google-access)

---

*Last Updated: December 28, 2025*
*Version: 1.2*
