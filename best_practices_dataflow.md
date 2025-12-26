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
✅ Use **VPC** for worker VMs
✅ Use **Private Google Access** (no public IPs)
✅ Configure **firewall rules** appropriately
✅ Use **VPC Service Controls** for perimeter security

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

## 11. Common Patterns

### Slowly Changing Dimensions (SCD)
```python
# Type 2 SCD - Track history with effective dates
def process_scd_type2(new_record, existing_records):
    # Close existing active record
    # Insert new record with current timestamp
    pass
```

### Data Enrichment
```python
# Join with side input
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

## 12. Common Anti-Patterns to Avoid

❌ **Using GroupByKey unnecessarily**: Use CombinePerKey when possible
❌ **Excessive logging**: Slows down pipeline
❌ **Large side inputs**: Broadcast to all workers (memory issues)
❌ **Unbounded PCollections in batch**: Use windowing
❌ **Not handling late data**: Data loss in streaming
❌ **Synchronous external calls**: Use batch API calls
❌ **Not using Streaming Engine**: Higher costs for streaming
❌ **Too many small files**: Combine before processing
❌ **Not setting max_num_workers**: Runaway costs
❌ **Using public IPs**: Unnecessary egress costs
❌ **Hot keys**: Causes worker stragglers
❌ **Not testing locally**: Issues found late in production

---

## Quick Reference Checklist

- [ ] Use appropriate runner (DirectRunner for testing, DataflowRunner for production)
- [ ] Configure temp and staging locations
- [ ] Set appropriate machine types and worker counts
- [ ] Enable autoscaling with max_num_workers
- [ ] Use Streaming Engine for streaming pipelines
- [ ] Implement error handling with dead letter queues
- [ ] Add custom metrics and logging
- [ ] Use efficient I/O connectors
- [ ] Optimize with CombinePerKey instead of GroupByKey
- [ ] Handle late data with appropriate triggers
- [ ] Use VPC with Private Google Access
- [ ] Test pipelines locally before deployment
- [ ] Monitor pipeline metrics in Cloud Monitoring
- [ ] Set up alerts for job failures
- [ ] Use appropriate windowing strategies
- [ ] Implement exactly-once processing semantics
- [ ] Use composite transforms for reusability

---

## Additional Resources

- [Dataflow Documentation](https://cloud.google.com/dataflow/docs)
- [Apache Beam Programming Guide](https://beam.apache.org/documentation/programming-guide/)
- [Dataflow Best Practices](https://cloud.google.com/dataflow/docs/guides/best-practices)
- [Pipeline Design Patterns](https://beam.apache.org/documentation/patterns/overview/)
- [Pricing Calculator](https://cloud.google.com/products/calculator)

---

*Last Updated: December 25, 2025*
