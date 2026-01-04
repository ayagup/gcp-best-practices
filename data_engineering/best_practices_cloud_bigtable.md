# Cloud Bigtable Best Practices

## Overview
Cloud Bigtable is a fully managed, scalable NoSQL wide-column database designed for low-latency, high-throughput workloads. This guide covers best practices for schema design, performance optimization, and operational excellence.

---

## 1. Schema Design Best Practices

### Row Key Design (Critical!)
The row key is the **most important design decision** in Bigtable. It determines:
- Data distribution across nodes
- Query performance
- Scalability

### Row Key Best Practices
✅ **Distribute reads/writes evenly**: Avoid hotspots
✅ **Design for query patterns**: Row keys determine what you can query efficiently
✅ **Keep row keys short**: 4KB max, but shorter is better (reduces storage and I/O)
✅ **Use human-readable format**: Easier to debug (but not required)
✅ **Avoid monotonically increasing values**: Timestamps, sequential IDs

### Row Key Anti-Patterns
❌ **Domain names**: `www.example.com` (creates hotspot at 'www')
❌ **Timestamps at start**: `20251225120000#user123` (all writes go to end)
❌ **Sequential IDs**: `user_00001`, `user_00002` (creates hotspot)
❌ **Single-value row keys**: Limited query flexibility

### Good Row Key Patterns

#### 1. Reverse Domain Names
```
❌ Bad: www.example.com
✅ Good: com.example.www
```

#### 2. Multi-Tenant Data
```
✅ tenant_id#entity_id
Example: tenant_a#user_12345
```

#### 3. Time-Series Data
```
✅ entity_id#reverse_timestamp
Example: sensor_42#9223370536854775807  (Long.MAX_VALUE - timestamp)

✅ bucket#entity_id#timestamp
Example: 2025_12#sensor_42#20251225120000
```

#### 4. User Activity Data
```
✅ user_id#reverse_timestamp#event_type
Example: user_789#9223370536854775807#click
```

#### 5. Salting (Add Random Prefix)
```
✅ hash(key)%100#actual_key#timestamp
Example: 42#sensor_data#20251225120000
```

### Field Promotion
✅ Promote important query fields to row key
✅ Order fields by query selectivity (most selective first)

```
Example: Querying user events by type and time
Row Key: user_id#event_type#reverse_timestamp
- Find all 'click' events for user_123: user_123#click#
- Find clicks in time range: user_123#click#<start> to user_123#click#<end>
```

---

## 2. Column Family Design

### Column Family Best Practices
✅ **Keep families small**: Typically 1-3 families (max 100, but don't approach that)
✅ **Group by access pattern**: Data accessed together goes in same family
✅ **Separate hot/cold data**: Frequently vs rarely accessed data
✅ **Name descriptively**: Use meaningful names

### Column Family Examples
```
Table: user_data

Column Families:
- profile: Basic user info (name, email) - accessed frequently
- metadata: Account settings - accessed occasionally  
- activity: User activity logs - append-only, rarely read

Table: sensor_data

Column Families:
- metrics: Sensor measurements
- metadata: Sensor configuration
- summary: Aggregated statistics
```

### Column Qualifiers
✅ Column qualifiers can be dynamic (not predefined)
✅ Use qualifiers as data (not just column names)
✅ Keep qualifier names short to save storage

```
Example: Time-series metrics
Column Family: metrics
Qualifiers: temperature, humidity, pressure (dynamic)

Example: Using qualifiers as data
Column Family: events
Qualifiers: click, view, purchase (event types)
```

---

## 3. Performance Optimization

### Read Performance

#### Use Row Key Ranges (Fastest)
```python
# Read single row (most efficient)
row = table.read_row(b'user_123#20251225')

# Read row range (scan)
rows = table.read_rows(
    start_key=b'user_123#20251225000000',
    end_key=b'user_123#20251225235959'
)
```

#### Filters for Efficiency
✅ Use **row filters** to reduce data transfer
✅ Apply filters at server side (not client)
✅ Use **cells-per-row limit** for pagination

```python
from google.cloud.bigtable import row_filters

# Only return latest version
filter_latest = row_filters.CellsColumnLimitFilter(1)

# Filter by column family
filter_family = row_filters.FamilyNameRegexFilter('profile')

# Filter by column qualifier
filter_column = row_filters.ColumnQualifierRegexFilter(b'email')

# Chain filters
filter_chain = row_filters.RowFilterChain([
    filter_family,
    filter_latest
])

# Apply filter to read
rows = table.read_rows(
    start_key=b'user_100',
    end_key=b'user_200',
    filter_=filter_chain
)
```

#### Batch Reads
✅ Read multiple rows in single request
✅ Use `read_rows()` with multiple row keys
✅ Reduces network overhead

```python
# Batch read specific rows
row_set = row_set_pb2.RowSet()
row_set.row_keys.append(b'user_123')
row_set.row_keys.append(b'user_456')
row_set.row_keys.append(b'user_789')

rows = table.read_rows(row_set=row_set)
```

### Write Performance

#### Batch Writes
✅ **Always use batch mutations** for multiple writes
✅ Batch size: 100-1000 mutations typically optimal
✅ Maximum request size: 100 MB

```python
# Batch write
rows = []
for i in range(1000):
    row = table.direct_row(f'user_{i}'.encode())
    row.set_cell('profile', 'name', f'User {i}', timestamp)
    rows.append(row)

# Write batch
table.mutate_rows(rows)
```

#### Async Writes
✅ Use async/bulk APIs for high throughput
✅ Don't wait for each write to complete

```python
# Python async example
from google.cloud.bigtable import Client
from concurrent.futures import ThreadPoolExecutor

def write_batch(rows):
    table.mutate_rows(rows)

# Parallel batch writes
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(write_batch, batch) for batch in batches]
```

#### Avoid Hotspots
✅ Monitor write distribution with Key Visualizer
✅ Ensure row keys distribute writes evenly
✅ Add salt/hash prefix if needed
✅ Pre-split tables for known key ranges

---

## 4. Cluster Configuration & Scaling

### Cluster Types

**SSD Cluster**
- ✅ Default choice for most workloads
- ✅ Low latency (single-digit milliseconds)
- ✅ High throughput
- Best for: Production workloads, real-time applications

**HDD Cluster**
- ✅ Lower cost (3x+ cheaper than SSD)
- ❌ Higher latency (10-50ms)
- ❌ Lower throughput
- Best for: Large datasets, infrequent access, batch analytics

### Node Sizing
✅ Each node provides:
  - **SSD**: ~10,000 QPS reads, ~10,000 QPS writes (1KB rows)
  - **HDD**: ~500 QPS reads, ~500 QPS writes
  - **Storage**: 8 TB per node (SSD), 16 TB per node (HDD)

### Scaling Best Practices
✅ **Minimum 3 nodes** for production (high availability)
✅ **Scale gradually**: Don't jump from 3 to 30 nodes instantly
✅ **Linear scaling**: Performance scales linearly with nodes
✅ **Storage-to-node ratio**: Keep storage < 70% of capacity per node
✅ **Monitor CPU**: Scale when CPU consistently > 70%

### When to Scale Up
- CPU utilization > 70% sustained
- Storage utilization > 70% of capacity
- QPS approaching node limits
- Latency increasing consistently

### Replication (Multi-Cluster)
✅ Use **replication** for:
  - Disaster recovery
  - Global low-latency reads
  - High availability
✅ Configure **app profiles** for routing requests
✅ Choose **single-cluster** or **multi-cluster** routing

```bash
# Create replicated instance
gcloud bigtable instances create my-instance \
  --display-name="My Instance" \
  --cluster=my-cluster-us \
  --cluster-zone=us-central1-b \
  --cluster-num-nodes=3 \
  --cluster=my-cluster-eu \
  --cluster-zone=europe-west1-b \
  --cluster-num-nodes=3
```

---

## 5. Data Modeling Patterns

### Time-Series Data
```
Row Key: entity_id#reverse_timestamp
Column Family: metrics
Qualifiers: metric1, metric2, metric3

Example:
Row Key: sensor_42#9223370536854775807
metrics:temperature = 72.5
metrics:humidity = 45.2
metrics:pressure = 1013.25
```

### Wide Rows (Event History)
```
Row Key: user_id
Column Family: events
Qualifiers: timestamp#event_type
Values: event_data

Example:
Row Key: user_123
events:20251225120000#click = "button_A"
events:20251225120015#view = "page_5"
events:20251225120030#purchase = "item_99"
```

### Entity Grouping
```
Row Key: group_id#entity_id
Column Family: data
Qualifiers: attributes

Example (E-commerce):
Row Key: order_12345#item_1
data:product_id = "SKU-789"
data:quantity = 2
data:price = 29.99
```

### Tall and Narrow vs Short and Wide

**Tall and Narrow** (Recommended for most cases)
```
Row Key: user_id#event_timestamp
Column Family: event
Qualifier: type
Value: event_data

Pros: Easy to scan time ranges, efficient storage
Cons: More rows to scan
```

**Short and Wide** (Use sparingly)
```
Row Key: user_id
Column Family: events
Qualifiers: timestamp1, timestamp2, timestamp3, ...
Values: event_data

Pros: Single row contains all data
Cons: Row size limits, harder to query time ranges
```

---

## 6. Garbage Collection & Retention

### Cell Versions
✅ Bigtable stores multiple **timestamped versions** of each cell
✅ Configure **garbage collection (GC) policies** to manage versions
✅ Reduce storage costs by cleaning old data

### GC Policy Types

**Max Versions**
```bash
# Keep only latest 3 versions
cbt setgcpolicy my-table cf1 maxversions=3
```

**Max Age**
```bash
# Keep data for 30 days
cbt setgcpolicy my-table cf1 maxage=30d
```

**Union (OR)**
```bash
# Keep if either condition is met (3 versions OR 7 days)
cbt setgcpolicy my-table cf1 "maxversions=3 or maxage=7d"
```

**Intersection (AND)**
```bash
# Keep only if both conditions are met (3 versions AND 7 days)
cbt setgcpolicy my-table cf1 "maxversions=3 and maxage=7d"
```

### GC Best Practices
✅ Set GC policy during table creation
✅ Use **maxage** for time-series data
✅ Use **maxversions=1** if only latest value needed
✅ Balance **storage cost vs data retention needs**
✅ GC runs automatically (not immediate)

---

## 7. Monitoring & Observability

### Key Metrics to Monitor
✅ **CPU utilization**: Per cluster (target < 70%)
✅ **Storage utilization**: Per node (target < 70% of capacity)
✅ **QPS**: Reads and writes per second
✅ **Latency**: p50, p95, p99 read/write latencies
✅ **Error rate**: Failed requests
✅ **Replication lag**: For multi-cluster instances

### Key Visualizer
✅ **Essential tool** for identifying hotspots
✅ Shows read/write patterns across row key ranges
✅ Visual representation of data distribution
✅ Use to validate row key design

#### Reading Key Visualizer
- **Vertical bands**: Hotspot (sequential writes)
- **Horizontal bands**: Good distribution
- **Dark areas**: High activity
- **Light areas**: Low activity

### Cloud Monitoring Setup
```python
# Set up alerting for high CPU
from google.cloud import monitoring_v3

client = monitoring_v3.AlertPolicyServiceClient()
alert_policy = monitoring_v3.AlertPolicy(
    display_name="Bigtable High CPU",
    conditions=[{
        "display_name": "CPU > 70%",
        "threshold_value": 0.7,
        "comparison": "COMPARISON_GT",
        "duration": {"seconds": 300}
    }]
)
```

### Monitoring Best Practices
✅ Set up **alerts** for CPU, storage, latency
✅ Monitor **storage growth rate** for capacity planning
✅ Track **QPS trends** to anticipate scaling needs
✅ Use **Cloud Logging** for detailed request logs
✅ Review **Key Visualizer weekly** for hotspots

---

## 8. Security Best Practices

### IAM & Access Control
✅ Use **IAM roles** for access management
✅ Implement **least privilege** principle
✅ Use **service accounts** for application access
✅ Separate **read and write permissions**

### IAM Roles
- `roles/bigtable.reader`: Read-only access to data
- `roles/bigtable.user`: Read/write access to data
- `roles/bigtable.admin`: Full control (including schema changes)
- `roles/bigtable.viewer`: View configuration (no data access)

### Encryption
✅ **At rest**: Automatic encryption with Google-managed keys
✅ **CMEK**: Customer-managed encryption keys for compliance
✅ **In transit**: TLS encryption (automatic)

### VPC Service Controls
✅ Implement **security perimeter** for sensitive data
✅ Prevent **data exfiltration**
✅ Control access from specific networks only

### Audit Logging
✅ Enable **Admin Activity logs** (always on)
✅ Enable **Data Access logs** for sensitive tables
✅ Export logs to **BigQuery** for analysis
✅ Monitor **unauthorized access attempts**

---

## 9. Backup & Disaster Recovery

### Backup Strategy
✅ Create **regular backups** using managed backups
✅ Backups are **full table snapshots** (consistent)
✅ Store in **separate instance** for DR
✅ Test **restore procedures** regularly

```bash
# Create backup
gcloud bigtable backups create my-backup \
  --instance=my-instance \
  --cluster=my-cluster \
  --table=my-table \
  --retention-period=30d

# List backups
gcloud bigtable backups list --instance=my-instance

# Restore from backup
gcloud bigtable backups restore my-backup \
  --destination-instance=my-instance \
  --destination-table=restored-table
```

### Backup Best Practices
✅ Schedule **automated backups** (using Cloud Scheduler + Cloud Functions)
✅ Keep backups in **different region** for DR
✅ Set appropriate **retention period** (7-365 days)
✅ Monitor **backup storage costs**
✅ Document **RTO/RPO** requirements

### Replication for DR
✅ Use **multi-cluster replication** for automatic failover
✅ **RPO**: Near zero (continuous replication)
✅ **RTO**: Automatic (seconds to minutes)
✅ Configure **app profiles** for failover routing

---

## 10. Cost Optimization

### Compute Costs
✅ **Right-size clusters**: Monitor CPU/storage, don't over-provision
✅ Use **HDD for cold data**: 3x+ cheaper than SSD
✅ Use **autoscaling** (manually or via scripts)
✅ Delete **unused instances** and clusters

### Storage Costs
✅ Implement **GC policies** to remove old data
✅ Archive historical data to **Cloud Storage** or **BigQuery**
✅ Monitor **storage utilization** per node
✅ Compress data in **application layer** if possible
✅ Keep row keys and qualifiers **short**

### Network Costs
✅ Keep compute and Bigtable in **same region**
✅ Minimize **cross-region traffic**
✅ Use **multi-cluster routing** strategically

### Backup Costs
✅ Set **appropriate retention** (not longer than needed)
✅ Delete **old backups** no longer required
✅ Consider **export to Cloud Storage** for long-term archival

---

## 11. Integration with Other GCP Services

### Dataflow Integration
✅ Use Bigtable connectors for **batch and streaming pipelines**
✅ Efficient **bulk reads and writes**
✅ Parallel processing across multiple workers

```python
# Apache Beam - Read from Bigtable
import apache_beam as beam
from apache_beam.io.gcp.bigtable import ReadFromBigtable

with beam.Pipeline() as pipeline:
    rows = (pipeline
            | ReadFromBigtable(
                project_id='my-project',
                instance_id='my-instance',
                table_id='my-table')
            | beam.Map(process_row))
```

### BigQuery Integration
✅ Export Bigtable data to **BigQuery for analytics**
✅ Use Dataflow for transformation
✅ Join BigQuery data with Bigtable lookups

### Cloud Functions Integration
✅ Trigger functions from **Pub/Sub** to write to Bigtable
✅ Use for **real-time data ingestion**
✅ Implement **connection pooling** in functions

---

## 12. Data Migration to Bigtable

### Migration Strategies

**1. Dataflow Migration** (Recommended)
```python
# Import from CSV/Avro/Parquet
pipeline | ReadFromSource() | Transform() | WriteToBigtable()
```

**2. HBase Import**
```bash
# For HBase migration
gcloud bigtable instances import my-table \
  --source-uri=gs://bucket/hbase-export/* \
  --source-format=hbase
```

**3. Streaming Replication**
- Use Datastream or custom CDC solution
- Dual-write during migration
- Cutover when synchronized

### Pre-Migration Checklist
- [ ] Analyze current data model and access patterns
- [ ] Design optimal row key structure for Bigtable
- [ ] Plan column families based on access patterns
- [ ] Estimate required cluster size
- [ ] Test performance with representative workload
- [ ] Plan for data validation post-migration
- [ ] Prepare rollback strategy

---

## 13. Common Anti-Patterns to Avoid

❌ **Sequential row keys**: Creates write hotspots (timestamps, IDs)
❌ **Too many column families**: > 100 families (hard limit)
❌ **Large rows**: > 100 MB per row (impacts performance)
❌ **Many column qualifiers per row**: > 100k qualifiers
❌ **Short-lived tables**: Bigtable not designed for temporary tables
❌ **Small datasets**: < 1 TB (consider Cloud SQL, Firestore instead)
❌ **Highly relational data**: Use Cloud SQL or Spanner
❌ **Ad-hoc queries**: Bigtable doesn't support SQL or secondary indexes
❌ **Single node cluster**: No high availability (minimum 3 for production)
❌ **Ignoring Key Visualizer**: Missing hotspot warnings
❌ **Not using batch operations**: Wasting resources on individual requests
❌ **Storing large objects**: Use Cloud Storage, store references in Bigtable
❌ **Complex transactions**: Bigtable only supports single-row transactions

---

## 14. Performance Benchmarking

### Testing Tools
✅ Use `cbt` command-line tool for testing
✅ Run load tests before production
✅ Test with realistic data distribution

```bash
# Simple read test
cbt read my-table count=1000

# Load test with cbt
for i in {1..10000}; do
  cbt set my-table "row_$i" cf1:col1="value_$i"
done

# Measure latency
time cbt read my-table start="row_1" end="row_1000"
```

### Performance Expectations
**SSD Cluster (per node)**
- Read latency: 5-10ms (p99)
- Write latency: 5-10ms (p99)
- Throughput: ~10k QPS per node

**HDD Cluster (per node)**
- Read latency: 10-50ms (p99)
- Write latency: 10-50ms (p99)
- Throughput: ~500 QPS per node

---

## 15. Troubleshooting Guide

### High Latency
1. Check **CPU utilization** (scale if > 70%)
2. Review **Key Visualizer** for hotspots
3. Analyze **row key design**
4. Check for **large rows** (> 1 MB)
5. Verify **cluster location** vs client location

### Write Hotspots
1. Review **Key Visualizer**
2. Check **row key design** (avoid sequential keys)
3. Add **salt/hash prefix** to row keys
4. Pre-split table if needed
5. Distribute writes across more key ranges

### Read Performance Issues
1. Ensure queries use **row key ranges**
2. Add **filters** to reduce data transfer
3. Check **CPU utilization**
4. Consider **caching** frequently accessed data
5. Review **application connection pooling**

### Storage Issues
1. Check **storage per node** (< 70% recommended)
2. Review **GC policies** (cleaning old data?)
3. Consider **archiving** historical data
4. Scale up **number of nodes**

---

## Quick Reference Checklist

- [ ] Design row keys for even distribution (avoid sequential)
- [ ] Use reverse timestamp for time-series data
- [ ] Keep column families to 1-3 (max 100)
- [ ] Implement appropriate GC policies
- [ ] Use batch operations for reads and writes
- [ ] Start with minimum 3 nodes for production
- [ ] Monitor CPU utilization (scale at 70%)
- [ ] Use Key Visualizer to detect hotspots
- [ ] Enable backups with appropriate retention
- [ ] Implement IAM least privilege access
- [ ] Use CMEK if compliance requires
- [ ] Set up monitoring and alerting
- [ ] Test performance with realistic workload
- [ ] Use multi-cluster for DR if needed
- [ ] Configure app profiles for routing
- [ ] Keep row keys and qualifiers short
- [ ] Archive old data to reduce costs
- [ ] Use HDD for cold data workloads

---

## Additional Resources

- [Cloud Bigtable Documentation](https://cloud.google.com/bigtable/docs)
- [Schema Design Guide](https://cloud.google.com/bigtable/docs/schema-design)
- [Performance Tuning](https://cloud.google.com/bigtable/docs/performance)
- [Key Visualizer](https://cloud.google.com/bigtable/docs/keyvis-overview)
- [cbt CLI Tool](https://cloud.google.com/bigtable/docs/cbt-reference)

---

*Last Updated: December 25, 2025*
