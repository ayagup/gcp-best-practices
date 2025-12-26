# Cloud Spanner Best Practices

## Overview
Cloud Spanner is Google Cloud's globally distributed, horizontally scalable, strongly consistent relational database. This guide covers best practices for schema design, performance, scalability, and cost optimization.

---

## 1. Schema Design Best Practices

### Primary Key Selection (Critical!)
✅ **Avoid monotonically increasing keys**: Timestamps, auto-incrementing IDs, sequential UUIDs
✅ **Use hash-based or random UUIDs**: Distributes writes across splits
✅ **Bit-reverse sequential IDs**: If you need sortability
✅ **Composite keys**: Combine multiple columns for distribution

### Bad vs Good Primary Key Examples
```sql
❌ BAD - Creates hotspots:
CREATE TABLE users (
  user_id INT64 NOT NULL,        -- Sequential, monotonic
  timestamp TIMESTAMP NOT NULL    -- Time-ordered
) PRIMARY KEY (user_id);

CREATE TABLE events (
  event_time TIMESTAMP NOT NULL,  -- Creates hotspot at current time
  event_id STRING(36)
) PRIMARY KEY (event_time, event_id);

✅ GOOD - Distributes writes:
CREATE TABLE users (
  user_id STRING(36) NOT NULL,    -- Random UUID
  created_at TIMESTAMP
) PRIMARY KEY (user_id);

CREATE TABLE events (
  shard_id INT64 NOT NULL,        -- Random 0-99 or hash-based
  event_time TIMESTAMP NOT NULL,
  event_id STRING(36) NOT NULL
) PRIMARY KEY (shard_id, event_time, event_id);

CREATE TABLE orders (
  customer_id STRING(36) NOT NULL,  -- Natural distribution
  order_id STRING(36) NOT NULL,
  order_time TIMESTAMP
) PRIMARY KEY (customer_id, order_id);
```

### UUID Generation Best Practices
```python
# Python - Use UUID4 (random)
import uuid
user_id = str(uuid.uuid4())  # e.g., 'f47ac10b-58cc-4372-a567-0e02b2c3d479'

# Go - Use UUID4
import "github.com/google/uuid"
userID := uuid.New().String()

# Java - Use UUID
import java.util.UUID;
String userId = UUID.randomUUID().toString();
```

### Interleaved Tables (Parent-Child Relationships)
✅ Use **INTERLEAVE IN PARENT** for 1-to-many relationships
✅ Co-locates child rows with parent rows (better performance)
✅ Best for data accessed together frequently
✅ Maximum 7 levels of interleaving

```sql
-- Parent table
CREATE TABLE customers (
  customer_id STRING(36) NOT NULL,
  name STRING(100),
  email STRING(100)
) PRIMARY KEY (customer_id);

-- Child table interleaved with parent
CREATE TABLE orders (
  customer_id STRING(36) NOT NULL,
  order_id STRING(36) NOT NULL,
  order_date DATE,
  total_amount NUMERIC
) PRIMARY KEY (customer_id, order_id),
  INTERLEAVE IN PARENT customers ON DELETE CASCADE;

-- Grandchild table
CREATE TABLE order_items (
  customer_id STRING(36) NOT NULL,
  order_id STRING(36) NOT NULL,
  item_id INT64 NOT NULL,
  product_id STRING(36),
  quantity INT64
) PRIMARY KEY (customer_id, order_id, item_id),
  INTERLEAVE IN PARENT orders ON DELETE CASCADE;
```

### When to Use Interleaving
✅ Parent-child data frequently accessed together
✅ Child rows always accessed via parent key
✅ Strong access patterns favor co-location
❌ Don't interleave if child table is large and independently queried
❌ Don't interleave more than 7 levels deep

### Data Types Best Practices
✅ **STRING**: Use appropriate length (e.g., STRING(36) for UUIDs, not STRING(MAX))
✅ **INT64**: Integers, avoid FLOAT64 for exact numbers
✅ **NUMERIC**: Financial data, precise decimal calculations
✅ **BYTES**: Binary data, use appropriate size limits
✅ **ARRAY**: Store lists within a row (avoid separate junction tables)
✅ **JSON**: Semi-structured data (use sparingly, limited query support)
✅ **TIMESTAMP**: Use for time data, automatically stores in UTC

---

## 2. Query Optimization

### Secondary Indexes
✅ Create indexes on **frequently filtered columns**
✅ Use **STORING clause** to avoid table lookups
✅ Create **covering indexes** for common queries
✅ Consider **NULL_FILTERED indexes** to exclude NULL values
✅ Limit indexes to avoid write overhead

```sql
-- Basic secondary index
CREATE INDEX idx_users_email ON users(email);

-- Covering index (stores additional columns)
CREATE INDEX idx_orders_customer_date 
  ON orders(customer_id, order_date) 
  STORING (total_amount, status);

-- NULL_FILTERED index (smaller, faster)
CREATE NULL_FILTERED INDEX idx_active_users 
  ON users(last_login) 
  WHERE active = true;

-- Composite index for multi-column queries
CREATE INDEX idx_events_type_time 
  ON events(event_type, event_time DESC);
```

### Index Design Considerations
✅ **Read-heavy workloads**: More indexes acceptable
✅ **Write-heavy workloads**: Minimize indexes (each index = additional write)
✅ **Index size**: Impacts storage costs and write performance
✅ **Index maintenance**: Automatically maintained by Spanner

### Query Best Practices
✅ Use **WHERE clauses** that leverage indexes
✅ **SELECT specific columns**, avoid SELECT *
✅ Use **LIMIT** for pagination
✅ Leverage **interleaving** for join-free queries
✅ Use **@{FORCE_INDEX=index_name}** hint when needed
✅ Batch reads using **IN clauses** or array parameters

```sql
-- Good: Uses index, specific columns
SELECT user_id, name, email 
FROM users 
WHERE email = @email 
LIMIT 1;

-- Good: Leverages interleaving (no join needed)
SELECT o.order_id, o.total_amount, c.name
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE c.customer_id = @customer_id;

-- Good: Batch read
SELECT user_id, name, email
FROM users
WHERE user_id IN UNNEST(@user_ids);

-- Force specific index if optimizer doesn't choose optimally
SELECT user_id, name
FROM users@{FORCE_INDEX=idx_users_email}
WHERE email = @email;
```

### Query Anti-Patterns
❌ **SELECT * FROM large_table**: Retrieves unnecessary data
❌ **Queries without indexes**: Full table scans
❌ **Large OFFSET values**: Inefficient pagination
❌ **Cartesian joins**: Explosive result sets
❌ **Unbounded queries**: No LIMIT, returns millions of rows

### Efficient Pagination
```sql
-- ❌ Bad: Large OFFSET (scans and discards rows)
SELECT * FROM orders ORDER BY order_date OFFSET 10000 LIMIT 100;

-- ✅ Good: Keyset pagination (uses WHERE clause)
SELECT * FROM orders 
WHERE order_date > @last_order_date 
   OR (order_date = @last_order_date AND order_id > @last_order_id)
ORDER BY order_date, order_id 
LIMIT 100;
```

---

## 3. Transaction Management

### Transaction Best Practices
✅ Keep transactions **short and focused**
✅ **Commit as soon as possible** to release locks
✅ Use **read-only transactions** when not modifying data
✅ Use **stale reads** for eventually consistent reads (lower latency)
✅ Avoid **long-running transactions** (10 seconds max recommended)
✅ Batch operations within transactions when possible

### Read-Only Transactions
```sql
-- Strong read (latest data, higher latency)
BEGIN TRANSACTION READ ONLY;
  SELECT * FROM users WHERE user_id = @user_id;
COMMIT TRANSACTION;

-- Stale read (slightly older data, lower latency)
SELECT * FROM users 
WHERE user_id = @user_id
@{read_timestamp=15s_ago};

-- Bounded staleness
SELECT * FROM users 
WHERE user_id = @user_id
@{max_staleness=10s};
```

### Read-Write Transactions
```sql
-- Keep transactions focused
BEGIN TRANSACTION;
  UPDATE accounts SET balance = balance - 100 WHERE account_id = @from_account;
  UPDATE accounts SET balance = balance + 100 WHERE account_id = @to_account;
  INSERT INTO transactions (transaction_id, from_account, to_account, amount)
    VALUES (@transaction_id, @from_account, @to_account, 100);
COMMIT TRANSACTION;
```

### Transaction Sizing
✅ **Mutations per transaction**: < 20,000 recommended
✅ **Transaction size**: < 100 MB recommended
✅ **Transaction duration**: < 10 seconds strongly recommended
✅ Break large operations into smaller batches

### Handling Aborted Transactions
✅ Implement **exponential backoff** retry logic
✅ Understand **abort reasons** (contention, deadline exceeded)
✅ Reduce **contention** by redesigning schema or access patterns

```python
# Python retry with exponential backoff
from google.api_core import retry
from google.cloud import spanner

@retry.Retry(predicate=retry.if_exception_type(
    spanner.exceptions.Aborted))
def transfer_funds(transaction):
    # Transaction logic here
    pass

database.run_in_transaction(transfer_funds)
```

---

## 4. Write Optimization

### Batch Operations
✅ Use **batch mutations** instead of individual writes
✅ Group related writes together
✅ Use **Mutation API** for better performance than DML
✅ Maximum 20,000 mutations per commit recommended

```python
# Python - Batch mutations
with database.batch() as batch:
    batch.insert(
        table='users',
        columns=['user_id', 'name', 'email'],
        values=[
            (uuid.uuid4().hex, 'Alice', 'alice@example.com'),
            (uuid.uuid4().hex, 'Bob', 'bob@example.com'),
            (uuid.uuid4().hex, 'Charlie', 'charlie@example.com'),
        ]
    )
```

### Avoiding Write Hotspots
✅ **Distribute writes** across key ranges
✅ Use **random UUIDs** or hash-based keys
✅ Avoid **timestamp-based primary keys**
✅ Add **shard column** if necessary
✅ Monitor **CPU utilization per split** in metrics

### Bulk Data Loading
✅ Use **Dataflow** for large-scale imports
✅ Parallelize writes across multiple workers
✅ Use **batch mutations** API
✅ Consider temporarily dropping secondary indexes during load
✅ Monitor and stay within rate limits

```python
# Apache Beam pipeline for bulk loading
import apache_beam as beam
from apache_beam.io.gcp.spanner import WriteToSpanner

with beam.Pipeline() as pipeline:
    (pipeline
     | 'Read CSV' >> beam.io.ReadFromText('gs://bucket/data.csv')
     | 'Parse' >> beam.Map(parse_csv_row)
     | 'Write to Spanner' >> WriteToSpanner(
         instance_id='my-instance',
         database_id='my-database',
         table='users',
         max_batch_size_bytes=5 * 1024 * 1024))  # 5MB batches
```

---

## 5. Instance Configuration & Scaling

### Node/Processing Unit Selection
✅ **Regional instances**: 1 node minimum = 1000 processing units (PUs)
✅ **Multi-region instances**: 2 nodes minimum = 2000 PUs
✅ Start small and scale based on monitoring
✅ Each node provides: ~10k QPS reads, ~2k QPS writes (rough estimates)

### Scaling Best Practices
✅ Monitor **CPU utilization** (< 65% recommended, < 75% high priority threshold)
✅ Monitor **storage utilization** (auto-scales, but check costs)
✅ **Scale proactively** before hitting limits
✅ Scale in **increments** (not drastic jumps)
✅ Allow **time to stabilize** after scaling (data rebalancing)

### Regional vs Multi-Region Configuration

**Regional (single region)**
- ✅ Lower latency for regional access
- ✅ Lower cost
- ❌ Regional failure affects availability
- Use for: Single-region applications

**Multi-Region (geo-replicated)**
- ✅ Global availability and disaster recovery
- ✅ Low-latency reads from multiple regions
- ✅ Automatic failover
- ❌ Higher cost (3x+ vs regional)
- ❌ Higher write latency (cross-region replication)
- Use for: Global applications, critical workloads

### Multi-Region Configuration Options
```
nam3 (North America): Iowa, South Carolina, Oregon
nam6: Iowa, South Carolina, Oklahoma, Oregon
eur3 (Europe): Belgium, Netherlands, Finland
eur5: London, Belgium, Finland, Warsaw
eur6: Belgium, Netherlands, Finland, Zurich, Frankfurt, Madrid
asia1: Tokyo, Osaka, Singapore (coming)
```

### Leader Placement
✅ Place **default leader** in region with most writes
✅ Reduces write latency for majority of transactions
✅ Configure per database, not per table

```sql
-- Set default leader region
ALTER DATABASE my_database SET OPTIONS (
  default_leader = 'us-central1'
);
```

---

## 6. Monitoring & Performance Tuning

### Key Metrics to Monitor
✅ **CPU utilization**: By instance and per node (target < 65%)
✅ **Storage utilization**: Track growth, forecast capacity
✅ **Read/Write latency**: 99th percentile important
✅ **QPS (queries per second)**: Read and write operations
✅ **Query performance**: Slow queries, execution plans
✅ **Lock waits**: Indicates contention
✅ **Transaction aborts**: High rate indicates issues

### CPU Utilization Best Practices
- **< 65%**: Recommended target (headroom for spikes)
- **65-75%**: High priority alert (scale up)
- **> 75%**: Critical (queries may timeout, scale immediately)
- **Regional difference**: If one region consistently high, review data/query distribution

### Query Insights
✅ Enable **Query Stats** for performance analysis
✅ Identify **top queries by latency, CPU, row counts**
✅ Review **execution plans** for optimization opportunities
✅ Track query trends over time

```sql
-- Query statistics from SPANNER_SYS tables
SELECT 
  text,
  avg_latency_seconds,
  avg_cpu_seconds,
  execution_count
FROM SPANNER_SYS.QUERY_STATS_TOP_MINUTE
ORDER BY avg_cpu_seconds DESC
LIMIT 10;
```

### Key Visualizer
✅ Use **Key Visualizer** to identify hotspots
✅ Visualize read/write distribution across key ranges
✅ Identify **hotspot patterns** (vertical bands = hotspot)
✅ Review regularly, especially after schema changes

### Troubleshooting Performance Issues
1. **High CPU**: Check for missing indexes, inefficient queries, hotspots
2. **High latency**: Check cross-region writes, large queries, lock contention
3. **Transaction aborts**: Reduce contention, optimize transaction scope
4. **Hotspots**: Review primary key design, add sharding column

---

## 7. Security Best Practices

### IAM & Access Control
✅ Use **IAM roles** for instance/database access
✅ Implement **least privilege** access
✅ Use **database roles** for fine-grained permissions
✅ Separate **read-only and read-write** access

### IAM Roles
- `roles/spanner.databaseReader`: Read-only access to data
- `roles/spanner.databaseUser`: Read/write access to data
- `roles/spanner.databaseAdmin`: Schema changes, backup/restore
- `roles/spanner.admin`: Full control over instances

### Encryption
✅ **Encryption at rest**: Automatic with Google-managed keys
✅ **CMEK**: Customer-managed encryption keys for compliance
✅ **Encryption in transit**: TLS for all connections (automatic)

### VPC Service Controls
✅ Implement **perimeter security** for sensitive data
✅ Prevent **data exfiltration**
✅ Control access from specific VPCs only

### Audit Logging
✅ Enable **Admin Activity logs** (always on)
✅ Enable **Data Access logs** for sensitive databases
✅ Export logs to **BigQuery** for analysis
✅ Monitor **failed access attempts**

---

## 8. Backup & Disaster Recovery

### Backup Strategy
✅ Enable **automatic backups** (retention up to 365 days)
✅ Take **on-demand backups** before major changes
✅ Understand **backup costs** (storage + restore operations)
✅ Test **restore procedures** regularly
✅ Document **RTO and RPO** requirements

### Point-in-Time Recovery (PITR)
✅ Restore to any point within version retention period (default 1 hour, max 7 days)
✅ Useful for **recovering from application errors** or **accidental deletes**
✅ Configure version retention based on recovery needs

```bash
# Create on-demand backup
gcloud spanner backups create BACKUP_ID \
  --instance=INSTANCE_ID \
  --database=DATABASE_ID \
  --retention-period=30d

# Restore from backup
gcloud spanner databases create RESTORED_DB \
  --instance=INSTANCE_ID \
  --backup=BACKUP_ID

# Restore to point-in-time
gcloud spanner databases create RESTORED_DB \
  --instance=INSTANCE_ID \
  --database=DATABASE_ID \
  --restore-time=2025-12-25T12:00:00Z
```

### Multi-Region DR
✅ Multi-region instances provide **automatic failover**
✅ **RPO**: Near zero (synchronous replication)
✅ **RTO**: Automatic, typically seconds to minutes
✅ Test failover by **relocating leader region**

---

## 9. Cost Optimization

### Instance Cost Management
✅ **Right-size instances**: Monitor CPU, don't over-provision
✅ Use **regional instances** if multi-region not required (3x+ cheaper)
✅ Start with **minimum nodes** and scale as needed
✅ Consider **processing units** for finer-grained scaling (100 PU increments)

### Storage Cost Optimization
✅ Monitor **storage growth** trends
✅ Implement **data retention policies** to delete old data
✅ Archive historical data to **Cloud Storage** or **BigQuery**
✅ Use **TTL** (time-to-live) for temporary data (when available)
✅ Review **index sizes** (indexes count toward storage)

### Backup Cost Management
✅ Set appropriate **backup retention periods**
✅ Delete **unnecessary on-demand backups**
✅ Monitor backup storage growth

### Network Cost Optimization
✅ Use **same region** for compute and database
✅ Minimize **cross-region queries** if possible
✅ Use **stale reads** to reduce leader region load

### Query Optimization for Cost
✅ Reduce **CPU usage** with efficient queries
✅ Minimize **data scanned** with proper indexing
✅ Use **read-only transactions** when possible (cheaper than read-write)

---

## 10. Common Anti-Patterns to Avoid

❌ **Monotonically increasing primary keys**: Creates write hotspots
❌ **Timestamp-based primary keys**: All writes go to end of key range
❌ **SELECT * queries**: Wastes CPU and network
❌ **Large unbounded queries**: Can timeout, waste resources
❌ **Too many indexes**: Slows down writes
❌ **Too few indexes**: Causes full table scans
❌ **Long-running transactions**: Holds locks, increases abort rate
❌ **Large transactions**: > 20k mutations or > 100MB
❌ **Not using interleaving**: Missing co-location benefits
❌ **Over-interleaving**: > 7 levels or inappropriate relationships
❌ **Using Spanner like OLAP**: Use BigQuery for analytics instead
❌ **Ignoring Key Visualizer warnings**: Hotspots harm performance
❌ **Not monitoring CPU utilization**: Can lead to outages
❌ **Using regional for global app**: Users experience high latency

---

## 11. Migration to Spanner

### Assessment Phase
✅ Analyze current database **schema, size, QPS**
✅ Identify **primary key design** issues
✅ Review **query patterns** and optimize for Spanner
✅ Evaluate **interleaving opportunities**
✅ Plan for **application changes** (if needed)

### Migration Strategies

**1. Offline Migration** (downtime acceptable)
- Export from source → Transform → Import to Spanner
- Simplest but requires downtime

**2. Online Migration** (minimal downtime)
- Use **Database Migration Service** or **Datastream**
- Dual-write to both databases
- Cutover when synchronized

**3. Dataflow Migration**
```python
# Use Dataflow for large-scale migration
pipeline | ReadFromJDBC(source) | Transform() | WriteToSpanner(destination)
```

### Schema Conversion
✅ Redesign **primary keys** for distribution
✅ Convert **foreign keys** to interleaved tables (where appropriate)
✅ Review **data types** (e.g., auto-increment → UUID)
✅ Plan **secondary indexes** based on query patterns
✅ Test schema with realistic workload

---

## 12. Application Development Best Practices

### Connection Management
✅ Use **connection pooling** (gRPC channels)
✅ Reuse **clients** across requests
✅ Set appropriate **timeouts**
✅ Handle **transient errors** with retry

### Client Libraries
```python
# Python example
from google.cloud import spanner

spanner_client = spanner.Client()
instance = spanner_client.instance('my-instance')
database = instance.database('my-database')

# Read operation
with database.snapshot() as snapshot:
    results = snapshot.execute_sql(
        'SELECT user_id, name FROM users WHERE email = @email',
        params={'email': 'user@example.com'},
        param_types={'email': spanner.param_types.STRING}
    )
    for row in results:
        print(row)

# Write operation
with database.batch() as batch:
    batch.insert(
        table='users',
        columns=['user_id', 'name', 'email'],
        values=[(uuid.uuid4().hex, 'Alice', 'alice@example.com')]
    )
```

### Error Handling
✅ Handle **ABORTED** errors with retry
✅ Handle **DEADLINE_EXCEEDED** with timeout increase or query optimization
✅ Handle **RESOURCE_EXHAUSTED** by scaling instance or reducing QPS
✅ Log errors with context for debugging

---

## Quick Reference Checklist

- [ ] Use random/hash-based primary keys (avoid sequential)
- [ ] Design schema with interleaving for parent-child relationships
- [ ] Create secondary indexes for frequently filtered columns
- [ ] Use STORING clause for covering indexes
- [ ] Keep transactions short (< 10 seconds)
- [ ] Use read-only transactions when not modifying data
- [ ] Batch mutations for better write performance
- [ ] Monitor CPU utilization (target < 65%)
- [ ] Use Key Visualizer to identify hotspots
- [ ] Configure appropriate instance size (regional vs multi-region)
- [ ] Enable automatic backups
- [ ] Implement proper IAM roles and permissions
- [ ] Use stale reads for eventually consistent queries
- [ ] Optimize queries with proper WHERE clauses and LIMIT
- [ ] Use keyset pagination instead of OFFSET
- [ ] Test schema design with realistic workload
- [ ] Document RTO/RPO requirements
- [ ] Set up monitoring and alerting

---

## Additional Resources

- [Cloud Spanner Documentation](https://cloud.google.com/spanner/docs)
- [Schema Design Best Practices](https://cloud.google.com/spanner/docs/schema-design)
- [Query Best Practices](https://cloud.google.com/spanner/docs/query-best-practices)
- [Key Visualizer](https://cloud.google.com/spanner/docs/key-visualizer)
- [Pricing Calculator](https://cloud.google.com/products/calculator)

---

*Last Updated: December 25, 2025*
