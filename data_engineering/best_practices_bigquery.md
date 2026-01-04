# BigQuery Best Practices

## Overview
BigQuery is Google Cloud's serverless, highly scalable enterprise data warehouse. This guide covers best practices for performance optimization, cost management, schema design, and query optimization.

---

## 1. Table Design & Schema Best Practices

### Partitioning

**When to Use Partitioning**
✅ **Time-series data**: Logs, events, sensor data
✅ **Large tables**: > 1 GB recommended
✅ **Query specific date ranges**: Reduces data scanned
✅ **Cost optimization**: Scan only relevant partitions

**Partition Types**

**1. Time-Unit Column Partitioning** (Recommended)
```sql
-- Partition by DATE column
CREATE TABLE my_dataset.events (
  event_id STRING,
  event_date DATE,
  user_id STRING,
  event_type STRING
)
PARTITION BY event_date
OPTIONS(
  partition_expiration_days=365,
  require_partition_filter=true
);

-- Partition by TIMESTAMP column
CREATE TABLE my_dataset.logs (
  log_id STRING,
  log_timestamp TIMESTAMP,
  severity STRING,
  message STRING
)
PARTITION BY DATE(log_timestamp)
OPTIONS(
  partition_expiration_days=90
);
```

**2. Ingestion-Time Partitioning**
```sql
-- Partition by _PARTITIONTIME pseudo-column
CREATE TABLE my_dataset.raw_data (
  id STRING,
  data STRING
)
PARTITION BY _PARTITIONDATE
OPTIONS(
  partition_expiration_days=7
);
```

**3. Integer Range Partitioning**
```sql
-- Partition by customer_id ranges
CREATE TABLE my_dataset.customer_data (
  customer_id INT64,
  name STRING,
  region STRING
)
PARTITION BY RANGE_BUCKET(customer_id, GENERATE_ARRAY(0, 100000, 1000));
```

### Partitioning Best Practices
✅ Use **DATE** partitioning for daily data
✅ Set **partition_expiration_days** for automatic cleanup
✅ Use **require_partition_filter=true** to enforce filter usage
✅ Limit to **4000 partitions** per table (soft limit)
✅ Partition on **frequently filtered columns**

### Clustering

**When to Use Clustering**
✅ **High cardinality columns**: user_id, product_id, country
✅ **Frequently filtered columns** in WHERE clauses
✅ **Combine with partitioning** for best performance
✅ **Large tables**: > 1 GB

```sql
-- Partition + Cluster (Best combination)
CREATE TABLE my_dataset.ecommerce_events (
  event_timestamp TIMESTAMP,
  user_id STRING,
  product_id STRING,
  event_type STRING,
  country STRING
)
PARTITION BY DATE(event_timestamp)
CLUSTER BY user_id, country
OPTIONS(
  partition_expiration_days=730
);

-- Query benefits from both partitioning and clustering
SELECT user_id, COUNT(*) as event_count
FROM my_dataset.ecommerce_events
WHERE DATE(event_timestamp) BETWEEN '2025-01-01' AND '2025-12-31'
  AND country = 'US'
GROUP BY user_id;
```

### Clustering Best Practices
✅ Cluster on **up to 4 columns** (order matters)
✅ Put **most filtered column first**
✅ Use for **high-cardinality columns**
✅ Combine with partitioning for **maximum performance**
✅ Automatic re-clustering (no maintenance needed)

### Column Ordering
✅ Place **frequently queried columns first**
✅ Group **related columns** together
✅ Consider **column compression** (similar values compress better)

---

## 2. Query Optimization

### SELECT Only Required Columns
```sql
-- ❌ Bad: Scans entire table
SELECT * FROM my_dataset.large_table;

-- ✅ Good: Scans only needed columns
SELECT user_id, event_type, event_date 
FROM my_dataset.large_table;
```

### Use Partitioning Filters
```sql
-- ❌ Bad: Scans all partitions
SELECT COUNT(*) 
FROM my_dataset.events
WHERE user_id = '12345';

-- ✅ Good: Scans single partition
SELECT COUNT(*) 
FROM my_dataset.events
WHERE DATE(event_timestamp) = '2025-12-25'
  AND user_id = '12345';
```

### Avoid SELECT DISTINCT on Large Datasets
```sql
-- ❌ Bad: Expensive operation
SELECT DISTINCT user_id FROM my_dataset.events;

-- ✅ Good: Use GROUP BY (same result, often faster)
SELECT user_id FROM my_dataset.events GROUP BY user_id;

-- ✅ Better: Use APPROX_COUNT_DISTINCT for counts
SELECT APPROX_COUNT_DISTINCT(user_id) FROM my_dataset.events;
```

### Optimize JOINs

**Join Order Matters**
```sql
-- ✅ Good: Put largest table first
SELECT 
  large.user_id,
  small.user_name
FROM my_dataset.large_fact_table AS large
JOIN my_dataset.small_dimension_table AS small
  ON large.user_id = small.user_id;
```

**Use INNER JOIN When Possible**
```sql
-- INNER JOIN is faster than LEFT/RIGHT JOIN when appropriate
SELECT a.id, b.name
FROM table_a AS a
INNER JOIN table_b AS b
  ON a.id = b.id;
```

**Broadcast Joins for Small Tables**
```sql
-- Use hint for small table (< 10 MB)
SELECT 
  large.user_id,
  small.user_name
FROM my_dataset.large_table AS large
JOIN my_dataset.small_table AS small
  ON large.user_id = small.user_id
OPTIONS(join_type='BROADCAST');
```

### Use Approximate Aggregation Functions
```sql
-- ✅ APPROX functions are much faster for large datasets
SELECT 
  APPROX_COUNT_DISTINCT(user_id) as unique_users,
  APPROX_QUANTILES(amount, 100)[OFFSET(50)] as median_amount,
  APPROX_TOP_COUNT(product_id, 10) as top_products
FROM my_dataset.events;
```

### Avoid Self-JOINs with Window Functions
```sql
-- ❌ Bad: Self-join for running total
SELECT 
  a.date,
  SUM(b.amount) as running_total
FROM table a
JOIN table b ON b.date <= a.date
GROUP BY a.date;

-- ✅ Good: Window function
SELECT 
  date,
  SUM(amount) OVER (ORDER BY date) as running_total
FROM table;
```

### Use LIMIT for Testing
```sql
-- Test queries with LIMIT first
SELECT * 
FROM my_dataset.large_table
LIMIT 100;

-- Use cost estimation before running
-- Check "bytes processed" in query validator
```

---

## 3. Cost Optimization

### Query Cost Management

**Use Query Dry Run**
```python
# Python: Estimate query cost before running
from google.cloud import bigquery

client = bigquery.Client()
job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

query_job = client.query(
    "SELECT COUNT(*) FROM `project.dataset.large_table`",
    job_config=job_config
)

print(f"This query will process {query_job.total_bytes_processed} bytes")
print(f"Estimated cost: ${(query_job.total_bytes_processed / 1e12) * 5:.2f}")
```

**Set Maximum Bytes Billed**
```sql
-- Limit query cost
SELECT COUNT(*) 
FROM my_dataset.large_table
OPTIONS(max_bytes_billed=1000000000); -- 1 GB limit
```

### Storage Cost Optimization

**Use Table Expiration**
```sql
-- Set table expiration (auto-delete after 90 days)
CREATE TABLE my_dataset.temp_data (
  id STRING,
  data STRING
)
OPTIONS(
  expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
);
```

**Long-Term Storage Discount**
✅ **Automatic discount**: 50% off for tables not modified for 90 days
✅ No action required
✅ Monitor with INFORMATION_SCHEMA

```sql
-- Find tables eligible for long-term storage
SELECT 
  table_name,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), TIMESTAMP_MILLIS(last_modified_time), DAY) as days_since_modified
FROM my_dataset.INFORMATION_SCHEMA.TABLES
WHERE TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), TIMESTAMP_MILLIS(last_modified_time), DAY) >= 90;
```

**Delete Unused Tables/Datasets**
```sql
-- Find large unused tables
SELECT 
  table_name,
  size_bytes / POW(10, 9) as size_gb,
  TIMESTAMP_MILLIS(last_modified_time) as last_modified
FROM my_dataset.INFORMATION_SCHEMA.TABLES
WHERE TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), TIMESTAMP_MILLIS(last_modified_time), DAY) > 180
ORDER BY size_bytes DESC;
```

### Partition Pruning
```sql
-- ✅ Partition filter in WHERE clause (prunes partitions)
SELECT COUNT(*) 
FROM my_dataset.events
WHERE event_date BETWEEN '2025-12-01' AND '2025-12-31';

-- ❌ Partition column in SELECT without WHERE (scans all)
SELECT event_date, COUNT(*) 
FROM my_dataset.events
GROUP BY event_date;
```

### Use Materialized Views
```sql
-- Create materialized view for frequently run aggregations
CREATE MATERIALIZED VIEW my_dataset.daily_metrics AS
SELECT 
  DATE(event_timestamp) as event_date,
  country,
  COUNT(*) as event_count,
  COUNT(DISTINCT user_id) as unique_users
FROM my_dataset.events
GROUP BY event_date, country;

-- Query materialized view (much cheaper)
SELECT * FROM my_dataset.daily_metrics
WHERE event_date = '2025-12-25';
```

### Use BI Engine for Dashboards
✅ **BI Engine**: In-memory analysis service
✅ **Cost**: $0.06 per GB per hour (much cheaper than repeated queries)
✅ **Use for**: Dashboards, Looker Studio, frequent queries
✅ **Capacity**: Reserve 1-100 GB

---

## 4. Data Loading Best Practices

### Batch Loading

**Use Appropriate File Format**
- **Avro** (Recommended): Self-describing, efficient, schema evolution
- **Parquet**: Columnar format, excellent compression, fast reads
- **ORC**: Columnar format, good compression
- **CSV**: Simple but slower, larger size
- **JSON**: Flexible but inefficient (use newline-delimited)

```python
# Load Parquet files (recommended for analytics)
from google.cloud import bigquery

client = bigquery.Client()
table_id = "my_dataset.my_table"

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.PARQUET,
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
)

uri = "gs://my-bucket/data/*.parquet"
load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
load_job.result()  # Wait for completion
```

**Batch Loading Best Practices**
✅ Load **10 GB - 1 TB** per load job (optimal)
✅ Use **wildcard URIs** to load multiple files
✅ Compress files (gzip for CSV/JSON)
✅ Use **autodetect** for schema (testing only)
✅ Explicitly specify schema in production
✅ Use **WRITE_APPEND** for incremental loads
✅ Use **WRITE_TRUNCATE** for full refreshes

### Streaming Inserts

**When to Use Streaming**
✅ **Real-time analytics**: Dashboard updates
✅ **Low-latency requirements**: < 1 second
✅ **Small batches**: Individual records or small batches

```python
# Streaming insert
from google.cloud import bigquery

client = bigquery.Client()
table_id = "my_dataset.my_table"

rows_to_insert = [
    {"user_id": "123", "event_type": "click", "timestamp": "2025-12-25T12:00:00"},
    {"user_id": "456", "event_type": "view", "timestamp": "2025-12-25T12:01:00"},
]

errors = client.insert_rows_json(table_id, rows_to_insert)
if errors:
    print(f"Errors: {errors}")
```

**Streaming Best Practices**
✅ Batch streaming inserts (up to 10,000 rows)
✅ Use **insertId** for deduplication
✅ Handle errors and retries
✅ Monitor streaming costs (higher than batch)
✅ Data available immediately but not in long-term storage for 90 minutes

**Cost Comparison**
- **Batch loading**: Free
- **Streaming**: $0.01 per 200 MB (more expensive)

### Storage Write API (Recommended for Streaming)
✅ **Lower cost**: 50% cheaper than legacy streaming
✅ **Higher throughput**: Better performance
✅ **Exactly-once semantics**: Automatic deduplication
✅ **Committed vs Pending**: Immediate vs buffered

```python
# Use Storage Write API via Dataflow or client libraries
# More efficient than legacy streaming inserts
```

---

## 5. Interactive vs Batch Query Comparison

BigQuery offers two query execution modes: **Interactive (on-demand)** and **Batch**. Understanding when to use each mode is critical for cost optimization and resource management.

### Overview of Query Modes

#### Interactive Queries (Default)
**Characteristics:**
- Executed immediately with high priority
- Results returned as soon as possible (typically seconds to minutes)
- Uses on-demand slot allocation from shared pool
- Subject to concurrent query limits per project
- Ideal for ad-hoc analysis and user-facing applications

#### Batch Queries
**Characteristics:**
- Queued and executed when resources become available
- Lower priority than interactive queries
- May wait in queue during peak times (up to 24 hours maximum)
- Still counted against on-demand slot usage and billing
- Free from concurrent query slot limits
- Ideal for ETL, scheduled reports, non-urgent analytics

### Key Differences

| Feature | Interactive Query | Batch Query |
|---------|------------------|-------------|
| **Priority** | High (default) | Low |
| **Execution** | Immediate | Queued, executes when resources available |
| **Typical Latency** | Seconds to minutes | Minutes to hours (depends on queue) |
| **Slot Allocation** | Shared on-demand pool | Opportunistic allocation |
| **Concurrent Limits** | 100 concurrent queries per project | Unlimited (queue depth) |
| **Max Queue Time** | N/A | 24 hours |
| **Cost** | Standard on-demand rates | Same as interactive (still billed) |
| **Use Case** | Dashboards, ad-hoc queries, APIs | ETL, batch processing, scheduled jobs |
| **Timeout** | 6 hours max | 6 hours max (after execution starts) |
| **Reservation Slots** | Uses reserved slots if available | Uses reserved slots if available |

### When to Use Interactive Queries

✅ **User-Facing Applications**
- Dashboards and reporting tools (Looker, Looker Studio)
- Web applications with real-time query requirements
- Interactive notebooks (Jupyter, Colab)

✅ **Ad-Hoc Analysis**
- Data exploration and discovery
- Business analyst queries
- Data science experimentation

✅ **Time-Sensitive Workloads**
- Real-time alerting systems
- APIs that require low-latency responses (<10 seconds)
- Customer-facing analytics features

✅ **Short-Running Queries**
- Simple aggregations (<1 minute execution)
- Queries scanning small datasets (<100 GB)

**Example: Interactive Query**
```python
from google.cloud import bigquery

client = bigquery.Client()

# Interactive query (default priority=INTERACTIVE)
query = """
    SELECT 
        DATE(order_timestamp) as order_date,
        COUNT(*) as order_count,
        SUM(total_amount) as revenue
    FROM `project.dataset.orders`
    WHERE DATE(order_timestamp) = CURRENT_DATE()
    GROUP BY order_date
"""

# Execute with default interactive priority
query_job = client.query(query)
results = query_job.result()  # Waits for completion

for row in results:
    print(f"Date: {row.order_date}, Orders: {row.order_count}, Revenue: ${row.revenue}")
```

### When to Use Batch Queries

✅ **ETL Pipelines**
- Nightly data transformations
- Large-scale data processing jobs
- Multi-stage data pipelines with Cloud Composer/Airflow

✅ **Scheduled Reports**
- Daily/weekly/monthly report generation
- Email reports sent during off-peak hours
- Pre-computed aggregations for dashboards

✅ **Long-Running Queries**
- Queries processing TBs of data (>1 TB)
- Complex joins across multiple large tables
- Historical data backfill operations

✅ **Cost Optimization**
- Non-urgent analytical workloads
- Queries that can tolerate delays
- Jobs running during off-peak hours

✅ **High-Volume Workloads**
- More than 100 concurrent queries needed
- Avoiding interactive query slot contention
- Bulk data processing tasks

**Example: Batch Query**
```python
from google.cloud import bigquery

client = bigquery.Client()

# Configure batch query
job_config = bigquery.QueryJobConfig(
    priority=bigquery.QueryPriority.BATCH,
    use_query_cache=True,
    labels={"job_type": "etl", "team": "analytics"}
)

query = """
    CREATE OR REPLACE TABLE `project.dataset.customer_aggregates` AS
    SELECT 
        customer_id,
        COUNT(DISTINCT order_id) as total_orders,
        SUM(total_amount) as lifetime_value,
        MAX(order_timestamp) as last_order_date,
        DATE_DIFF(CURRENT_DATE(), DATE(MAX(order_timestamp)), DAY) as days_since_last_order
    FROM `project.dataset.orders`
    GROUP BY customer_id
"""

# Execute as batch query
query_job = client.query(query, job_config=job_config)

print(f"Job ID: {query_job.job_id}")
print(f"Job State: {query_job.state}")
print(f"Priority: BATCH")

# Optional: Wait for completion
query_job.result()
print(f"Batch job completed. Rows written: {query_job.num_dml_affected_rows}")
```

### Performance Considerations

#### Interactive Query Performance Factors
**What Affects Performance:**
- Query complexity and data volume scanned
- Number of concurrent queries in project
- Partition/cluster pruning effectiveness
- Join strategies and shuffling requirements
- BI Engine availability (for eligible queries)

**Optimization Tips:**
✅ Enable **BI Engine** for sub-second dashboard queries
✅ Use **materialized views** for frequently accessed data
✅ Implement **query caching** (24-hour default)
✅ Design tables with **partitioning and clustering**
✅ Limit result set size with appropriate filters

#### Batch Query Performance Factors
**What Affects Performance:**
- Queue wait time (depends on resource availability)
- Query complexity once execution begins (same as interactive)
- Time of day (less queue during off-peak hours)

**Optimization Tips:**
✅ Schedule batch jobs during **off-peak hours** (e.g., 2-6 AM in your region)
✅ Use **Cloud Composer** to orchestrate dependencies
✅ Monitor queue time with **INFORMATION_SCHEMA.JOBS**
✅ Break large jobs into **smaller parallelizable tasks**
✅ Use **CREATE TABLE AS SELECT** for intermediate results

### Cost Implications

#### On-Demand Pricing
Both interactive and batch queries use the **same pricing model**:
- **$6.25 per TB** processed (US multi-region)
- **First 1 TB per month**: Free
- Pricing varies by region

**Key Insight:** Batch queries do NOT save money on a per-TB basis, but they can:
- Reduce slot contention, improving overall efficiency
- Enable better scheduling to avoid peak-time resource costs
- Allow more queries within flat-rate reservation budgets

#### Flat-Rate Pricing (Reservations)
If using **BigQuery Editions** or **slot reservations**:
- Both query types consume **slots** from your reservation
- Batch queries can fill unused capacity without affecting interactive workloads
- Better slot utilization = better ROI on reservation

```sql
-- Check slot usage by priority
SELECT 
    priority,
    COUNT(*) as query_count,
    SUM(total_slot_ms) / 1000 / 3600 as total_slot_hours,
    AVG(total_slot_ms) / 1000 as avg_slot_seconds
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE DATE(creation_time) = CURRENT_DATE()
    AND job_type = 'QUERY'
    AND state = 'DONE'
GROUP BY priority
ORDER BY total_slot_hours DESC;
```

### Monitoring and Troubleshooting

#### Check Query Priority
```sql
-- Query job history with priority information
SELECT 
    job_id,
    user_email,
    priority,
    state,
    total_bytes_processed / POW(10, 12) as tb_processed,
    TIMESTAMP_DIFF(end_time, start_time, SECOND) as duration_seconds,
    TIMESTAMP_DIFF(start_time, creation_time, SECOND) as queue_wait_seconds,
    error_result.message as error_message
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE DATE(creation_time) = CURRENT_DATE()
    AND job_type = 'QUERY'
ORDER BY creation_time DESC
LIMIT 100;
```

#### Monitor Batch Query Queue Times
```sql
-- Analyze batch query queue performance
SELECT 
    DATE(creation_time) as job_date,
    EXTRACT(HOUR FROM creation_time) as hour,
    COUNT(*) as batch_queries,
    AVG(TIMESTAMP_DIFF(start_time, creation_time, SECOND)) as avg_queue_wait_seconds,
    MAX(TIMESTAMP_DIFF(start_time, creation_time, SECOND)) as max_queue_wait_seconds,
    PERCENTILE_CONT(TIMESTAMP_DIFF(start_time, creation_time, SECOND), 0.95) 
        OVER (PARTITION BY DATE(creation_time)) as p95_queue_wait_seconds
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE job_type = 'QUERY'
    AND priority = 'BATCH'
    AND state = 'DONE'
    AND DATE(creation_time) >= CURRENT_DATE() - 7
GROUP BY job_date, hour
ORDER BY job_date DESC, hour;
```

#### Set Up Alerting for Long Queue Times
```python
# Cloud Monitoring alert policy for batch query queue times
from google.cloud import monitoring_v3

client = monitoring_v3.AlertPolicyServiceClient()
project_name = f"projects/{project_id}"

# Alert if batch queries wait >30 minutes in queue
alert_policy = monitoring_v3.AlertPolicy(
    display_name="BigQuery Batch Query Long Queue Time",
    conditions=[{
        "display_name": "Queue wait time >30 minutes",
        "condition_threshold": {
            "filter": 'resource.type="bigquery_project" AND metric.type="bigquery.googleapis.com/job/queue_time"',
            "comparison": "COMPARISON_GT",
            "threshold_value": 1800,  # 30 minutes in seconds
            "duration": {"seconds": 300},  # 5-minute window
        }
    }],
    notification_channels=[notification_channel_id],
)

policy = client.create_alert_policy(name=project_name, alert_policy=alert_policy)
```

### Best Practices Summary

#### Interactive Query Best Practices
✅ Use for user-facing applications and dashboards
✅ Enable **BI Engine** for repeated queries (<100 GB scanned)
✅ Implement **query caching** to avoid redundant processing
✅ Set **query timeouts** to prevent runaway queries
✅ Monitor **concurrent query limits** (100 per project)
✅ Use **parameterized queries** to improve cache hit rates
✅ Apply **row-level filters** early in query execution
✅ Consider **materialized views** for frequently accessed aggregations

#### Batch Query Best Practices
✅ Use for **ETL pipelines** and scheduled data processing
✅ Schedule during **off-peak hours** to minimize queue time
✅ Set appropriate **job labels** for tracking and cost attribution
✅ Monitor **queue wait times** and adjust scheduling
✅ Use **Cloud Composer/Airflow** for dependency management
✅ Break large jobs into **parallelizable smaller tasks**
✅ Set **max_bytes_billed** to prevent cost overruns
✅ Use **CREATE TABLE AS SELECT** to persist intermediate results
✅ Implement **retry logic** with exponential backoff

### Common Anti-Patterns

❌ **Using interactive for long-running ETL**: Wastes high-priority slots
❌ **Using batch for real-time dashboards**: Unacceptable latency
❌ **Not monitoring queue times**: Batch jobs may wait 24 hours
❌ **Running 100+ concurrent interactive queries**: Hits project limits
❌ **No retry logic for batch queries**: Jobs can fail after long waits
❌ **Not using labels to track job types**: Poor cost visibility
❌ **Scheduling all batch jobs at same time**: Creates queue congestion

### Migration Strategy: Interactive to Batch

If you're experiencing slot contention or hitting concurrent query limits:

**Step 1: Identify Candidates**
```sql
-- Find queries suitable for batch conversion
SELECT 
    user_email,
    query,
    COUNT(*) as execution_count,
    AVG(total_slot_ms) / 1000 / 3600 as avg_slot_hours,
    AVG(TIMESTAMP_DIFF(end_time, start_time, SECOND)) as avg_duration_seconds,
    SUM(total_bytes_processed) / POW(10, 12) as total_tb_processed
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE DATE(creation_time) >= CURRENT_DATE() - 30
    AND job_type = 'QUERY'
    AND priority = 'INTERACTIVE'
    AND state = 'DONE'
GROUP BY user_email, query
HAVING avg_duration_seconds > 300  -- Queries longer than 5 minutes
ORDER BY avg_slot_hours DESC
LIMIT 50;
```

**Step 2: Convert to Batch**
- Update application code to set `priority=BATCH`
- Add monitoring for queue wait times
- Implement retry logic for transient failures
- Test during expected queue conditions

**Step 3: Monitor and Optimize**
- Track queue times and adjust scheduling
- Use Cloud Composer for complex dependencies
- Monitor slot utilization improvements
- Validate cost savings with flat-rate pricing

---

## 6. BigQuery ML Best Practices

### Model Creation
```sql
-- Create logistic regression model
CREATE OR REPLACE MODEL my_dataset.churn_model
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['churned'],
  max_iterations=10
) AS
SELECT 
  user_age,
  account_balance,
  num_transactions,
  churned
FROM my_dataset.customer_features
WHERE date >= '2024-01-01';
```

### Model Types
- **Linear Regression**: Numerical predictions
- **Logistic Regression**: Binary classification
- **Multiclass Logistic Regression**: Multi-class classification
- **K-Means**: Clustering
- **Matrix Factorization**: Recommendations
- **Time Series**: ARIMA_PLUS for forecasting
- **Boosted Trees**: XGBoost for classification/regression
- **DNN**: Deep neural networks
- **AutoML**: Automatic model selection

### Model Evaluation
```sql
-- Evaluate model
SELECT * FROM ML.EVALUATE(MODEL my_dataset.churn_model);

-- Get feature importance
SELECT * FROM ML.FEATURE_INFO(MODEL my_dataset.churn_model);

-- Make predictions
SELECT 
  user_id,
  predicted_churned,
  predicted_churned_probs
FROM ML.PREDICT(MODEL my_dataset.churn_model, (
  SELECT * FROM my_dataset.new_customers
));
```

### BQML Best Practices
✅ Use **train/test splits** for validation
✅ Start with **simple models** (logistic regression)
✅ Use **AutoML** for automatic model selection
✅ Monitor **model performance** over time
✅ Retrain models regularly with **fresh data**
✅ Use **TRANSFORM** clause for feature engineering

---

## 7. Security & Access Control

### IAM Roles
✅ Use **predefined roles** when possible
✅ Implement **least privilege** access
✅ Use **dataset-level** and **table-level** permissions

**Common Roles**
- `roles/bigquery.dataViewer`: Read data only
- `roles/bigquery.dataEditor`: Read and write data
- `roles/bigquery.user`: Run queries, limited dataset access
- `roles/bigquery.jobUser`: Run jobs only
- `roles/bigquery.admin`: Full control

### Row-Level Security
```sql
-- Create row access policy
CREATE ROW ACCESS POLICY regional_filter
ON my_dataset.sales
GRANT TO ('group:analysts@company.com')
FILTER USING (region = 'US');

-- Users see only their region's data
SELECT * FROM my_dataset.sales;  -- Automatically filtered
```

### Column-Level Security
```sql
-- Create taxonomy and policy tag
-- In Data Catalog, create:
-- Taxonomy: "SensitiveData"
-- Policy Tag: "PII"

-- Apply to column
ALTER TABLE my_dataset.customers
ALTER COLUMN email SET OPTIONS (policy_tags=['projects/PROJECT/locations/LOCATION/taxonomies/TAXONOMY/policyTags/TAG']);

-- Grant access to policy tag
-- In Data Catalog, grant access to specific groups
```

### Encryption

**Storage-Level Encryption**
✅ **Default encryption**: Google-managed keys (automatic)
✅ **CMEK**: Customer-managed encryption keys
✅ **Encryption in transit**: TLS (automatic)

```sql
-- Create table with CMEK (encrypts entire table at rest)
CREATE TABLE my_dataset.encrypted_table (
  id INT64,
  data STRING
)
OPTIONS(
  kms_key_name='projects/PROJECT/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY'
);
```

### Application-Layer Encryption with AEAD Functions

**Understanding AEAD (Authenticated Encryption with Associated Data)**

AEAD provides application-layer encryption for sensitive data within BigQuery tables. Unlike CMEK which encrypts entire tables, AEAD encrypts individual column values.

**Key Benefits of AEAD**
✅ **Column-level encryption**: Encrypt specific sensitive fields
✅ **Deterministic or non-deterministic**: Choose based on needs
✅ **Key management**: Integration with Cloud KMS
✅ **Searchable encryption**: Deterministic allows equality searches
✅ **Additional authentication**: Associated data (AAD) prevents tampering
✅ **Compliance**: Meet regulatory requirements for sensitive data

**AEAD Function Types**

1. **AEAD.ENCRYPT**: Standard (non-deterministic) encryption
2. **KEYS.NEW_KEYSET**: Generate keysets for encryption
3. **KEYS.ADD_KEY_FROM_RAW_BYTES**: Create keys from raw bytes

**Deterministic vs Non-Deterministic Encryption**

| Feature | Deterministic | Non-Deterministic |
|---------|--------------|-------------------|
| **Same input → Same output** | ✅ Yes | ❌ No (different each time) |
| **Searchable** | ✅ Can use = operator | ❌ Cannot search encrypted values |
| **Security** | ⚠️ Less secure (pattern analysis) | ✅ More secure |
| **Use case** | Need to search/join | Maximum security, no search needed |

### Setting Up AEAD Encryption

**Step 1: Create KMS Keyring and Key**
```bash
# Create keyring
gcloud kms keyrings create bigquery-aead-keyring \
  --location=us-central1

# Create encryption key
gcloud kms keys create aead-encryption-key \
  --keyring=bigquery-aead-keyring \
  --location=us-central1 \
  --purpose=encryption

# Grant BigQuery access to the key
gcloud kms keys add-iam-policy-binding aead-encryption-key \
  --keyring=bigquery-aead-keyring \
  --location=us-central1 \
  --member="serviceAccount:bq-SERVICE-ACCOUNT@bigquery-encryption.iam.gserviceaccount.com" \
  --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

**Step 2: Create Keyset in BigQuery**
```sql
-- Create keyset table to store encryption keysets
CREATE TABLE my_dataset.keysets (
  keyset_name STRING,
  keyset BYTES
);

-- Generate and store a new keyset
DECLARE new_keyset BYTES;

SET new_keyset = KEYS.NEW_KEYSET('AEAD_AES_GCM_256');

INSERT INTO my_dataset.keysets (keyset_name, keyset)
VALUES ('customer_data_keyset', new_keyset);
```

**Step 3: Encrypt Data with AEAD**
```sql
-- Encrypt sensitive data when inserting
INSERT INTO my_dataset.customers (
  customer_id,
  name,
  email_encrypted,
  ssn_encrypted,
  credit_card_encrypted
)
SELECT
  customer_id,
  name,
  -- Non-deterministic encryption (most secure, not searchable)
  AEAD.ENCRYPT(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset'),
    email,
    customer_id  -- Additional Authenticated Data (AAD)
  ) AS email_encrypted,
  
  AEAD.ENCRYPT(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset'),
    ssn,
    customer_id
  ) AS ssn_encrypted,
  
  AEAD.ENCRYPT(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset'),
    credit_card_number,
    customer_id
  ) AS credit_card_encrypted
FROM my_dataset.customers_raw;
```

**Step 4: Decrypt Data with AEAD**
```sql
-- Decrypt when querying
SELECT
  customer_id,
  name,
  AEAD.DECRYPT_STRING(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset'),
    email_encrypted,
    customer_id  -- Must match AAD used during encryption
  ) AS email,
  
  AEAD.DECRYPT_STRING(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset'),
    ssn_encrypted,
    customer_id
  ) AS ssn
FROM my_dataset.customers
WHERE customer_id = '12345';
```

### Deterministic Encryption (Searchable)

**When to Use Deterministic Encryption**
✅ Need to search encrypted values (WHERE, JOIN)
✅ Need to group by encrypted columns
✅ Less sensitive data (e.g., product IDs, not SSNs)
⚠️ Accept lower security (pattern analysis possible)

**Create Deterministic Keyset**
```sql
-- Create deterministic keyset
DECLARE det_keyset BYTES;

SET det_keyset = KEYS.NEW_KEYSET('DETERMINISTIC_AEAD_AES_SIV_CMAC_256');

INSERT INTO my_dataset.keysets (keyset_name, keyset)
VALUES ('searchable_data_keyset', det_keyset);
```

**Encrypt with Deterministic Encryption**
```sql
-- Encrypt customer email for searchable queries
CREATE TABLE my_dataset.customers_searchable AS
SELECT
  customer_id,
  name,
  -- Deterministic encryption (searchable)
  KEYS.KEYSET_TO_JSON(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'searchable_data_keyset')
  ) AS keyset_json,
  
  DETERMINISTIC_ENCRYPT(
    KEYS.KEYSET_CHAIN(
      'gcp-kms://projects/PROJECT/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY',
      (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'searchable_data_keyset')
    ),
    email,
    customer_id  -- AAD
  ) AS email_encrypted
FROM my_dataset.customers_raw;
```

**Search Encrypted Data**
```sql
-- Search using encrypted value
DECLARE encrypted_email BYTES;

-- Encrypt the search term
SET encrypted_email = DETERMINISTIC_ENCRYPT(
  KEYS.KEYSET_CHAIN(
    'gcp-kms://projects/PROJECT/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY',
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'searchable_data_keyset')
  ),
  'user@example.com',  -- The email to search for
  NULL  -- No AAD for search
);

-- Query using encrypted value
SELECT
  customer_id,
  name,
  DETERMINISTIC_DECRYPT_STRING(
    KEYS.KEYSET_CHAIN(
      'gcp-kms://projects/PROJECT/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY',
      (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'searchable_data_keyset')
    ),
    email_encrypted,
    customer_id
  ) AS email
FROM my_dataset.customers_searchable
WHERE email_encrypted = encrypted_email;
```

### Advanced AEAD Patterns

**Pattern 1: Encryption with Cloud KMS Integration**
```sql
-- Use Cloud KMS wrapped keyset for better security
CREATE TABLE my_dataset.customers_kms_encrypted AS
SELECT
  customer_id,
  name,
  AEAD.ENCRYPT(
    KEYS.KEYSET_CHAIN(
      'gcp-kms://projects/PROJECT/locations/us-central1/keyRings/bigquery-aead-keyring/cryptoKeys/aead-encryption-key',
      (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset')
    ),
    email,
    customer_id
  ) AS email_encrypted,
  
  AEAD.ENCRYPT(
    KEYS.KEYSET_CHAIN(
      'gcp-kms://projects/PROJECT/locations/us-central1/keyRings/bigquery-aead-keyring/cryptoKeys/aead-encryption-key',
      (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset')
    ),
    ssn,
    customer_id
  ) AS ssn_encrypted
FROM my_dataset.customers_raw;

-- Decrypt with KMS
SELECT
  customer_id,
  AEAD.DECRYPT_STRING(
    KEYS.KEYSET_CHAIN(
      'gcp-kms://projects/PROJECT/locations/us-central1/keyRings/bigquery-aead-keyring/cryptoKeys/aead-encryption-key',
      (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset')
    ),
    email_encrypted,
    customer_id
  ) AS email
FROM my_dataset.customers_kms_encrypted;
```

**Pattern 2: Selective Column Encryption**
```sql
-- Encrypt only PII columns, leave non-sensitive data unencrypted
CREATE OR REPLACE TABLE my_dataset.customers_selective AS
SELECT
  customer_id,
  name,  -- Not encrypted (not sensitive)
  region,  -- Not encrypted
  account_created_date,  -- Not encrypted
  
  -- Encrypt PII
  AEAD.ENCRYPT(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset'),
    email,
    customer_id
  ) AS email_encrypted,
  
  AEAD.ENCRYPT(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset'),
    phone,
    customer_id
  ) AS phone_encrypted,
  
  AEAD.ENCRYPT(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset'),
    ssn,
    customer_id
  ) AS ssn_encrypted
FROM my_dataset.customers_raw;
```

**Pattern 3: Encryption with Different AAD**
```sql
-- Use different AAD for different security contexts
CREATE OR REPLACE TABLE my_dataset.transactions_encrypted AS
SELECT
  transaction_id,
  customer_id,
  
  -- Encrypt amount with transaction_id as AAD
  AEAD.ENCRYPT(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'transaction_keyset'),
    CAST(amount AS STRING),
    transaction_id  -- AAD: transaction context
  ) AS amount_encrypted,
  
  -- Encrypt credit card with customer_id as AAD
  AEAD.ENCRYPT(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'payment_keyset'),
    credit_card_number,
    customer_id  -- AAD: customer context
  ) AS card_encrypted,
  
  transaction_date
FROM my_dataset.transactions_raw;
```

**Pattern 4: Key Rotation**
```sql
-- Create new keyset for rotation
DECLARE new_keyset BYTES;
SET new_keyset = KEYS.NEW_KEYSET('AEAD_AES_GCM_256');

INSERT INTO my_dataset.keysets (keyset_name, keyset)
VALUES ('customer_data_keyset_v2', new_keyset);

-- Re-encrypt data with new keyset
CREATE OR REPLACE TABLE my_dataset.customers_reencrypted AS
SELECT
  customer_id,
  name,
  
  -- Decrypt with old key, encrypt with new key
  AEAD.ENCRYPT(
    (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset_v2'),
    AEAD.DECRYPT_STRING(
      (SELECT keyset FROM my_dataset.keysets WHERE keyset_name = 'customer_data_keyset'),
      email_encrypted,
      customer_id
    ),
    customer_id
  ) AS email_encrypted
FROM my_dataset.customers_encrypted;
```

### AEAD Best Practices

**Security Best Practices**
✅ **Use non-deterministic encryption** for highly sensitive data (SSN, credit cards)
✅ **Use deterministic only when necessary** (searchable columns)
✅ **Store keysets securely** with appropriate IAM controls
✅ **Use Cloud KMS integration** for enterprise-grade key management
✅ **Implement key rotation** regularly (annually or per policy)
✅ **Use meaningful AAD** (additional authenticated data) for context binding
✅ **Separate keysets** for different data sensitivity levels
✅ **Audit keyset access** using Cloud Audit Logs

**Performance Best Practices**
✅ **Encrypt at ingestion time** (one-time cost)
✅ **Decrypt only when needed** (not in every query)
✅ **Create views for decryption** to centralize logic
✅ **Use deterministic for JOIN columns** if search is needed
✅ **Consider partitioning** encrypted tables for performance
✅ **Avoid encrypting columns** used in GROUP BY (use deterministic if needed)

**Operational Best Practices**
✅ **Document encryption strategy** clearly
✅ **Maintain keyset inventory** with metadata
✅ **Test decryption before deleting old keysets**
✅ **Monitor keyset usage** via audit logs
✅ **Plan for key rotation** before implementation
✅ **Backup keysets securely** (encrypted backups)
✅ **Implement access controls** on keyset tables

### AEAD Use Cases

**Use Case 1: PII Protection**
```sql
-- Encrypt customer PII for GDPR compliance
CREATE TABLE my_dataset.customers_gdpr_compliant (
  customer_id STRING,
  name STRING,
  email_encrypted BYTES,  -- Encrypted email
  phone_encrypted BYTES,  -- Encrypted phone
  address_encrypted BYTES,  -- Encrypted address
  consent_date DATE,
  data_retention_date DATE
);
```

**Use Case 2: Healthcare Data (HIPAA)**
```sql
-- Encrypt patient health information
CREATE TABLE my_dataset.patient_records (
  patient_id STRING,
  name STRING,  -- Not PHI if de-identified
  diagnosis_encrypted BYTES,  -- Encrypted diagnosis
  treatment_encrypted BYTES,  -- Encrypted treatment
  medication_encrypted BYTES,  -- Encrypted medications
  insurance_number_encrypted BYTES,  -- Encrypted insurance
  record_date DATE
);
```

**Use Case 3: Financial Data (PCI DSS)**
```sql
-- Encrypt payment card data
CREATE TABLE my_dataset.payment_cards (
  card_token STRING,  -- Unencrypted token (for reference)
  card_number_encrypted BYTES,  -- Encrypted card number
  cvv_encrypted BYTES,  -- Encrypted CVV
  cardholder_name_encrypted BYTES,  -- Encrypted name
  expiry_date_encrypted BYTES,  -- Encrypted expiry
  created_timestamp TIMESTAMP
);
```

**Use Case 4: Searchable Encrypted Logs**
```sql
-- Encrypt user IDs but keep them searchable
CREATE TABLE my_dataset.audit_logs_encrypted (
  log_id STRING,
  timestamp TIMESTAMP,
  user_id_encrypted BYTES,  -- Deterministic encryption (searchable)
  action STRING,  -- Not encrypted (not sensitive)
  resource STRING,  -- Not encrypted
  ip_address_encrypted BYTES,  -- Non-deterministic encryption
  user_agent_encrypted BYTES  -- Non-deterministic encryption
);
```

### AEAD Limitations and Considerations

**Limitations**
❌ **Cannot use in WHERE with non-deterministic**: Must decrypt all rows
❌ **Cannot index encrypted columns**: Impacts query performance
❌ **Cannot use in JOIN with non-deterministic**: Must decrypt first
❌ **Increased storage**: Encrypted values larger than plaintext
❌ **Decryption overhead**: CPU cost for decryption operations

**Considerations**
⚠️ **Key management complexity**: Must manage keysets carefully
⚠️ **Backup strategy**: Encrypted data useless without keys
⚠️ **Migration complexity**: Re-encrypting large tables takes time
⚠️ **Query performance**: Decryption adds latency
⚠️ **Testing requirements**: Must test encryption/decryption thoroughly

### Monitoring AEAD Usage

**Audit Keyset Access**
```sql
-- Query audit logs for keyset access
SELECT
  timestamp,
  protopayload_auditlog.authenticationInfo.principalEmail as user,
  protopayload_auditlog.resourceName as resource,
  protopayload_auditlog.methodName as method
FROM `project.dataset.cloudaudit_googleapis_com_data_access`
WHERE protopayload_auditlog.resourceName LIKE '%keysets%'
  AND DATE(timestamp) >= CURRENT_DATE() - 7
ORDER BY timestamp DESC;
```

**Monitor Encryption Operations**
```sql
-- Track encryption/decryption usage in queries
SELECT
  job_id,
  user_email,
  query,
  total_bytes_processed,
  total_slot_ms
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE query LIKE '%AEAD.ENCRYPT%' OR query LIKE '%AEAD.DECRYPT%'
  AND DATE(creation_time) >= CURRENT_DATE() - 7
ORDER BY creation_time DESC;
```

### Audit Logging
✅ Enable **Data Access logs** for sensitive datasets
✅ Export logs to **BigQuery** for analysis
✅ Monitor **query patterns** and access

```sql
-- Analyze audit logs
SELECT 
  protopayload_auditlog.authenticationInfo.principalEmail as user,
  protopayload_auditlog.resourceName as resource,
  COUNT(*) as query_count
FROM `project.dataset.cloudaudit_googleapis_com_data_access`
WHERE DATE(timestamp) = CURRENT_DATE()
GROUP BY user, resource
ORDER BY query_count DESC;
```

---

## 8. Performance Monitoring

### Query Execution Plan
```sql
-- View execution plan
-- In BigQuery Console: Check "Execution details" tab after query

-- Or use INFORMATION_SCHEMA
SELECT 
  job_id,
  query,
  total_bytes_processed,
  total_slot_ms,
  total_bytes_billed
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE DATE(creation_time) = CURRENT_DATE()
ORDER BY total_bytes_processed DESC
LIMIT 10;
```

### Slot Usage Monitoring
```sql
-- Monitor slot usage
SELECT 
  TIMESTAMP_TRUNC(period_start, HOUR) as hour,
  project_id,
  AVG(total_slot_ms) / 1000 / 60 as avg_slot_minutes
FROM `region-us`.INFORMATION_SCHEMA.JOBS_TIMELINE_BY_PROJECT
WHERE DATE(period_start) = CURRENT_DATE()
GROUP BY hour, project_id
ORDER BY hour DESC;
```

### Table Statistics
```sql
-- Get table size and row count
SELECT 
  table_name,
  row_count,
  size_bytes / POW(10, 9) as size_gb,
  TIMESTAMP_MILLIS(last_modified_time) as last_modified
FROM my_dataset.INFORMATION_SCHEMA.TABLES
ORDER BY size_bytes DESC;

-- Get partition information
SELECT 
  table_name,
  partition_id,
  total_rows,
  total_logical_bytes / POW(10, 9) as size_gb
FROM my_dataset.INFORMATION_SCHEMA.PARTITIONS
WHERE table_name = 'my_partitioned_table'
ORDER BY partition_id DESC;
```

---

## 9. Advanced Features

### Federated Queries
```sql
-- Query Cloud Storage directly
SELECT * FROM 
EXTERNAL_QUERY(
  'projects/PROJECT/locations/LOCATION/connections/CONNECTION',
  '''SELECT * FROM table'''
);

-- Query Cloud SQL
SELECT * FROM 
EXTERNAL_QUERY(
  'cloud_sql_connection',
  '''SELECT * FROM mysql_table WHERE date = '2025-12-25' '''
);

-- Query Cloud Spanner
SELECT * FROM 
EXTERNAL_QUERY(
  'spanner_connection',
  '''SELECT * FROM spanner_table LIMIT 1000'''
);
```

### User-Defined Functions (UDFs)
```sql
-- JavaScript UDF
CREATE TEMP FUNCTION multiply(x FLOAT64, y FLOAT64)
RETURNS FLOAT64
LANGUAGE js AS """
  return x * y;
""";

SELECT multiply(5, 10) as result;

-- SQL UDF (preferred - better performance)
CREATE TEMP FUNCTION calculate_discount(price FLOAT64, discount_pct FLOAT64)
RETURNS FLOAT64 AS (
  price * (1 - discount_pct / 100)
);

SELECT 
  product_id,
  price,
  calculate_discount(price, 10) as discounted_price
FROM my_dataset.products;
```

### Stored Procedures
```sql
-- Create stored procedure
CREATE OR REPLACE PROCEDURE my_dataset.refresh_daily_metrics(date_param DATE)
BEGIN
  -- Delete existing data for date
  DELETE FROM my_dataset.daily_metrics
  WHERE metric_date = date_param;
  
  -- Insert fresh data
  INSERT INTO my_dataset.daily_metrics
  SELECT 
    date_param as metric_date,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users
  FROM my_dataset.events
  WHERE DATE(event_timestamp) = date_param;
END;

-- Call procedure
CALL my_dataset.refresh_daily_metrics('2025-12-25');
```

### Scheduled Queries
```sql
-- Create scheduled query in BigQuery Console
-- Or use API:

-- Refresh materialized view daily
-- Schedule: "every day 02:00"
CREATE OR REPLACE TABLE my_dataset.daily_summary AS
SELECT 
  DATE(event_timestamp) as event_date,
  COUNT(*) as event_count
FROM my_dataset.events
WHERE DATE(event_timestamp) = CURRENT_DATE() - 1
GROUP BY event_date;
```

---

## 10. Data Transfer & Migration

### Data Transfer Service
```bash
# Transfer from S3 to BigQuery
gcloud transfer jobs create s3://my-bucket \
  --destination-table=my_dataset.my_table \
  --schedule="every day 03:00"
```

### Export Data
```sql
-- Export to Cloud Storage (Avro format recommended)
EXPORT DATA OPTIONS(
  uri='gs://my-bucket/export/*.avro',
  format='AVRO',
  overwrite=true
) AS
SELECT * FROM my_dataset.my_table
WHERE date >= '2025-01-01';
```

### Cross-Region Data Transfer
```sql
-- Copy table to different region
CREATE TABLE `us-dataset.table` 
AS SELECT * FROM `eu-dataset.table`;

-- Or use bq command-line tool
-- bq cp eu-dataset.table us-dataset.table
```

---

## 11. Common Anti-Patterns to Avoid

### Query Anti-Patterns
❌ **SELECT * FROM large_table**: Scans entire table
❌ **Not using partition filters**: Scans all partitions
❌ **Using LIMIT without ORDER BY**: Non-deterministic results
❌ **Nested subqueries**: Use CTEs (WITH clauses) instead
❌ **Self-joins for running totals**: Use window functions
❌ **Cross-joins**: Usually indicates query error
❌ **Not using approximate functions**: COUNT DISTINCT on billions of rows
❌ **Querying INFORMATION_SCHEMA frequently**: Cache results

### Data Loading Anti-Patterns
❌ **Loading CSV files for production**: Use Parquet or Avro instead
❌ **Streaming small volumes**: Use batch loading (free)
❌ **Not compressing files**: Wasting bandwidth and time
❌ **Loading without schema validation**: Data quality issues

### Schema Design Anti-Patterns
❌ **Not setting table expiration**: Wasting storage on temp tables
❌ **Using too many clustering columns**: Max 4 columns
❌ **Not partitioning large tables**: Higher query costs
❌ **UPDATE/DELETE on large tables**: Use partition replacement instead

### Security & Encryption Anti-Patterns
❌ **Storing sensitive data unencrypted**: Compliance violations
❌ **Using deterministic encryption for highly sensitive data**: Pattern analysis risk
❌ **Not using AAD with AEAD**: Missing tamper detection
❌ **Hardcoding encryption keys**: Security vulnerability
❌ **Not rotating encryption keys**: Long-term exposure risk
❌ **Encrypting all columns unnecessarily**: Performance and cost overhead
❌ **Not backing up keysets**: Risk of permanent data loss
❌ **Using same keyset for different sensitivity levels**: Poor security hygiene
❌ **Decrypting data in every query**: Unnecessary performance overhead
❌ **Not testing key rotation procedures**: Downtime risk
❌ **Storing keysets without access controls**: Unauthorized access risk

### Query Priority & Execution Anti-Patterns
❌ **Using interactive for long-running ETL jobs**: Wastes high-priority slots
❌ **Using batch for real-time dashboards**: Unacceptable latency
❌ **Not monitoring batch query queue times**: Jobs may wait 24 hours
❌ **Running 100+ concurrent interactive queries**: Hits project limits
❌ **No retry logic for batch queries**: Jobs can fail after long waits
❌ **Not using labels to track job types**: Poor cost visibility
❌ **Scheduling all batch jobs at same time**: Creates queue congestion
❌ **Not setting max_bytes_billed**: Risk of cost overruns

---

## 12. Query Optimization Checklist

- [ ] Select only required columns (no SELECT *)
- [ ] Use WHERE clause with partition filter
- [ ] Use clustering columns in WHERE clause
- [ ] Use APPROX functions for large aggregations
- [ ] Avoid ORDER BY on large result sets
- [ ] Use LIMIT for testing queries
- [ ] Pre-aggregate data in materialized views
- [ ] Use window functions instead of self-joins
- [ ] Use WITH clauses for readable complex queries
- [ ] Filter early in query (before joins)
- [ ] Use appropriate JOIN types
- [ ] Avoid UDFs when SQL functions available
- [ ] Use BI Engine for repeated dashboard queries
- [ ] Monitor bytes processed before running
- [ ] Cache frequently run queries
- [ ] Use selective decryption (only decrypt needed columns)
- [ ] Apply filters before decryption to reduce overhead

---

## 13. Cost Optimization Checklist

- [ ] Use partitioning on large tables (> 1 GB)
- [ ] Use clustering on high-cardinality columns
- [ ] Set partition_expiration_days
- [ ] Delete unused tables and datasets
- [ ] Use materialized views for frequent aggregations
- [ ] Use BI Engine for dashboards
- [ ] Batch load data (don't stream small volumes)
- [ ] Use Avro or Parquet file formats
- [ ] Set max_bytes_billed for cost control
- [ ] Monitor query costs with dry runs
- [ ] Use approximate functions
- [ ] Enable long-term storage discount (auto)
- [ ] Use slots efficiently (monitor INFORMATION_SCHEMA)
- [ ] Archive old data to Cloud Storage
- [ ] Use table expiration for temporary data

---

## 14. Security & Encryption Checklist

- [ ] Enable CMEK for table-level encryption
- [ ] Implement column-level security with policy tags
- [ ] Configure VPC Service Controls for private endpoints
- [ ] Use authorized views for row-level security
- [ ] Enable Data Catalog for metadata management
- [ ] Configure audit logs for all data access
- [ ] Implement least privilege IAM roles
- [ ] Set up DLP scans for PII detection
- [ ] Identify columns requiring AEAD encryption
- [ ] Choose appropriate encryption type (deterministic vs non-deterministic)
- [ ] Set up Cloud KMS with proper IAM permissions
- [ ] Generate and securely store keysets
- [ ] Test encryption/decryption performance
- [ ] Document encryption strategy and key locations
- [ ] Implement key rotation schedule (quarterly/annually)
- [ ] Backup keysets to secure location
- [ ] Configure monitoring for key usage
- [ ] Test disaster recovery procedures
- [ ] Verify compliance with regulations (GDPR, HIPAA, PCI DSS)
- [ ] Encrypt data in ETL pipelines before loading

---

## Additional Resources

- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Best Practices for Performance](https://cloud.google.com/bigquery/docs/best-practices-performance-overview)
- [Cost Optimization](https://cloud.google.com/bigquery/docs/best-practices-costs)
- [SQL Reference](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax)
- [Pricing Calculator](https://cloud.google.com/products/calculator)
- [Interactive vs Batch Queries](https://cloud.google.com/bigquery/docs/running-queries#batch)
- [Query Priority Documentation](https://cloud.google.com/bigquery/docs/query-priority)

---

*Last Updated: December 27, 2025*
*Version: 1.3*
