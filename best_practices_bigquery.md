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

## 5. BigQuery ML Best Practices

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

## 6. Security & Access Control

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
✅ **Default encryption**: Google-managed keys (automatic)
✅ **CMEK**: Customer-managed encryption keys
✅ **Encryption in transit**: TLS (automatic)

```sql
-- Create table with CMEK
CREATE TABLE my_dataset.encrypted_table (
  id INT64,
  data STRING
)
OPTIONS(
  kms_key_name='projects/PROJECT/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY'
);
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

## 7. Performance Monitoring

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

## 8. Advanced Features

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

## 9. Data Transfer & Migration

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

## 10. Common Anti-Patterns to Avoid

❌ **SELECT * FROM large_table**: Scans entire table
❌ **Not using partition filters**: Scans all partitions
❌ **Using LIMIT without ORDER BY**: Non-deterministic results
❌ **Nested subqueries**: Use CTEs (WITH clauses) instead
❌ **Self-joins for running totals**: Use window functions
❌ **Loading CSV files**: Use Parquet or Avro instead
❌ **Streaming small volumes**: Use batch loading (free)
❌ **Not setting table expiration**: Wasting storage on temp tables
❌ **Using too many clustering columns**: Max 4 columns
❌ **Querying INFORMATION_SCHEMA frequently**: Cache results
❌ **Not using approximate functions**: COUNT DISTINCT on billions of rows
❌ **UPDATE/DELETE on large tables**: Use partition replacement instead
❌ **Cross-joins**: Usually indicates query error

---

## 11. Query Optimization Checklist

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

---

## 12. Cost Optimization Checklist

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

## Additional Resources

- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Best Practices for Performance](https://cloud.google.com/bigquery/docs/best-practices-performance-overview)
- [Cost Optimization](https://cloud.google.com/bigquery/docs/best-practices-costs)
- [SQL Reference](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax)
- [Pricing Calculator](https://cloud.google.com/products/calculator)

---

*Last Updated: December 25, 2025*
