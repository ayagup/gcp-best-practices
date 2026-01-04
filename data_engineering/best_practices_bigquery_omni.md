# BigQuery Omni Best Practices

## Overview
BigQuery Omni is a multi-cloud analytics solution that enables you to analyze data across Google Cloud, AWS, and Azure using BigQuery's interface and capabilities. This document covers best practices for deploying and optimizing BigQuery Omni for cross-cloud analytics.

---

## 1. Multi-Cloud Architecture Design

### 1.1 Understanding BigQuery Omni
- **Multi-cloud query engine**: Run BigQuery queries on data in AWS S3 and Azure Blob Storage
- **Single interface**: Use familiar BigQuery SQL across all clouds
- **Unified governance**: Centralized access control and monitoring
- **No data movement**: Query data in place without copying

### 1.2 Use Cases
- **Cross-cloud analytics**: Analyze data distributed across multiple clouds
- **Cloud migration**: Query data during migration without moving it
- **Vendor diversification**: Avoid cloud vendor lock-in
- **Regulatory compliance**: Keep data in specific regions/clouds
- **Multi-cloud data lakes**: Unified analytics across distributed storage

### 1.3 Architecture Patterns

**Pattern 1: Hybrid Analytics**
```
AWS S3 Data → BigQuery Omni (us-east-1)
Azure Blob Data → BigQuery Omni (eastus)  } → Unified Analytics
GCS Data → BigQuery (us-central1)
```

**Pattern 2: Migration Analytics**
```
Legacy AWS Data → BigQuery Omni → Gradual Migration → BigQuery Native
```

---

## 2. Connection Setup and Configuration

### 2.1 AWS Connection Setup

**Create AWS Connection**
```sql
-- Create connection to AWS
CREATE EXTERNAL CONNECTION aws_connection
OPTIONS (
  type = 'CLOUD_RESOURCE',
  provider = 'AWS',
  location = 'aws-us-east-1'
);
```

**Required AWS IAM Permissions**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ]
    }
  ]
}
```

**Grant BigQuery Omni Access**
```bash
# Get the service account from connection
bq show --connection --location=aws-us-east-1 \
  --project_id=my-project aws_connection

# Configure AWS IAM role trust policy
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "accounts.google.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "accounts.google.com:sub": "SERVICE_ACCOUNT_ID"
        }
      }
    }
  ]
}
```

### 2.2 Azure Connection Setup

**Create Azure Connection**
```sql
-- Create connection to Azure
CREATE EXTERNAL CONNECTION azure_connection
OPTIONS (
  type = 'CLOUD_RESOURCE',
  provider = 'AZURE',
  location = 'azure-eastus'
);
```

**Configure Azure Storage Access**
```bash
# Assign Storage Blob Data Reader role
az role assignment create \
  --assignee SERVICE_ACCOUNT_ID \
  --role "Storage Blob Data Reader" \
  --scope /subscriptions/SUBSCRIPTION_ID/resourceGroups/RG_NAME/providers/Microsoft.Storage/storageAccounts/STORAGE_ACCOUNT
```

### 2.3 Connection Best Practices
- ✅ **Use dedicated connections** per cloud provider and region
- ✅ **Apply least privilege** IAM permissions
- ✅ **Separate connections by environment** (dev/staging/prod)
- ✅ **Document service account mappings**
- ✅ **Monitor connection health** regularly
- ❌ **Don't share connections** across security domains
- ❌ **Avoid overly permissive** IAM policies

---

## 3. External Table Management

### 3.1 Creating External Tables on AWS S3

**Parquet Format**
```sql
CREATE EXTERNAL TABLE `project.dataset.aws_sales_data`
(
  order_id STRING,
  customer_id STRING,
  order_date DATE,
  amount NUMERIC(10,2),
  region STRING
)
WITH CONNECTION `project.aws-us-east-1.aws_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['s3://my-bucket/sales/year=2024/*/*.parquet'],
  hive_partition_uri_prefix = 's3://my-bucket/sales',
  require_hive_partition_filter = true
);
```

**CSV Format**
```sql
CREATE EXTERNAL TABLE `project.dataset.aws_logs`
(
  timestamp TIMESTAMP,
  user_id STRING,
  action STRING,
  ip_address STRING
)
WITH CONNECTION `project.aws-us-east-1.aws_connection`
OPTIONS (
  format = 'CSV',
  uris = ['s3://my-bucket/logs/*.csv'],
  skip_leading_rows = 1,
  field_delimiter = ',',
  max_bad_records = 100
);
```

### 3.2 Creating External Tables on Azure Blob Storage

**Parquet on Azure**
```sql
CREATE EXTERNAL TABLE `project.dataset.azure_events`
(
  event_id STRING,
  event_type STRING,
  event_time TIMESTAMP,
  user_id STRING,
  metadata JSON
)
WITH CONNECTION `project.azure-eastus.azure_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['https://myaccount.blob.core.windows.net/container/events/*.parquet']
);
```

**Delta Lake on Azure**
```sql
CREATE EXTERNAL TABLE `project.dataset.azure_delta_table`
WITH CONNECTION `project.azure-eastus.azure_connection`
OPTIONS (
  format = 'DELTA_LAKE',
  uris = ['https://myaccount.blob.core.windows.net/container/delta_table']
);
```

### 3.3 Partitioned External Tables

**Time-Based Partitioning**
```sql
CREATE EXTERNAL TABLE `project.dataset.partitioned_events`
(
  event_id STRING,
  event_type STRING,
  amount NUMERIC
)
WITH PARTITION COLUMNS (
  event_date DATE,
  region STRING
)
WITH CONNECTION `project.aws-us-east-1.aws_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['s3://my-bucket/events/event_date=*/region=*/*.parquet'],
  hive_partition_uri_prefix = 's3://my-bucket/events',
  require_hive_partition_filter = true
);
```

---

## 4. Query Optimization

### 4.1 Partition Pruning
- **Always use partition filters** when querying partitioned tables
- **Benefits**: Reduces data scanned, lowers costs, improves performance

```sql
-- ✅ Good: Uses partition filter
SELECT *
FROM `project.dataset.partitioned_events`
WHERE event_date = '2024-12-01'
  AND region = 'us-east-1';

-- ❌ Bad: No partition filter (scans all data)
SELECT *
FROM `project.dataset.partitioned_events`
WHERE event_type = 'purchase';
```

### 4.2 File Format Selection
- **Parquet**: Best for analytical workloads (columnar storage)
- **ORC**: Good alternative to Parquet
- **Avro**: Good for schema evolution
- **CSV/JSON**: Avoid for large-scale analytics (slower, larger scans)

```sql
-- Performance comparison
-- Parquet: 10GB scanned, 2 seconds
-- CSV: 50GB scanned, 15 seconds (same data)
```

### 4.3 Query Best Practices
- ✅ **Select only needed columns** (especially with columnar formats)
- ✅ **Use WHERE clauses** to filter data early
- ✅ **Leverage partition pruning** aggressively
- ✅ **Use LIMIT** for exploratory queries
- ✅ **Cache frequently accessed data** in BigQuery native tables
- ❌ **Avoid SELECT *** on large external tables
- ❌ **Don't query without partition filters** on partitioned tables

### 4.4 Query Patterns

**Efficient Column Selection**
```sql
-- ✅ Good: Select specific columns
SELECT order_id, amount, region
FROM `project.dataset.aws_sales_data`
WHERE order_date = '2024-12-01';

-- ❌ Bad: Select all columns
SELECT *
FROM `project.dataset.aws_sales_data`
WHERE order_date = '2024-12-01';
```

**Materialized Results**
```sql
-- Create native BigQuery table from external data
CREATE TABLE `project.dataset.sales_summary` AS
SELECT 
  region,
  DATE_TRUNC(order_date, MONTH) as month,
  SUM(amount) as total_sales,
  COUNT(DISTINCT customer_id) as unique_customers
FROM `project.dataset.aws_sales_data`
WHERE order_date >= '2024-01-01'
GROUP BY region, month;
```

---

## 5. Performance Optimization

### 5.1 Data Layout Optimization
- **Use columnar formats**: Parquet, ORC for better performance
- **Partition strategically**: By date, region, or frequently filtered columns
- **Optimize file sizes**: 128MB to 1GB per file ideal
- **Avoid small files**: Consolidate to reduce metadata overhead
- **Use compression**: Snappy or GZIP for Parquet

### 5.2 Caching Strategies

**Materialize Hot Data**
```sql
-- Cache frequently queried data in BigQuery
CREATE TABLE `project.dataset.hot_data_cache`
PARTITION BY order_date
CLUSTER BY customer_id, region
AS
SELECT *
FROM `project.dataset.aws_sales_data`
WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY);

-- Schedule daily refresh
CREATE OR REPLACE PROCEDURE RefreshCache()
BEGIN
  TRUNCATE TABLE `project.dataset.hot_data_cache`;
  
  INSERT INTO `project.dataset.hot_data_cache`
  SELECT *
  FROM `project.dataset.aws_sales_data`
  WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY);
END;
```

### 5.3 Federated Query Optimization
- **Minimize cross-cloud joins**: Expensive due to network latency
- **Pre-aggregate in source cloud**: Reduce data transfer
- **Use query cache**: BigQuery caches external table queries
- **Monitor query execution**: Use INFORMATION_SCHEMA.JOBS

**Optimized Cross-Cloud Join**
```sql
-- ✅ Good: Aggregate first, then join
WITH aws_summary AS (
  SELECT region, SUM(amount) as total_sales
  FROM `project.dataset.aws_sales_data`
  WHERE order_date = CURRENT_DATE()
  GROUP BY region
),
gcp_summary AS (
  SELECT region, SUM(amount) as total_sales
  FROM `project.dataset.gcp_sales_data`
  WHERE order_date = CURRENT_DATE()
  GROUP BY region
)
SELECT 
  COALESCE(a.region, g.region) as region,
  IFNULL(a.total_sales, 0) as aws_sales,
  IFNULL(g.total_sales, 0) as gcp_sales
FROM aws_summary a
FULL OUTER JOIN gcp_summary g USING (region);
```

---

## 6. Cost Optimization

### 6.1 Cost Considerations
- **Query pricing**: Based on data scanned from external sources
- **Network egress**: Data transferred between clouds
- **Compute costs**: BigQuery Omni compute in each cloud region
- **Storage costs**: Managed by source cloud (AWS/Azure)

### 6.2 Cost Reduction Strategies

**Use Partitioning**
```sql
-- Without partition filter: Scans 1TB ($5)
SELECT * FROM external_table WHERE status = 'completed';

-- With partition filter: Scans 10GB ($0.05)
SELECT * FROM external_table 
WHERE date_partition = '2024-12-01' 
  AND status = 'completed';
```

**Materialize Frequently Accessed Data**
```sql
-- Instead of querying external table repeatedly
-- Create a BigQuery native table (one-time scan cost)
CREATE TABLE native_table AS
SELECT * FROM external_table
WHERE date_partition >= '2024-01-01';

-- Future queries on native_table are much cheaper
SELECT * FROM native_table WHERE status = 'completed';
```

### 6.3 Cost Monitoring

**Track Query Costs**
```sql
-- Monitor BigQuery Omni query costs
SELECT
  user_email,
  project_id,
  DATE(creation_time) as query_date,
  COUNT(*) as query_count,
  SUM(total_bytes_processed) / POW(10, 12) as tb_processed,
  SUM(total_bytes_processed) / POW(10, 12) * 5 as estimated_cost_usd
FROM `region-aws-us-east-1`.INFORMATION_SCHEMA.JOBS
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND statement_type = 'SELECT'
GROUP BY user_email, project_id, query_date
ORDER BY tb_processed DESC;
```

### 6.4 Cost Best Practices
- ✅ **Set query cost limits** using custom quotas
- ✅ **Use partition filters** to reduce data scanned
- ✅ **Cache hot data** in BigQuery native tables
- ✅ **Monitor usage patterns** and optimize accordingly
- ✅ **Use scheduled queries** to pre-compute aggregations
- ❌ **Avoid repeated queries** on large external tables
- ❌ **Don't scan unnecessary columns** with columnar formats

---

## 7. Security and Access Control

### 7.1 IAM and Authentication

**BigQuery Omni IAM Roles**
```bash
# Grant access to external connection
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:user@example.com" \
  --role="roles/bigquery.connectionUser"

# Grant query access
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:user@example.com" \
  --role="roles/bigquery.dataViewer"
```

### 7.2 Column-Level Security

**Implement Column Masking**
```sql
-- Create policy tag for sensitive data
CREATE POLICY TAG sensitive_data_tag;

-- Apply to external table columns
CREATE EXTERNAL TABLE `project.dataset.customer_data`
(
  customer_id STRING,
  name STRING,
  email STRING OPTIONS(policy_tags=['projects/PROJECT/locations/LOCATION/taxonomies/TAXONOMY/policyTags/TAG']),
  ssn STRING OPTIONS(policy_tags=['projects/PROJECT/locations/LOCATION/taxonomies/TAXONOMY/policyTags/TAG'])
)
WITH CONNECTION `project.aws-us-east-1.aws_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['s3://my-bucket/customers/*.parquet']
);
```

### 7.3 Row-Level Security

**Implement Row-Level Policies**
```sql
-- Create authorized view for row-level access
CREATE VIEW `project.dataset.regional_sales_view` AS
SELECT *
FROM `project.dataset.aws_sales_data`
WHERE region = (
  SELECT region 
  FROM `project.dataset.user_regions`
  WHERE user_email = SESSION_USER()
);

-- Grant access to view, not underlying table
GRANT `roles/bigquery.dataViewer` 
ON TABLE `project.dataset.regional_sales_view`
TO "user:analyst@example.com";
```

### 7.4 Encryption
- **Data in transit**: Encrypted by default (TLS)
- **Data at rest**: Managed by source cloud (AWS/Azure encryption)
- **Cross-cloud transfer**: Encrypted end-to-end
- **Keys**: Use cloud provider's KMS for source data

### 7.5 Security Best Practices
- ✅ **Use least privilege** IAM policies
- ✅ **Implement column-level security** for PII
- ✅ **Apply row-level security** for multi-tenant data
- ✅ **Audit access logs** regularly
- ✅ **Encrypt data at source** using cloud provider KMS
- ✅ **Use VPC Service Controls** for additional security
- ❌ **Don't grant broad access** to external connections
- ❌ **Avoid storing unencrypted PII** in external storage

---

## 8. Monitoring and Observability

### 8.1 Query Performance Monitoring

**Monitor Query Execution**
```sql
-- Analyze query performance
SELECT
  job_id,
  user_email,
  statement_type,
  start_time,
  end_time,
  TIMESTAMP_DIFF(end_time, start_time, SECOND) as duration_seconds,
  total_bytes_processed / POW(10, 9) as gb_processed,
  total_slot_ms / 1000 as slot_seconds
FROM `region-aws-us-east-1`.INFORMATION_SCHEMA.JOBS
WHERE DATE(creation_time) = CURRENT_DATE()
ORDER BY total_bytes_processed DESC
LIMIT 20;
```

### 8.2 Connection Health Monitoring

**Monitor External Connections**
```sql
-- Check connection status
SELECT
  connection_id,
  location,
  connection_type,
  creation_time,
  last_modified_time
FROM `project.region-aws-us-east-1`.INFORMATION_SCHEMA.CONNECTIONS;
```

### 8.3 Data Scan Monitoring

**Track Data Scanned by Table**
```sql
-- Monitor external table usage
SELECT
  referenced_tables[SAFE_OFFSET(0)].table_id as table_name,
  COUNT(*) as query_count,
  SUM(total_bytes_processed) / POW(10, 12) as tb_scanned,
  AVG(TIMESTAMP_DIFF(end_time, start_time, SECOND)) as avg_duration_sec
FROM `region-aws-us-east-1`.INFORMATION_SCHEMA.JOBS,
UNNEST(referenced_tables) as referenced_tables
WHERE DATE(creation_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY table_name
ORDER BY tb_scanned DESC;
```

### 8.4 Alerting

**Set Up Cloud Monitoring Alerts**
```yaml
# Alert on high query costs
displayName: "High BigQuery Omni Query Costs"
conditions:
  - displayName: "Daily cost exceeds threshold"
    conditionThreshold:
      filter: 'resource.type="bigquery_project"'
      comparison: COMPARISON_GT
      thresholdValue: 1000
      duration: 86400s
      aggregations:
        - alignmentPeriod: 3600s
          perSeriesAligner: ALIGN_SUM
```

### 8.5 Monitoring Best Practices
- ✅ **Monitor query performance** regularly
- ✅ **Track data scanned** per table and user
- ✅ **Set up cost alerts** to avoid surprises
- ✅ **Monitor connection failures** and errors
- ✅ **Create dashboards** for key metrics
- ✅ **Review slow queries** and optimize

---

## 9. Data Migration Patterns

### 9.1 Lift and Shift Pattern

**Query Data in Place**
```sql
-- Phase 1: Query data in AWS without moving
SELECT 
  product_id,
  SUM(quantity) as total_sold,
  AVG(price) as avg_price
FROM `project.dataset.aws_sales_external`
WHERE sale_date >= '2024-01-01'
GROUP BY product_id;
```

### 9.2 Incremental Migration

**Migrate Historical Data**
```sql
-- Step 1: Migrate historical data (one-time)
CREATE TABLE `project.dataset.sales_native`
PARTITION BY sale_date
CLUSTER BY product_id, region
AS
SELECT *
FROM `project.dataset.aws_sales_external`
WHERE sale_date < '2024-01-01';

-- Step 2: Query recent data from AWS, historical from BigQuery
SELECT *
FROM `project.dataset.sales_native`
WHERE sale_date < '2024-01-01'
UNION ALL
SELECT *
FROM `project.dataset.aws_sales_external`
WHERE sale_date >= '2024-01-01';
```

### 9.3 Hybrid Analytics Pattern

**Unified View Across Clouds**
```sql
-- Create view combining multiple clouds
CREATE VIEW `project.dataset.unified_sales` AS
SELECT *, 'aws' as source_cloud
FROM `project.dataset.aws_sales_external`
UNION ALL
SELECT *, 'azure' as source_cloud
FROM `project.dataset.azure_sales_external`
UNION ALL
SELECT *, 'gcp' as source_cloud
FROM `project.dataset.gcp_sales_native`;
```

---

## 10. Integration Patterns

### 10.1 With BigQuery Native Tables

**Federated Queries**
```sql
-- Join external and native tables
SELECT 
  e.customer_id,
  e.order_id,
  e.amount,
  c.customer_name,
  c.segment
FROM `project.dataset.aws_orders_external` e
JOIN `project.dataset.customers_native` c
  ON e.customer_id = c.customer_id
WHERE e.order_date = CURRENT_DATE();
```

### 10.2 With BI Tools

**Connect Looker/Looker Studio**
```sql
-- Create optimized view for BI
CREATE MATERIALIZED VIEW `project.dataset.sales_dashboard_mv`
PARTITION BY DATE_TRUNC(order_date, MONTH)
CLUSTER BY region, product_category
AS
SELECT
  order_date,
  region,
  product_category,
  SUM(amount) as total_sales,
  COUNT(DISTINCT customer_id) as unique_customers,
  COUNT(*) as order_count
FROM `project.dataset.aws_sales_external`
WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
GROUP BY order_date, region, product_category;
```

### 10.3 With Data Pipelines

**Scheduled ETL from External Sources**
```sql
-- Scheduled query to refresh native table
CREATE OR REPLACE TABLE `project.dataset.daily_summary`
PARTITION BY report_date
AS
SELECT
  CURRENT_DATE() as report_date,
  region,
  COUNT(*) as transactions,
  SUM(amount) as revenue
FROM `project.dataset.aws_transactions_external`
WHERE transaction_date = CURRENT_DATE()
GROUP BY region;
```

---

## 11. Troubleshooting

### 11.1 Common Issues

**Connection Failures**
```
Error: Connection to AWS S3 failed
Solution:
1. Verify IAM role trust relationship
2. Check service account permissions
3. Confirm S3 bucket permissions
4. Verify network connectivity
```

**Slow Query Performance**
```
Issue: Queries taking too long
Diagnosis:
- Check if partition filters are used
- Verify file format (prefer Parquet)
- Check file sizes (avoid small files)
- Monitor bytes scanned in job info

Solution:
- Add partition filters
- Consolidate small files
- Convert to columnar format
- Materialize hot data
```

**High Costs**
```
Issue: Unexpected BigQuery Omni charges
Diagnosis:
SELECT 
  user_email,
  SUM(total_bytes_processed) / POW(10, 12) as tb_processed
FROM `region-aws-us-east-1`.INFORMATION_SCHEMA.JOBS
WHERE DATE(creation_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY user_email
ORDER BY tb_processed DESC;

Solution:
- Identify heavy users/queries
- Implement partition filtering
- Materialize frequently accessed data
- Set query cost controls
```

### 11.2 Debugging Queries

**Enable Query Execution Details**
```sql
-- Check query execution plan
SELECT
  job_id,
  query,
  total_bytes_processed,
  total_slot_ms,
  cache_hit
FROM `region-aws-us-east-1`.INFORMATION_SCHEMA.JOBS
WHERE job_id = 'YOUR_JOB_ID';
```

---

## 12. Best Practices Summary

### 12.1 Design Best Practices
- ✅ **Use columnar formats** (Parquet, ORC) for external data
- ✅ **Partition data** by frequently filtered columns
- ✅ **Optimize file sizes** (128MB-1GB per file)
- ✅ **Implement proper IAM** with least privilege
- ✅ **Create materialized views** for hot data
- ✅ **Use connections per region** and cloud provider

### 12.2 Query Best Practices
- ✅ **Always use partition filters** on partitioned tables
- ✅ **Select specific columns** instead of SELECT *
- ✅ **Pre-aggregate data** before cross-cloud joins
- ✅ **Monitor query costs** regularly
- ✅ **Use query cache** for repeated queries
- ✅ **Implement incremental processing** where possible

### 12.3 Cost Best Practices
- ✅ **Materialize frequently queried data** in BigQuery native tables
- ✅ **Set up cost alerts** and quotas
- ✅ **Use partition pruning** aggressively
- ✅ **Consolidate small files** to reduce overhead
- ✅ **Monitor and optimize** heavy queries
- ✅ **Educate users** on cost-efficient query patterns

### 12.4 Security Best Practices
- ✅ **Implement least privilege** IAM policies
- ✅ **Use column and row-level security** for sensitive data
- ✅ **Encrypt data at rest** in source cloud
- ✅ **Audit access regularly** using logs
- ✅ **Use VPC Service Controls** for additional isolation
- ✅ **Rotate credentials** and review permissions periodically

---

## 13. Anti-Patterns to Avoid

### 13.1 Design Anti-Patterns
- ❌ **Using CSV/JSON** for large-scale analytics (use Parquet instead)
- ❌ **Creating many small files** (consolidate to 128MB-1GB)
- ❌ **Not partitioning large tables** (adds unnecessary cost)
- ❌ **Granting overly broad IAM permissions** (security risk)
- ❌ **Mixing data formats** in same table (inconsistent performance)

### 13.2 Query Anti-Patterns
- ❌ **SELECT * on large external tables** (scans unnecessary data)
- ❌ **No partition filters** on partitioned tables (expensive)
- ❌ **Repeated queries** on same external data (cache or materialize)
- ❌ **Complex joins across clouds** without pre-aggregation (slow and costly)
- ❌ **Not using query cache** (waste of compute)

### 13.3 Operational Anti-Patterns
- ❌ **No cost monitoring** (budget surprises)
- ❌ **No query optimization** (wasted resources)
- ❌ **Ignoring slow queries** (poor user experience)
- ❌ **Not documenting external table schemas** (maintenance issues)
- ❌ **No governance policies** (security and compliance risks)

---

## Quick Reference Checklist

### Initial Setup
- [ ] Enable BigQuery Omni in required regions
- [ ] Create external connections for AWS/Azure
- [ ] Configure IAM roles and trust relationships
- [ ] Grant appropriate permissions to users
- [ ] Set up audit logging
- [ ] Create initial external tables
- [ ] Test connectivity and queries
- [ ] Document connection configurations
- [ ] Establish cost monitoring
- [ ] Train users on best practices

### Ongoing Operations
- [ ] Monitor query performance weekly
- [ ] Review and optimize expensive queries
- [ ] Check cost trends and set alerts
- [ ] Audit IAM permissions quarterly
- [ ] Update external table schemas as needed
- [ ] Materialize hot data to native tables
- [ ] Consolidate small files regularly
- [ ] Review and optimize partition strategies
- [ ] Update documentation
- [ ] Conduct user training sessions

### Before Production Launch
- [ ] Partition large external tables
- [ ] Implement column and row-level security
- [ ] Create materialized views for dashboards
- [ ] Set up monitoring and alerting
- [ ] Document query patterns and best practices
- [ ] Establish cost budgets and quotas
- [ ] Test failover and disaster recovery
- [ ] Create runbooks for common issues
- [ ] Train support team
- [ ] Conduct security review

---

## Resources

### Official Documentation
- [BigQuery Omni Overview](https://cloud.google.com/bigquery/docs/omni-introduction)
- [BigQuery Omni on AWS](https://cloud.google.com/bigquery/docs/omni-aws-introduction)
- [BigQuery Omni on Azure](https://cloud.google.com/bigquery/docs/omni-azure-introduction)
- [External Tables](https://cloud.google.com/bigquery/docs/external-tables)
- [External Connections](https://cloud.google.com/bigquery/docs/external-data-cloud-storage)

### Tools and Integrations
- BigQuery Console for query execution
- Cloud Monitoring for observability
- IAM for access control
- Data Catalog for metadata management
- Looker/Looker Studio for visualization

### Pricing
- [BigQuery Omni Pricing](https://cloud.google.com/bigquery/pricing#omni)
- Network egress charges apply for cross-cloud data transfer

---

*Last Updated: December 27, 2025*
*Version: 1.0*
