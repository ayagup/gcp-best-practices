# BigQuery BigLake Best Practices

## Overview
BigLake is a storage engine that provides unified, fine-grained access control across data lakes (Cloud Storage, AWS S3, Azure Blob Storage) and BigQuery. It enables query acceleration, caching, and advanced security features for external data without moving it into BigQuery.

---

## 1. BigLake Architecture and Concepts

### 1.1 Understanding BigLake
- **Unified access control**: Single security model across multi-cloud data lakes
- **Query acceleration**: Performance optimization for external data
- **Metadata caching**: Faster query planning and execution
- **Column-level security**: Fine-grained access control on external data
- **Row-level security**: Filter data based on user identity
- **Dynamic data masking**: Protect sensitive data without copying

### 1.2 BigLake vs Standard External Tables

**Standard External Tables**
- Direct access to Cloud Storage
- No metadata caching
- Table-level security only
- No query acceleration
- No dynamic data masking

**BigLake Tables**
- Access via BigLake connection
- Metadata caching enabled
- Column and row-level security
- Query acceleration available
- Dynamic data masking supported
- Fine-grained access control

### 1.3 Use Cases
- **Data lake governance**: Unified security across storage systems
- **Performance optimization**: Accelerate queries on external data
- **PII protection**: Mask sensitive data without copying
- **Multi-cloud analytics**: Secure access to data across clouds
- **Zero-copy analytics**: Query data in place with security
- **Compliance**: Fine-grained access control for regulations

### 1.4 Architecture Pattern
```
Cloud Storage/S3/Azure Blob
    ↓
BigLake Connection (with service account)
    ↓
BigLake Table (metadata cache + security)
    ↓
BigQuery Query Engine
    ↓
Results (with security applied)
```

---

## 2. BigLake Connection Setup

### 2.1 Create BigLake Connection for Cloud Storage

**Basic Connection**
```sql
-- Create BigLake connection
CREATE EXTERNAL CONNECTION my_biglake_connection
OPTIONS (
  type = 'CLOUD_RESOURCE'
);
```

**Get Connection Service Account**
```bash
# Retrieve service account
bq show --connection --location=us \
  --project_id=my-project my_biglake_connection

# Output: Service account like:
# bqcx-123456789-abcd@gcp-sa-bigquery-condel.iam.gserviceaccount.com
```

**Grant Storage Access**
```bash
# Grant Cloud Storage read access
gcloud storage buckets add-iam-policy-binding gs://my-bucket \
  --member="serviceAccount:SERVICE_ACCOUNT" \
  --role="roles/storage.objectViewer"
```

### 2.2 Create BigLake Connection for AWS

**AWS Connection with Credentials**
```sql
CREATE EXTERNAL CONNECTION aws_biglake_connection
OPTIONS (
  type = 'AWS',
  properties = [
    ('roleArn', 'arn:aws:iam::ACCOUNT_ID:role/BigLakeRole'),
    ('roleSessionName', 'biglake-session')
  ]
);
```

**AWS IAM Role Configuration**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "glue:GetDatabase",
        "glue:GetTable",
        "glue:GetPartitions"
      ],
      "Resource": "*"
    }
  ]
}
```

### 2.3 Create BigLake Connection for Azure

**Azure Connection**
```sql
CREATE EXTERNAL CONNECTION azure_biglake_connection
OPTIONS (
  type = 'AZURE',
  properties = [
    ('storageAccountId', '/subscriptions/SUB_ID/resourceGroups/RG/providers/Microsoft.Storage/storageAccounts/ACCOUNT')
  ]
);
```

**Azure RBAC Configuration**
```bash
# Assign Storage Blob Data Reader
az role assignment create \
  --assignee SERVICE_ACCOUNT \
  --role "Storage Blob Data Reader" \
  --scope STORAGE_ACCOUNT_ID
```

### 2.4 Connection Best Practices
- ✅ **Use dedicated connections** per data source and region
- ✅ **Apply least privilege** to service accounts
- ✅ **Separate connections** by environment (dev/prod)
- ✅ **Document service account mappings**
- ✅ **Monitor connection health** regularly
- ✅ **Rotate credentials** periodically for AWS/Azure
- ❌ **Don't share connections** across security boundaries
- ❌ **Avoid overly permissive** IAM/RBAC roles

---

## 3. Creating BigLake Tables

### 3.1 BigLake Table on Cloud Storage

**Parquet Format**
```sql
CREATE EXTERNAL TABLE `project.dataset.customer_data`
(
  customer_id STRING,
  name STRING,
  email STRING,
  phone STRING,
  address STRING,
  credit_score INT64,
  account_balance NUMERIC(15,2)
)
WITH CONNECTION `project.us.my_biglake_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://my-bucket/customers/*.parquet'],
  metadata_cache_mode = 'AUTOMATIC',
  max_staleness = INTERVAL 4 HOUR
);
```

**Partitioned BigLake Table**
```sql
CREATE EXTERNAL TABLE `project.dataset.sales_data`
(
  transaction_id STRING,
  customer_id STRING,
  amount NUMERIC(10,2),
  product_id STRING
)
WITH PARTITION COLUMNS (
  transaction_date DATE,
  region STRING
)
WITH CONNECTION `project.us.my_biglake_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://my-bucket/sales/transaction_date=*/region=*/*.parquet'],
  hive_partition_uri_prefix = 'gs://my-bucket/sales',
  metadata_cache_mode = 'AUTOMATIC',
  max_staleness = INTERVAL 1 HOUR,
  require_partition_filter = true
);
```

### 3.2 BigLake Table on AWS S3

**S3 with Parquet**
```sql
CREATE EXTERNAL TABLE `project.dataset.aws_events`
(
  event_id STRING,
  event_type STRING,
  user_id STRING,
  event_data JSON,
  created_at TIMESTAMP
)
WITH CONNECTION `project.us.aws_biglake_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['s3://my-bucket/events/*.parquet'],
  metadata_cache_mode = 'AUTOMATIC'
);
```

**S3 with Hive Metastore**
```sql
CREATE EXTERNAL TABLE `project.dataset.hive_table`
WITH CONNECTION `project.us.aws_biglake_connection`
OPTIONS (
  format = 'HIVE',
  hive_database_name = 'my_database',
  hive_table_name = 'my_table',
  hive_uri = 'thrift://metastore-host:9083'
);
```

### 3.3 BigLake Table on Azure Blob Storage

**Azure with Delta Lake**
```sql
CREATE EXTERNAL TABLE `project.dataset.azure_delta`
WITH CONNECTION `project.us.azure_biglake_connection`
OPTIONS (
  format = 'DELTA_LAKE',
  uris = ['abfss://container@account.dfs.core.windows.net/delta-table'],
  metadata_cache_mode = 'AUTOMATIC'
);
```

### 3.4 Metadata Cache Configuration

**Cache Modes**
```sql
-- AUTOMATIC: BigQuery manages cache refresh
metadata_cache_mode = 'AUTOMATIC'

-- MANUAL: You control cache refresh
metadata_cache_mode = 'MANUAL'
```

**Staleness Configuration**
```sql
-- Allow 4 hours of staleness
max_staleness = INTERVAL 4 HOUR

-- Allow 30 minutes of staleness
max_staleness = INTERVAL 30 MINUTE

-- Always fresh (expensive)
max_staleness = INTERVAL 0 MINUTE
```

**Manual Cache Refresh**
```sql
-- Refresh metadata cache
ALTER TABLE `project.dataset.biglake_table`
REFRESH_METADATA_CACHE;
```

---

## 4. Security and Access Control

### 4.1 Column-Level Security

**Define Policy Tags**
```bash
# Create policy taxonomy
gcloud data-catalog taxonomies create sensitive_data \
  --location=us \
  --display-name="Sensitive Data Classification"

# Create policy tags
gcloud data-catalog taxonomies policy-tags create pii \
  --taxonomy=sensitive_data \
  --location=us \
  --display-name="PII Data"

gcloud data-catalog taxonomies policy-tags create financial \
  --taxonomy=sensitive_data \
  --location=us \
  --display-name="Financial Data"
```

**Apply to BigLake Table**
```sql
CREATE EXTERNAL TABLE `project.dataset.customers_secure`
(
  customer_id STRING,
  name STRING,
  email STRING OPTIONS(
    policy_tags = ['projects/PROJECT/locations/us/taxonomies/TAX_ID/policyTags/PII_TAG']
  ),
  ssn STRING OPTIONS(
    policy_tags = ['projects/PROJECT/locations/us/taxonomies/TAX_ID/policyTags/PII_TAG']
  ),
  income NUMERIC OPTIONS(
    policy_tags = ['projects/PROJECT/locations/us/taxonomies/TAX_ID/policyTags/FIN_TAG']
  )
)
WITH CONNECTION `project.us.my_biglake_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://my-bucket/customers/*.parquet']
);
```

**Grant Fine-Grained Access**
```bash
# Grant access to specific policy tag
gcloud data-catalog taxonomies policy-tags set-iam-policy PII_TAG \
  --policy-file=policy.json

# policy.json
{
  "bindings": [
    {
      "role": "roles/datacatalog.categoryFineGrainedReader",
      "members": [
        "user:analyst@example.com",
        "group:data-team@example.com"
      ]
    }
  ]
}
```

### 4.2 Row-Level Security

**Create Row Access Policy**
```sql
-- Create policy for regional access
CREATE ROW ACCESS POLICY regional_filter
ON `project.dataset.sales_data`
GRANT TO ('user:analyst@example.com')
FILTER USING (region = (
  SELECT region 
  FROM `project.dataset.user_regions`
  WHERE user_email = SESSION_USER()
));

-- Create policy for manager access
CREATE ROW ACCESS POLICY manager_access
ON `project.dataset.employee_data`
GRANT TO ('group:managers@example.com')
FILTER USING (
  employee_id IN (
    SELECT employee_id
    FROM `project.dataset.manager_assignments`
    WHERE manager_email = SESSION_USER()
  )
  OR SESSION_USER() = 'user:hr-admin@example.com'
);
```

**List and Manage Policies**
```sql
-- List row access policies
SELECT * 
FROM `project.dataset`.INFORMATION_SCHEMA.ROW_ACCESS_POLICIES
WHERE table_name = 'sales_data';

-- Drop row access policy
DROP ROW ACCESS POLICY regional_filter
ON `project.dataset.sales_data`;
```

### 4.3 Dynamic Data Masking

**Create Masked View**
```sql
CREATE VIEW `project.dataset.customers_masked` AS
SELECT
  customer_id,
  name,
  -- Mask email (show domain only)
  CONCAT('***', SUBSTR(email, STRPOS(email, '@'))) as email,
  -- Mask phone (show last 4 digits)
  CONCAT('XXX-XXX-', SUBSTR(phone, -4)) as phone,
  -- Hash SSN
  TO_HEX(SHA256(ssn)) as ssn_hash,
  -- Anonymize address
  CONCAT(SPLIT(address, ',')[OFFSET(1)], ', ***') as city_only,
  -- Round credit score to range
  CAST(FLOOR(credit_score / 50) * 50 AS INT64) as credit_score_range,
  -- Show balance in ranges
  CASE
    WHEN account_balance < 1000 THEN '< 1K'
    WHEN account_balance < 10000 THEN '1K-10K'
    WHEN account_balance < 100000 THEN '10K-100K'
    ELSE '100K+'
  END as balance_range
FROM `project.dataset.customer_data`;
```

**Conditional Masking Based on User**
```sql
CREATE VIEW `project.dataset.customers_conditional_mask` AS
SELECT
  customer_id,
  name,
  -- Show full email to authorized users, masked to others
  CASE
    WHEN SESSION_USER() IN (
      SELECT user_email 
      FROM `project.dataset.authorized_users`
    )
    THEN email
    ELSE CONCAT('***', SUBSTR(email, STRPOS(email, '@')))
  END as email,
  -- Full SSN for authorized, masked for others
  CASE
    WHEN SESSION_USER() IN (
      SELECT user_email 
      FROM `project.dataset.authorized_users`
    )
    THEN ssn
    ELSE CONCAT('XXX-XX-', SUBSTR(ssn, -4))
  END as ssn
FROM `project.dataset.customer_data`;
```

### 4.4 Encryption
- **Data at rest**: Encrypted by default in Cloud Storage/S3/Azure
- **Data in transit**: TLS encryption for all connections
- **Customer-managed encryption keys (CMEK)**: Supported for Cloud Storage
- **Client-side encryption**: Supported for additional security

**Using CMEK**
```bash
# Create Cloud KMS key
gcloud kms keys create biglake-key \
  --location=us-central1 \
  --keyring=biglake-keyring \
  --purpose=encryption

# Grant BigLake service account access
gcloud kms keys add-iam-policy-binding biglake-key \
  --location=us-central1 \
  --keyring=biglake-keyring \
  --member="serviceAccount:SERVICE_ACCOUNT" \
  --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

---

## 5. Performance Optimization

### 5.1 Metadata Caching

**Optimal Cache Settings**
```sql
-- High-frequency updates (near real-time)
CREATE EXTERNAL TABLE `project.dataset.realtime_events`
OPTIONS (
  metadata_cache_mode = 'AUTOMATIC',
  max_staleness = INTERVAL 5 MINUTE
);

-- Daily batch updates
CREATE EXTERNAL TABLE `project.dataset.daily_reports`
OPTIONS (
  metadata_cache_mode = 'AUTOMATIC',
  max_staleness = INTERVAL 4 HOUR
);

-- Static reference data
CREATE EXTERNAL TABLE `project.dataset.reference_data`
OPTIONS (
  metadata_cache_mode = 'AUTOMATIC',
  max_staleness = INTERVAL 24 HOUR
);
```

**Cache Monitoring**
```sql
-- Check cache effectiveness
SELECT
  table_name,
  metadata_cache_mode,
  max_staleness,
  last_metadata_refresh_time
FROM `project.dataset`.INFORMATION_SCHEMA.TABLES
WHERE table_type = 'EXTERNAL';
```

### 5.2 Query Acceleration

**Partition Pruning**
```sql
-- ✅ Good: Uses partition filter
SELECT *
FROM `project.dataset.sales_data`
WHERE transaction_date = '2024-12-01'
  AND region = 'us-west';

-- ❌ Bad: No partition filter
SELECT *
FROM `project.dataset.sales_data`
WHERE amount > 1000;
```

**Column Selection**
```sql
-- ✅ Good: Select only needed columns (columnar format benefit)
SELECT customer_id, amount, transaction_date
FROM `project.dataset.sales_data`
WHERE transaction_date >= '2024-01-01';

-- ❌ Bad: Select all columns
SELECT *
FROM `project.dataset.sales_data`
WHERE transaction_date >= '2024-01-01';
```

### 5.3 File Organization

**Optimal File Sizes**
- **Ideal**: 128MB to 1GB per file
- **Too small**: < 10MB causes overhead
- **Too large**: > 5GB limits parallelism

**File Consolidation**
```bash
# Use Cloud Storage to consolidate small files
gsutil compose gs://bucket/small-file-1.parquet \
  gs://bucket/small-file-2.parquet \
  ... \
  gs://bucket/consolidated-file.parquet
```

**Partition Strategy**
```sql
-- ✅ Good: Reasonable partition count (365 days)
PARTITION BY DATE(transaction_date)

-- ❌ Bad: Too many partitions (hourly = 8,760/year)
PARTITION BY DATETIME_TRUNC(transaction_timestamp, HOUR)

-- ✅ Good: Multi-level partitioning
PARTITION BY DATE_TRUNC(transaction_date, MONTH)
CLUSTER BY customer_id, region
```

### 5.4 Data Format Selection

**Format Comparison**
| Format | Read Speed | Compression | Schema Evolution | BigLake Support |
|--------|-----------|-------------|------------------|-----------------|
| Parquet | Fast | Excellent | Good | ✅ Full |
| ORC | Fast | Excellent | Good | ✅ Full |
| Avro | Medium | Good | Excellent | ✅ Full |
| Delta Lake | Fast | Excellent | Good | ✅ Full |
| Iceberg | Fast | Excellent | Good | ✅ Full |
| CSV | Slow | Poor | None | ✅ Basic |
| JSON | Slow | Poor | Flexible | ✅ Basic |

**Recommended Format**
```sql
-- ✅ Best: Parquet for analytics
CREATE EXTERNAL TABLE `project.dataset.analytics_data`
WITH CONNECTION `project.us.my_biglake_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://bucket/data/*.parquet']
);

-- ✅ Good: Delta Lake for ACID and updates
CREATE EXTERNAL TABLE `project.dataset.transactional_data`
WITH CONNECTION `project.us.my_biglake_connection`
OPTIONS (
  format = 'DELTA_LAKE',
  uris = ['gs://bucket/delta-table']
);
```

---

## 6. Data Governance and Compliance

### 6.1 Data Classification

**Implement Taxonomy**
```bash
# Create data classification taxonomy
gcloud data-catalog taxonomies create data_classification \
  --location=us \
  --display-name="Data Classification"

# Create classification levels
gcloud data-catalog taxonomies policy-tags create public \
  --taxonomy=data_classification \
  --display-name="Public"

gcloud data-catalog taxonomies policy-tags create internal \
  --taxonomy=data_classification \
  --display-name="Internal"

gcloud data-catalog taxonomies policy-tags create confidential \
  --taxonomy=data_classification \
  --display-name="Confidential"

gcloud data-catalog taxonomies policy-tags create restricted \
  --taxonomy=data_classification \
  --display-name="Restricted"
```

### 6.2 Audit Logging

**Enable Data Access Logs**
```yaml
# In organization policy or project IAM settings
auditConfigs:
  - service: bigquery.googleapis.com
    auditLogConfigs:
      - logType: DATA_READ
      - logType: DATA_WRITE
      - logType: ADMIN_READ
```

**Query Access Logs**
```sql
-- Analyze BigLake table access
SELECT
  protopayload_auditlog.authenticationInfo.principalEmail as user,
  protopayload_auditlog.resourceName as resource,
  protopayload_auditlog.methodName as method,
  timestamp
FROM `project.dataset._AllLogs`
WHERE
  resource.type = 'bigquery_resource'
  AND protopayload_auditlog.resourceName LIKE '%biglake_table%'
  AND DATE(timestamp) >= CURRENT_DATE() - 7
ORDER BY timestamp DESC;
```

### 6.3 Data Lineage

**Track Data Flow**
```sql
-- Query lineage information
SELECT
  source_table,
  target_table,
  job_id,
  creation_time
FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  referenced_tables LIKE '%biglake%'
  AND DATE(creation_time) >= CURRENT_DATE() - 30;
```

**Document in Data Catalog**
```bash
# Add table description and tags
gcloud data-catalog entries update \
  --lookup-entry='bigquery:project.dataset.biglake_table' \
  --description="Customer data from data lake with PII masking"

# Add custom tags
gcloud data-catalog entries tags create \
  --entry='bigquery:project.dataset.biglake_table' \
  --tag-template=data_governance \
  --tag-template-location=us \
  --tag-file=tag.json
```

### 6.4 Compliance Features

**GDPR Right to Erasure**
```sql
-- Delete customer data (if supported by format like Delta Lake)
DELETE FROM `project.dataset.delta_customers`
WHERE customer_id = 'customer-to-delete';

-- For Parquet/immutable formats, recreate without deleted records
CREATE OR REPLACE EXTERNAL TABLE `project.dataset.customers_updated`
WITH CONNECTION `project.us.my_biglake_connection`
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://bucket/customers-updated/*.parquet']
)
AS
SELECT *
FROM `project.dataset.customers_original`
WHERE customer_id != 'customer-to-delete';
```

---

## 7. Cost Optimization

### 7.1 Cost Components
- **Query costs**: Based on data scanned from BigLake tables
- **Metadata caching**: Minimal cost for cache storage and refresh
- **Storage costs**: Paid to cloud provider (GCS/S3/Azure)
- **Network egress**: Data transfer costs for cross-region queries

### 7.2 Cost Reduction Strategies

**Use Partition Filtering**
```sql
-- Expensive: Scans all data
SELECT * FROM biglake_table WHERE status = 'active';

-- Cheap: Scans only one partition
SELECT * FROM biglake_table 
WHERE date_partition = '2024-12-01' 
  AND status = 'active';
```

**Materialize Hot Data**
```sql
-- Create native BigQuery table for frequently accessed data
CREATE TABLE `project.dataset.hot_data`
PARTITION BY DATE(transaction_date)
CLUSTER BY customer_id
AS
SELECT *
FROM `project.dataset.biglake_sales`
WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY);

-- Schedule daily refresh
CREATE OR REPLACE PROCEDURE RefreshHotData()
BEGIN
  CREATE OR REPLACE TABLE `project.dataset.hot_data`
  PARTITION BY DATE(transaction_date)
  CLUSTER BY customer_id
  AS
  SELECT *
  FROM `project.dataset.biglake_sales`
  WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY);
END;
```

**Optimize Metadata Staleness**
```sql
-- Balance freshness vs. cost
-- Less frequent refresh = lower cost
CREATE EXTERNAL TABLE `project.dataset.reference_data`
OPTIONS (
  metadata_cache_mode = 'AUTOMATIC',
  max_staleness = INTERVAL 24 HOUR  -- Daily refresh acceptable
);

-- More frequent refresh = higher cost
CREATE EXTERNAL TABLE `project.dataset.realtime_data`
OPTIONS (
  metadata_cache_mode = 'AUTOMATIC',
  max_staleness = INTERVAL 5 MINUTE  -- Near real-time needed
);
```

### 7.3 Cost Monitoring

**Track Query Costs**
```sql
SELECT
  user_email,
  DATE(creation_time) as query_date,
  COUNT(*) as query_count,
  SUM(total_bytes_processed) / POW(10, 12) as tb_processed,
  SUM(total_bytes_processed) / POW(10, 12) * 6.25 as estimated_cost_usd
FROM `region-us`.INFORMATION_SCHEMA.JOBS
WHERE
  DATE(creation_time) >= CURRENT_DATE() - 30
  AND referenced_tables LIKE '%biglake%'
GROUP BY user_email, query_date
ORDER BY tb_processed DESC;
```

---

## 8. Integration Patterns

### 8.1 With BigQuery Native Tables

**Federated Queries**
```sql
-- Join BigLake and native tables
SELECT
  bl.customer_id,
  bl.name,
  bl.email,
  n.order_count,
  n.total_spent
FROM `project.dataset.biglake_customers` bl
LEFT JOIN `project.dataset.native_order_summary` n
  ON bl.customer_id = n.customer_id
WHERE bl.region = 'us-west';
```

### 8.2 With Data Catalog

**Register BigLake Tables**
```bash
# BigLake tables automatically registered in Data Catalog
# Query Data Catalog
gcloud data-catalog entries lookup \
  --bigquery-table='project.dataset.biglake_table'
```

### 8.3 With Dataplex

**Manage with Dataplex**
```bash
# Create Dataplex lake
gcloud dataplex lakes create my-lake \
  --location=us-central1 \
  --display-name="My Data Lake"

# Create zone for BigLake tables
gcloud dataplex zones create biglake-zone \
  --lake=my-lake \
  --location=us-central1 \
  --type=CURATED \
  --resource-location-type=SINGLE_REGION

# Discover BigLake tables
gcloud dataplex assets create biglake-asset \
  --lake=my-lake \
  --zone=biglake-zone \
  --location=us-central1 \
  --resource-type=BIGQUERY_DATASET \
  --resource-name=projects/PROJECT/datasets/DATASET
```

### 8.4 With BI Tools

**Connect Looker**
```sql
-- Create materialized view for dashboard
CREATE MATERIALIZED VIEW `project.dataset.dashboard_mv`
PARTITION BY DATE_TRUNC(order_date, MONTH)
CLUSTER BY region
AS
SELECT
  DATE_TRUNC(order_date, DAY) as order_date,
  region,
  product_category,
  SUM(amount) as total_sales,
  COUNT(DISTINCT customer_id) as unique_customers
FROM `project.dataset.biglake_orders`
WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
GROUP BY order_date, region, product_category;
```

---

## 9. Troubleshooting

### 9.1 Common Issues

**Access Denied Errors**
```
Error: Access Denied: BigQuery BigLake: Permission denied on Cloud Storage bucket

Solution:
1. Verify BigLake connection service account
2. Check Cloud Storage IAM permissions
3. Ensure objectViewer role granted
4. Verify bucket location matches connection region
```

**Metadata Cache Issues**
```
Issue: Queries returning stale data

Solution:
# Manually refresh metadata
ALTER TABLE `project.dataset.biglake_table`
REFRESH_METADATA_CACHE;

# Or adjust max_staleness
ALTER TABLE `project.dataset.biglake_table`
SET OPTIONS (max_staleness = INTERVAL 5 MINUTE);
```

**Performance Issues**
```
Issue: Slow query performance

Diagnosis:
1. Check if partition filters are used
2. Verify file format (use Parquet)
3. Check file sizes
4. Review metadata cache settings

Solution:
- Add partition filters
- Consolidate small files
- Use columnar format
- Enable metadata caching
```

### 9.2 Debug Queries

**Check Table Configuration**
```sql
SELECT
  table_name,
  table_type,
  ddl
FROM `project.dataset`.INFORMATION_SCHEMA.TABLES
WHERE table_name = 'biglake_table';
```

**Monitor Query Performance**
```sql
SELECT
  job_id,
  user_email,
  total_bytes_processed,
  total_slot_ms,
  TIMESTAMP_DIFF(end_time, start_time, SECOND) as duration_sec
FROM `region-us`.INFORMATION_SCHEMA.JOBS
WHERE
  job_id = 'YOUR_JOB_ID';
```

---

## Quick Reference Checklist

### Initial Setup
- [ ] Enable BigQuery API
- [ ] Create BigLake connections
- [ ] Configure IAM permissions
- [ ] Grant storage access to service accounts
- [ ] Create policy taxonomies for security
- [ ] Set up audit logging
- [ ] Test connectivity and basic queries
- [ ] Document connection configurations
- [ ] Establish governance policies
- [ ] Train users on BigLake features

### Creating BigLake Tables
- [ ] Choose appropriate file format (Parquet recommended)
- [ ] Implement partitioning for large tables
- [ ] Configure metadata caching
- [ ] Set appropriate max_staleness
- [ ] Apply column-level security tags
- [ ] Implement row-level policies if needed
- [ ] Test queries with different users
- [ ] Document table schemas and security
- [ ] Optimize file organization
- [ ] Verify performance

### Security Configuration
- [ ] Define policy taxonomies
- [ ] Create and apply policy tags
- [ ] Set up row access policies
- [ ] Implement data masking views
- [ ] Configure audit logging
- [ ] Grant fine-grained permissions
- [ ] Test security with different users
- [ ] Document access control model
- [ ] Review permissions quarterly
- [ ] Conduct security audits

### Ongoing Operations
- [ ] Monitor query performance
- [ ] Track costs and optimize
- [ ] Refresh metadata cache as needed
- [ ] Review and update security policies
- [ ] Consolidate small files regularly
- [ ] Update partitioning strategies
- [ ] Audit access logs
- [ ] Optimize slow queries
- [ ] Update documentation
- [ ] Train new users

---

## Resources

### Official Documentation
- [BigLake Overview](https://cloud.google.com/bigquery/docs/biglake-intro)
- [BigLake Tables](https://cloud.google.com/bigquery/docs/biglake-tables)
- [BigLake Security](https://cloud.google.com/bigquery/docs/biglake-security)
- [Column-Level Security](https://cloud.google.com/bigquery/docs/column-level-security)
- [Row-Level Security](https://cloud.google.com/bigquery/docs/row-level-security)

### Tools and Integrations
- BigQuery Console for table management
- Data Catalog for metadata
- Dataplex for data lake governance
- Cloud Monitoring for observability
- Policy Tag Manager for security

### Pricing
- [BigQuery Pricing](https://cloud.google.com/bigquery/pricing)
- BigLake query costs same as BigQuery
- Metadata cache storage costs minimal

---

*Last Updated: December 27, 2025*
*Version: 1.0*
