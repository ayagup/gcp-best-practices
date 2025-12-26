# Datastream Best Practices

*Last Updated: December 25, 2025*

## Overview

Datastream is a serverless change data capture (CDC) and replication service that enables real-time, reliable data movement from operational databases into BigQuery, Cloud Storage, Cloud SQL, and other targets with minimal latency.

---

## 1. Source Configuration

### Supported Source Databases

**Oracle Configuration:**
```sql
-- Enable supplemental logging (required for CDC)
ALTER DATABASE ADD SUPPLEMENTAL LOG DATA;
ALTER DATABASE ADD SUPPLEMENTAL LOG DATA (PRIMARY KEY) COLUMNS;

-- Create Datastream user
CREATE USER datastream_user IDENTIFIED BY secure_password;

-- Grant necessary privileges
GRANT CREATE SESSION TO datastream_user;
GRANT SELECT ON DBA_TABLES TO datastream_user;
GRANT SELECT_CATALOG_ROLE TO datastream_user;
GRANT EXECUTE_CATALOG_ROLE TO datastream_user;
GRANT SELECT ANY TRANSACTION TO datastream_user;
GRANT SELECT ANY TABLE TO datastream_user;
GRANT LOGMINING TO datastream_user;

-- For specific schemas
GRANT SELECT ON schema_name.table_name TO datastream_user;
```

**MySQL Configuration:**
```sql
-- Enable binary logging (required for CDC)
-- In my.cnf or my.ini:
[mysqld]
server-id = 1
log_bin = mysql-bin
binlog_format = ROW
binlog_row_image = FULL
expire_logs_days = 7

-- Create replication user
CREATE USER 'datastream_user'@'%' IDENTIFIED BY 'secure_password';

-- Grant replication privileges
GRANT SELECT, REPLICATION CLIENT, REPLICATION SLAVE ON *.* TO 'datastream_user'@'%';

-- For specific databases
GRANT SELECT ON database_name.* TO 'datastream_user'@'%';

FLUSH PRIVILEGES;
```

**PostgreSQL Configuration:**
```sql
-- Enable logical replication (postgresql.conf)
wal_level = logical
max_replication_slots = 10
max_wal_senders = 10

-- Create replication user
CREATE USER datastream_user WITH REPLICATION PASSWORD 'secure_password';

-- Grant schema permissions
GRANT USAGE ON SCHEMA schema_name TO datastream_user;
GRANT SELECT ON ALL TABLES IN SCHEMA schema_name TO datastream_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA schema_name 
    GRANT SELECT ON TABLES TO datastream_user;

-- Create publication (for logical replication)
CREATE PUBLICATION datastream_pub FOR ALL TABLES;
```

**AlloyDB Configuration:**
```sql
-- Similar to PostgreSQL
-- Enable logical replication
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_replication_slots = 10;

-- Create replication user
CREATE USER datastream_user WITH REPLICATION PASSWORD 'secure_password';

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO datastream_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO datastream_user;
```

---

## 2. Connection Profile Setup

### Network Connectivity

**Best Practices:**
- Use Private Connectivity (VPC peering) for production
- Configure IP allowlisting for public IP connections
- Test connectivity before creating streams

**Private Connectivity (Recommended):**
```bash
# Create private connection
gcloud datastream private-connections create my-private-connection \
    --location=us-central1 \
    --display-name="Private connection to on-prem" \
    --vpc-peering-config=subnet=10.0.0.0/29,vpc=my-vpc

# Create connection profile with private connectivity
gcloud datastream connection-profiles create oracle-source \
    --location=us-central1 \
    --type=ORACLE \
    --oracle-profile=hostname=oracle.internal,port=1521,username=datastream_user,password=secure_password,database=ORCL \
    --display-name="Oracle Source" \
    --private-connection=projects/my-project/locations/us-central1/privateConnections/my-private-connection
```

**Forward SSH Tunnel:**
```bash
# For sources behind firewall
gcloud datastream connection-profiles create mysql-source-ssh \
    --location=us-central1 \
    --type=MYSQL \
    --mysql-profile=hostname=mysql.internal,port=3306,username=datastream_user,password=secure_password \
    --display-name="MySQL with SSH" \
    --forward-ssh-connectivity=hostname=bastion.example.com,port=22,username=tunnel_user,private-key=$(cat ~/.ssh/id_rsa)
```

---

## 3. Stream Configuration

### Stream Creation

**Best Practices:**
- Start with backfill for initial data load
- Configure appropriate refresh intervals
- Use object exclusion/inclusion patterns efficiently

```bash
# Create stream with BigQuery destination
gcloud datastream streams create mysql-to-bigquery \
    --location=us-central1 \
    --display-name="MySQL to BigQuery Stream" \
    --source=projects/my-project/locations/us-central1/connectionProfiles/mysql-source \
    --destination=projects/my-project/locations/us-central1/connectionProfiles/bigquery-dest \
    --backfill-none \
    --mysql-source-config=include-objects=database1.table1,database1.table2,exclude-objects=database1.temp_* \
    --bigquery-destination-config=data-freshness=900
```

**Python SDK Configuration:**
```python
from google.cloud import datastream_v1

def create_datastream(project_id, location, stream_id):
    """Create a Datastream stream."""
    client = datastream_v1.DatastreamClient()
    parent = f"projects/{project_id}/locations/{location}"
    
    # MySQL source configuration
    mysql_source_config = datastream_v1.MysqlSourceConfig(
        include_objects=datastream_v1.MysqlRdbms(
            mysql_databases=[
                datastream_v1.MysqlDatabase(
                    database="ecommerce",
                    mysql_tables=[
                        datastream_v1.MysqlTable(
                            table="orders",
                            mysql_columns=[
                                datastream_v1.MysqlColumn(column="order_id"),
                                datastream_v1.MysqlColumn(column="customer_id"),
                                datastream_v1.MysqlColumn(column="amount"),
                                datastream_v1.MysqlColumn(column="order_date"),
                            ]
                        ),
                        datastream_v1.MysqlTable(table="customers"),
                    ]
                )
            ]
        ),
        # Exclude temporary tables
        exclude_objects=datastream_v1.MysqlRdbms(
            mysql_databases=[
                datastream_v1.MysqlDatabase(
                    database="ecommerce",
                    mysql_tables=[
                        datastream_v1.MysqlTable(table="temp_*"),
                        datastream_v1.MysqlTable(table="staging_*"),
                    ]
                )
            ]
        ),
    )
    
    # BigQuery destination configuration
    bigquery_destination_config = datastream_v1.BigQueryDestinationConfig(
        data_freshness=900,  # 15 minutes
        source_hierarchy_datasets=datastream_v1.SourceHierarchyDatasets(
            dataset_template=datastream_v1.DatasetTemplate(
                location="us-central1",
                dataset_id_prefix="datastream_",
            )
        ),
    )
    
    # Stream configuration
    stream = datastream_v1.Stream(
        display_name="MySQL to BigQuery CDC",
        source_config=datastream_v1.SourceConfig(
            source_connection_profile=f"{parent}/connectionProfiles/mysql-source",
            mysql_source_config=mysql_source_config,
        ),
        destination_config=datastream_v1.DestinationConfig(
            destination_connection_profile=f"{parent}/connectionProfiles/bigquery-dest",
            bigquery_destination_config=bigquery_destination_config,
        ),
        backfill_all=datastream_v1.Stream.BackfillAllStrategy(),
    )
    
    # Create stream
    operation = client.create_stream(
        parent=parent,
        stream=stream,
        stream_id=stream_id,
    )
    
    response = operation.result()
    print(f"Stream created: {response.name}")
    return response
```

---

## 4. Backfill Strategy

### Initial Data Load

**Best Practices:**
- Perform backfill during low-traffic periods
- Monitor source database load during backfill
- Use backfill-none for tables that don't need historical data

**Backfill All:**
```bash
# Backfill all historical data
gcloud datastream streams create stream-with-backfill \
    --location=us-central1 \
    --source=projects/my-project/locations/us-central1/connectionProfiles/mysql-source \
    --destination=projects/my-project/locations/us-central1/connectionProfiles/bigquery-dest \
    --backfill-all \
    --mysql-source-config=include-objects=database.*
```

**Selective Backfill:**
```python
# Backfill specific tables only
stream = datastream_v1.Stream(
    backfill_all=datastream_v1.Stream.BackfillAllStrategy(
        mysql_excluded_objects=datastream_v1.MysqlRdbms(
            mysql_databases=[
                datastream_v1.MysqlDatabase(
                    database="ecommerce",
                    mysql_tables=[
                        # Exclude large log tables from backfill
                        datastream_v1.MysqlTable(table="audit_logs"),
                        datastream_v1.MysqlTable(table="system_logs"),
                    ]
                )
            ]
        )
    )
)
```

---

## 5. Destination Configuration

### BigQuery Destination

**Best Practices:**
- Use data freshness configuration appropriately
- Enable merge for updates (automatic deduplication)
- Configure partitioning for large tables

```bash
# BigQuery destination with optimizations
gcloud datastream streams create optimized-bq-stream \
    --location=us-central1 \
    --source=... \
    --destination=... \
    --bigquery-destination-config=\
data-freshness=300,\
source-hierarchy-datasets-dataset-id-prefix=cdc_
```

**Schema Evolution:**
```python
# BigQuery handles schema changes automatically
# New columns in source → automatically added to BigQuery
# Supports:
# - Adding columns
# - Changing column types (compatible changes)
# - NOT supported: Dropping columns, renaming columns
```

### Cloud Storage Destination

**Best Practices:**
- Use Avro format for structured data
- Use JSON for flexibility
- Organize by date partitions

```bash
# Cloud Storage destination
gcloud datastream streams create mysql-to-gcs \
    --location=us-central1 \
    --source=projects/my-project/locations/us-central1/connectionProfiles/mysql-source \
    --destination=projects/my-project/locations/us-central1/connectionProfiles/gcs-dest \
    --gcs-destination-config=\
path=cdc-data/,\
file-rotation-mb=100,\
file-rotation-interval=60s,\
avro-file-format='{}'
```

**Python Configuration:**
```python
# Cloud Storage with JSON Lines format
gcs_destination_config = datastream_v1.GcsDestinationConfig(
    path="cdc-data/",
    file_rotation_mb=100,
    file_rotation_interval="60s",
    json_file_format=datastream_v1.JsonFileFormat(
        schema_file_format=datastream_v1.JsonFileFormat.SchemaFileFormat.NO_SCHEMA_FILE,
        compression=datastream_v1.JsonFileFormat.JsonCompression.GZIP,
    ),
)
```

---

## 6. Monitoring and Troubleshooting

### Key Metrics

**Monitor These Metrics:**
- Stream throughput (events/second)
- Replication lag
- Error rate
- Source database load

**Cloud Monitoring Setup:**
```python
from google.cloud import monitoring_v3

def create_datastream_alert(project_id):
    """Create alert for high replication lag."""
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"
    
    alert_policy = monitoring_v3.AlertPolicy(
        display_name="Datastream High Lag Alert",
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Lag > 5 minutes",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='metric.type="datastream.googleapis.com/stream/lag" '
                           'resource.type="datastream.googleapis.com/Stream"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=300,  # 5 minutes
                    duration={"seconds": 180},
                ),
            )
        ],
        notification_channels=[],
        alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
            auto_close="1800s"
        ),
    )
    
    return client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
```

### Log Analysis

**Query Stream Logs:**
```python
from google.cloud import logging

def query_datastream_logs(project_id, stream_name):
    """Query Datastream logs for errors."""
    client = logging.Client(project=project_id)
    
    filter_str = f'''
    resource.type="datastream.googleapis.com/Stream"
    resource.labels.stream_id="{stream_name}"
    severity>=ERROR
    '''
    
    for entry in client.list_entries(filter_=filter_str, page_size=100):
        print(f"Timestamp: {entry.timestamp}")
        print(f"Severity: {entry.severity}")
        print(f"Message: {entry.payload}")
        print("---")
```

---

## 7. Performance Optimization

### Network Optimization

**Best Practices:**
- Use Private Connectivity for better throughput
- Configure appropriate network bandwidth
- Monitor network utilization

### Source Database Optimization

**Reduce Impact on Source:**
```sql
-- Oracle: Optimize LogMiner performance
ALTER SYSTEM SET "_log_read_buffers"=8;

-- MySQL: Optimize binary log configuration
SET GLOBAL binlog_cache_size = 2097152;  -- 2MB
SET GLOBAL max_binlog_size = 1073741824;  -- 1GB

-- PostgreSQL: Tune WAL settings
ALTER SYSTEM SET max_wal_size = '2GB';
ALTER SYSTEM SET checkpoint_timeout = '15min';
```

### Stream Optimization

**Batch Configuration:**
```python
# For Cloud Storage destinations
gcs_destination_config = datastream_v1.GcsDestinationConfig(
    path="cdc-data/",
    file_rotation_mb=500,  # Larger files = fewer writes
    file_rotation_interval="300s",  # 5 minutes
)
```

---

## 8. Schema Management

### Handling Schema Changes

**Supported Changes:**
- ✅ Adding columns (automatically propagated)
- ✅ Widening column types (e.g., INT to BIGINT)
- ❌ Dropping columns (requires stream recreation)
- ❌ Renaming columns (appears as drop + add)

**Best Practices:**
```sql
-- Make compatible changes to source
-- Example: Adding a new column
ALTER TABLE orders ADD COLUMN shipping_address VARCHAR(500);

-- Datastream automatically detects and propagates to BigQuery
-- No manual intervention needed
```

---

## 9. Data Validation

### Validation Queries

**Count Validation:**
```sql
-- Source (MySQL)
SELECT COUNT(*) as source_count
FROM orders
WHERE created_at >= '2025-01-01';

-- Destination (BigQuery)
SELECT COUNT(*) as dest_count
FROM `project.datastream_database.orders`
WHERE _metadata_timestamp >= '2025-01-01';
```

**Checksum Validation:**
```python
import hashlib

def validate_data_integrity(source_conn, bq_client, table_name):
    """Compare checksums between source and destination."""
    
    # Source checksum (MySQL example)
    source_query = f"""
        SELECT MD5(GROUP_CONCAT(
            CONCAT_WS(',', column1, column2, column3)
            ORDER BY id
        )) as checksum
        FROM {table_name}
    """
    
    # BigQuery checksum
    bq_query = f"""
        SELECT MD5(STRING_AGG(
            CONCAT(CAST(column1 AS STRING), ',', 
                   CAST(column2 AS STRING), ',',
                   CAST(column3 AS STRING))
            ORDER BY id
        )) as checksum
        FROM `project.dataset.{table_name}`
        WHERE _metadata_deleted = FALSE
    """
    
    # Compare checksums
    source_checksum = execute_query(source_conn, source_query)
    bq_checksum = execute_query(bq_client, bq_query)
    
    if source_checksum == bq_checksum:
        print(f"✓ Data integrity verified for {table_name}")
    else:
        print(f"✗ Data mismatch detected in {table_name}")
```

---

## 10. Security Best Practices

### Authentication

**Service Account Setup:**
```bash
# Create service account for Datastream
gcloud iam service-accounts create datastream-sa \
    --display-name="Datastream Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding my-project \
    --member=serviceAccount:datastream-sa@my-project.iam.gserviceaccount.com \
    --role=roles/datastream.admin

# For BigQuery destination
gcloud projects add-iam-policy-binding my-project \
    --member=serviceAccount:datastream-sa@my-project.iam.gserviceaccount.com \
    --role=roles/bigquery.dataEditor
```

### Encryption

**Data Encryption:**
- In-transit: TLS 1.2+ automatically enabled
- At-rest: Automatic encryption in destination
- CMEK support for BigQuery and Cloud Storage

```bash
# Create BigQuery dataset with CMEK
bq mk --dataset \
    --default_kms_key=projects/my-project/locations/us-central1/keyRings/my-keyring/cryptoKeys/my-key \
    my-project:datastream_database
```

---

## 11. Disaster Recovery

### Stream Recreation

**Backup Stream Configuration:**
```python
def backup_stream_config(project_id, location, stream_id):
    """Export stream configuration for disaster recovery."""
    client = datastream_v1.DatastreamClient()
    stream_name = f"projects/{project_id}/locations/{location}/streams/{stream_id}"
    
    stream = client.get_stream(name=stream_name)
    
    # Export configuration
    config = {
        'display_name': stream.display_name,
        'source_config': stream.source_config,
        'destination_config': stream.destination_config,
        'backfill_strategy': stream.backfill_all or stream.backfill_none,
    }
    
    # Save to Cloud Storage or local file
    import json
    with open(f'{stream_id}_backup.json', 'w') as f:
        json.dump(config, f, default=str, indent=2)
    
    print(f"Configuration backed up for {stream_id}")
```

### Failover Strategy

**Multi-Region Setup:**
```bash
# Primary stream in us-central1
gcloud datastream streams create primary-stream \
    --location=us-central1 \
    --source=... \
    --destination=...

# Standby stream in us-east1
gcloud datastream streams create standby-stream \
    --location=us-east1 \
    --source=... \
    --destination=...
```

---

## 12. Common Use Cases

### Real-Time Analytics

**Pattern: CDC to BigQuery:**
```sql
-- Query latest data with CDC metadata
SELECT 
    order_id,
    customer_id,
    amount,
    status,
    _metadata_timestamp as replicated_at,
    _metadata_deleted as is_deleted
FROM `project.datastream_database.orders`
WHERE _metadata_deleted = FALSE
    AND _metadata_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
ORDER BY _metadata_timestamp DESC;
```

### Data Lake Ingestion

**Pattern: CDC to Cloud Storage to BigQuery:**
```bash
# 1. Stream to Cloud Storage (Avro format)
gcloud datastream streams create mysql-to-datalake \
    --location=us-central1 \
    --source=... \
    --destination=gcs-connection-profile \
    --gcs-destination-config=path=raw/cdc/,avro-file-format='{}'

# 2. Load from Cloud Storage to BigQuery (scheduled)
# Use Cloud Composer or Cloud Scheduler with Dataflow
```

### Cross-Cloud Replication

**Pattern: On-Prem to GCP:**
```python
# Stream from on-premises Oracle to Cloud SQL
def create_cross_cloud_stream(project_id):
    """Replicate on-prem database to Cloud SQL."""
    client = datastream_v1.DatastreamClient()
    
    stream = datastream_v1.Stream(
        display_name="On-Prem Oracle to Cloud SQL",
        source_config=datastream_v1.SourceConfig(
            source_connection_profile="projects/.../connectionProfiles/on-prem-oracle",
            oracle_source_config=datastream_v1.OracleSourceConfig(
                include_objects=datastream_v1.OracleRdbms(...)
            ),
        ),
        destination_config=datastream_v1.DestinationConfig(
            destination_connection_profile="projects/.../connectionProfiles/cloud-sql-postgres",
        ),
        backfill_all=datastream_v1.Stream.BackfillAllStrategy(),
    )
    
    # Create stream
    operation = client.create_stream(...)
    return operation.result()
```

---

## 13. Common Anti-Patterns

### ❌ Anti-Pattern 1: Not Using Private Connectivity
**Problem:** Public IP connections have lower throughput and security concerns
**Solution:** Use VPC peering for production workloads

### ❌ Anti-Pattern 2: Ignoring Replication Lag
**Problem:** Lag accumulates and impacts analytics freshness
**Solution:** Monitor lag metrics and optimize source database

### ❌ Anti-Pattern 3: Backfilling Everything
**Problem:** Unnecessary load on source database
**Solution:** Selectively backfill only needed tables

### ❌ Anti-Pattern 4: Not Validating Data
**Problem:** Undetected data inconsistencies
**Solution:** Implement automated validation checks

### ❌ Anti-Pattern 5: Insufficient Source Permissions
**Problem:** Stream failures due to permission issues
**Solution:** Grant all required permissions upfront

---

## 14. Quick Reference Checklist

### Source Configuration
- [ ] Enable CDC on source database (binary logs, supplemental logging, etc.)
- [ ] Create dedicated replication user
- [ ] Grant minimum required permissions
- [ ] Configure network connectivity
- [ ] Test connection before creating stream

### Stream Setup
- [ ] Choose appropriate backfill strategy
- [ ] Configure object inclusion/exclusion
- [ ] Set data freshness for BigQuery
- [ ] Enable monitoring and alerting
- [ ] Document stream configuration

### Security
- [ ] Use Private Connectivity for production
- [ ] Apply least privilege IAM roles
- [ ] Enable encryption (CMEK if required)
- [ ] Rotate credentials regularly
- [ ] Audit access logs

### Monitoring
- [ ] Monitor replication lag
- [ ] Track error rates
- [ ] Alert on high lag
- [ ] Validate data periodically
- [ ] Monitor source database impact

### Performance
- [ ] Optimize source database configuration
- [ ] Use appropriate file rotation settings
- [ ] Monitor network bandwidth
- [ ] Tune batch sizes for Cloud Storage
- [ ] Review and optimize large table replication

---

*Best Practices for Google Cloud Data Engineer Certification*
