# BigQuery Data Transfer Service Best Practices

*Last Updated: December 25, 2025*

## Overview

BigQuery Data Transfer Service automates data movement into BigQuery from SaaS applications (Google Ads, YouTube, Google Play), cloud storage platforms, and other Google services on a scheduled, managed basis. It eliminates the need for custom data pipeline code.

---

## 1. Supported Data Sources

### Google Marketing Platform

**Google Ads:**
```bash
# Create Google Ads transfer
gcloud transfer runs create \
    --data-source=google_ads \
    --display-name="Daily Google Ads Import" \
    --schedule="every day 02:00" \
    --destination-dataset-id=google_ads_data \
    --params='{
        "customer_id": "1234567890",
        "include_pmax": true,
        "conversion_window_days": 60
    }'
```

**Campaign Manager 360:**
```bash
# Create CM360 transfer
gcloud transfer runs create \
    --data-source=dcm_dt \
    --display-name="Daily CM360 Import" \
    --schedule="every day 03:00" \
    --destination-dataset-id=cm360_data \
    --params='{
        "profile_id": "1234567",
        "network_id": "7890123",
        "file_name_prefix": "cm360_"
    }'
```

**Google Analytics 4:**
```bash
# Create GA4 transfer
gcloud transfer runs create \
    --data-source=google_analytics_4 \
    --display-name="Daily GA4 Import" \
    --schedule="every day 04:00" \
    --destination-dataset-id=analytics_data \
    --params='{
        "property_id": "123456789"
    }'
```

### YouTube Reporting

**YouTube Channel Reports:**
```python
from google.cloud import bigquery_datatransfer_v1

def create_youtube_transfer(project_id, dataset_id, channel_id):
    """Create YouTube channel data transfer."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    # Transfer configuration
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="YouTube Channel Analytics",
        data_source_id="youtube_channel",
        destination_dataset_id=dataset_id,
        schedule="every day 05:00",
        params={
            "channel_id": channel_id,
            "table_suffix": "_YYYYMMDD",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    print(f"Created transfer: {response.name}")
    return response
```

### Google Play

**Play Console Reports:**
```python
def create_play_transfer(project_id, dataset_id, bucket_name):
    """Create Google Play transfer from Cloud Storage."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="Google Play Reports",
        data_source_id="play",
        destination_dataset_id=dataset_id,
        schedule="every day 06:00",
        params={
            "bucket": bucket_name,
            "file_prefix": "sales/",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    return response
```

---

## 2. Cloud Storage Transfers

### Scheduled Loads from Cloud Storage

**Best Practices:**
- Use consistent file naming patterns
- Organize files by date partitions
- Use appropriate file formats (Parquet, Avro recommended)
- Implement file validation before loading

```python
def create_gcs_transfer(project_id, dataset_id, table_id, bucket_path):
    """Create scheduled transfer from Cloud Storage to BigQuery."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name=f"Daily load {table_id}",
        data_source_id="google_cloud_storage",
        destination_dataset_id=dataset_id,
        schedule="every day 07:00",
        params={
            "data_path_template": f"gs://{bucket_path}/data/{{run_date}}/*.parquet",
            "destination_table_name_template": table_id,
            "file_format": "PARQUET",
            "max_bad_records": "100",
            "write_disposition": "WRITE_APPEND",
            "delete_source_files": "false",
        },
        # Notification via Pub/Sub
        notification_pubsub_topic=f"projects/{project_id}/topics/bq-transfer-notifications",
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    print(f"Created GCS transfer: {response.name}")
    return response
```

**File Pattern Examples:**
```bash
# Date-based partitioning
gs://my-bucket/data/{run_date}/*.parquet
gs://my-bucket/data/{run_date|"YYYY/MM/DD"}/*.csv

# Time-based partitioning
gs://my-bucket/data/{run_time|"YYYY/MM/DD/HH"}/*.json

# Static pattern
gs://my-bucket/daily_export/*.parquet
```

---

## 3. Amazon S3 Transfers

### S3 to BigQuery

**Prerequisites:**
- AWS IAM role with S3 read permissions
- Cross-account access configured
- S3 bucket in supported region

```python
def create_s3_transfer(project_id, dataset_id, table_id, s3_bucket, aws_role_arn):
    """Create transfer from Amazon S3 to BigQuery."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name=f"S3 to BigQuery - {table_id}",
        data_source_id="amazon_s3",
        destination_dataset_id=dataset_id,
        schedule="every day 08:00",
        params={
            "data_path": f"s3://{s3_bucket}/data/*/*.parquet",
            "destination_table_name_template": table_id,
            "file_format": "PARQUET",
            "write_disposition": "WRITE_TRUNCATE",
            "access_key_id": "",  # Use IAM role instead
            "secret_access_key": "",  # Use IAM role instead
            "role_arn": aws_role_arn,
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    return response
```

**AWS IAM Policy:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ]
        }
    ]
}
```

---

## 4. Data Warehouse Migration

### Teradata to BigQuery Migration

BigQuery Data Transfer Service supports two primary methods for Teradata migration:
1. **Teradata Parallel Transporter (TPT)** - High-performance bulk data extraction
2. **JDBC Connection** - Standard database connectivity for smaller datasets

#### Method 1: Teradata TPT (Recommended for Large-Scale Migrations)

**Overview:**
Teradata Parallel Transporter (TPT) is Teradata's utility for high-speed data loading and exporting. It provides parallel processing capabilities for optimal performance during large-scale migrations.

**Key Benefits:**
- **High Performance**: Parallel extraction using multiple sessions
- **Large Volume Support**: Optimized for TB-scale data transfers
- **Data Partitioning**: Automatic workload distribution
- **Restart Capability**: Resume failed transfers from checkpoint
- **Transformation Support**: Apply transformations during extraction

**Prerequisites:**
```bash
# Install Teradata TPT on migration server/VM
# TPT utilities required:
# - tbuild (TPT Builder)
# - tlogview (TPT Log Viewer)
# - tdicu (Teradata ICU Library)

# Verify TPT installation
tbuild -v
# Output: Teradata Parallel Transporter Version 17.00.00.xx

# Verify ODBC connectivity
tdpid
# Should list available Teradata systems
```

**TPT Migration Transfer Configuration:**
```python
from google.cloud import bigquery_datatransfer_v1

def create_teradata_tpt_transfer(
    project_id,
    dataset_id,
    teradata_host,
    teradata_database,
    tables_config
):
    """
    Create Teradata migration using TPT for high-performance extraction.
    
    Args:
        project_id: GCP project ID
        dataset_id: Target BigQuery dataset
        teradata_host: Teradata server hostname or IP
        teradata_database: Source Teradata database name
        tables_config: List of tables with extraction parameters
    """
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="Teradata TPT Migration - Production",
        data_source_id="teradata",
        destination_dataset_id=dataset_id,
        schedule="every day 01:00",
        params={
            # Connection parameters
            "teradata_host": teradata_host,
            "teradata_port": "1025",  # Default Teradata port
            "teradata_database": teradata_database,
            "teradata_username": "migration_user",
            "teradata_password_secret": f"projects/{project_id}/secrets/teradata-password/versions/latest",
            
            # TPT-specific parameters
            "use_tpt": "true",  # Enable TPT mode
            "tpt_sessions": "8",  # Number of parallel TPT sessions (4-16 recommended)
            "tpt_instances": "2",  # Number of TPT instances
            "tpt_buffer_size": "64000",  # Buffer size in bytes (default: 64KB)
            "tpt_max_sessions": "16",  # Maximum concurrent sessions
            
            # Table selection and filtering
            "table_list": ",".join([t["name"] for t in tables_config]),
            "schema_list": "dbc,production",  # Teradata schemas to migrate
            
            # Incremental load configuration
            "partition_column": "last_modified_date",  # Column for incremental loads
            "partition_start_date": "2024-01-01",
            "partition_end_date": "",  # Empty = current date
            
            # Data transformation
            "column_mapping": "teradata_col:bigquery_col,id:customer_id",
            "exclude_columns": "internal_audit_col,temp_col",
            
            # Performance tuning
            "export_rate_limit": "100",  # MB/s per session (optional)
            "batch_size": "10000",  # Rows per batch
            "max_bad_records": "1000",  # Error tolerance
            
            # Output configuration
            "export_format": "PARQUET",  # PARQUET (recommended), AVRO, CSV
            "compression": "SNAPPY",  # SNAPPY, GZIP, NONE
            "staging_bucket": f"gs://{project_id}-teradata-staging",
            
            # Advanced options
            "use_fastexport": "true",  # Use Teradata FastExport API
            "char_encoding": "UTF8",  # Character encoding
            "date_format": "YYYY-MM-DD",
            "timestamp_format": "YYYY-MM-DD HH24:MI:SS.F6",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    print(f"Created Teradata TPT transfer: {response.name}")
    return response


# Example usage with detailed table configuration
tables_config = [
    {
        "name": "production.customer",
        "partition_column": "created_date",
        "where_clause": "status = 'ACTIVE'",
    },
    {
        "name": "production.orders",
        "partition_column": "order_date",
        "where_clause": "amount > 0",
    },
    {
        "name": "production.transactions",
        "partition_column": "transaction_timestamp",
        "where_clause": "",
    },
]

transfer = create_teradata_tpt_transfer(
    project_id="my-gcp-project",
    dataset_id="teradata_migration",
    teradata_host="tdprod.company.com",
    teradata_database="production",
    tables_config=tables_config
)
```

**TPT Script Example (Optional - for custom TPT jobs):**
```sql
-- tpt_export.sql: Custom TPT script for complex extractions
DEFINE JOB EXPORT_TO_GCS
DESCRIPTION 'Export Teradata tables to GCS via TPT'
(
    DEFINE SCHEMA customer_schema
    (
        customer_id     INTEGER,
        customer_name   VARCHAR(100),
        email          VARCHAR(200),
        created_date   DATE,
        last_updated   TIMESTAMP(6)
    );
    
    DEFINE OPERATOR export_operator
    TYPE EXPORT
    SCHEMA customer_schema
    ATTRIBUTES
    (
        VARCHAR TdpId = 'tdprod',
        VARCHAR UserName = 'migration_user',
        VARCHAR UserPassword = '****',
        VARCHAR SelectStmt = 'SELECT * FROM production.customer 
                              WHERE created_date >= CURRENT_DATE - 30',
        INTEGER Sessions = 8,
        INTEGER MaxSessions = 16,
        VARCHAR DateForm = 'ANSIDATE',
        VARCHAR ExportFmt = 'DELIMITED'
    );
    
    DEFINE OPERATOR file_writer
    TYPE DATACONNECTOR PRODUCER
    SCHEMA customer_schema
    ATTRIBUTES
    (
        VARCHAR DirectoryPath = '/staging/teradata/',
        VARCHAR FileName = 'customer_export',
        VARCHAR Format = 'DELIMITED',
        VARCHAR OpenMode = 'Write',
        INTEGER BufferSize = 64000
    );
    
    APPLY TO OPERATOR (file_writer)
    SELECT * FROM OPERATOR (export_operator);
);
```

**TPT Performance Tuning:**
```python
def optimize_tpt_parameters(table_size_gb, network_bandwidth_mbps):
    """
    Calculate optimal TPT parameters based on table size and network.
    
    Args:
        table_size_gb: Size of table in GB
        network_bandwidth_mbps: Available network bandwidth in Mbps
    
    Returns:
        Dictionary of optimized TPT parameters
    """
    
    # Calculate optimal session count
    if table_size_gb < 10:
        sessions = 4
    elif table_size_gb < 100:
        sessions = 8
    elif table_size_gb < 1000:
        sessions = 12
    else:
        sessions = 16
    
    # Calculate buffer size (64KB - 256KB)
    buffer_size = min(256000, max(64000, (table_size_gb / sessions) * 1024))
    
    # Calculate rate limit based on network
    rate_limit_per_session = (network_bandwidth_mbps * 0.8) / sessions  # 80% of bandwidth
    
    return {
        "tpt_sessions": str(sessions),
        "tpt_instances": "2",
        "tpt_buffer_size": str(int(buffer_size)),
        "export_rate_limit": str(int(rate_limit_per_session)),
        "batch_size": "10000" if table_size_gb < 100 else "50000",
    }

# Example
params = optimize_tpt_parameters(table_size_gb=500, network_bandwidth_mbps=1000)
print(params)
# Output: {'tpt_sessions': '12', 'tpt_instances': '2', 
#          'tpt_buffer_size': '43690', 'export_rate_limit': '66', 
#          'batch_size': '50000'}
```

#### Method 2: JDBC Connection (For Smaller Datasets and Ad-Hoc Queries)

**Overview:**
JDBC (Java Database Connectivity) provides a standard interface for connecting to Teradata. Best suited for smaller datasets (<100 GB) or when TPT is not available.

**Key Benefits:**
- **Simple Setup**: Standard JDBC driver installation
- **No TPT Required**: Works with standard Teradata installation
- **Query Flexibility**: Direct SQL query execution
- **Wide Compatibility**: Works with any JDBC-compliant tool

**Prerequisites:**
```bash
# Download Teradata JDBC driver
# From Teradata Downloads: terajdbc4.jar and tdgssconfig.jar

# Verify JDBC driver
java -cp terajdbc4.jar com.teradata.jdbc.TeraDriver
# Should print Teradata JDBC Driver version

# Test JDBC connection
java -cp "terajdbc4.jar:tdgssconfig.jar" \
  -Djdbc.drivers=com.teradata.jdbc.TeraDriver \
  TestConnection
```

**JDBC Migration Transfer Configuration:**
```python
def create_teradata_jdbc_transfer(
    project_id,
    dataset_id,
    teradata_host,
    teradata_database,
    table_queries
):
    """
    Create Teradata migration using JDBC connection.
    
    Args:
        project_id: GCP project ID
        dataset_id: Target BigQuery dataset
        teradata_host: Teradata server hostname
        teradata_database: Source database
        table_queries: Dict of table names to custom SQL queries
    """
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="Teradata JDBC Transfer",
        data_source_id="teradata",
        destination_dataset_id=dataset_id,
        schedule="every day 02:00",
        params={
            # JDBC connection parameters
            "connection_type": "JDBC",  # Specify JDBC mode
            "jdbc_url": f"jdbc:teradata://{teradata_host}/DATABASE={teradata_database}",
            "driver_class": "com.teradata.jdbc.TeraDriver",
            "teradata_username": "jdbc_user",
            "teradata_password_secret": f"projects/{project_id}/secrets/teradata-jdbc-password/versions/latest",
            
            # JDBC-specific settings
            "jdbc_connection_pool_size": "5",  # Number of pooled connections
            "jdbc_fetch_size": "10000",  # Rows fetched per batch
            "jdbc_query_timeout": "3600",  # Timeout in seconds
            
            # Advanced JDBC properties
            "jdbc_properties": (
                "CHARSET=UTF8,"
                "TMODE=ANSI,"
                "TYPE=FASTEXPORT,"  # Use FastExport for better performance
                "SESSIONS=4,"
                "MAYBENULL=ON"
            ),
            
            # Table/Query configuration
            "table_list": ",".join(table_queries.keys()),
            "custom_queries": ";".join([
                f"{table}:{query}" 
                for table, query in table_queries.items()
            ]),
            
            # Incremental load
            "partition_column": "last_modified_timestamp",
            "partition_start_date": "2024-01-01",
            
            # Data type mapping
            "type_conversion": (
                "DECIMAL:NUMERIC,"
                "INTEGER:INT64,"
                "BYTEINT:INT64,"
                "SMALLINT:INT64,"
                "DATE:DATE,"
                "TIMESTAMP:TIMESTAMP,"
                "CHAR:STRING,"
                "VARCHAR:STRING"
            ),
            
            # Performance and reliability
            "max_retries": "3",
            "retry_delay_seconds": "60",
            "max_bad_records": "100",
            
            # Output format
            "export_format": "PARQUET",
            "compression": "SNAPPY",
            "staging_bucket": f"gs://{project_id}-jdbc-staging",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    print(f"Created Teradata JDBC transfer: {response.name}")
    return response


# Example with custom queries
table_queries = {
    "customer_dim": """
        SELECT 
            customer_id,
            customer_name,
            email,
            created_date,
            CAST(last_updated AS TIMESTAMP(0)) as last_updated
        FROM production.customer
        WHERE status = 'ACTIVE'
    """,
    "orders_fact": """
        SELECT 
            order_id,
            customer_id,
            order_date,
            total_amount,
            CAST(order_timestamp AS TIMESTAMP(0)) as order_timestamp
        FROM production.orders
        WHERE order_date >= CURRENT_DATE - 90
    """,
    "product_dim": """
        SELECT *
        FROM production.products
        WHERE is_deleted = 'N'
    """,
}

transfer = create_teradata_jdbc_transfer(
    project_id="my-gcp-project",
    dataset_id="teradata_jdbc_data",
    teradata_host="tdprod.company.com",
    teradata_database="production",
    table_queries=table_queries
)
```

**JDBC Connection String Examples:**
```python
def build_teradata_jdbc_url(host, database, **kwargs):
    """
    Build Teradata JDBC connection URL with advanced options.
    
    Common parameters:
    - CHARSET: Character set (UTF8, UTF16, ASCII)
    - TMODE: Transaction mode (ANSI, TERA)
    - TYPE: Connection type (DEFAULT, FASTEXPORT, FASTLOAD)
    - SESSIONS: Number of sessions (1-16)
    - LOGMECH: Authentication mechanism (TD2, LDAP, KRB5)
    - ENCRYPTDATA: Encrypt data in transit (ON, OFF)
    """
    
    base_url = f"jdbc:teradata://{host}/DATABASE={database}"
    
    # Common configurations
    configs = {
        "standard": {
            "CHARSET": "UTF8",
            "TMODE": "ANSI",
        },
        "high_performance": {
            "CHARSET": "UTF8",
            "TMODE": "ANSI",
            "TYPE": "FASTEXPORT",
            "SESSIONS": "8",
        },
        "secure": {
            "CHARSET": "UTF8",
            "TMODE": "ANSI",
            "ENCRYPTDATA": "ON",
            "LOGMECH": "LDAP",
        },
        "bulk_load": {
            "CHARSET": "UTF8",
            "TYPE": "FASTLOAD",
            "SESSIONS": "4",
            "MAXERRORS": "100",
        },
    }
    
    # Select configuration
    config_type = kwargs.get("config_type", "standard")
    params = configs.get(config_type, configs["standard"])
    
    # Build parameter string
    param_string = ",".join([f"{k}={v}" for k, v in params.items()])
    
    return f"{base_url},{param_string}"


# Examples
print(build_teradata_jdbc_url("tdprod.company.com", "production", config_type="high_performance"))
# jdbc:teradata://tdprod.company.com/DATABASE=production,CHARSET=UTF8,TMODE=ANSI,TYPE=FASTEXPORT,SESSIONS=8

print(build_teradata_jdbc_url("tdprod.company.com", "production", config_type="secure"))
# jdbc:teradata://tdprod.company.com/DATABASE=production,CHARSET=UTF8,TMODE=ANSI,ENCRYPTDATA=ON,LOGMECH=LDAP
```

#### Comparison: TPT vs JDBC

| Feature | Teradata TPT | JDBC Connection |
|---------|--------------|-----------------|
| **Performance** | Excellent (parallel extraction) | Good (single-threaded) |
| **Best For** | Large datasets (>100 GB) | Small to medium datasets (<100 GB) |
| **Complexity** | Higher (requires TPT installation) | Lower (standard JDBC driver) |
| **Session Count** | 4-16 parallel sessions | 1-5 pooled connections |
| **Throughput** | 500+ MB/s (depends on config) | 50-100 MB/s |
| **Restart Capability** | Yes (checkpoint support) | Limited (query-level retry) |
| **Resource Usage** | Higher (multiple sessions) | Lower (fewer connections) |
| **Network Efficiency** | Optimized (bulk transfer) | Standard (row-by-row fetch) |
| **Use Case** | Full database migration | Ad-hoc queries, small tables |
| **Cost** | More Teradata resources | Less Teradata resources |

#### Hybrid Approach: TPT + JDBC

```python
def create_hybrid_teradata_migration(project_id, dataset_id, table_metadata):
    """
    Use TPT for large tables and JDBC for small tables.
    
    Args:
        table_metadata: List of dicts with table name and size
    """
    
    # Classify tables by size
    large_tables = [t for t in table_metadata if t["size_gb"] >= 100]
    small_tables = [t for t in table_metadata if t["size_gb"] < 100]
    
    transfers = []
    
    # Create TPT transfer for large tables
    if large_tables:
        tpt_transfer = create_teradata_tpt_transfer(
            project_id=project_id,
            dataset_id=dataset_id,
            teradata_host="tdprod.company.com",
            teradata_database="production",
            tables_config=[{"name": t["name"]} for t in large_tables]
        )
        transfers.append(("TPT", tpt_transfer))
        print(f"Created TPT transfer for {len(large_tables)} large tables")
    
    # Create JDBC transfer for small tables
    if small_tables:
        table_queries = {
            t["name"]: f"SELECT * FROM {t['name']}"
            for t in small_tables
        }
        jdbc_transfer = create_teradata_jdbc_transfer(
            project_id=project_id,
            dataset_id=dataset_id,
            teradata_host="tdprod.company.com",
            teradata_database="production",
            table_queries=table_queries
        )
        transfers.append(("JDBC", jdbc_transfer))
        print(f"Created JDBC transfer for {len(small_tables)} small tables")
    
    return transfers


# Example usage
table_metadata = [
    {"name": "production.customer", "size_gb": 250},  # TPT
    {"name": "production.orders", "size_gb": 500},    # TPT
    {"name": "production.products", "size_gb": 5},    # JDBC
    {"name": "production.categories", "size_gb": 0.1}, # JDBC
]

transfers = create_hybrid_teradata_migration(
    project_id="my-project",
    dataset_id="teradata_data",
    table_metadata=table_metadata
)
```

#### Monitoring Teradata Migrations

```python
def monitor_teradata_migration(transfer_config_name):
    """Monitor Teradata migration progress and performance."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    
    # Get recent transfer runs
    request = bigquery_datatransfer_v1.ListTransferRunsRequest(
        parent=transfer_config_name,
        page_size=10,
    )
    
    runs = client.list_transfer_runs(request=request)
    
    for run in runs:
        print(f"\n=== Transfer Run: {run.name} ===")
        print(f"State: {run.state.name}")
        print(f"Schedule Time: {run.schedule_time}")
        
        if run.start_time:
            print(f"Start Time: {run.start_time}")
        if run.end_time:
            print(f"End Time: {run.end_time}")
            duration = (run.end_time.timestamp() - run.start_time.timestamp()) / 60
            print(f"Duration: {duration:.2f} minutes")
        
        # Check for errors
        if run.state == bigquery_datatransfer_v1.TransferState.FAILED:
            print(f"Error: {run.error_status.message}")
            
            # Get detailed error information
            if "TPT" in run.error_status.message:
                print("TPT-specific error detected")
                print("Check: TPT session limits, Teradata resource availability")
            elif "JDBC" in run.error_status.message:
                print("JDBC-specific error detected")
                print("Check: Connection pool, network connectivity, query timeout")
        
        # Display performance metrics
        if run.state == bigquery_datatransfer_v1.TransferState.SUCCEEDED:
            print(f"✓ Migration completed successfully")
            # Note: Detailed metrics available in Cloud Monitoring


# Monitor both TPT and JDBC transfers
monitor_teradata_migration("projects/123/locations/us/transferConfigs/tpt_config")
monitor_teradata_migration("projects/123/locations/us/transferConfigs/jdbc_config")
```

#### Troubleshooting Common Issues

**TPT-Specific Issues:**
```python
def troubleshoot_tpt_issues():
    """
    Common TPT issues and solutions:
    
    1. Session Limit Exceeded
       Error: "Maximum number of sessions exceeded"
       Solution: Reduce tpt_sessions or tpt_max_sessions parameter
    
    2. Buffer Overflow
       Error: "Buffer size too small"
       Solution: Increase tpt_buffer_size (64KB to 256KB)
    
    3. Network Timeout
       Error: "Connection timeout during export"
       Solution: Reduce tpt_sessions, check network stability
    
    4. Permission Denied
       Error: "User does not have EXPORT privilege"
       Solution: Grant EXPORT permission to migration user:
                 GRANT EXPORT ON database TO user;
    
    5. Disk Space Issues
       Error: "No space left on staging bucket"
       Solution: Increase staging bucket size, enable lifecycle policies
    """
    pass


def troubleshoot_jdbc_issues():
    """
    Common JDBC issues and solutions:
    
    1. Driver Not Found
       Error: "Class not found: com.teradata.jdbc.TeraDriver"
       Solution: Ensure terajdbc4.jar is in classpath
    
    2. Connection Pool Exhausted
       Error: "Unable to acquire connection from pool"
       Solution: Increase jdbc_connection_pool_size
    
    3. Query Timeout
       Error: "Query execution timeout"
       Solution: Increase jdbc_query_timeout or optimize query
    
    4. Type Conversion Error
       Error: "Cannot convert Teradata type to BigQuery type"
       Solution: Use explicit CAST in custom_queries or adjust type_conversion
    
    5. Memory Issues
       Error: "OutOfMemoryError during fetch"
       Solution: Reduce jdbc_fetch_size (default: 10000)
    """
    pass
```

### Redshift to BigQuery

**Migration Pattern:**
```python
def create_redshift_transfer(
    project_id,
    dataset_id,
    redshift_cluster,
    redshift_database,
    redshift_user
):
    """Create Redshift to BigQuery migration."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="Redshift Migration",
        data_source_id="amazon_redshift",
        destination_dataset_id=dataset_id,
        schedule="every 6 hours",
        params={
            "cluster_identifier": redshift_cluster,
            "database": redshift_database,
            "username": redshift_user,
            "password_secret_version": f"projects/{project_id}/secrets/redshift-password/versions/latest",
            "s3_bucket": "redshift-unload-bucket",
            "kms_key_name": "",  # Optional: encryption key
            "tables": "schema1.table1,schema1.table2,schema2.table3",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    return response
```

### Generic JDBC Connections for Other Databases

BigQuery Data Transfer Service supports JDBC connections to various relational databases including Oracle, SQL Server, MySQL, PostgreSQL, and other JDBC-compliant databases.

#### Supported JDBC Databases

**Common JDBC Data Sources:**
- Oracle Database
- Microsoft SQL Server
- MySQL / MariaDB
- PostgreSQL
- IBM DB2
- SAP HANA
- Snowflake (via JDBC)
- Any JDBC-compliant database

#### Oracle Database via JDBC

**Oracle JDBC Transfer Configuration:**
```python
def create_oracle_jdbc_transfer(
    project_id,
    dataset_id,
    oracle_host,
    oracle_port,
    oracle_service_name,
    oracle_schema
):
    """
    Create Oracle database transfer using JDBC.
    
    Prerequisites:
    - Oracle JDBC driver (ojdbc8.jar or ojdbc11.jar)
    - Network connectivity to Oracle database
    - Read permissions on source tables
    """
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="Oracle JDBC Transfer",
        data_source_id="jdbc",  # Generic JDBC source
        destination_dataset_id=dataset_id,
        schedule="every day 03:00",
        params={
            # JDBC connection
            "jdbc_url": f"jdbc:oracle:thin:@{oracle_host}:{oracle_port}/{oracle_service_name}",
            "driver_class": "oracle.jdbc.OracleDriver",
            "username": "migration_user",
            "password_secret": f"projects/{project_id}/secrets/oracle-password/versions/latest",
            
            # Connection pool settings
            "connection_pool_size": "5",
            "connection_timeout": "30",  # seconds
            "idle_timeout": "600",  # seconds
            
            # Query configuration
            "fetch_size": "10000",
            "query_timeout": "7200",  # 2 hours
            
            # Tables to transfer
            "tables": f"{oracle_schema}.CUSTOMER,{oracle_schema}.ORDERS,{oracle_schema}.PRODUCTS",
            
            # Or use custom queries
            "custom_queries": (
                f"{oracle_schema}.CUSTOMER:SELECT * FROM {oracle_schema}.CUSTOMER WHERE CREATED_DATE >= ADD_MONTHS(SYSDATE, -1);"
                f"{oracle_schema}.ORDERS:SELECT * FROM {oracle_schema}.ORDERS WHERE ORDER_DATE >= TRUNC(SYSDATE) - 90"
            ),
            
            # Incremental load
            "incremental_column": "LAST_MODIFIED_DATE",
            "incremental_mode": "APPEND",  # APPEND or REPLACE
            
            # Data type mapping (Oracle to BigQuery)
            "type_mapping": (
                "NUMBER:NUMERIC,"
                "VARCHAR2:STRING,"
                "CHAR:STRING,"
                "DATE:DATE,"
                "TIMESTAMP:TIMESTAMP,"
                "CLOB:STRING,"
                "BLOB:BYTES"
            ),
            
            # Performance options
            "parallel_degree": "4",  # Oracle parallel query
            "partition_query": "true",  # Use partitioning hints
            
            # Output
            "output_format": "PARQUET",
            "compression": "SNAPPY",
            "staging_location": f"gs://{project_id}-oracle-staging",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    print(f"Created Oracle JDBC transfer: {response.name}")
    return response


# Example usage
transfer = create_oracle_jdbc_transfer(
    project_id="my-project",
    dataset_id="oracle_data",
    oracle_host="oracledb.company.com",
    oracle_port="1521",
    oracle_service_name="PRODDB",
    oracle_schema="SALES"
)
```

**Oracle JDBC URL Variations:**
```python
def build_oracle_jdbc_url(connection_type, **kwargs):
    """
    Build Oracle JDBC URL for different connection types.
    
    Connection types:
    - thin: Direct TCP connection (recommended)
    - oci: Oracle Call Interface (requires Oracle client)
    - tns: TNS names resolution
    """
    
    urls = {
        "thin_service": f"jdbc:oracle:thin:@{kwargs['host']}:{kwargs['port']}/{kwargs['service_name']}",
        "thin_sid": f"jdbc:oracle:thin:@{kwargs['host']}:{kwargs['port']}:{kwargs['sid']}",
        "tns": f"jdbc:oracle:thin:@{kwargs['tns_entry']}",
        "rac": f"jdbc:oracle:thin:@(DESCRIPTION=(LOAD_BALANCE=ON)(ADDRESS=(PROTOCOL=TCP)(HOST={kwargs['host1']})(PORT=1521))(ADDRESS=(PROTOCOL=TCP)(HOST={kwargs['host2']})(PORT=1521))(CONNECT_DATA=(SERVICE_NAME={kwargs['service_name']})))",
    }
    
    return urls.get(connection_type, urls["thin_service"])


# Examples
print(build_oracle_jdbc_url("thin_service", host="oracledb", port="1521", service_name="PRODDB"))
# jdbc:oracle:thin:@oracledb:1521/PRODDB

print(build_oracle_jdbc_url("thin_sid", host="oracledb", port="1521", sid="PROD"))
# jdbc:oracle:thin:@oracledb:1521:PROD
```

#### SQL Server via JDBC

**SQL Server JDBC Transfer:**
```python
def create_sqlserver_jdbc_transfer(
    project_id,
    dataset_id,
    sqlserver_host,
    sqlserver_database,
    schema="dbo"
):
    """
    Create SQL Server transfer using JDBC.
    
    Prerequisites:
    - Microsoft SQL Server JDBC driver (mssql-jdbc-x.x.x.jre8.jar)
    - Network connectivity to SQL Server
    """
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="SQL Server JDBC Transfer",
        data_source_id="jdbc",
        destination_dataset_id=dataset_id,
        schedule="every 6 hours",
        params={
            # JDBC connection
            "jdbc_url": f"jdbc:sqlserver://{sqlserver_host}:1433;databaseName={sqlserver_database};encrypt=true;trustServerCertificate=false",
            "driver_class": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
            "username": "sa",
            "password_secret": f"projects/{project_id}/secrets/sqlserver-password/versions/latest",
            
            # Connection settings
            "connection_pool_size": "5",
            "fetch_size": "5000",
            "query_timeout": "3600",
            
            # Tables
            "tables": f"{schema}.Customer,{schema}.Orders,{schema}.OrderDetails",
            
            # Custom queries with SQL Server syntax
            "custom_queries": (
                f"{schema}.Customer:SELECT * FROM {schema}.Customer WHERE ModifiedDate >= DATEADD(day, -30, GETDATE());"
                f"{schema}.Orders:SELECT * FROM {schema}.Orders WHERE OrderDate >= DATEADD(day, -90, GETDATE())"
            ),
            
            # Data type mapping (SQL Server to BigQuery)
            "type_mapping": (
                "INT:INT64,"
                "BIGINT:INT64,"
                "SMALLINT:INT64,"
                "TINYINT:INT64,"
                "BIT:BOOL,"
                "FLOAT:FLOAT64,"
                "REAL:FLOAT64,"
                "DECIMAL:NUMERIC,"
                "MONEY:NUMERIC,"
                "VARCHAR:STRING,"
                "CHAR:STRING,"
                "NVARCHAR:STRING,"
                "NCHAR:STRING,"
                "TEXT:STRING,"
                "NTEXT:STRING,"
                "DATE:DATE,"
                "DATETIME:TIMESTAMP,"
                "DATETIME2:TIMESTAMP,"
                "DATETIMEOFFSET:TIMESTAMP,"
                "BINARY:BYTES,"
                "VARBINARY:BYTES"
            ),
            
            # Incremental load
            "incremental_column": "ModifiedDate",
            "incremental_mode": "APPEND",
            
            # Output
            "output_format": "PARQUET",
            "staging_location": f"gs://{project_id}-sqlserver-staging",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    return response
```

#### MySQL / MariaDB via JDBC

**MySQL JDBC Transfer:**
```python
def create_mysql_jdbc_transfer(
    project_id,
    dataset_id,
    mysql_host,
    mysql_database,
    mysql_port=3306
):
    """
    Create MySQL/MariaDB transfer using JDBC.
    
    Prerequisites:
    - MySQL Connector/J (mysql-connector-java-x.x.xx.jar)
    - Or MariaDB Connector/J (mariadb-java-client-x.x.x.jar)
    """
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="MySQL JDBC Transfer",
        data_source_id="jdbc",
        destination_dataset_id=dataset_id,
        schedule="every 4 hours",
        params={
            # JDBC connection (MySQL)
            "jdbc_url": f"jdbc:mysql://{mysql_host}:{mysql_port}/{mysql_database}?useSSL=true&serverTimezone=UTC",
            "driver_class": "com.mysql.cj.jdbc.Driver",  # MySQL 8.x
            # For MySQL 5.x: "com.mysql.jdbc.Driver"
            # For MariaDB: "org.mariadb.jdbc.Driver"
            
            "username": "replication_user",
            "password_secret": f"projects/{project_id}/secrets/mysql-password/versions/latest",
            
            # Connection pool
            "connection_pool_size": "5",
            "connection_properties": (
                "useServerPrepStmts=true,"
                "cachePrepStmts=true,"
                "prepStmtCacheSize=250,"
                "prepStmtCacheSqlLimit=2048,"
                "useCompression=true"
            ),
            
            # Query configuration
            "fetch_size": "10000",
            "query_timeout": "1800",
            
            # Tables
            "tables": "users,orders,products,inventory",
            
            # Custom queries
            "custom_queries": (
                "users:SELECT * FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY);"
                "orders:SELECT * FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)"
            ),
            
            # Data type mapping (MySQL to BigQuery)
            "type_mapping": (
                "TINYINT:INT64,"
                "SMALLINT:INT64,"
                "MEDIUMINT:INT64,"
                "INT:INT64,"
                "INTEGER:INT64,"
                "BIGINT:INT64,"
                "FLOAT:FLOAT64,"
                "DOUBLE:FLOAT64,"
                "DECIMAL:NUMERIC,"
                "VARCHAR:STRING,"
                "CHAR:STRING,"
                "TEXT:STRING,"
                "MEDIUMTEXT:STRING,"
                "LONGTEXT:STRING,"
                "DATE:DATE,"
                "DATETIME:TIMESTAMP,"
                "TIMESTAMP:TIMESTAMP,"
                "BINARY:BYTES,"
                "VARBINARY:BYTES,"
                "BLOB:BYTES"
            ),
            
            # Incremental load
            "incremental_column": "updated_at",
            "incremental_mode": "APPEND",
            
            # Output
            "output_format": "PARQUET",
            "staging_location": f"gs://{project_id}-mysql-staging",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    return response
```

#### PostgreSQL via JDBC

**PostgreSQL JDBC Transfer:**
```python
def create_postgres_jdbc_transfer(
    project_id,
    dataset_id,
    postgres_host,
    postgres_database,
    postgres_schema="public"
):
    """
    Create PostgreSQL transfer using JDBC.
    
    Prerequisites:
    - PostgreSQL JDBC driver (postgresql-x.x.x.jar)
    """
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="PostgreSQL JDBC Transfer",
        data_source_id="jdbc",
        destination_dataset_id=dataset_id,
        schedule="every 8 hours",
        params={
            # JDBC connection
            "jdbc_url": f"jdbc:postgresql://{postgres_host}:5432/{postgres_database}?ssl=true&sslmode=require",
            "driver_class": "org.postgresql.Driver",
            "username": "readonly_user",
            "password_secret": f"projects/{project_id}/secrets/postgres-password/versions/latest",
            
            # Connection settings
            "connection_pool_size": "5",
            "connection_properties": (
                "ApplicationName=BigQueryDataTransfer,"
                "tcpKeepAlive=true,"
                "loginTimeout=30"
            ),
            
            # Query configuration
            "fetch_size": "10000",
            "query_timeout": "3600",
            
            # Schema and tables
            "schema": postgres_schema,
            "tables": f"{postgres_schema}.customers,{postgres_schema}.orders,{postgres_schema}.line_items",
            
            # Custom queries with PostgreSQL syntax
            "custom_queries": (
                f"{postgres_schema}.customers:SELECT * FROM {postgres_schema}.customers WHERE created_at >= NOW() - INTERVAL '30 days';"
                f"{postgres_schema}.orders:SELECT * FROM {postgres_schema}.orders WHERE order_date >= CURRENT_DATE - INTERVAL '90 days'"
            ),
            
            # Data type mapping (PostgreSQL to BigQuery)
            "type_mapping": (
                "SMALLINT:INT64,"
                "INTEGER:INT64,"
                "BIGINT:INT64,"
                "REAL:FLOAT64,"
                "DOUBLE PRECISION:FLOAT64,"
                "NUMERIC:NUMERIC,"
                "DECIMAL:NUMERIC,"
                "VARCHAR:STRING,"
                "CHAR:STRING,"
                "TEXT:STRING,"
                "DATE:DATE,"
                "TIMESTAMP:TIMESTAMP,"
                "TIMESTAMPTZ:TIMESTAMP,"
                "BOOLEAN:BOOL,"
                "BYTEA:BYTES,"
                "UUID:STRING,"
                "JSON:STRING,"
                "JSONB:STRING,"
                "ARRAY:STRING"  # Arrays converted to JSON strings
            ),
            
            # Incremental load
            "incremental_column": "updated_at",
            "incremental_mode": "APPEND",
            
            # Output
            "output_format": "PARQUET",
            "compression": "SNAPPY",
            "staging_location": f"gs://{project_id}-postgres-staging",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    return response
```

#### Generic JDBC Transfer Template

**Universal JDBC Transfer Function:**
```python
def create_generic_jdbc_transfer(
    project_id,
    dataset_id,
    jdbc_url,
    driver_class,
    username,
    password_secret,
    tables_or_queries,
    **optional_params
):
    """
    Generic JDBC transfer function for any JDBC-compliant database.
    
    Args:
        project_id: GCP project ID
        dataset_id: Target BigQuery dataset
        jdbc_url: Full JDBC connection URL
        driver_class: JDBC driver class name
        username: Database username
        password_secret: Secret Manager path to password
        tables_or_queries: List of table names or dict of table:query mappings
        optional_params: Additional transfer parameters
    """
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    # Build base parameters
    params = {
        "jdbc_url": jdbc_url,
        "driver_class": driver_class,
        "username": username,
        "password_secret": password_secret,
        "connection_pool_size": optional_params.get("connection_pool_size", "5"),
        "fetch_size": optional_params.get("fetch_size", "10000"),
        "query_timeout": optional_params.get("query_timeout", "3600"),
        "output_format": optional_params.get("output_format", "PARQUET"),
        "compression": optional_params.get("compression", "SNAPPY"),
        "staging_location": optional_params.get("staging_location", f"gs://{project_id}-jdbc-staging"),
    }
    
    # Add tables or custom queries
    if isinstance(tables_or_queries, list):
        params["tables"] = ",".join(tables_or_queries)
    elif isinstance(tables_or_queries, dict):
        params["custom_queries"] = ";".join([
            f"{table}:{query}" for table, query in tables_or_queries.items()
        ])
    
    # Add optional parameters
    if "type_mapping" in optional_params:
        params["type_mapping"] = optional_params["type_mapping"]
    if "incremental_column" in optional_params:
        params["incremental_column"] = optional_params["incremental_column"]
    if "incremental_mode" in optional_params:
        params["incremental_mode"] = optional_params["incremental_mode"]
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name=optional_params.get("display_name", "JDBC Transfer"),
        data_source_id="jdbc",
        destination_dataset_id=dataset_id,
        schedule=optional_params.get("schedule", "every day 02:00"),
        params=params,
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    print(f"Created JDBC transfer: {response.name}")
    return response


# Example: IBM DB2 Transfer
db2_transfer = create_generic_jdbc_transfer(
    project_id="my-project",
    dataset_id="db2_data",
    jdbc_url="jdbc:db2://db2server:50000/SAMPLE",
    driver_class="com.ibm.db2.jcc.DB2Driver",
    username="db2user",
    password_secret="projects/my-project/secrets/db2-password/versions/latest",
    tables_or_queries=["SCHEMA1.TABLE1", "SCHEMA1.TABLE2"],
    display_name="DB2 Transfer",
    schedule="every 12 hours",
)

# Example: SAP HANA Transfer
hana_transfer = create_generic_jdbc_transfer(
    project_id="my-project",
    dataset_id="hana_data",
    jdbc_url="jdbc:sap://hanaserver:30015/?databaseName=SYSTEMDB",
    driver_class="com.sap.db.jdbc.Driver",
    username="hanauser",
    password_secret="projects/my-project/secrets/hana-password/versions/latest",
    tables_or_queries={"SALES": "SELECT * FROM SALES WHERE YEAR = YEAR(CURRENT_DATE)"},
    display_name="SAP HANA Transfer",
)
```

#### JDBC Driver Management

**Required JDBC Drivers by Database:**
```python
def get_jdbc_driver_info(database_type):
    """
    Get JDBC driver information for different databases.
    
    Returns driver class, Maven coordinates, and download URL.
    """
    
    drivers = {
        "teradata": {
            "driver_class": "com.teradata.jdbc.TeraDriver",
            "jar_files": ["terajdbc4.jar", "tdgssconfig.jar"],
            "download": "https://downloads.teradata.com/download/connectivity/jdbc-driver",
            "maven": "com.teradata.jdbc:terajdbc4:17.20.00.15",
        },
        "oracle": {
            "driver_class": "oracle.jdbc.OracleDriver",
            "jar_files": ["ojdbc8.jar"],  # or ojdbc11.jar for Java 11+
            "download": "https://www.oracle.com/database/technologies/jdbc-downloads.html",
            "maven": "com.oracle.database.jdbc:ojdbc8:21.1.0.0",
        },
        "sqlserver": {
            "driver_class": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
            "jar_files": ["mssql-jdbc-9.4.1.jre8.jar"],
            "download": "https://docs.microsoft.com/en-us/sql/connect/jdbc/download-microsoft-jdbc-driver-for-sql-server",
            "maven": "com.microsoft.sqlserver:mssql-jdbc:9.4.1.jre8",
        },
        "mysql": {
            "driver_class": "com.mysql.cj.jdbc.Driver",
            "jar_files": ["mysql-connector-java-8.0.28.jar"],
            "download": "https://dev.mysql.com/downloads/connector/j/",
            "maven": "mysql:mysql-connector-java:8.0.28",
        },
        "postgresql": {
            "driver_class": "org.postgresql.Driver",
            "jar_files": ["postgresql-42.3.3.jar"],
            "download": "https://jdbc.postgresql.org/download.html",
            "maven": "org.postgresql:postgresql:42.3.3",
        },
        "mariadb": {
            "driver_class": "org.mariadb.jdbc.Driver",
            "jar_files": ["mariadb-java-client-3.0.3.jar"],
            "download": "https://mariadb.com/kb/en/about-mariadb-connector-j/",
            "maven": "org.mariadb.jdbc:mariadb-java-client:3.0.3",
        },
        "db2": {
            "driver_class": "com.ibm.db2.jcc.DB2Driver",
            "jar_files": ["db2jcc4.jar"],
            "download": "https://www.ibm.com/support/pages/db2-jdbc-driver-versions-and-downloads",
            "maven": "com.ibm.db2:jcc:11.5.7.0",
        },
        "sap_hana": {
            "driver_class": "com.sap.db.jdbc.Driver",
            "jar_files": ["ngdbc.jar"],
            "download": "https://tools.hana.ondemand.com/#hanatools",
            "maven": "com.sap.cloud.db.jdbc:ngdbc:2.12.9",
        },
    }
    
    return drivers.get(database_type, {})


# Example usage
oracle_info = get_jdbc_driver_info("oracle")
print(f"Driver Class: {oracle_info['driver_class']}")
print(f"JAR Files: {oracle_info['jar_files']}")
print(f"Maven: {oracle_info['maven']}")
```

---

## 5. Schedule Configuration

### Schedule Patterns

**Common Schedules:**
```python
# Daily at specific time
schedule = "every day 02:00"

# Every X hours
schedule = "every 6 hours"
schedule = "every 12 hours"

# Weekly on specific day
schedule = "every monday 03:00"
schedule = "every sunday 00:00"

# First day of month
schedule = "first day of month 01:00"

# Custom interval (seconds)
schedule = "every 3600 seconds"  # Every hour

# Specific timezone
schedule = "every day 02:00 America/New_York"
schedule = "every day 14:00 Europe/London"
```

**Python Schedule Examples:**
```python
def create_transfer_with_schedule(
    project_id,
    dataset_id,
    data_source_id,
    schedule_pattern,
    params
):
    """Create transfer with custom schedule."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name=f"Transfer with {schedule_pattern}",
        data_source_id=data_source_id,
        destination_dataset_id=dataset_id,
        schedule=schedule_pattern,
        params=params,
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    return response

# Examples
# Hourly transfer during business hours
create_transfer_with_schedule(
    project_id="my-project",
    dataset_id="realtime_data",
    data_source_id="google_cloud_storage",
    schedule_pattern="every hour from 09:00 to 17:00",
    params={...}
)

# Off-peak batch transfer
create_transfer_with_schedule(
    project_id="my-project",
    dataset_id="batch_data",
    data_source_id="google_cloud_storage",
    schedule_pattern="every day 02:00",
    params={...}
)
```

---

## 6. Transfer Management

### Manual Transfer Runs

**Trigger Immediate Run:**
```python
def trigger_transfer_run(transfer_config_name, run_time=None):
    """Manually trigger a transfer run."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    
    # Optional: Specify run time for backfill
    if run_time:
        from google.protobuf import timestamp_pb2
        requested_run_time = timestamp_pb2.Timestamp()
        requested_run_time.FromDatetime(run_time)
    else:
        from datetime import datetime, timezone
        requested_run_time = timestamp_pb2.Timestamp()
        requested_run_time.FromDatetime(datetime.now(timezone.utc))
    
    response = client.start_manual_transfer_runs(
        parent=transfer_config_name,
        requested_run_time=requested_run_time,
    )
    
    for run in response.runs:
        print(f"Started transfer run: {run.name}")
        print(f"  State: {run.state}")
    
    return response
```

**Backfill Historical Data:**
```python
from datetime import datetime, timedelta

def backfill_historical_data(transfer_config_name, start_date, end_date):
    """Backfill data for a date range."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    
    current_date = start_date
    runs = []
    
    while current_date <= end_date:
        print(f"Triggering backfill for {current_date.date()}")
        
        response = trigger_transfer_run(
            transfer_config_name=transfer_config_name,
            run_time=current_date
        )
        
        runs.extend(response.runs)
        current_date += timedelta(days=1)
    
    print(f"Triggered {len(runs)} backfill runs")
    return runs

# Example: Backfill last 30 days
backfill_historical_data(
    transfer_config_name="projects/123/locations/us/transferConfigs/abc",
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)
```

### Monitor Transfer Runs

**Check Transfer Status:**
```python
def monitor_transfer_runs(transfer_config_name, max_results=10):
    """Monitor recent transfer runs."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    
    request = bigquery_datatransfer_v1.ListTransferRunsRequest(
        parent=transfer_config_name,
        states=[
            bigquery_datatransfer_v1.TransferState.PENDING,
            bigquery_datatransfer_v1.TransferState.RUNNING,
            bigquery_datatransfer_v1.TransferState.SUCCEEDED,
            bigquery_datatransfer_v1.TransferState.FAILED,
        ],
        page_size=max_results,
    )
    
    runs = client.list_transfer_runs(request=request)
    
    for run in runs:
        print(f"Run: {run.name}")
        print(f"  Schedule: {run.schedule_time}")
        print(f"  State: {run.state.name}")
        print(f"  Start: {run.start_time}")
        print(f"  End: {run.end_time}")
        
        if run.state == bigquery_datatransfer_v1.TransferState.FAILED:
            print(f"  Error: {run.error_status.message}")
        
        print("---")
    
    return runs
```

---

## 7. Error Handling

### Notification Configuration

**Configure Pub/Sub Notifications:**
```python
def configure_transfer_notifications(project_id, transfer_config_name, topic_name):
    """Configure Pub/Sub notifications for transfer."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    
    # Create or get existing transfer config
    transfer_config = client.get_transfer_config(name=transfer_config_name)
    
    # Update with notification topic
    transfer_config.notification_pubsub_topic = f"projects/{project_id}/topics/{topic_name}"
    
    update_mask = {"paths": ["notification_pubsub_topic"]}
    
    updated_config = client.update_transfer_config(
        transfer_config=transfer_config,
        update_mask=update_mask,
    )
    
    print(f"Notifications configured: {updated_config.notification_pubsub_topic}")
    return updated_config
```

**Process Notifications:**
```python
from google.cloud import pubsub_v1
import json

def process_transfer_notifications(project_id, subscription_id):
    """Process transfer notifications from Pub/Sub."""
    
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    
    def callback(message):
        try:
            # Parse notification
            data = json.loads(message.data.decode('utf-8'))
            
            transfer_run_name = data.get('name')
            state = data.get('state')
            
            print(f"Transfer Run: {transfer_run_name}")
            print(f"State: {state}")
            
            if state == 'FAILED':
                error_status = data.get('errorStatus', {})
                print(f"Error: {error_status.get('message')}")
                
                # Send alert
                send_alert(f"Transfer failed: {transfer_run_name}")
            
            elif state == 'SUCCEEDED':
                print("Transfer completed successfully")
            
            message.ack()
            
        except Exception as e:
            print(f"Error processing notification: {e}")
            message.nack()
    
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    
    print(f"Listening for notifications on {subscription_path}")
    
    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()

def send_alert(message):
    """Send alert via email, Slack, etc."""
    print(f"ALERT: {message}")
    # Implement actual alerting logic
```

---

## 8. Performance Optimization

### Parallel Loading

**Best Practices:**
- Split large datasets into multiple smaller files
- Use file patterns for parallel loading
- Optimize file formats (Parquet > Avro > JSON > CSV)

```python
# Good: Multiple files loaded in parallel
data_path = "gs://my-bucket/data/2025/01/01/*.parquet"  # Many files

# Less optimal: Single large file
data_path = "gs://my-bucket/data/2025/01/01/all_data.csv"  # One file
```

### File Format Selection

**Performance Comparison:**
```python
def compare_file_formats():
    """
    File format recommendations for BigQuery Data Transfer:
    
    1. Parquet (Best)
       - Columnar format
       - Built-in compression
       - Schema embedded
       - Fastest load times
    
    2. Avro (Good)
       - Row-based format
       - Schema embedded
       - Good compression
       - Fast load times
    
    3. ORC (Good)
       - Columnar format
       - Optimized for Hive
       - Good for data warehouse migrations
    
    4. JSON (Acceptable)
       - Flexible schema
       - Human readable
       - Slower load times
       - Larger file sizes
    
    5. CSV (Least preferred)
       - No schema
       - No compression
       - Slowest load times
       - Encoding issues
    """
    
    recommendations = {
        "google_cloud_storage": "PARQUET",
        "amazon_s3": "PARQUET",
        "data_warehouse_migration": "PARQUET or ORC",
        "log_files": "JSON",
        "legacy_systems": "CSV (convert to Parquet if possible)",
    }
    
    return recommendations
```

---

## 9. Security Best Practices

### Credential Management

**Use Secret Manager:**
```python
def create_transfer_with_secrets(project_id, dataset_id):
    """Create transfer using Secret Manager for credentials."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="Secure Transfer with Secret Manager",
        data_source_id="amazon_s3",
        destination_dataset_id=dataset_id,
        schedule="every day 02:00",
        params={
            "data_path": "s3://my-bucket/data/*.parquet",
            "destination_table_name_template": "imported_data",
            "file_format": "PARQUET",
            # Reference secrets instead of plain credentials
            "access_key_id_secret_version": 
                f"projects/{project_id}/secrets/s3-access-key/versions/latest",
            "secret_access_key_secret_version": 
                f"projects/{project_id}/secrets/s3-secret-key/versions/latest",
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    return response
```

### IAM Permissions

**Required Roles:**
```bash
# Service account for Data Transfer Service
gcloud iam service-accounts create bq-transfer-sa \
    --display-name="BigQuery Data Transfer Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding my-project \
    --member=serviceAccount:bq-transfer-sa@my-project.iam.gserviceaccount.com \
    --role=roles/bigquery.admin

# For Cloud Storage source
gsutil iam ch \
    serviceAccount:bq-transfer-sa@my-project.iam.gserviceaccount.com:roles/storage.objectViewer \
    gs://source-bucket

# For Secret Manager
gcloud secrets add-iam-policy-binding my-secret \
    --member=serviceAccount:bq-transfer-sa@my-project.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor
```

---

## 10. Cost Optimization

### Minimize Transfer Costs

**Best Practices:**
```python
def optimize_transfer_costs():
    """
    Cost optimization strategies:
    
    1. Schedule during off-peak hours
       - Reduce impact on production systems
       - Potential cost savings from source systems
    
    2. Use incremental transfers
       - Only transfer changed data
       - Reduces data volume and processing time
    
    3. Optimize file formats
       - Compressed formats (Parquet, Avro)
       - Reduce storage and transfer costs
    
    4. Right-size transfer frequency
       - Balance freshness vs. cost
       - Not all data needs real-time updates
    
    5. Clean up failed runs
       - Remove incomplete or failed data
       - Avoid redundant storage costs
    
    6. Use appropriate storage tiers
       - Standard for frequently accessed data
       - Long-term storage for archival data
    """
    
    pass

# Example: Incremental transfer configuration
def create_incremental_transfer(project_id, dataset_id):
    """Configure incremental data transfer."""
    
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    parent = f"projects/{project_id}/locations/us"
    
    transfer_config = bigquery_datatransfer_v1.TransferConfig(
        display_name="Incremental Transfer",
        data_source_id="google_cloud_storage",
        destination_dataset_id=dataset_id,
        schedule="every 6 hours",
        params={
            # Only load new files
            "data_path_template": "gs://my-bucket/incremental/{run_date}/*.parquet",
            "destination_table_name_template": "incremental_data",
            "write_disposition": "WRITE_APPEND",  # Append, don't overwrite
            "delete_source_files": "true",  # Clean up after successful load
        },
    )
    
    response = client.create_transfer_config(
        parent=parent,
        transfer_config=transfer_config
    )
    
    return response
```

---

## 11. Common Anti-Patterns

### ❌ Anti-Pattern 1: Hardcoded Credentials
**Problem:** Security risk, difficult to rotate
**Solution:** Use Secret Manager for all credentials

### ❌ Anti-Pattern 2: No Error Monitoring
**Problem:** Failed transfers go unnoticed
**Solution:** Configure Pub/Sub notifications and alerting

### ❌ Anti-Pattern 3: Inappropriate Schedule
**Problem:** Transfers during peak hours impact performance
**Solution:** Schedule during off-peak hours

### ❌ Anti-Pattern 4: Large Single Files
**Problem:** Slow load times, no parallelization
**Solution:** Split into multiple smaller files

### ❌ Anti-Pattern 5: Ignoring Failed Runs
**Problem:** Data gaps and inconsistencies
**Solution:** Monitor and retry failed runs

### ❌ Anti-Pattern 6: Wrong Connection Method
**Problem:** Using JDBC for large datasets or TPT for small tables
**Solution:** Use TPT (4-16 sessions) for datasets >100 GB, JDBC for smaller datasets

### ❌ Anti-Pattern 7: Insufficient Connection Pool
**Problem:** JDBC connection pool exhaustion causing timeouts
**Solution:** Size connection pool appropriately (5-10 connections typical)

### ❌ Anti-Pattern 8: Not Optimizing Fetch Size
**Problem:** Memory issues or slow performance with default fetch size
**Solution:** Tune fetch_size parameter (5,000-50,000 rows based on row size)

### ❌ Anti-Pattern 9: Missing Type Mappings
**Problem:** Data type conversion errors during transfer
**Solution:** Explicitly define type_mapping for source to BigQuery conversions

### ❌ Anti-Pattern 10: No Incremental Load Strategy
**Problem:** Full table scans on every transfer, wasting resources
**Solution:** Use partition_column or incremental_column for incremental loads

---

## 12. Quick Reference Checklist

### Setup
- [ ] Identify data source and destination dataset
- [ ] Configure source system permissions
- [ ] Set up service account with appropriate IAM roles
- [ ] Store credentials in Secret Manager
- [ ] Create destination dataset and tables
- [ ] Install required JDBC drivers (if using JDBC)
- [ ] Verify network connectivity to source databases

### Database Migration (Teradata/JDBC)
- [ ] Determine migration method: TPT (>100 GB) or JDBC (<100 GB)
- [ ] For TPT: Install TPT utilities (tbuild, tlogview)
- [ ] For JDBC: Download and configure appropriate JDBC driver
- [ ] Optimize TPT session count (4-16 based on table size)
- [ ] Configure JDBC connection pool size (5-10 typical)
- [ ] Set appropriate fetch size (5,000-50,000 rows)
- [ ] Define type_mapping for data type conversions
- [ ] Configure incremental load with partition_column
- [ ] Set up staging bucket in Cloud Storage
- [ ] Test with small table before full migration

### Configuration
- [ ] Choose appropriate schedule (off-peak preferred)
- [ ] Configure file format and path patterns
- [ ] Set write disposition (APPEND, TRUNCATE, or EMPTY)
- [ ] Enable Pub/Sub notifications
- [ ] Test with manual run before scheduling
- [ ] Configure query timeout appropriately
- [ ] Set max_bad_records tolerance

### Monitoring
- [ ] Set up Pub/Sub subscription for notifications
- [ ] Configure Cloud Monitoring alerts
- [ ] Monitor transfer run history
- [ ] Track data freshness
- [ ] Review error logs regularly
- [ ] Monitor TPT session utilization (if applicable)
- [ ] Check JDBC connection pool health

### Security
- [ ] Use Secret Manager for credentials
- [ ] Apply least privilege IAM roles
- [ ] Encrypt sensitive data in transit and at rest
- [ ] Audit access logs
- [ ] Rotate credentials regularly
- [ ] Use secure JDBC URLs (SSL/TLS enabled)
- [ ] Restrict source database user permissions to read-only

### Optimization
- [ ] Use compressed file formats (Parquet, Avro)
- [ ] Split large files for parallel loading
- [ ] Configure incremental transfers
- [ ] Schedule during off-peak hours
- [ ] Clean up old transfer runs
- [ ] Tune TPT buffer size and parallel degree
- [ ] Optimize JDBC fetch size based on row width
- [ ] Use custom queries to filter data at source

### Performance Tuning
- [ ] For TPT: Adjust tpt_sessions based on table size
- [ ] For TPT: Configure tpt_buffer_size (64KB-256KB)
- [ ] For JDBC: Tune connection_pool_size
- [ ] For JDBC: Optimize fetch_size parameter
- [ ] Use PARQUET format with SNAPPY compression
- [ ] Enable parallel query execution (where supported)
- [ ] Monitor and optimize query timeout settings

---

## Additional Resources

- [BigQuery Data Transfer Service Documentation](https://cloud.google.com/bigquery-transfer/docs)
- [Teradata to BigQuery Migration Guide](https://cloud.google.com/architecture/teradata-to-bigquery-migration-guide)
- [JDBC Data Sources](https://cloud.google.com/bigquery-transfer/docs/jdbc-sources)
- [Teradata TPT Reference](https://docs.teradata.com/r/Teradata-Parallel-Transporter-Reference)
- [JDBC Driver Downloads](https://cloud.google.com/bigquery-transfer/docs/jdbc-drivers)
- [Secret Manager Best Practices](https://cloud.google.com/secret-manager/docs/best-practices)
- [Transfer Service Pricing](https://cloud.google.com/bigquery-transfer/pricing)

---

*Last Updated: December 27, 2025*
*Version: 1.1*

*Best Practices for Google Cloud Data Engineer Certification*
