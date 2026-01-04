# AlloyDB Best Practices

## Overview
AlloyDB for PostgreSQL is Google Cloud's fully managed, PostgreSQL-compatible database service designed for demanding transactional and analytical workloads. It offers enterprise-grade performance, availability, and scale while maintaining 100% PostgreSQL compatibility. This guide covers best practices for optimal performance, high availability, security, and cost management.

---

## 1. Architecture Design Best Practices

### Cluster Architecture

**AlloyDB Cluster Components:**
- **Primary Instance**: Handles read-write operations
- **Read Pool Instances**: Handle read-only queries with autoscaling
- **Cross-Region Replica**: Disaster recovery and read scaling
- **Backup and Recovery**: Automated continuous backups

### High Availability Configuration

✅ **Enable High Availability (HA)**: Automatic failover with zero data loss
✅ **Use Read Pool**: Offload read traffic from primary instance
✅ **Configure Cross-Region Replicas**: For disaster recovery (RPO < 1 second)
✅ **Multi-zone deployment**: Primary and standby in different zones
✅ **Connection pooling**: Use PgBouncer or built-in connection pooling

**Create HA Cluster:**
```bash
# Create AlloyDB cluster with HA enabled
gcloud alloydb clusters create production-cluster \
    --region=us-central1 \
    --network=projects/PROJECT_ID/global/networks/vpc-network \
    --enable-private-service-connect \
    --password=STRONG_PASSWORD

# Create primary instance with HA
gcloud alloydb instances create primary-instance \
    --cluster=production-cluster \
    --region=us-central1 \
    --instance-type=PRIMARY \
    --cpu-count=8 \
    --availability-type=REGIONAL \
    --database-flags=max_connections=1000
```

### Read Pool Configuration

**Benefits of Read Pools:**
- ✅ Horizontal read scaling without impacting primary
- ✅ Automatic load balancing across read replicas
- ✅ Sub-second replication lag
- ✅ Autoscaling based on CPU utilization

```bash
# Create read pool with autoscaling
gcloud alloydb instances create read-pool-1 \
    --cluster=production-cluster \
    --region=us-central1 \
    --instance-type=READ_POOL \
    --read-pool-node-count=2 \
    --cpu-count=4 \
    --enable-autoscaling \
    --min-node-count=2 \
    --max-node-count=10 \
    --autoscaling-cpu-target=70
```

**Read Pool Best Practices:**
✅ Start with 2-3 read pool nodes for redundancy
✅ Set autoscaling CPU target to 70% for optimal performance
✅ Monitor read pool node count and adjust limits based on traffic patterns
✅ Use read pool for reporting, analytics, and read-heavy queries
✅ Keep write operations on primary instance

---

## 2. Performance Optimization

### Query Performance

#### Columnar Engine for Analytics

AlloyDB's columnar engine accelerates analytical queries up to 100x faster than traditional row storage.

**Enable Columnar Engine:**
```sql
-- Enable columnar engine at database level
ALTER DATABASE mydb SET google_columnar_engine.enabled = on;

-- Enable for specific session
SET google_columnar_engine.enabled = on;

-- Verify columnar engine is active
SHOW google_columnar_engine.enabled;
```

**When to Use Columnar Engine:**
✅ Complex analytical queries with aggregations
✅ Queries scanning large tables (millions+ rows)
✅ OLAP workloads with few column access
✅ Reporting and business intelligence queries
❌ Simple OLTP queries (overhead not worth it)
❌ Queries accessing most columns

**Example - Analytical Query:**
```sql
-- Enable columnar engine for analytics
SET google_columnar_engine.enabled = on;

-- Complex analytical query (100x faster with columnar engine)
SELECT 
    DATE_TRUNC('month', order_date) as month,
    product_category,
    SUM(revenue) as total_revenue,
    AVG(revenue) as avg_revenue,
    COUNT(DISTINCT customer_id) as unique_customers
FROM orders
WHERE order_date >= '2024-01-01'
GROUP BY 1, 2
ORDER BY 1 DESC, 3 DESC;
```

#### Indexing Best Practices

```sql
-- Create B-tree index for equality and range queries
CREATE INDEX idx_orders_customer_date 
ON orders(customer_id, order_date DESC);

-- Create partial index for frequently filtered data
CREATE INDEX idx_active_orders 
ON orders(order_date) 
WHERE status = 'active';

-- Create GIN index for JSONB columns
CREATE INDEX idx_metadata_gin 
ON products USING GIN(metadata);

-- Create covering index (index-only scan)
CREATE INDEX idx_orders_covering 
ON orders(customer_id, order_date) 
INCLUDE (total_amount, status);

-- Monitor index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

#### Query Optimization Techniques

```sql
-- Use EXPLAIN ANALYZE to understand query plans
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM orders 
WHERE customer_id = 12345 
AND order_date > '2024-01-01';

-- Use CTEs for complex queries
WITH customer_stats AS (
    SELECT 
        customer_id,
        COUNT(*) as order_count,
        SUM(total_amount) as lifetime_value
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '1 year'
    GROUP BY customer_id
)
SELECT 
    c.customer_name,
    cs.order_count,
    cs.lifetime_value
FROM customers c
JOIN customer_stats cs ON c.customer_id = cs.customer_id
WHERE cs.lifetime_value > 10000;

-- Use window functions efficiently
SELECT 
    order_id,
    customer_id,
    order_date,
    total_amount,
    AVG(total_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date 
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) as moving_avg
FROM orders;
```

### Connection Management

**Connection Pooling Best Practices:**

```python
import psycopg2
from psycopg2 import pool

# Create connection pool
connection_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=5,
    maxconn=20,
    host='10.0.0.5',  # AlloyDB private IP
    port=5432,
    database='mydb',
    user='app_user',
    password='PASSWORD'
)

def execute_query(query, params=None):
    """Execute query using connection pool."""
    conn = None
    try:
        # Get connection from pool
        conn = connection_pool.getconn()
        cursor = conn.cursor()
        
        cursor.execute(query, params)
        
        if query.strip().upper().startswith('SELECT'):
            result = cursor.fetchall()
            return result
        else:
            conn.commit()
            
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            # Return connection to pool
            connection_pool.putconn(conn)
```

**Connection Pooler Configuration (PgBouncer):**
```ini
[databases]
mydb = host=10.0.0.5 port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
reserve_pool_timeout = 3
server_idle_timeout = 600
```

### Memory and Resource Configuration

```sql
-- Optimize memory settings for your workload
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';

-- Connection settings
ALTER SYSTEM SET max_connections = 1000;
ALTER SYSTEM SET max_prepared_transactions = 100;

-- Query optimization
ALTER SYSTEM SET random_page_cost = 1.1;  -- For SSD storage
ALTER SYSTEM SET effective_io_concurrency = 200;

-- WAL settings for performance
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET checkpoint_timeout = '10min';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- Reload configuration
SELECT pg_reload_conf();
```

---

## 3. Data Migration Best Practices

### Migration Strategies

#### Option 1: Database Migration Service (DMS)

**Recommended for:**
- ✅ Large databases (100GB+)
- ✅ Minimal downtime requirements
- ✅ Continuous replication needed
- ✅ Heterogeneous migrations (Oracle, MySQL → AlloyDB)

```bash
# Create migration job using DMS
gcloud database-migration migration-jobs create postgres-to-alloydb \
    --region=us-central1 \
    --type=CONTINUOUS \
    --source=projects/PROJECT/locations/us-central1/connectionProfiles/source-postgres \
    --destination=projects/PROJECT/locations/us-central1/connectionProfiles/alloydb-dest \
    --display-name="Production Migration"
```

#### Option 2: pg_dump and pg_restore

**Recommended for:**
- ✅ Small to medium databases (<100GB)
- ✅ One-time migration with planned downtime
- ✅ Same PostgreSQL version

```bash
# Dump source database with custom format
pg_dump -Fc \
    -h source-host \
    -U postgres \
    -d source_db \
    -f /backup/source_db.dump \
    --verbose

# Restore to AlloyDB
pg_restore -h 10.0.0.5 \
    -U postgres \
    -d target_db \
    -j 8 \
    --verbose \
    /backup/source_db.dump

# Verify row counts
psql -h 10.0.0.5 -U postgres -d target_db -c \
"SELECT schemaname, tablename, n_live_tup 
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC;"
```

#### Option 3: Logical Replication

**Recommended for:**
- ✅ Near-zero downtime migrations
- ✅ Gradual cutover
- ✅ PostgreSQL 10+ source

```sql
-- On source PostgreSQL instance
-- Enable logical replication
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_replication_slots = 10;
ALTER SYSTEM SET max_wal_senders = 10;

-- Restart PostgreSQL (if needed)

-- Create publication
CREATE PUBLICATION alloydb_migration FOR ALL TABLES;

-- On AlloyDB instance
-- Create subscription
CREATE SUBSCRIPTION alloydb_sub
CONNECTION 'host=source-host port=5432 dbname=source_db user=replication_user password=PASSWORD'
PUBLICATION alloydb_migration;

-- Monitor replication lag
SELECT 
    slot_name,
    confirmed_flush_lsn,
    pg_current_wal_lsn(),
    (pg_current_wal_lsn() - confirmed_flush_lsn) AS lag_bytes
FROM pg_replication_slots;
```

### Post-Migration Validation

```sql
-- Compare row counts
SELECT 
    'source' as system, 
    schemaname, 
    tablename, 
    n_live_tup 
FROM pg_stat_user_tables
UNION ALL
SELECT 
    'alloydb', 
    schemaname, 
    tablename, 
    n_live_tup 
FROM pg_stat_user_tables;

-- Verify data integrity with checksums
SELECT 
    tablename,
    COUNT(*) as row_count,
    MD5(STRING_AGG(column_name::text, '' ORDER BY column_name)) as schema_checksum
FROM information_schema.columns
WHERE table_schema = 'public'
GROUP BY tablename;

-- Analyze all tables after migration
ANALYZE VERBOSE;

-- Update statistics
VACUUM ANALYZE;
```

---

## 4. Security Best Practices

### Network Security

**Private Service Connect (Recommended):**
```bash
# Create AlloyDB cluster with Private Service Connect
gcloud alloydb clusters create secure-cluster \
    --region=us-central1 \
    --network=projects/PROJECT_ID/global/networks/vpc-network \
    --enable-private-service-connect

# No public IP exposure - accessible only via VPC
```

**VPC Peering (Alternative):**
```bash
# Create peering connection
gcloud services vpc-peerings connect \
    --service=servicenetworking.googleapis.com \
    --ranges=alloydb-ip-range \
    --network=vpc-network
```

### Authentication and Authorization

**IAM-based Authentication:**
```bash
# Grant AlloyDB IAM roles
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:app-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/alloydb.client"

# Connect using IAM authentication
gcloud alloydb instances describe primary-instance \
    --cluster=production-cluster \
    --region=us-central1 \
    --format="value(ipAddress)"
```

**Database-level Security:**
```sql
-- Create roles with least privilege
CREATE ROLE app_reader;
GRANT CONNECT ON DATABASE mydb TO app_reader;
GRANT USAGE ON SCHEMA public TO app_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_reader;

CREATE ROLE app_writer;
GRANT app_reader TO app_writer;
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_writer;

-- Create users
CREATE USER app_user WITH PASSWORD 'STRONG_PASSWORD';
GRANT app_writer TO app_user;

-- Row-level security
CREATE POLICY customer_isolation ON orders
FOR ALL
TO app_user
USING (customer_id = current_setting('app.current_customer_id')::int);

ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
```

### Encryption

**Encryption at Rest:**
```bash
# Use CMEK (Customer-Managed Encryption Keys)
gcloud alloydb clusters create encrypted-cluster \
    --region=us-central1 \
    --network=projects/PROJECT_ID/global/networks/vpc-network \
    --kms-key=projects/PROJECT/locations/us-central1/keyRings/alloydb-keyring/cryptoKeys/alloydb-key
```

**Encryption in Transit:**
```python
import psycopg2

# Connect with SSL/TLS encryption
conn = psycopg2.connect(
    host='10.0.0.5',
    port=5432,
    database='mydb',
    user='app_user',
    password='PASSWORD',
    sslmode='require',  # Enforce SSL
    sslrootcert='/path/to/server-ca.pem',
    sslcert='/path/to/client-cert.pem',
    sslkey='/path/to/client-key.pem'
)
```

### Audit Logging

```bash
# Enable Cloud Audit Logs
gcloud logging read "resource.type=alloydb.googleapis.com/Instance" \
    --limit=50 \
    --format=json

# Export audit logs to BigQuery
gcloud logging sinks create alloydb-audit-sink \
    bigquery.googleapis.com/projects/PROJECT_ID/datasets/alloydb_logs \
    --log-filter='resource.type="alloydb.googleapis.com/Instance"'
```

**Query Audit Logs:**
```sql
-- Query audit logs in BigQuery
SELECT
  timestamp,
  protoPayload.authenticationInfo.principalEmail,
  protoPayload.methodName,
  protoPayload.resourceName,
  protoPayload.status.message
FROM `project.alloydb_logs.cloudaudit_googleapis_com_activity_*`
WHERE DATE(_PARTITIONTIME) >= CURRENT_DATE() - 7
  AND protoPayload.serviceName = 'alloydb.googleapis.com'
ORDER BY timestamp DESC
LIMIT 100;
```

---

## 5. Backup and Disaster Recovery

### Automated Backups

**Continuous Backup (Recommended):**
```bash
# Backups are automatic and continuous
# Retention period: 1-35 days (default: 14 days)

# Configure backup retention
gcloud alloydb clusters update production-cluster \
    --region=us-central1 \
    --automated-backup-retention-period=35d \
    --automated-backup-window-start-time=02:00 \
    --automated-backup-window-duration=4h
```

**On-Demand Backups:**
```bash
# Create manual backup
gcloud alloydb backups create manual-backup-20250128 \
    --cluster=production-cluster \
    --region=us-central1

# List backups
gcloud alloydb backups list \
    --cluster=production-cluster \
    --region=us-central1

# Restore from backup
gcloud alloydb clusters restore production-cluster-restored \
    --region=us-central1 \
    --backup=projects/PROJECT/locations/us-central1/backups/manual-backup-20250128
```

### Point-in-Time Recovery (PITR)

```bash
# Restore to specific timestamp (within retention period)
gcloud alloydb clusters restore production-cluster-pitr \
    --region=us-central1 \
    --source-cluster=production-cluster \
    --point-in-time=2025-01-28T14:30:00Z

# Verify restored cluster
gcloud alloydb clusters describe production-cluster-pitr \
    --region=us-central1
```

### Cross-Region Replicas for DR

```bash
# Create cross-region replica for disaster recovery
gcloud alloydb clusters create dr-cluster \
    --region=us-east1 \
    --secondary-cluster \
    --primary-cluster=projects/PROJECT/locations/us-central1/clusters/production-cluster

# Promote cross-region replica to primary (during DR)
gcloud alloydb clusters promote dr-cluster \
    --region=us-east1
```

### Disaster Recovery Testing

```sql
-- DR Drill Script
-- 1. Test read access to cross-region replica
SELECT COUNT(*) FROM orders;

-- 2. Verify replication lag
SELECT 
    NOW() - pg_last_xact_replay_timestamp() AS replication_lag;

-- 3. Test application connectivity
\conninfo

-- 4. Validate data consistency
SELECT 
    schemaname,
    tablename,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

---

## 6. Monitoring and Observability

### Key Metrics to Monitor

**Cloud Monitoring Metrics:**
```bash
# CPU utilization
gcloud monitoring time-series list \
    --filter='metric.type="alloydb.googleapis.com/instance/cpu/utilization"' \
    --interval-start-time="2025-01-28T00:00:00Z" \
    --interval-end-time="2025-01-28T23:59:59Z"

# Memory utilization
# Connection count
# Transaction rate
# Replication lag (for read pools and replicas)
```

**Custom Monitoring Dashboard:**
```python
from google.cloud import monitoring_v3
import time

def create_alloydb_dashboard(project_id):
    """Create Cloud Monitoring dashboard for AlloyDB."""
    
    client = monitoring_v3.DashboardsServiceClient()
    project_name = f"projects/{project_id}"
    
    dashboard = monitoring_v3.Dashboard(
        display_name="AlloyDB Production Monitoring",
        grid_layout=monitoring_v3.GridLayout(
            widgets=[
                # CPU utilization widget
                monitoring_v3.Widget(
                    title="CPU Utilization",
                    xy_chart=monitoring_v3.XyChart(
                        data_sets=[
                            monitoring_v3.XyChart.DataSet(
                                time_series_query=monitoring_v3.TimeSeriesQuery(
                                    time_series_filter=monitoring_v3.TimeSeriesFilter(
                                        filter='metric.type="alloydb.googleapis.com/instance/cpu/utilization"'
                                    )
                                )
                            )
                        ]
                    )
                ),
                # Connection count widget
                monitoring_v3.Widget(
                    title="Active Connections",
                    xy_chart=monitoring_v3.XyChart(
                        data_sets=[
                            monitoring_v3.XyChart.DataSet(
                                time_series_query=monitoring_v3.TimeSeriesQuery(
                                    time_series_filter=monitoring_v3.TimeSeriesFilter(
                                        filter='metric.type="alloydb.googleapis.com/instance/postgres/connections"'
                                    )
                                )
                            )
                        ]
                    )
                )
            ]
        )
    )
    
    dashboard = client.create_dashboard(
        name=project_name,
        dashboard=dashboard
    )
    
    print(f"Created dashboard: {dashboard.name}")
    return dashboard
```

### Performance Insights

**Query Performance Statistics:**
```sql
-- Install pg_stat_statements extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Top 10 slowest queries
SELECT 
    queryid,
    LEFT(query, 100) as query_preview,
    calls,
    mean_exec_time,
    total_exec_time,
    stddev_exec_time,
    rows
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Queries with high I/O
SELECT 
    queryid,
    LEFT(query, 100) as query_preview,
    calls,
    shared_blks_hit,
    shared_blks_read,
    (shared_blks_hit::float / NULLIF(shared_blks_hit + shared_blks_read, 0) * 100)::numeric(5,2) as cache_hit_ratio
FROM pg_stat_statements
WHERE shared_blks_read > 0
ORDER BY shared_blks_read DESC
LIMIT 10;

-- Reset statistics (after optimization)
SELECT pg_stat_statements_reset();
```

### Alerting Configuration

```bash
# Create alert policy for high CPU
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="AlloyDB High CPU Alert" \
    --condition-display-name="CPU > 80%" \
    --condition-threshold-value=0.8 \
    --condition-threshold-duration=300s \
    --condition-filter='metric.type="alloydb.googleapis.com/instance/cpu/utilization" resource.type="alloydb.googleapis.com/Instance"'

# Create alert for high connection count
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="AlloyDB High Connections" \
    --condition-display-name="Connections > 900" \
    --condition-threshold-value=900 \
    --condition-threshold-duration=120s \
    --condition-filter='metric.type="alloydb.googleapis.com/instance/postgres/connections" resource.type="alloydb.googleapis.com/Instance"'
```

---

## 7. Cost Optimization

### Right-sizing Instances

**CPU and Memory Sizing:**
```bash
# Start with appropriate instance size
# 2 vCPUs = 16 GB RAM
# 4 vCPUs = 32 GB RAM
# 8 vCPUs = 64 GB RAM
# 16 vCPUs = 128 GB RAM

# Create instance with appropriate sizing
gcloud alloydb instances create primary-instance \
    --cluster=production-cluster \
    --region=us-central1 \
    --instance-type=PRIMARY \
    --cpu-count=4 \
    --availability-type=REGIONAL
```

**Monitoring for Right-sizing:**
```sql
-- Check current resource utilization
SELECT 
    NOW() as timestamp,
    (SELECT setting FROM pg_settings WHERE name = 'max_connections') as max_connections,
    (SELECT count(*) FROM pg_stat_activity) as current_connections,
    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
    pg_size_pretty(pg_database_size(current_database())) as database_size;
```

### Read Pool Optimization

```bash
# Configure autoscaling to match workload
gcloud alloydb instances update read-pool-1 \
    --cluster=production-cluster \
    --region=us-central1 \
    --enable-autoscaling \
    --min-node-count=2 \
    --max-node-count=5 \
    --autoscaling-cpu-target=75
```

### Storage Optimization

```sql
-- Identify large tables
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 20;

-- Reclaim space from deleted rows
VACUUM FULL VERBOSE tablename;

-- Archive old data
CREATE TABLE orders_archive AS 
SELECT * FROM orders WHERE order_date < '2023-01-01';

DELETE FROM orders WHERE order_date < '2023-01-01';

VACUUM ANALYZE orders;
```

### Backup Cost Optimization

```bash
# Optimize backup retention (balance cost vs recovery needs)
gcloud alloydb clusters update production-cluster \
    --region=us-central1 \
    --automated-backup-retention-period=14d  # Reduce from 35d if appropriate

# Delete old manual backups
gcloud alloydb backups delete old-backup-id \
    --region=us-central1
```

---

## 8. High Availability and Failover

### Automatic Failover

**HA Configuration:**
```bash
# HA is enabled via availability-type=REGIONAL
# Automatic failover to standby in different zone
# RTO: typically 60-120 seconds
# RPO: zero data loss (synchronous replication)

# Monitor failover events
gcloud logging read \
    'resource.type="alloydb.googleapis.com/Instance"
     AND (protoPayload.methodName="PromoteReplica" OR 
          protoPayload.methodName="Failover")' \
    --limit=10 \
    --format=json
```

### Application-Level Failover Handling

```python
import psycopg2
from psycopg2 import OperationalError
import time

def connect_with_retry(max_retries=5, retry_delay=2):
    """Connect to AlloyDB with automatic retry on failover."""
    
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host='10.0.0.5',
                port=5432,
                database='mydb',
                user='app_user',
                password='PASSWORD',
                connect_timeout=10,
                options='-c statement_timeout=30000'
            )
            
            # Verify connection
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            cursor.close()
            
            print(f"Connected successfully on attempt {attempt + 1}")
            return conn
            
        except OperationalError as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                raise Exception("Failed to connect after maximum retries")

# Usage
try:
    conn = connect_with_retry()
    # Use connection
except Exception as e:
    print(f"Connection failed: {e}")
    # Alert operations team
```

### Health Checks

```python
def check_alloydb_health(host, port, database, user, password):
    """Comprehensive health check for AlloyDB."""
    
    health_status = {
        'connectivity': False,
        'replication_lag': None,
        'active_connections': 0,
        'max_connections': 0,
        'connection_utilization': 0
    }
    
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connect_timeout=5
        )
        
        health_status['connectivity'] = True
        cursor = conn.cursor()
        
        # Check replication lag (for read pools)
        cursor.execute("""
            SELECT EXTRACT(EPOCH FROM (NOW() - pg_last_xact_replay_timestamp()))::INT 
            AS lag_seconds
        """)
        lag = cursor.fetchone()
        if lag and lag[0] is not None:
            health_status['replication_lag'] = lag[0]
        
        # Check connections
        cursor.execute("""
            SELECT 
                (SELECT count(*) FROM pg_stat_activity) as active,
                (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max
        """)
        row = cursor.fetchone()
        health_status['active_connections'] = row[0]
        health_status['max_connections'] = row[1]
        health_status['connection_utilization'] = (row[0] / row[1]) * 100
        
        cursor.close()
        conn.close()
        
        return health_status
        
    except Exception as e:
        health_status['error'] = str(e)
        return health_status

# Usage
health = check_alloydb_health('10.0.0.5', 5432, 'mydb', 'monitor_user', 'PASSWORD')
print(f"Health Status: {health}")

# Alert if unhealthy
if not health['connectivity'] or health['connection_utilization'] > 90:
    # Send alert
    print("ALERT: AlloyDB unhealthy!")
```

---

## 9. Maintenance Best Practices

### Vacuum and Analyze

```sql
-- Regular maintenance schedule
-- Vacuum: Reclaim storage and prevent transaction ID wraparound
-- Analyze: Update statistics for query planner

-- Manual vacuum (run during low-traffic periods)
VACUUM VERBOSE ANALYZE;

-- Aggressive vacuum for heavily updated tables
VACUUM FULL VERBOSE tablename;

-- Monitor vacuum progress
SELECT 
    schemaname,
    tablename,
    last_vacuum,
    last_autovacuum,
    n_dead_tup,
    n_live_tup,
    (n_dead_tup::float / NULLIF(n_live_tup, 0) * 100)::numeric(5,2) as dead_tuple_percent
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY dead_tuple_percent DESC;

-- Configure autovacuum (if needed)
ALTER TABLE orders SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05,
    autovacuum_vacuum_cost_delay = 10
);
```

### Index Maintenance

```sql
-- Rebuild indexes to reduce bloat
REINDEX TABLE orders;
REINDEX INDEX CONCURRENTLY idx_orders_customer_date;

-- Identify bloated indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;

-- Drop unused indexes
DROP INDEX IF EXISTS idx_unused_index;
```

### Database Upgrades

```bash
# AlloyDB handles minor version upgrades automatically
# Major version upgrades require planning

# Check current version
gcloud alloydb instances describe primary-instance \
    --cluster=production-cluster \
    --region=us-central1 \
    --format="value(databaseVersion)"

# Schedule maintenance window
gcloud alloydb instances update primary-instance \
    --cluster=production-cluster \
    --region=us-central1 \
    --maintenance-window-day=SUNDAY \
    --maintenance-window-hour=2
```

---

## 10. Common Anti-Patterns to Avoid

### Architecture Anti-Patterns
❌ **Using single-zone deployment for production**: No automatic failover
❌ **Not using read pools for read-heavy workloads**: Overloading primary instance
❌ **Direct connections from applications**: Connection exhaustion without pooling
❌ **Not configuring autoscaling for read pools**: Manual scaling overhead
❌ **Public IP exposure**: Security risk and performance impact

### Performance Anti-Patterns
❌ **Not enabling columnar engine for analytics**: Missing 100x performance gains
❌ **Missing indexes on frequently queried columns**: Slow queries
❌ **Using SELECT * in queries**: Unnecessary data transfer
❌ **Not using prepared statements**: SQL injection risk and parsing overhead
❌ **Large transactions holding locks**: Blocking other queries
❌ **Not monitoring query performance**: Can't identify slow queries

### Security Anti-Patterns
❌ **Using default passwords**: Security vulnerability
❌ **Granting superuser to application accounts**: Excessive privileges
❌ **Not enabling SSL/TLS**: Data exposure in transit
❌ **Sharing database credentials across applications**: Poor access control
❌ **Not enabling audit logging**: No visibility into data access

### Backup and DR Anti-Patterns
❌ **Not testing restore procedures**: Failed recovery during disaster
❌ **Short backup retention (< 7 days)**: Limited recovery options
❌ **No cross-region replica for critical data**: Regional failure risk
❌ **Not monitoring replication lag**: Stale read data
❌ **Assuming zero RPO without testing**: May not meet SLA

### Cost Anti-Patterns
❌ **Over-provisioned instances**: Wasting resources and money
❌ **Not using autoscaling for read pools**: Paying for idle capacity
❌ **Long backup retention for non-critical data**: Unnecessary storage costs
❌ **Not archiving old data**: Growing storage costs
❌ **Running analytics on primary instance**: Impacting production performance

### Migration Anti-Patterns
❌ **Not testing migration in staging**: Production migration failures
❌ **Insufficient downtime planning**: Extended outages
❌ **Not validating data integrity post-migration**: Data loss or corruption
❌ **Migrating without application testing**: Application failures
❌ **Not having rollback plan**: Unable to revert if needed

---

## Quick Reference Checklist

### Initial Setup
- [ ] Create AlloyDB cluster in appropriate region
- [ ] Enable high availability (REGIONAL availability type)
- [ ] Configure Private Service Connect or VPC peering
- [ ] Set up connection pooling (application-level or PgBouncer)
- [ ] Create read pool with autoscaling enabled
- [ ] Configure automated backups with appropriate retention
- [ ] Set up cross-region replica for disaster recovery

### Security Configuration
- [ ] Enable IAM-based authentication
- [ ] Create database users with least privilege
- [ ] Implement row-level security where appropriate
- [ ] Enable SSL/TLS for all connections
- [ ] Configure CMEK for encryption at rest
- [ ] Enable Cloud Audit Logs
- [ ] Set up VPC Service Controls for perimeter security
- [ ] Implement database-level firewall rules

### Performance Optimization
- [ ] Enable columnar engine for analytical workloads
- [ ] Create appropriate indexes (B-tree, GIN, partial)
- [ ] Configure shared_buffers and work_mem appropriately
- [ ] Install and configure pg_stat_statements
- [ ] Set up connection pooling with appropriate pool size
- [ ] Monitor and optimize slow queries
- [ ] Configure autovacuum settings for high-write tables

### Monitoring & Alerting
- [ ] Set up Cloud Monitoring dashboard
- [ ] Configure alerts for CPU utilization (>80%)
- [ ] Configure alerts for connection count (>90% of max)
- [ ] Configure alerts for replication lag (>5 seconds)
- [ ] Monitor disk space utilization
- [ ] Set up query performance monitoring
- [ ] Export audit logs to BigQuery for analysis

### Backup & Disaster Recovery
- [ ] Verify automated backups are running
- [ ] Test point-in-time recovery procedure
- [ ] Create manual backups before major changes
- [ ] Test cross-region replica promotion
- [ ] Document disaster recovery runbook
- [ ] Schedule regular DR drills
- [ ] Monitor backup success and failures

### Cost Optimization
- [ ] Right-size primary instance based on actual utilization
- [ ] Configure read pool autoscaling limits appropriately
- [ ] Set backup retention based on actual recovery needs
- [ ] Archive old data to Cloud Storage or BigQuery
- [ ] Monitor and optimize storage utilization
- [ ] Review and remove unused indexes
- [ ] Use labels for cost tracking and attribution

### Maintenance
- [ ] Schedule maintenance windows for off-peak hours
- [ ] Monitor vacuum and analyze operations
- [ ] Review and optimize query performance monthly
- [ ] Update database statistics regularly
- [ ] Rebuild bloated indexes periodically
- [ ] Review and clean up old backups
- [ ] Test application failover procedures quarterly

---

## Additional Resources

- [AlloyDB Documentation](https://cloud.google.com/alloydb/docs)
- [AlloyDB for PostgreSQL Overview](https://cloud.google.com/alloydb/docs/overview)
- [Columnar Engine Guide](https://cloud.google.com/alloydb/docs/columnar-engine)
- [Migration Best Practices](https://cloud.google.com/alloydb/docs/migration-overview)
- [High Availability Configuration](https://cloud.google.com/alloydb/docs/instance-high-availability)
- [Database Migration Service](https://cloud.google.com/database-migration/docs)
- [Connection Pooling Best Practices](https://cloud.google.com/alloydb/docs/connect-instances)
- [Query Insights](https://cloud.google.com/alloydb/docs/query-insights)
- [Pricing Calculator](https://cloud.google.com/products/calculator)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Cloud Monitoring for AlloyDB](https://cloud.google.com/alloydb/docs/monitoring)
- [Security Best Practices](https://cloud.google.com/alloydb/docs/security-best-practices)

---

*Last Updated: December 28, 2025*
*Version: 1.0*
