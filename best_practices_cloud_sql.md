# Cloud SQL Best Practices

## Overview
Cloud SQL is a fully managed relational database service for MySQL, PostgreSQL, and SQL Server. This guide covers best practices for performance, availability, security, and cost optimization.

---

## 1. Instance Configuration & Sizing

### Choose the Right Database Engine
- **MySQL**: General-purpose applications, WordPress, e-commerce
- **PostgreSQL**: Complex queries, geospatial data, JSON support, analytical workloads
- **SQL Server**: Windows/.NET applications, enterprise applications

### Machine Type Selection
✅ **Shared-core**: Development/testing environments (db-f1-micro, db-g1-small)
✅ **Standard**: Production workloads (db-n1-standard-1 to db-n1-standard-96)
✅ **High-memory**: Memory-intensive workloads (db-n1-highmem-2 to db-n1-highmem-96)
✅ **Custom**: Specific CPU/memory requirements

### Sizing Best Practices
✅ Start with monitoring baseline workload performance
✅ Use **Cloud Monitoring** to track CPU, memory, and disk usage
✅ Right-size based on actual usage patterns (not peak capacity)
✅ Enable **automatic storage increase** to prevent disk full errors
✅ Leave 20-30% headroom for CPU and memory for spikes

### Storage Configuration
✅ Use **SSD storage** for production (better performance than HDD)
✅ Start with 10GB minimum, scale up as needed
✅ Enable **automatic storage increase** (recommended)
✅ Plan for 2-3x data size for indexes, temp tables, transaction logs
✅ Monitor storage I/O metrics (IOPS, throughput)

---

## 2. High Availability & Disaster Recovery

### High Availability Configuration
✅ Enable **Regional HA** (99.95% SLA) for production instances
✅ Configure HA in same region, different zones
✅ Understand automatic failover triggers (instance/zone failure)
✅ Test failover procedures regularly
✅ Plan for ~1-2 minute failover time

### HA Best Practices
```
Primary Instance (Zone A) ──> Synchronous Replication ──> Standby Instance (Zone B)
                          ──> Connection String (single endpoint)
```

✅ Use **Cloud SQL Proxy** or **Private IP** for connections
✅ Implement **connection pooling** to handle failover reconnections
✅ Set appropriate **application timeouts** for failover scenarios
✅ Use **exponential backoff** for connection retries
❌ Don't use public IP for production workloads

### Backup Strategy
✅ Enable **automated backups** (daily, retention up to 365 days)
✅ Set backup window during low-traffic periods
✅ Enable **binary logging** (required for point-in-time recovery)
✅ Configure appropriate **transaction log retention** (1-7 days)
✅ Test restore procedures regularly
✅ Use **on-demand backups** before major changes

### Point-in-Time Recovery (PITR)
```bash
# Restore to specific timestamp
gcloud sql backups create --instance=INSTANCE_NAME \
  --recovery-time=2025-12-25T12:00:00Z
```

✅ Understand PITR limitations (within transaction log retention)
✅ Document recovery time objectives (RTO) and recovery point objectives (RPO)
✅ Keep binary logs for desired PITR window

### Read Replicas
✅ Use **read replicas** to offload read traffic from primary
✅ Create replicas in same region for low latency
✅ Use **cross-region replicas** for disaster recovery
✅ Configure **replica lag monitoring** and alerting
✅ Maximum 10 read replicas per instance
✅ Support for **cascading replicas** (replicas of replicas)
✅ Enable **HA on read replicas** for high availability

### Replica Use Cases
- **Load balancing**: Distribute read queries across replicas
- **Reporting/Analytics**: Isolate heavy queries from production
- **Disaster Recovery**: Cross-region failover capability
- **Geographic distribution**: Serve users from closest region

### Standard Read Replicas

**Create Basic Read Replica**
```bash
# Create read replica in same region
gcloud sql instances create read-replica-1 \
  --master-instance-name=primary-instance \
  --tier=db-n1-standard-2 \
  --region=us-central1

# Create cross-region read replica
gcloud sql instances create read-replica-cross-region \
  --master-instance-name=primary-instance \
  --tier=db-n1-standard-2 \
  --region=europe-west1
```

**Read Replica Characteristics**
- Asynchronous replication from primary
- Eventually consistent (replication lag possible)
- Read-only access
- Can be promoted to standalone instance
- Inherits primary's backup configuration

### Cascading Read Replicas

**Understanding Cascading Replication**
Cascading replicas are read replicas created from other read replicas (not directly from the primary). This creates a replication chain.

```
Primary Instance (us-central1)
    ↓ (replication)
Read Replica 1 (us-central1)
    ↓ (cascading replication)
Read Replica 2 (us-east1)
    ↓ (cascading replication)
Read Replica 3 (us-west1)
```

**Benefits of Cascading Replicas**
- **Reduced load on primary**: Primary replicates to fewer direct replicas
- **Cost optimization**: Reduce cross-region replication costs
- **Geographic distribution**: Create regional replica hierarchies
- **Scale read capacity**: Distribute replication load across tiers

**Create Cascading Replica**
```bash
# Step 1: Create first-tier read replica
gcloud sql instances create replica-tier1-uscentral \
  --master-instance-name=primary-instance \
  --tier=db-n1-standard-4 \
  --region=us-central1

# Step 2: Create cascading replica from first-tier replica
gcloud sql instances create replica-tier2-useast \
  --master-instance-name=replica-tier1-uscentral \
  --tier=db-n1-standard-2 \
  --region=us-east1

# Step 3: Create another cascading replica
gcloud sql instances create replica-tier2-uswest \
  --master-instance-name=replica-tier1-uscentral \
  --tier=db-n1-standard-2 \
  --region=us-west1
```

**Cascading Replica Best Practices**
✅ **Limit cascade depth**: Recommended maximum 2-3 levels deep
✅ **Monitor lag accumulation**: Each tier adds latency
✅ **Size tier-1 replicas appropriately**: They handle replication to downstream replicas
✅ **Use for geographic distribution**: Place tier-1 in regional hubs
✅ **Plan for failures**: Understand impact if tier-1 replica fails
✅ **Consider network topology**: Minimize cross-region hops

**Cascading Replica Use Cases**
- **Multi-region architecture**: Hub-and-spoke replication model
- **Cost optimization**: Single cross-region link, multiple regional replicas
- **Hierarchical read scaling**: Distribute load across tiers
- **Geographic proximity**: Serve local regions from local replicas

**Cascading Replica Limitations**
❌ Cannot cascade from cross-region replicas in some configurations
❌ Increased replication lag with each tier
❌ Tier-1 replica failure impacts all downstream replicas
❌ More complex failover scenarios

### High Availability Read Replicas

**Understanding HA Read Replicas**
HA read replicas are read replicas with regional high availability enabled, providing 99.95% SLA for read workloads.

```
Primary HA Instance (us-central1)
    ↓ (replication)
HA Read Replica (us-central1)
    ├─ Replica Primary (zone a)
    └─ Replica Standby (zone b) [automatic failover]
```

**Enable HA on Read Replica**
```bash
# Create read replica with HA enabled
gcloud sql instances create ha-read-replica \
  --master-instance-name=primary-instance \
  --tier=db-n1-standard-2 \
  --region=us-central1 \
  --availability-type=REGIONAL

# Enable HA on existing read replica
gcloud sql instances patch existing-read-replica \
  --availability-type=REGIONAL
```

**HA Read Replica Architecture**
```
Primary Instance
    ↓
HA Read Replica
    ├─ Read Replica Primary (Zone A)
    │   ↓ (synchronous replication)
    └─ Read Replica Standby (Zone B)
         ↑ (automatic failover)
```

**HA Read Replica Benefits**
- **High availability for reads**: 99.95% SLA
- **Automatic failover**: Seamless failover within replica
- **Zero data loss**: Synchronous replication between replica zones
- **Independent HA**: Replica HA independent from primary HA
- **Production-grade reads**: Critical read workloads protected

**HA Read Replica Best Practices**
✅ **Use for critical read workloads**: Dashboards, APIs, reporting
✅ **Monitor both primary and standby**: Ensure both healthy
✅ **Test replica failover**: Understand behavior during outages
✅ **Consider cost**: HA replicas ~2x cost of standard replicas
✅ **Plan connection strings**: Use replica endpoint (handles failover)
✅ **Monitor failover events**: Track and analyze failover occurrences

**HA Read Replica Configuration**
```bash
# Check HA status of read replica
gcloud sql instances describe ha-read-replica \
  --format="value(settings.availabilityType)"

# Output: REGIONAL (HA enabled) or ZONAL (HA disabled)

# View replica configuration
gcloud sql instances describe ha-read-replica \
  --format="json" | jq '.settings.backupConfiguration'
```

### Combining Cascading and HA Replicas

**Hybrid Architecture**
```
Primary HA Instance (us-central1)
    ↓
HA Read Replica (Tier 1 - us-central1) [Regional HA]
    ├─ Primary (Zone A)
    └─ Standby (Zone B)
    ↓ (cascading replication)
Standard Read Replica (Tier 2 - us-east1)
    ↓ (cascading replication)
Standard Read Replica (Tier 2 - us-west1)
```

**Create Hybrid Architecture**
```bash
# Step 1: Create HA tier-1 replica
gcloud sql instances create tier1-ha-replica \
  --master-instance-name=primary-instance \
  --tier=db-n1-standard-4 \
  --region=us-central1 \
  --availability-type=REGIONAL

# Step 2: Create standard cascading replicas from HA replica
gcloud sql instances create tier2-east-replica \
  --master-instance-name=tier1-ha-replica \
  --tier=db-n1-standard-2 \
  --region=us-east1 \
  --availability-type=ZONAL

gcloud sql instances create tier2-west-replica \
  --master-instance-name=tier1-ha-replica \
  --tier=db-n1-standard-2 \
  --region=us-west1 \
  --availability-type=ZONAL
```

**Hybrid Architecture Benefits**
- **Critical replicas protected**: Tier-1 HA replicas highly available
- **Cost-effective scaling**: Standard replicas for non-critical reads
- **Regional resilience**: HA tier-1 survives zone failures
- **Geographic distribution**: Cascade to distant regions

### Read Replica Monitoring

**Key Metrics to Monitor**
```sql
-- Check replication lag (MySQL)
SHOW SLAVE STATUS\G

-- Key fields to monitor:
-- Seconds_Behind_Master: Replication lag in seconds
-- Slave_IO_Running: Should be 'Yes'
-- Slave_SQL_Running: Should be 'Yes'

-- Check replication lag (PostgreSQL)
SELECT
  client_addr,
  state,
  sent_lsn,
  write_lsn,
  flush_lsn,
  replay_lsn,
  sync_state,
  EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds
FROM pg_stat_replication;
```

**Cloud Monitoring Metrics**
```bash
# Monitor replica lag
gcloud monitoring time-series list \
  --filter='metric.type="cloudsql.googleapis.com/database/replication/replica_lag"' \
  --format="table(metric.labels.database_id,points)"

# Key replica metrics:
# - database/replication/replica_lag (seconds)
# - database/replication/network_lag (seconds)  
# - database/replication/replica_byte_lag (bytes)
```

**Alerting Configuration**
```yaml
# Alert policy for high replica lag
displayName: "High Read Replica Lag"
conditions:
  - displayName: "Replica lag > 10 seconds"
    conditionThreshold:
      filter: 'resource.type="cloudsql_database" AND metric.type="cloudsql.googleapis.com/database/replication/replica_lag"'
      comparison: COMPARISON_GT
      thresholdValue: 10
      duration: 300s
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_MEAN
notificationChannels:
  - projects/PROJECT_ID/notificationChannels/CHANNEL_ID
```

### Read Replica Failover and Promotion

**Promote Read Replica to Standalone**
```bash
# Promote replica (breaks replication link)
gcloud sql instances promote-replica ha-read-replica

# Warning: This action:
# - Converts replica to standalone instance
# - Breaks replication from primary
# - Cannot be undone
# - Use for disaster recovery or migration
```

**Replica Failover Scenarios**

**Scenario 1: Primary Instance Failure**
- Read replicas continue serving reads (with lag)
- Can promote replica to new primary
- Update application to write to promoted replica
- Repoint remaining replicas to new primary

**Scenario 2: HA Read Replica Failover**
- Automatic failover to standby within replica
- ~1-2 minute failover time
- No manual intervention needed
- Application reconnects to same endpoint

**Scenario 3: Cascading Replica Failure**
- Tier-1 failure: Downstream replicas stop receiving updates
- Tier-2+ failure: No impact on other replicas
- Recover or promote tier-2 to tier-1

### Advanced Replication Topologies

**Regional Hub-and-Spoke**
```
Primary (us-central1)
    ├─ HA Replica Hub (us-central1) [Regional HA]
    │   ├─ Spoke 1 (us-east1)
    │   └─ Spoke 2 (us-west1)
    ├─ HA Replica Hub (europe-west1) [Regional HA]
    │   ├─ Spoke 1 (europe-west2)
    │   └─ Spoke 2 (europe-north1)
    └─ HA Replica Hub (asia-southeast1) [Regional HA]
        ├─ Spoke 1 (asia-northeast1)
        └─ Spoke 2 (asia-south1)
```

**Multi-Tier Cascading**
```
Primary Instance
    ↓
Tier-1 Replica (Regional Distribution)
    ├─ Tier-2 Replica (Country A)
    │   ├─ Tier-3 Replica (City 1)
    │   └─ Tier-3 Replica (City 2)
    └─ Tier-2 Replica (Country B)
        ├─ Tier-3 Replica (City 3)
        └─ Tier-3 Replica (City 4)
```

**Disaster Recovery with HA Replicas**
```
Production Region (us-central1)
    Primary HA Instance
        ↓
DR Region (europe-west1)
    HA Read Replica [Can be promoted during disaster]
        ├─ Replica Primary (Zone A)
        └─ Replica Standby (Zone B)
```

### Replication Strategy Decision Matrix

| Requirement | Standard Replica | HA Replica | Cascading Replica | Recommendation |
|-------------|------------------|------------|-------------------|----------------|
| **Low latency reads** | ✅ Same region | ✅ Same region | ⚠️ Increased lag | Standard or HA |
| **High availability reads** | ❌ Single zone | ✅ Multi-zone | ❌ Single zone | HA Replica |
| **Cost optimization** | ✅ Low cost | ❌ 2x cost | ✅ Reduces primary load | Cascading |
| **DR/Failover** | ✅ Can promote | ✅ Can promote + HA | ✅ Can promote | HA for critical |
| **Global distribution** | ⚠️ All from primary | ⚠️ All from primary | ✅ Regional hubs | Cascading |
| **Critical workloads** | ⚠️ No zone redundancy | ✅ Zone redundant | ⚠️ No zone redundancy | HA Replica |
| **Read scaling** | ✅ Good | ✅ Good | ✅ Excellent | Depends on need |
| **Replication lag** | Low | Low | Higher with depth | Consider topology |

**When to Use Standard Read Replicas**
- Non-critical read workloads
- Cost-sensitive deployments
- Simple scaling requirements
- Development/testing environments

**When to Use HA Read Replicas**
- Critical read workloads requiring high availability
- Production APIs and dashboards
- Customer-facing applications
- Workloads with read SLA requirements
- When read downtime is unacceptable

**When to Use Cascading Replicas**
- Global/multi-region deployments
- Need to scale beyond 10 direct replicas
- Regional hub-and-spoke architectures
- Cost optimization for multiple regional replicas
- Reducing load on primary instance

**When to Combine HA + Cascading**
- Critical regional hubs need HA
- Global distribution with high availability
- Tier-1 replicas serve critical regions
- Cost-effective expansion to additional regions

### Replication Performance Tuning

**Optimize Replication Lag**
```sql
-- MySQL: Check replication configuration
SHOW VARIABLES LIKE 'slave%';

-- Key variables to tune:
-- slave_parallel_workers (parallel replication threads)
SET GLOBAL slave_parallel_workers = 4;

-- slave_parallel_type (logical_clock or database)
SET GLOBAL slave_parallel_type = 'LOGICAL_CLOCK';

-- PostgreSQL: Check replication slots
SELECT * FROM pg_replication_slots;

-- Monitor WAL sender processes
SELECT * FROM pg_stat_replication;
```

**Network Optimization**
✅ **Use same region**: Minimize network latency
✅ **Private IP**: Better performance than public IP
✅ **Adequate bandwidth**: Ensure sufficient network capacity
✅ **Monitor network lag**: Separate from replica lag

**Write Load Management**
✅ **Batch writes**: Reduce replication overhead
✅ **Avoid large transactions**: Break into smaller chunks
✅ **Index appropriately**: Balance write vs read performance
✅ **Monitor write throughput**: Ensure replicas can keep up

---

## 3. Security Best Practices

### Network Security
✅ Use **Private IP** (VPC peering) for private connectivity
✅ Avoid public IP unless absolutely necessary
✅ Enable **authorized networks** if using public IP
✅ Use **Cloud SQL Proxy** for secure connections
✅ Implement **VPC Service Controls** for perimeter security

### Authentication & Authorization
✅ Use **Cloud SQL IAM authentication** (passwordless)
✅ Implement **least privilege** for database users
✅ Rotate passwords regularly for SQL authentication
✅ Disable default 'root' or 'postgres' users
✅ Create separate users for applications and administrators

### IAM Roles
- `roles/cloudsql.client`: Connect to instances via Cloud SQL Proxy
- `roles/cloudsql.instanceUser`: IAM-based database login
- `roles/cloudsql.editor`: Modify instance configuration
- `roles/cloudsql.admin`: Full control over instances

### Encryption
✅ **At rest**: Automatic encryption with Google-managed keys
✅ **CMEK**: Customer-managed encryption keys for compliance
✅ **In transit**: SSL/TLS for all connections (enforce SSL)
✅ **Client certificates**: Additional security layer

### Enforce SSL Connections
```sql
-- MySQL: Require SSL for user
ALTER USER 'myuser'@'%' REQUIRE SSL;

-- PostgreSQL: Force SSL in pg_hba.conf
-- hostssl all all 0.0.0.0/0 md5
```

### Audit & Compliance
✅ Enable **audit logging** (PostgreSQL pgAudit, MySQL Enterprise Audit)
✅ Enable **Cloud Logging** for connection logs
✅ Monitor **failed login attempts**
✅ Use **Cloud Audit Logs** for administrative operations
✅ Implement **data access logging** for sensitive tables

---

## 4. Performance Optimization

### Connection Management
✅ Use **connection pooling** (reduces connection overhead)
✅ Recommended pool size: (cores × 2) + effective_spindle_count
✅ Set appropriate **connection timeouts**
✅ Close connections properly to avoid leaks
✅ Use **Cloud SQL Proxy** for connection management

### Connection Pooling Configuration
```python
# Python example with SQLAlchemy
engine = create_engine(
    'postgresql+pg8000://user:pass@/dbname',
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600
)
```

### Query Optimization
✅ **Create appropriate indexes** on frequently queried columns
✅ Use **EXPLAIN** to analyze query plans
✅ Avoid **SELECT \*** (select only needed columns)
✅ Use **prepared statements** to reduce parsing overhead
✅ Implement **pagination** for large result sets
✅ Optimize **JOIN operations** with proper indexing

### Indexing Best Practices
```sql
-- Create index on frequently filtered columns
CREATE INDEX idx_users_email ON users(email);

-- Composite index for multi-column queries
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);

-- Partial index for specific conditions
CREATE INDEX idx_active_users ON users(email) WHERE active = true;
```

✅ Monitor **unused indexes** (they slow down writes)
✅ Use **covering indexes** for frequently accessed column sets
✅ Avoid **over-indexing** (balance reads vs writes)
✅ Regularly analyze and rebuild fragmented indexes

### Query Performance Monitoring
✅ Enable **Query Insights** for slow query identification
✅ Set **slow query log threshold** appropriately
✅ Use **pg_stat_statements** (PostgreSQL) for query analysis
✅ Monitor **long-running queries** and kill if necessary
✅ Review and optimize top N slowest queries regularly

### Database Configuration Tuning

#### MySQL Configuration
```sql
-- Buffer pool (70-80% of available memory)
SET GLOBAL innodb_buffer_pool_size = 8589934592; -- 8GB

-- Connection settings
SET GLOBAL max_connections = 100;
SET GLOBAL wait_timeout = 600;

-- Query cache (use with caution)
SET GLOBAL query_cache_size = 0; -- Disabled in MySQL 8.0+
```

#### PostgreSQL Configuration
```sql
-- Shared buffers (25% of memory)
ALTER SYSTEM SET shared_buffers = '2GB';

-- Work memory (for sorting/hashing)
ALTER SYSTEM SET work_mem = '64MB';

-- Effective cache size (50-75% of memory)
ALTER SYSTEM SET effective_cache_size = '6GB';

-- Reload configuration
SELECT pg_reload_conf();
```

### Caching Strategies
✅ Implement **application-level caching** (Redis, Memcached)
✅ Use **query result caching** for frequently accessed data
✅ Cache **computed values** and aggregations
✅ Set appropriate **cache TTL** based on data freshness requirements
✅ Invalidate cache on data updates

---

## 5. Maintenance & Updates

### Maintenance Windows
✅ Configure **maintenance window** during low-traffic periods
✅ Understand that minor updates may cause brief unavailability
✅ Enable **maintenance notifications** for advance warning
✅ Test updates in non-production environments first
✅ Review release notes before major version upgrades

### Database Maintenance Tasks
✅ **VACUUM** (PostgreSQL): Reclaim storage, update statistics
✅ **ANALYZE**: Update query planner statistics
✅ **REINDEX**: Rebuild fragmented indexes
✅ **OPTIMIZE TABLE** (MySQL): Defragment tables
✅ Schedule maintenance during off-peak hours

### Version Management
✅ Keep database version up-to-date for security patches
✅ Test **major version upgrades** in staging first
✅ Review **compatibility changes** before upgrading
✅ Plan downtime for major upgrades (or use read replicas)
✅ Take backup before any major changes

---

## 6. Monitoring & Alerting

### Key Metrics to Monitor
✅ **CPU utilization**: Alert at 80%+
✅ **Memory utilization**: Alert at 85%+
✅ **Disk utilization**: Alert at 80%+, enable auto-increase
✅ **IOPS and throughput**: Monitor against limits
✅ **Connection count**: Alert at 90% of max_connections
✅ **Replication lag**: Alert at 5+ seconds
✅ **Slow queries**: Track queries > 1 second
✅ **Error rates**: 5xx errors, failed connections

### Cloud Monitoring Setup
```bash
# Example alert policy for CPU
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="High CPU Alert" \
  --condition-display-name="CPU > 80%" \
  --condition-threshold-value=0.8 \
  --condition-threshold-duration=300s
```

### Logging Best Practices
✅ Enable **error logs** (always on)
✅ Enable **slow query logs** (set appropriate threshold)
✅ Enable **general logs** only for troubleshooting (high overhead)
✅ Export logs to **Cloud Logging** for centralized management
✅ Set log retention policies appropriately

### Dashboard Essentials
- Instance health and availability
- CPU, memory, disk, network utilization
- Active connections and connection pool status
- Query performance (avg query time, slow queries)
- Replication lag (for HA and read replicas)
- Backup status and age of last backup

---

## 7. Cost Optimization

### Right-Sizing Strategies
✅ Monitor actual resource usage over time
✅ **Downsize** over-provisioned instances
✅ Use **shared-core** instances for dev/test
✅ Schedule **instance stop/start** for non-production
✅ Delete unused instances and replicas

### Storage Optimization
✅ Enable **automatic storage increase** to avoid manual resizing fees
✅ Archive old data to **Cloud Storage** for cheaper storage
✅ Implement **data retention policies** to delete old data
✅ Use **table partitioning** for efficient archival
✅ Monitor storage growth trends

### Backup Cost Management
✅ Set appropriate **backup retention** (7-35 days typical)
✅ Longer retention = higher costs
✅ Delete old **on-demand backups** no longer needed
✅ Consider exporting to Cloud Storage for long-term archival

### Network Cost Reduction
✅ Use **Private IP** to avoid egress charges within VPC
✅ Keep instances and compute in **same region**
✅ Use **read replicas** in same region as clients
✅ Minimize cross-region data transfer

### Reserved Capacity
✅ Use **committed use discounts** for predictable workloads (1 or 3 years)
✅ Save up to 52% for 3-year commitments
✅ Evaluate cost savings vs flexibility tradeoff

---

## 8. Migration Best Practices

### Migration Planning
✅ **Assess** current database (size, schema, dependencies)
✅ Choose appropriate **migration strategy** (online vs offline)
✅ Test migration in **staging environment** first
✅ Plan for **rollback** in case of issues
✅ Communicate downtime windows to stakeholders

### Migration Methods

#### Database Migration Service (DMS)
✅ **Online migration**: Minimal downtime using CDC
✅ Supports **MySQL, PostgreSQL, SQL Server, Oracle**
✅ Continuous replication during migration
✅ Switchover when cutover ready

#### Manual Migration
```bash
# Export from source
mysqldump -u user -p --databases mydb > dump.sql

# Import to Cloud SQL
gcloud sql import sql INSTANCE_NAME gs://BUCKET/dump.sql \
  --database=mydb

# Or use Cloud SQL proxy
mysql -h 127.0.0.1 -u user -p mydb < dump.sql
```

### Pre-Migration Checklist
- [ ] Backup source database
- [ ] Test connectivity to Cloud SQL
- [ ] Verify schema compatibility
- [ ] Check for deprecated features
- [ ] Identify and resolve potential blockers
- [ ] Plan for character set/collation differences
- [ ] Test application with Cloud SQL instance
- [ ] Prepare rollback plan

### Post-Migration Tasks
✅ Verify data integrity (row counts, checksums)
✅ Update **application connection strings**
✅ Test application functionality thoroughly
✅ Monitor performance and optimize as needed
✅ Update **database statistics** (ANALYZE)
✅ Rebuild indexes if necessary
✅ Implement ongoing backup strategy

---

## 9. Application Integration

### Connection Best Practices
✅ Use **Cloud SQL Proxy** for secure, managed connections
✅ Implement **connection pooling** at application layer
✅ Use **Private IP** for VPC-native applications
✅ Handle **transient failures** with retry logic
✅ Set appropriate **connection timeouts**

### Cloud SQL Proxy Setup
```bash
# Download and run proxy
cloud_sql_proxy -instances=PROJECT:REGION:INSTANCE=tcp:3306 &

# Connect through proxy
mysql -h 127.0.0.1 -u myuser -p mydatabase
```

### Application Code Best Practices
```python
# Python example with connection pooling
import sqlalchemy
from google.cloud.sql.connector import Connector

connector = Connector()

def getconn():
    return connector.connect(
        "project:region:instance",
        "pg8000",
        user="myuser",
        password="mypassword",
        db="mydb"
    )

pool = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=getconn,
    pool_size=5,
    max_overflow=2,
    pool_timeout=30,
    pool_recycle=1800
)
```

### Error Handling
✅ Catch and handle **database exceptions** gracefully
✅ Implement **exponential backoff** for retries
✅ Log errors with sufficient context
✅ Differentiate **transient vs permanent** failures
✅ Fail gracefully with user-friendly messages

---

## 10. Serverless & Cloud Run Integration

### Cloud Run to Cloud SQL
✅ Use **Cloud SQL Proxy sidecar** or **Unix socket**
✅ Configure **instance connection name** in Cloud Run
✅ Use **Secret Manager** for database credentials
✅ Set **connection limits** appropriate for Cloud Run concurrency
✅ Handle **cold starts** with connection pooling

### Cloud Functions Integration
✅ Use **connection pooling** across function invocations
✅ Reuse connections in **global scope**
✅ Set **max instances** to avoid overwhelming database
✅ Consider **connection limits** per function instance

---

## 11. Compliance & Governance

### Data Residency
✅ Choose **region** based on compliance requirements
✅ Use **regional instances** for data locality
✅ Implement **Organization Policies** to restrict regions
✅ Document data location for auditors

### Compliance Standards
✅ **HIPAA**: Enable encryption, audit logs, BAA with Google
✅ **PCI DSS**: Network isolation, encryption, access controls
✅ **GDPR**: Data encryption, right to deletion, audit trails
✅ **SOC 2/3**: Audit logs, access controls, monitoring

### Access Control
✅ Implement **least privilege** access
✅ Use **IAM** for instance management
✅ Use **database users/roles** for data access
✅ Regular **access reviews** and audits
✅ Separate **admin and application accounts**

---

## 12. Common Anti-Patterns to Avoid

### General Anti-Patterns
❌ **Using public IP for production**: Security and performance risks
❌ **Not using connection pooling**: Connection overhead and exhaustion
❌ **Ignoring slow queries**: Performance degradation over time
❌ **Over-provisioning**: Wasting money on unused capacity
❌ **Not enabling HA for production**: Single point of failure
❌ **Storing large BLOBs in database**: Use Cloud Storage instead
❌ **Running analytics on production**: Use read replicas or export to BigQuery
❌ **Not testing failover**: Surprises during actual incidents
❌ **Hardcoding credentials**: Use Secret Manager or IAM auth
❌ **Not implementing backups**: Data loss risk
❌ **Ignoring maintenance windows**: Unexpected downtime

### Read Replica Anti-Patterns
❌ **Not monitoring replica lag**: Stale reads affecting application
❌ **Using standard replicas for critical reads**: No HA protection
❌ **Cascading too deep**: Excessive lag accumulation (>3 levels)
❌ **Writing to read replicas**: Read replicas are read-only
❌ **Not considering lag in application logic**: Expecting immediate consistency
❌ **Creating all replicas from primary**: Missing cascading optimization opportunities
❌ **No alerting on replica failures**: Undetected replica outages
❌ **Not testing replica promotion**: Unprepared for disaster recovery
❌ **Ignoring replica lag spikes**: Performance degradation warnings
❌ **Using same tier for all replicas**: Not matching workload requirements

### HA Configuration Anti-Patterns
❌ **Skipping HA for production primaries**: Risking extended downtime
❌ **Not understanding failover time**: Unrealistic RTO expectations
❌ **No application retry logic**: Failures during automatic failover
❌ **Using HA everywhere**: Overspending on non-critical workloads
❌ **Not testing HA failover**: Discovering issues during real outages
❌ **Ignoring connection drops during failover**: Poor user experience

### Cascading Replica Anti-Patterns
❌ **Cascading from cross-region replicas in complex topologies**: Unsupported configurations
❌ **Not sizing tier-1 replicas properly**: Bottleneck for downstream replicas
❌ **Cascading critical workloads deeply**: Unacceptable lag for production
❌ **No fallback plan for tier-1 failures**: Entire branch goes offline
❌ **Creating cascades without monitoring**: No visibility into multi-tier lag
❌ **Mixing HA and cascading without planning**: Configuration complexity

---

## 13. Troubleshooting Guide

### High CPU Usage
1. Check for missing indexes
2. Identify slow queries using Query Insights
3. Look for full table scans
4. Check for lock contention
5. Consider scaling up or optimizing queries

### High Memory Usage
1. Review buffer pool configuration
2. Check for memory leaks in connections
3. Identify queries with large result sets
4. Consider increasing instance memory

### Connection Issues
1. Verify network connectivity (VPC peering, firewall rules)
2. Check authorized networks (if using public IP)
3. Verify connection limits not exceeded
4. Check Cloud SQL Proxy configuration
5. Review IAM permissions

### Slow Queries
1. Use EXPLAIN to analyze query plans
2. Check for missing indexes
3. Review table statistics freshness
4. Look for lock contention
5. Consider query rewriting or schema changes

### Replication Lag
1. Check network connectivity between zones/regions
2. Identify long-running transactions on primary
3. Review write workload intensity
4. Consider scaling up replica instance
5. Check for DDL operations blocking replication
6. For cascading replicas, check each tier independently
7. Monitor network lag separately from replica lag

### Cascading Replica Issues
```bash
# Check replication chain status
gcloud sql instances describe replica-tier2 \
  --format="value(masterInstanceName)"

# Output should show tier-1 replica name

# Check each tier's lag
gcloud sql instances describe replica-tier1 \
  --format="value(replicaConfiguration.replicaLag)"

gcloud sql instances describe replica-tier2 \
  --format="value(replicaConfiguration.replicaLag)"
```

**Tier-1 Replica Failure**
1. Downstream cascading replicas stop receiving updates
2. Promote a tier-2 replica to become new tier-1
3. Repoint other tier-2 replicas to new tier-1
4. Or connect tier-2 replicas directly to primary (if within limits)

**Excessive Cascading Lag**
1. Check if tier-1 replica is undersized
2. Review write workload on primary
3. Consider reducing cascade depth
4. Monitor each tier's individual lag
5. Check network connectivity between tiers

### HA Replica Failover Issues
```bash
# Check HA status
gcloud sql instances describe ha-read-replica \
  --format="value(settings.availabilityType)"

# Review recent failover events
gcloud logging read "resource.type=cloudsql_database
  AND protoPayload.methodName=cloudsql.instances.failover
  AND resource.labels.database_id=PROJECT:ha-read-replica" \
  --limit=10 \
  --format=json
```

**HA Failover Not Working**
1. Verify HA is enabled (REGIONAL, not ZONAL)
2. Check standby instance health
3. Review recent configuration changes
4. Ensure sufficient capacity in zone
5. Check for ongoing maintenance

**Post-Failover Connection Issues**
1. Verify application uses replica connection string
2. Check connection pooling configuration
3. Review retry logic in application
4. Confirm DNS resolution working
5. Check for firewall rule changes

---

## Quick Reference Checklist

### Initial Setup
- [ ] Choose appropriate machine type and storage
- [ ] Enable Regional HA for production instances
- [ ] Configure automated backups with appropriate retention
- [ ] Enable binary logging for PITR
- [ ] Use Private IP and Cloud SQL Proxy
- [ ] Implement connection pooling
- [ ] Enable SSL/TLS enforcement
- [ ] Use IAM authentication where possible

### Read Replica Configuration
- [ ] Create read replicas to offload read traffic
- [ ] Enable HA on read replicas for critical read workloads
- [ ] Consider cascading replicas for global distribution
- [ ] Monitor replica lag and set up alerts
- [ ] Test read replica failover procedures
- [ ] Document replication topology
- [ ] Size tier-1 replicas appropriately for cascading
- [ ] Plan replica promotion strategy for DR

### Performance & Monitoring
- [ ] Create appropriate indexes
- [ ] Enable Query Insights
- [ ] Configure monitoring and alerting
- [ ] Monitor primary and replica lag
- [ ] Track cascading replica lag per tier
- [ ] Set up alerts for high replica lag (>10s)
- [ ] Monitor HA failover events

### High Availability & DR
- [ ] Enable Regional HA for production
- [ ] Configure cross-region replicas for DR
- [ ] Test primary failover procedures
- [ ] Test HA replica failover
- [ ] Test cascading replica recovery
- [ ] Document failover runbooks
- [ ] Verify application retry logic

### Operations
- [ ] Set maintenance window during off-peak hours
- [ ] Implement cost optimization strategies
- [ ] Enable audit logging for compliance
- [ ] Document backup and recovery procedures
- [ ] Right-size instances based on actual usage
- [ ] Review and optimize replication topology
- [ ] Test replica promotion procedures quarterly

---

## Additional Resources

- [Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [Best Practices for Cloud SQL](https://cloud.google.com/sql/docs/postgres/best-practices)
- [Query Insights](https://cloud.google.com/sql/docs/postgres/using-query-insights)
- [Cloud SQL Proxy](https://cloud.google.com/sql/docs/mysql/sql-proxy)
- [Pricing Calculator](https://cloud.google.com/products/calculator)

---

*Last Updated: December 25, 2025*
