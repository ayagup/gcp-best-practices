# Cloud Data Fusion Best Practices

## Overview
Cloud Data Fusion is a fully managed, cloud-native data integration service for building ETL/ELT pipelines using a visual interface. Built on CDAP (Cask Data Application Platform), it provides pre-built connectors and transformations for common data integration tasks.

---

## 1. Instance Configuration

### Instance Types

**Basic Edition**
✅ Single-tenant instance
✅ Basic features and connectors
✅ Lower cost
✅ Best for: Development, testing, simple pipelines

**Enterprise Edition** (Recommended for Production)
✅ High availability and SLA
✅ Advanced features (lineage, triggers, metadata)
✅ Custom connectors and plugins
✅ Team collaboration features
✅ Best for: Production workloads

**Developer Edition** (Free Tier)
✅ Limited capacity (50 node hours/month)
✅ Basic features
✅ Best for: Learning, prototyping

### Instance Creation
```bash
# Create Data Fusion instance
gcloud data-fusion instances create my-instance \
  --location=us-central1 \
  --edition=enterprise \
  --enable-stackdriver-logging \
  --enable-stackdriver-monitoring \
  --network=projects/PROJECT/global/networks/VPC
```

### Instance Configuration Best Practices
✅ Use **Enterprise edition** for production
✅ Enable **Stackdriver logging and monitoring**
✅ Configure **VPC peering** for private connectivity
✅ Set appropriate **version** (6.x recommended)
✅ Configure **namespace** for multi-tenancy

---

## 2. Pipeline Design Best Practices

### Pipeline Architecture

**Source → Transform → Sink Pattern**
```
BigQuery → Wrangler → Joiner → Cloud Storage
Pub/Sub → JavaScript → Filter → BigQuery
Cloud Storage → Python → Group By → Bigtable
```

### Visual Pipeline Builder
✅ Drag-and-drop interface
✅ Pre-built connectors (150+)
✅ Built-in transformations
✅ Real-time validation
✅ Visual data lineage

### Pipeline Components

**Sources (Input)**
- BigQuery
- Cloud Storage (CSV, JSON, Avro, Parquet)
- Cloud SQL
- Pub/Sub
- Spanner
- Database plugins (Oracle, MySQL, PostgreSQL)
- HTTP/REST APIs

**Transforms**
- Wrangler (data preparation)
- JavaScript transform
- Python evaluator
- Joiner
- Group By
- Deduplicate
- Normalize
- Projection
- Filter

**Sinks (Output)**
- BigQuery
- Cloud Storage
- Cloud Bigtable
- Cloud SQL
- Spanner

### Best Practice Pipeline Structure
```
Source(s)
  ↓
Data Quality Checks
  ↓
Transformations (Wrangler/JavaScript)
  ↓
Business Logic
  ↓
Enrichment (Joins with reference data)
  ↓
Aggregation (if needed)
  ↓
Error Handling (Error Collector)
  ↓
Sink(s)
```

---

## 3. Wrangler (Data Preparation)

### Wrangler Best Practices
✅ Use for **data cleaning and transformation**
✅ **Interactive interface** for data exploration
✅ Apply **directives** for transformations
✅ Preview data before applying changes
✅ Export directives for reuse

### Common Wrangler Directives
```
# Parse CSV
parse-as-csv :body ',' false

# Drop columns
drop :column1,:column2

# Rename column
rename :old_name :new_name

# Filter rows
filter-rows-on condition-false :age > 18

# Fill null values
fill-null-or-empty :column 'default_value'

# Change data type
set-type :column integer

# Split column
split-to-columns :full_name ' ' :first_name,:last_name

# Merge columns
merge :first_name,:last_name :full_name ' '

# Extract with regex
extract-regex-groups :email '^(.+)@(.+)$' :username,:domain

# Format date
format-date :timestamp 'yyyy-MM-dd HH:mm:ss'

# Deduplicate
deduplicate :column1,:column2
```

### Wrangler Performance Tips
✅ **Sample data** for faster development
✅ Apply **filters early** to reduce data volume
✅ Use **built-in directives** instead of custom code
✅ **Test on small datasets** before full pipeline run
✅ Export and version control wrangler recipes

---

## 4. Pipeline Execution

### Execution Modes

**Batch Pipeline**
✅ Scheduled or on-demand execution
✅ Process historical data
✅ Use Dataproc or Spark clusters

```json
{
  "schedule": "0 2 * * *",  // Daily at 2 AM
  "engine": "spark",
  "resources": {
    "virtualCores": 4,
    "memoryMB": 8192
  }
}
```

**Realtime Pipeline**
✅ Continuous processing
✅ Low-latency data ingestion
✅ Stream from Pub/Sub, Kafka

```json
{
  "instances": 2,
  "resources": {
    "virtualCores": 2,
    "memoryMB": 4096
  }
}
```

### Resource Configuration
```json
{
  "driverResources": {
    "virtualCores": 2,
    "memoryMB": 4096
  },
  "executorResources": {
    "virtualCores": 4,
    "memoryMB": 8192
  },
  "numExecutors": 4
}
```

### Execution Best Practices
✅ Set **appropriate resource allocation** based on data volume
✅ Use **schedule** for regular batch jobs
✅ Enable **auto-tuning** for resource optimization
✅ Monitor **pipeline metrics** for performance issues
✅ Use **incremental processing** when possible

---

## 5. Error Handling & Data Quality

### Error Collectors
✅ Capture **invalid records** without failing pipeline
✅ Route errors to separate sink
✅ Analyze error patterns

```
Source → Transform → Error Collector → Valid Sink
                     ↓
                  Error Sink
```

### Data Quality Checks
```
# Add validator plugins
- Required fields check
- Data type validation
- Range validation
- Regex pattern matching
- Custom business rules
```

### Error Handling Best Practices
✅ **Always use error collectors** for production pipelines
✅ Store errors in **separate BigQuery table** or Cloud Storage
✅ Set up **alerts** for high error rates
✅ Implement **retry logic** for transient failures
✅ Log error details for **troubleshooting**

---

## 6. Plugin Management

### Pre-Built Plugins
✅ 150+ connectors available
✅ Google Cloud services
✅ Databases (Oracle, SQL Server, MySQL, PostgreSQL)
✅ SaaS applications (Salesforce, SAP)
✅ File systems (S3, Azure Blob)

### Custom Plugins
```java
// Example custom transform plugin
@Plugin(type = BatchTransform.PLUGIN_TYPE)
@Name("MyCustomTransform")
public class MyCustomTransform extends Transform {
    
    @Override
    public void transform(StructuredRecord input, Emitter<StructuredRecord> emitter) {
        // Custom transformation logic
        StructuredRecord.Builder builder = StructuredRecord.builder(outputSchema);
        // Process fields
        emitter.emit(builder.build());
    }
}
```

### Plugin Best Practices
✅ Use **Hub** to discover and install plugins
✅ Keep plugins **up-to-date**
✅ Test custom plugins in **development environment**
✅ Version control custom plugin code
✅ Document custom plugin usage

---

## 7. Integration with GCP Services

### BigQuery Integration
```json
{
  "name": "BigQuerySource",
  "plugin": {
    "name": "BigQueryTable",
    "type": "batchsource",
    "properties": {
      "project": "my-project",
      "dataset": "my_dataset",
      "table": "my_table",
      "filter": "date >= '2025-01-01'"
    }
  }
}
```

### Cloud Storage Integration
```json
{
  "name": "GCSSource",
  "plugin": {
    "name": "GCS",
    "type": "batchsource",
    "properties": {
      "path": "gs://my-bucket/data/*.parquet",
      "format": "parquet",
      "schema": "..."
    }
  }
}
```

### Pub/Sub Integration
```json
{
  "name": "PubSubSource",
  "plugin": {
    "name": "GoogleSubscriber",
    "type": "streamingsource",
    "properties": {
      "project": "my-project",
      "subscription": "my-subscription",
      "format": "json"
    }
  }
}
```

### Cloud SQL Integration
```json
{
  "name": "CloudSQLSource",
  "plugin": {
    "name": "Database",
    "type": "batchsource",
    "properties": {
      "connectionString": "jdbc:mysql://CLOUD_SQL_IP:3306/database",
      "importQuery": "SELECT * FROM table WHERE date = '${logicalStartTime}'",
      "username": "user",
      "password": "password"
    }
  }
}
```

---

## 8. Scheduling & Orchestration

### Pipeline Scheduling
```json
{
  "schedule": {
    "name": "DailySchedule",
    "cron": "0 2 * * *",  // Daily at 2 AM
    "properties": {
      "date": "${logicalStartTime(yyyy-MM-dd)}"
    }
  }
}
```

### Triggers
✅ **Time-based**: Cron schedules
✅ **Event-based**: Cloud Storage, Pub/Sub
✅ **Data-dependent**: After other pipeline completion

### Macros & Runtime Arguments
```json
{
  "runtimeArguments": {
    "input.path": "gs://bucket/data/${date}",
    "output.table": "dataset.table_${date}",
    "date": "${logicalStartTime(yyyy-MM-dd)}"
  }
}
```

### Orchestration Best Practices
✅ Use **runtime arguments** for parameterization
✅ Use **macros** for dynamic values
✅ Set appropriate **schedule** based on data availability
✅ Implement **dependencies** between pipelines
✅ Use **Cloud Composer** for complex workflows

---

## 9. Monitoring & Logging

### Pipeline Metrics
✅ **Records processed**: Input/output counts
✅ **Execution time**: Total and per stage
✅ **Error rate**: Failed records
✅ **Resource usage**: CPU, memory
✅ **Data latency**: For streaming pipelines

### Cloud Monitoring Integration
```bash
# View pipeline metrics
gcloud monitoring time-series list \
  --filter='resource.type="data_fusion_instance"' \
  --format=json
```

### Logging Best Practices
✅ Enable **Stackdriver logging** on instance
✅ Monitor **pipeline logs** for errors
✅ Set up **alerts** for failures
✅ Use **log-based metrics** for custom monitoring
✅ Export logs to **BigQuery** for analysis

### Key Metrics to Monitor
- Pipeline success/failure rate
- Average execution time
- Records processed per run
- Error record count
- Resource utilization
- Data freshness (for streaming)

---

## 10. Security Best Practices

### IAM & Access Control
✅ Use **IAM roles** for instance access
✅ Implement **least privilege** principle
✅ Use **service accounts** for pipeline execution

**Common Roles**
- `roles/datafusion.admin`: Full control
- `roles/datafusion.editor`: Create and manage pipelines
- `roles/datafusion.viewer`: View-only access

### Network Security
✅ Use **VPC peering** for private connectivity
✅ Configure **firewall rules** appropriately
✅ Enable **Private Google Access**
✅ Use **Private IP** for data sources

```bash
# Create instance with VPC peering
gcloud data-fusion instances create my-instance \
  --location=us-central1 \
  --network=projects/PROJECT/global/networks/VPC \
  --enable-stackdriver-logging
```

### Credential Management
✅ Use **Secret Manager** for credentials
✅ Don't hardcode passwords in pipelines
✅ Use **service account keys** securely
✅ Rotate credentials regularly

### Encryption
✅ **Data in transit**: TLS (automatic)
✅ **Data at rest**: CMEK for instance data
✅ **Credentials**: Encrypted in metadata store

---

## 11. Performance Optimization

### Pipeline Optimization
✅ **Filter early**: Reduce data volume early in pipeline
✅ **Push down predicates**: Filter at source when possible
✅ **Partition data**: Use partitioned tables/files
✅ **Optimize joins**: Use broadcast joins for small tables
✅ **Cache reference data**: Reuse lookup tables

### Resource Optimization
✅ Start with **small resources**, scale up as needed
✅ Monitor **resource utilization** metrics
✅ Use **auto-tuning** for automatic optimization
✅ Increase **parallelism** for large datasets
✅ Use **incremental processing** instead of full loads

### Data Format Optimization
✅ Use **Parquet** or **Avro** for Cloud Storage (not CSV)
✅ Use **columnar formats** for analytics workloads
✅ **Compress data** to reduce I/O
✅ Use **partitioning** in BigQuery and Cloud Storage

---

## 12. Cost Optimization

### Instance Cost Management
✅ Use **Basic edition** for dev/test
✅ Use **Enterprise edition** only for production
✅ **Delete unused instances**
✅ Share instances across **multiple teams**

### Pipeline Cost Optimization
✅ Use **incremental loads** instead of full refreshes
✅ **Filter at source** to reduce data processing
✅ Optimize **resource allocation** (don't over-provision)
✅ Use **scheduled pipelines** during off-peak hours
✅ Monitor **Dataproc cluster usage** (Data Fusion uses Dataproc)

### Data Transfer Cost Reduction
✅ Keep data and compute in **same region**
✅ Use **regional Cloud Storage** buckets
✅ Minimize **cross-region transfers**
✅ Use **Private Google Access** (no egress charges)

---

## 13. Testing & Deployment

### Pipeline Testing
✅ Test with **small datasets** in development
✅ Validate **schema mappings**
✅ Test **error handling** scenarios
✅ Verify **data quality** checks
✅ Test with **realistic data volumes**

### CI/CD Integration
```bash
# Export pipeline
cdap-cli.sh export artifact pipeline-name > pipeline.json

# Import to another instance
cdap-cli.sh import artifact pipeline.json

# Deploy via API
curl -X PUT \
  http://DATA_FUSION_ENDPOINT/v3/namespaces/default/apps/my-pipeline \
  -H 'Content-Type: application/json' \
  -d @pipeline.json
```

### Deployment Best Practices
✅ Use **separate instances** for dev/staging/prod
✅ Version control **pipeline JSON** definitions
✅ Use **CI/CD pipelines** for deployments
✅ Test in staging before **production deployment**
✅ Use **Blue-Green deployment** for zero-downtime updates

---

## 14. Migration from On-Premises ETL Tools

### Assessment Phase
✅ **Inventory existing ETL jobs**
✅ Identify **source and target systems**
✅ Analyze **transformation logic**
✅ Document **scheduling dependencies**
✅ Identify **custom code** that needs migration

### Migration Strategies

**Lift and Shift**
- Recreate ETL jobs in Data Fusion
- Use similar logic and patterns
- Minimal redesign

**Modernize**
- Redesign for cloud-native architecture
- Use managed services (BigQuery, Cloud Storage)
- Optimize for cloud performance

### Common Migration Patterns

**Informatica → Data Fusion**
- Mappings → Pipelines
- Transformations → Wrangler/Transforms
- Workflows → Pipeline dependencies

**Talend → Data Fusion**
- Jobs → Pipelines
- Components → Plugins
- Contexts → Runtime arguments

**SSIS → Data Fusion**
- Packages → Pipelines
- Data flow tasks → Transform plugins
- Control flow → Pipeline logic

---

## 15. Common Anti-Patterns to Avoid

❌ **Not using error collectors**: Pipeline fails on bad data
❌ **Full table loads**: Use incremental processing
❌ **Hardcoded values**: Use runtime arguments
❌ **No data quality checks**: Bad data in production
❌ **Over-provisioning resources**: Wasting money
❌ **Not monitoring pipelines**: Miss failures and performance issues
❌ **Complex transformations in JavaScript**: Use Wrangler or custom plugins
❌ **Not testing with realistic data**: Surprises in production
❌ **Storing credentials in pipeline**: Security risk
❌ **Not using VPC peering**: Security and performance issues
❌ **Ignoring data lineage**: Difficult troubleshooting
❌ **Not versioning pipelines**: Cannot rollback changes

---

## Quick Reference Checklist

- [ ] Use Enterprise edition for production
- [ ] Enable Stackdriver logging and monitoring
- [ ] Configure VPC peering for private connectivity
- [ ] Use error collectors in all pipelines
- [ ] Implement data quality checks
- [ ] Use Wrangler for data preparation
- [ ] Filter data early in pipeline
- [ ] Use runtime arguments for parameterization
- [ ] Set up pipeline scheduling
- [ ] Monitor pipeline metrics and logs
- [ ] Use appropriate resource allocation
- [ ] Store credentials in Secret Manager
- [ ] Test pipelines with small datasets first
- [ ] Version control pipeline definitions
- [ ] Use incremental processing when possible
- [ ] Optimize data formats (Parquet, Avro)
- [ ] Set up alerts for pipeline failures
- [ ] Document pipeline logic and dependencies

---

## Additional Resources

- [Cloud Data Fusion Documentation](https://cloud.google.com/data-fusion/docs)
- [Pipeline Design Guide](https://cloud.google.com/data-fusion/docs/how-to/pipeline-design)
- [Plugin Hub](https://hub.cdap.io/)
- [Wrangler Directives Reference](https://cloud.google.com/data-fusion/docs/reference/directives)
- [Pricing Calculator](https://cloud.google.com/products/calculator)

---

*Last Updated: December 25, 2025*
