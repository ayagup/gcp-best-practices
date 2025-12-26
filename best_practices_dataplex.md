# Dataplex Best Practices

*Last Updated: December 25, 2025*

## Overview

Dataplex is an intelligent data fabric that provides unified data management across data lakes, data warehouses, and data marts. It enables organizations to centrally discover, manage, monitor, and govern their data with automated data quality and lifecycle management.

---

## 1. Lake and Zone Architecture

### Lake Organization

**Best Practices:**
- Create lakes based on organizational boundaries or data domains
- Use zones to separate raw, curated, and consumption-ready data
- Implement clear naming conventions
- Document lake and zone purposes

```bash
# Create a Dataplex lake
gcloud dataplex lakes create production-data-lake \
    --location=us-central1 \
    --display-name="Production Data Lake" \
    --description="Primary production data lake for analytics" \
    --labels=environment=production,team=data-engineering

# Create zones within the lake
# Raw zone for unprocessed data
gcloud dataplex zones create raw-zone \
    --location=us-central1 \
    --lake=production-data-lake \
    --type=RAW \
    --resource-location-type=SINGLE_REGION \
    --discovery-enabled \
    --discovery-schedule="0 */12 * * *"

# Curated zone for processed data
gcloud dataplex zones create curated-zone \
    --location=us-central1 \
    --lake=production-data-lake \
    --type=CURATED \
    --resource-location-type=SINGLE_REGION \
    --discovery-enabled \
    --discovery-schedule="0 */6 * * *"
```

**Python SDK Configuration:**
```python
from google.cloud import dataplex_v1

def create_data_lake(project_id, location, lake_id):
    """Create a Dataplex lake with best practices."""
    
    client = dataplex_v1.DataplexServiceClient()
    parent = f"projects/{project_id}/locations/{location}"
    
    lake = dataplex_v1.Lake(
        display_name="Enterprise Data Lake",
        description="Centralized data lake for all business data",
        labels={
            "environment": "production",
            "cost-center": "data-platform",
            "compliance": "pii-data"
        },
    )
    
    operation = client.create_lake(
        parent=parent,
        lake_id=lake_id,
        lake=lake
    )
    
    result = operation.result()
    print(f"Created lake: {result.name}")
    return result

def create_zones(project_id, location, lake_id):
    """Create zones for different data stages."""
    
    client = dataplex_v1.DataplexServiceClient()
    parent = f"projects/{project_id}/locations/{location}/lakes/{lake_id}"
    
    zones = [
        {
            "zone_id": "raw-ingestion",
            "type": dataplex_v1.Zone.Type.RAW,
            "description": "Raw data from various sources",
            "discovery_spec": {
                "enabled": True,
                "schedule": "0 */12 * * *"  # Every 12 hours
            }
        },
        {
            "zone_id": "bronze-landing",
            "type": dataplex_v1.Zone.Type.RAW,
            "description": "Validated and standardized raw data",
            "discovery_spec": {
                "enabled": True,
                "schedule": "0 */6 * * *"
            }
        },
        {
            "zone_id": "silver-processed",
            "type": dataplex_v1.Zone.Type.CURATED,
            "description": "Cleaned and enriched data",
            "discovery_spec": {
                "enabled": True,
                "schedule": "0 */4 * * *"
            }
        },
        {
            "zone_id": "gold-analytics",
            "type": dataplex_v1.Zone.Type.CURATED,
            "description": "Business-ready aggregated data",
            "discovery_spec": {
                "enabled": True,
                "schedule": "0 */2 * * *"
            }
        }
    ]
    
    created_zones = []
    for zone_config in zones:
        zone = dataplex_v1.Zone(
            type_=zone_config["type"],
            description=zone_config["description"],
            resource_spec=dataplex_v1.Zone.ResourceSpec(
                location_type=dataplex_v1.Zone.ResourceSpec.LocationType.SINGLE_REGION
            ),
            discovery_spec=dataplex_v1.Zone.DiscoverySpec(
                enabled=zone_config["discovery_spec"]["enabled"],
                schedule=zone_config["discovery_spec"]["schedule"]
            )
        )
        
        operation = client.create_zone(
            parent=parent,
            zone_id=zone_config["zone_id"],
            zone=zone
        )
        
        result = operation.result()
        created_zones.append(result)
        print(f"Created zone: {result.name}")
    
    return created_zones
```

---

## 2. Asset Management

### Adding Assets to Zones

**Best Practices:**
- Attach Cloud Storage buckets to raw zones
- Attach BigQuery datasets to curated zones
- Enable automatic discovery for metadata
- Use consistent asset naming

```bash
# Add Cloud Storage bucket as asset
gcloud dataplex assets create raw-data-bucket \
    --location=us-central1 \
    --lake=production-data-lake \
    --zone=raw-zone \
    --resource-type=STORAGE_BUCKET \
    --resource-name=projects/my-project/buckets/raw-data-bucket \
    --discovery-enabled \
    --discovery-schedule="0 */6 * * *"

# Add BigQuery dataset as asset
gcloud dataplex assets create analytics-dataset \
    --location=us-central1 \
    --lake=production-data-lake \
    --zone=gold-analytics \
    --resource-type=BIGQUERY_DATASET \
    --resource-name=projects/my-project/datasets/analytics \
    --discovery-enabled \
    --discovery-schedule="0 */2 * * *"
```

**Python Asset Configuration:**
```python
def add_storage_asset(project_id, location, lake_id, zone_id, bucket_name):
    """Add Cloud Storage bucket as Dataplex asset."""
    
    client = dataplex_v1.DataplexServiceClient()
    parent = f"projects/{project_id}/locations/{location}/lakes/{lake_id}/zones/{zone_id}"
    
    asset = dataplex_v1.Asset(
        resource_spec=dataplex_v1.Asset.ResourceSpec(
            name=f"projects/{project_id}/buckets/{bucket_name}",
            type_=dataplex_v1.Asset.ResourceSpec.Type.STORAGE_BUCKET,
        ),
        discovery_spec=dataplex_v1.Asset.DiscoverySpec(
            enabled=True,
            schedule="0 */6 * * *",
            include_patterns=["*.parquet", "*.avro", "*.csv"],
            exclude_patterns=["temp/*", "*.tmp", "_SUCCESS"],
        ),
        resource_status=dataplex_v1.Asset.ResourceStatus(
            state=dataplex_v1.Asset.ResourceStatus.State.READY
        )
    )
    
    operation = client.create_asset(
        parent=parent,
        asset_id=f"{bucket_name.replace('_', '-')}-asset",
        asset=asset
    )
    
    result = operation.result()
    print(f"Created asset: {result.name}")
    return result

def add_bigquery_asset(project_id, location, lake_id, zone_id, dataset_id):
    """Add BigQuery dataset as Dataplex asset."""
    
    client = dataplex_v1.DataplexServiceClient()
    parent = f"projects/{project_id}/locations/{location}/lakes/{lake_id}/zones/{zone_id}"
    
    asset = dataplex_v1.Asset(
        resource_spec=dataplex_v1.Asset.ResourceSpec(
            name=f"projects/{project_id}/datasets/{dataset_id}",
            type_=dataplex_v1.Asset.ResourceSpec.Type.BIGQUERY_DATASET,
        ),
        discovery_spec=dataplex_v1.Asset.DiscoverySpec(
            enabled=True,
            schedule="0 */2 * * *"
        )
    )
    
    operation = client.create_asset(
        parent=parent,
        asset_id=f"{dataset_id}-asset",
        asset=asset
    )
    
    result = operation.result()
    return result
```

---

## 3. Data Discovery and Metadata

### Automatic Discovery

**Best Practices:**
- Enable discovery on all assets
- Set appropriate discovery schedules
- Use include/exclude patterns for files
- Monitor discovery job results

**Discovery Configuration:**
```python
def configure_discovery(project_id, location, lake_id, zone_id, asset_id):
    """Configure discovery settings for an asset."""
    
    client = dataplex_v1.DataplexServiceClient()
    asset_name = f"projects/{project_id}/locations/{location}/lakes/{lake_id}/zones/{zone_id}/assets/{asset_id}"
    
    # Get existing asset
    asset = client.get_asset(name=asset_name)
    
    # Update discovery spec
    asset.discovery_spec = dataplex_v1.Asset.DiscoverySpec(
        enabled=True,
        schedule="0 */4 * * *",  # Every 4 hours
        include_patterns=[
            "*.parquet",
            "*.avro",
            "*.orc",
            "*.json",
            "*.csv"
        ],
        exclude_patterns=[
            "temp/*",
            "_temporary/*",
            "*.tmp",
            "*.log",
            "_SUCCESS"
        ],
        csv_options=dataplex_v1.Asset.DiscoverySpec.CsvOptions(
            header_rows=1,
            delimiter=",",
            encoding="UTF-8"
        ),
        json_options=dataplex_v1.Asset.DiscoverySpec.JsonOptions(
            encoding="UTF-8"
        )
    )
    
    # Update asset
    update_mask = {"paths": ["discovery_spec"]}
    operation = client.update_asset(
        asset=asset,
        update_mask=update_mask
    )
    
    result = operation.result()
    print(f"Updated discovery configuration for: {result.name}")
    return result
```

### Metadata Management

**Query Discovered Metadata:**
```python
from google.cloud import datacatalog_v1

def search_dataplex_metadata(project_id, query):
    """Search metadata discovered by Dataplex."""
    
    datacatalog_client = datacatalog_v1.DataCatalogClient()
    scope = datacatalog_v1.SearchCatalogRequest.Scope(
        include_project_ids=[project_id]
    )
    
    request = datacatalog_v1.SearchCatalogRequest(
        scope=scope,
        query=query,
        page_size=100
    )
    
    results = datacatalog_client.search_catalog(request=request)
    
    for result in results:
        print(f"Resource: {result.relative_resource_name}")
        print(f"Type: {result.search_result_type}")
        print(f"Description: {result.description}")
        print("---")
    
    return results

# Example searches
# Find all tables in a specific lake
search_dataplex_metadata(
    project_id="my-project",
    query="system=dataplex lake=production-data-lake"
)

# Find Parquet files
search_dataplex_metadata(
    project_id="my-project",
    query="system=dataplex type=FILE format=PARQUET"
)
```

---

## 4. Data Quality

### Data Quality Rules

**Best Practices:**
- Define data quality rules for critical datasets
- Implement completeness, validity, and consistency checks
- Monitor data quality scores
- Set up alerts for quality issues

```python
from google.cloud import dataplex_v1

def create_data_quality_scan(
    project_id,
    location,
    lake_id,
    zone_id,
    asset_id,
    table_name
):
    """Create data quality scan for a table."""
    
    client = dataplex_v1.DataScanServiceClient()
    parent = f"projects/{project_id}/locations/{location}"
    
    # Define data quality rules
    data_quality_spec = dataplex_v1.DataQualitySpec(
        rules=[
            # Completeness check
            dataplex_v1.DataQualityRule(
                dimension="COMPLETENESS",
                name="customer_id_not_null",
                description="Customer ID should not be null",
                non_null_expectation=dataplex_v1.DataQualityRule.NonNullExpectation(
                    column="customer_id"
                ),
                threshold=0.95  # 95% of records must pass
            ),
            # Uniqueness check
            dataplex_v1.DataQualityRule(
                dimension="UNIQUENESS",
                name="customer_id_unique",
                description="Customer ID should be unique",
                uniqueness_expectation=dataplex_v1.DataQualityRule.UniquenessExpectation(
                    column="customer_id"
                ),
                threshold=1.0  # 100% must be unique
            ),
            # Range check
            dataplex_v1.DataQualityRule(
                dimension="VALIDITY",
                name="age_range_check",
                description="Age should be between 0 and 120",
                range_expectation=dataplex_v1.DataQualityRule.RangeExpectation(
                    column="age",
                    min_value="0",
                    max_value="120"
                ),
                threshold=0.99
            ),
            # Regex pattern check
            dataplex_v1.DataQualityRule(
                dimension="VALIDITY",
                name="email_format",
                description="Email should match valid format",
                regex_expectation=dataplex_v1.DataQualityRule.RegexExpectation(
                    column="email",
                    regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                ),
                threshold=0.98
            ),
            # Set membership check
            dataplex_v1.DataQualityRule(
                dimension="VALIDITY",
                name="status_values",
                description="Status should be one of valid values",
                set_expectation=dataplex_v1.DataQualityRule.SetExpectation(
                    column="status",
                    values=["active", "inactive", "pending", "deleted"]
                ),
                threshold=1.0
            ),
            # Row condition check
            dataplex_v1.DataQualityRule(
                dimension="CONSISTENCY",
                name="order_date_before_ship_date",
                description="Order date should be before ship date",
                sql_assertion=dataplex_v1.DataQualityRule.SqlAssertion(
                    sql_statement="SELECT COUNT(*) FROM `{table}` WHERE order_date > ship_date"
                ),
                threshold=1.0
            ),
            # Statistical check
            dataplex_v1.DataQualityRule(
                dimension="VALIDITY",
                name="order_amount_statistics",
                description="Order amount should be within expected range",
                statistic_range_expectation=dataplex_v1.DataQualityRule.StatisticRangeExpectation(
                    column="order_amount",
                    statistic=dataplex_v1.DataQualityRule.StatisticRangeExpectation.Statistic.MEAN,
                    min_value="50",
                    max_value="500"
                ),
                threshold=0.95
            ),
        ]
    )
    
    # Create data scan
    data_scan = dataplex_v1.DataScan(
        description=f"Data quality scan for {table_name}",
        data=dataplex_v1.DataSource(
            resource=f"//bigquery.googleapis.com/projects/{project_id}/datasets/my_dataset/tables/{table_name}"
        ),
        data_quality_spec=data_quality_spec,
        execution_spec=dataplex_v1.DataScan.ExecutionSpec(
            trigger=dataplex_v1.DataScan.ExecutionSpec.Trigger(
                schedule=dataplex_v1.DataScan.ExecutionSpec.Trigger.Schedule(
                    cron="0 2 * * *"  # Daily at 2 AM
                )
            )
        ),
        labels={
            "environment": "production",
            "criticality": "high"
        }
    )
    
    operation = client.create_data_scan(
        parent=parent,
        data_scan=data_scan,
        data_scan_id=f"{table_name}-quality-scan"
    )
    
    result = operation.result()
    print(f"Created data quality scan: {result.name}")
    return result
```

### Monitor Data Quality

**Query Data Quality Results:**
```python
def get_data_quality_results(project_id, location, data_scan_id):
    """Get data quality scan results."""
    
    client = dataplex_v1.DataScanServiceClient()
    parent = f"projects/{project_id}/locations/{location}/dataScans/{data_scan_id}"
    
    # List scan jobs
    jobs = client.list_data_scan_jobs(parent=parent)
    
    for job in jobs:
        print(f"Job: {job.name}")
        print(f"State: {job.state.name}")
        print(f"Start time: {job.start_time}")
        print(f"End time: {job.end_time}")
        
        if job.data_quality_result:
            result = job.data_quality_result
            print(f"Passed: {result.passed}")
            print(f"Score: {result.score}")
            print(f"Dimensions:")
            
            for dimension, dimension_result in result.dimensions.items():
                print(f"  {dimension}: {dimension_result.passed} (score: {dimension_result.score})")
            
            # Show failed rules
            if result.rules:
                print("Failed rules:")
                for rule in result.rules:
                    if not rule.passed:
                        print(f"  - {rule.rule.name}: {rule.failing_rows_count} failing rows")
        
        print("---")
    
    return jobs
```

---

## 5. Data Profiling

### Create Profiling Scans

**Best Practices:**
- Profile new datasets to understand data characteristics
- Schedule regular profiling for monitoring data drift
- Use profiling insights for data quality rules

```python
def create_data_profile_scan(project_id, location, dataset_id, table_id):
    """Create data profiling scan."""
    
    client = dataplex_v1.DataScanServiceClient()
    parent = f"projects/{project_id}/locations/{location}"
    
    # Data profiling specification
    data_profile_spec = dataplex_v1.DataProfileSpec(
        sampling_percent=100,  # Profile 100% of data
        row_filter="date >= CURRENT_DATE() - 30"  # Last 30 days only
    )
    
    data_scan = dataplex_v1.DataScan(
        description=f"Data profile for {dataset_id}.{table_id}",
        data=dataplex_v1.DataSource(
            resource=f"//bigquery.googleapis.com/projects/{project_id}/datasets/{dataset_id}/tables/{table_id}"
        ),
        data_profile_spec=data_profile_spec,
        execution_spec=dataplex_v1.DataScan.ExecutionSpec(
            trigger=dataplex_v1.DataScan.ExecutionSpec.Trigger(
                schedule=dataplex_v1.DataScan.ExecutionSpec.Trigger.Schedule(
                    cron="0 3 * * 0"  # Weekly on Sunday at 3 AM
                )
            )
        )
    )
    
    operation = client.create_data_scan(
        parent=parent,
        data_scan=data_scan,
        data_scan_id=f"{table_id}-profile-scan"
    )
    
    result = operation.result()
    return result

def get_profile_results(project_id, location, data_scan_id):
    """Get data profiling results."""
    
    client = dataplex_v1.DataScanServiceClient()
    parent = f"projects/{project_id}/locations/{location}/dataScans/{data_scan_id}"
    
    jobs = client.list_data_scan_jobs(parent=parent)
    
    for job in jobs:
        if job.data_profile_result:
            profile = job.data_profile_result.profile
            
            print(f"Table: {job.name}")
            print(f"Row count: {profile.row_count}")
            
            print("\nColumn Profiles:")
            for field_profile in profile.fields:
                print(f"\n  Column: {field_profile.name}")
                print(f"    Type: {field_profile.type_}")
                print(f"    Mode: {field_profile.mode}")
                
                if field_profile.profile:
                    col_profile = field_profile.profile
                    print(f"    Null ratio: {col_profile.null_ratio}")
                    print(f"    Distinct ratio: {col_profile.distinct_ratio}")
                    
                    # Numeric statistics
                    if col_profile.HasField('mean'):
                        print(f"    Mean: {col_profile.mean}")
                        print(f"    Min: {col_profile.min}")
                        print(f"    Max: {col_profile.max}")
                        print(f"    Std dev: {col_profile.standard_deviation}")
                    
                    # Top values
                    if col_profile.top_n_values:
                        print(f"    Top values:")
                        for top_value in col_profile.top_n_values[:5]:
                            print(f"      {top_value.value}: {top_value.count}")
    
    return jobs
```

---

## 6. Data Lifecycle Management

### Lifecycle Policies

**Best Practices:**
- Define retention policies based on data classification
- Automate data archival and deletion
- Comply with regulatory requirements

```python
def set_lifecycle_policy(bucket_name, rules):
    """Set lifecycle policy on Cloud Storage bucket."""
    
    from google.cloud import storage
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    bucket.lifecycle_rules = rules
    bucket.patch()
    
    print(f"Lifecycle policy set on {bucket_name}")
    return bucket

# Example lifecycle rules for data zones
raw_zone_rules = [
    {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {
            "age": 30,  # Move to NEARLINE after 30 days
            "matchesPrefix": ["raw/"]
        }
    },
    {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {
            "age": 90,  # Move to COLDLINE after 90 days
            "matchesPrefix": ["raw/"]
        }
    },
    {
        "action": {"type": "Delete"},
        "condition": {
            "age": 365,  # Delete after 1 year
            "matchesPrefix": ["raw/temp/"]
        }
    }
]

curated_zone_rules = [
    {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {
            "age": 90,  # Keep in STANDARD for 90 days
            "matchesPrefix": ["curated/"]
        }
    },
    {
        "action": {"type": "SetStorageClass", "storageClass": "ARCHIVE"},
        "condition": {
            "age": 730,  # Archive after 2 years
            "matchesPrefix": ["curated/historical/"]
        }
    }
]

# Apply policies
set_lifecycle_policy("raw-data-bucket", raw_zone_rules)
set_lifecycle_policy("curated-data-bucket", curated_zone_rules)
```

---

## 7. Security and Access Control

### IAM Configuration

**Best Practices:**
- Use predefined Dataplex roles
- Implement least privilege access
- Separate read and write permissions
- Use service accounts for automation

```bash
# Grant Dataplex viewer role
gcloud dataplex lakes add-iam-policy-binding production-data-lake \
    --location=us-central1 \
    --member=user:analyst@example.com \
    --role=roles/dataplex.viewer

# Grant Dataplex editor role
gcloud dataplex lakes add-iam-policy-binding production-data-lake \
    --location=us-central1 \
    --member=group:data-engineers@example.com \
    --role=roles/dataplex.editor

# Grant zone-specific access
gcloud dataplex zones add-iam-policy-binding gold-analytics \
    --location=us-central1 \
    --lake=production-data-lake \
    --member=group:analysts@example.com \
    --role=roles/dataplex.dataReader
```

**Python IAM Management:**
```python
def grant_lake_access(project_id, location, lake_id, member, role):
    """Grant IAM access to a Dataplex lake."""
    
    client = dataplex_v1.DataplexServiceClient()
    resource = f"projects/{project_id}/locations/{location}/lakes/{lake_id}"
    
    policy = client.get_iam_policy(resource=resource)
    
    # Add binding
    binding = {
        "role": role,
        "members": [member]
    }
    
    policy.bindings.append(binding)
    
    updated_policy = client.set_iam_policy(
        resource=resource,
        policy=policy
    )
    
    print(f"Granted {role} to {member} on {resource}")
    return updated_policy
```

### Data Classification

**Implement Data Classification:**
```python
def classify_data(project_id, location, lake_id):
    """Apply data classification tags."""
    
    from google.cloud import datacatalog_v1
    
    datacatalog_client = datacatalog_v1.DataCatalogClient()
    tag_template_client = datacatalog_v1.PolicyTagManagerClient()
    
    # Create taxonomy for data classification
    taxonomy = {
        "display_name": "Data Classification",
        "description": "Classification levels for data sensitivity",
        "activated_policy_types": ["FINE_GRAINED_ACCESS_CONTROL"]
    }
    
    # Policy tags: Public, Internal, Confidential, Restricted
    policy_tags = [
        {"display_name": "Public", "description": "Publicly accessible data"},
        {"display_name": "Internal", "description": "Internal use only"},
        {"display_name": "Confidential", "description": "Confidential business data"},
        {"display_name": "Restricted", "description": "Highly sensitive data (PII, PHI)"}
    ]
    
    print("Data classification taxonomy created")
    return taxonomy, policy_tags
```

---

## 8. Monitoring and Logging

### Monitor Dataplex Operations

**Best Practices:**
- Enable Cloud Logging for Dataplex
- Monitor discovery job status
- Track data quality metrics
- Set up alerts for failures

```python
from google.cloud import logging

def query_dataplex_logs(project_id):
    """Query Dataplex operation logs."""
    
    client = logging.Client(project=project_id)
    
    filter_str = """
    resource.type="dataplex.googleapis.com/Lake"
    severity>=WARNING
    """
    
    for entry in client.list_entries(filter_=filter_str, page_size=100):
        print(f"Timestamp: {entry.timestamp}")
        print(f"Severity: {entry.severity}")
        print(f"Log: {entry.payload}")
        print("---")

def create_dataplex_alerts(project_id):
    """Create alerts for Dataplex issues."""
    
    from google.cloud import monitoring_v3
    
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"
    
    # Alert for failed discovery jobs
    alert_policy = monitoring_v3.AlertPolicy(
        display_name="Dataplex Discovery Failures",
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Failed discovery jobs",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='resource.type="dataplex.googleapis.com/Lake" '
                           'metric.type="dataplex.googleapis.com/lake/discovery/failed_count"',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=0,
                    duration={"seconds": 300},
                ),
            )
        ],
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content="Dataplex discovery jobs are failing. Check logs for details.",
        ),
    )
    
    created_policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
    
    return created_policy
```

---

## 9. Integration Patterns

### Dataplex with BigQuery

**Query Dataplex-Managed Tables:**
```sql
-- Query tables across multiple zones
SELECT 
    table_schema,
    table_name,
    creation_time,
    row_count,
    size_bytes
FROM 
    `project.dataplex_lake.INFORMATION_SCHEMA.TABLES`
WHERE 
    table_schema IN ('raw_zone', 'curated_zone', 'gold_zone')
ORDER BY 
    creation_time DESC;

-- Query with Dataplex metadata
SELECT 
    t.table_name,
    t.row_count,
    COALESCE(dq.data_quality_score, 0) as quality_score
FROM 
    `project.dataplex_lake.INFORMATION_SCHEMA.TABLES` t
LEFT JOIN 
    `project.dataplex_metadata.data_quality_scores` dq
    ON t.table_name = dq.table_name;
```

### Dataplex with Data Catalog

**Enrich Metadata:**
```python
def enrich_dataplex_metadata(project_id, location, entry_name):
    """Add custom metadata to Dataplex entities."""
    
    from google.cloud import datacatalog_v1
    
    client = datacatalog_v1.DataCatalogClient()
    
    # Create custom tag template
    tag_template = datacatalog_v1.TagTemplate(
        display_name="Data Ownership",
        fields={
            "owner": datacatalog_v1.TagTemplateField(
                display_name="Data Owner",
                type_=datacatalog_v1.FieldType(
                    primitive_type=datacatalog_v1.FieldType.PrimitiveType.STRING
                ),
                is_required=True
            ),
            "team": datacatalog_v1.TagTemplateField(
                display_name="Responsible Team",
                type_=datacatalog_v1.FieldType(
                    primitive_type=datacatalog_v1.FieldType.PrimitiveType.STRING
                )
            ),
            "update_frequency": datacatalog_v1.TagTemplateField(
                display_name="Update Frequency",
                type_=datacatalog_v1.FieldType(
                    enum_type=datacatalog_v1.FieldType.EnumType(
                        allowed_values=[
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Real-time"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Hourly"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Daily"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Weekly")
                        ]
                    )
                )
            )
        }
    )
    
    # Attach tag to entry
    tag = datacatalog_v1.Tag(
        template=f"projects/{project_id}/locations/{location}/tagTemplates/data_ownership",
        fields={
            "owner": datacatalog_v1.TagField(string_value="data-team@example.com"),
            "team": datacatalog_v1.TagField(string_value="Data Engineering"),
            "update_frequency": datacatalog_v1.TagField(enum_value=datacatalog_v1.TagField.EnumValue(display_name="Daily"))
        }
    )
    
    return tag
```

---

## 10. Cost Optimization

### Optimize Storage Costs

**Best Practices:**
```python
def analyze_storage_costs(project_id, lake_id):
    """Analyze and optimize storage costs in Dataplex."""
    
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    
    # Query to find large tables
    query = f"""
    SELECT
        table_schema as zone,
        table_name,
        ROUND(size_bytes / POW(10, 9), 2) as size_gb,
        ROUND(size_bytes / POW(10, 12), 4) as size_tb,
        row_count,
        creation_time,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), creation_time, DAY) as age_days,
        ROUND(size_bytes / POW(10, 9) * 0.02, 2) as monthly_cost_standard,
        ROUND(size_bytes / POW(10, 9) * 0.01, 2) as monthly_cost_nearline,
        ROUND(size_bytes / POW(10, 9) * 0.004, 2) as monthly_cost_coldline
    FROM
        `{project_id}.{lake_id}.INFORMATION_SCHEMA.TABLES`
    WHERE
        table_type = 'BASE TABLE'
    ORDER BY
        size_bytes DESC
    LIMIT 100
    """
    
    results = client.query(query).result()
    
    print("Top storage consumers:")
    print(f"{'Zone':<20} {'Table':<30} {'Size (GB)':<12} {'Age (days)':<12} {'Monthly Cost'}")
    print("-" * 100)
    
    total_size = 0
    total_cost = 0
    
    for row in results:
        total_size += row.size_gb
        total_cost += row.monthly_cost_standard
        
        print(f"{row.zone:<20} {row.table_name:<30} {row.size_gb:<12.2f} {row.age_days:<12} ${row.monthly_cost_standard:.2f}")
        
        # Suggest cost optimization
        if row.age_days > 90 and row.size_gb > 100:
            savings = row.monthly_cost_standard - row.monthly_cost_nearline
            print(f"  💡 Potential savings: ${savings:.2f}/month by moving to Nearline")
    
    print("-" * 100)
    print(f"Total size: {total_size:.2f} GB")
    print(f"Total monthly cost: ${total_cost:.2f}")
    
    return results
```

---

## 11. Common Anti-Patterns

### ❌ Anti-Pattern 1: Flat Lake Structure
**Problem:** All data in single zone without organization
**Solution:** Use multiple zones (raw, bronze, silver, gold)

### ❌ Anti-Pattern 2: No Data Quality Checks
**Problem:** Poor data quality propagates through pipeline
**Solution:** Implement data quality scans at each zone

### ❌ Anti-Pattern 3: Manual Metadata Management
**Problem:** Outdated or missing metadata
**Solution:** Enable automatic discovery and profiling

### ❌ Anti-Pattern 4: Overly Permissive Access
**Problem:** Security and compliance risks
**Solution:** Implement fine-grained IAM and data classification

### ❌ Anti-Pattern 5: Ignoring Costs
**Problem:** Unnecessary storage of raw data
**Solution:** Implement lifecycle policies and monitor costs

---

## 12. Quick Reference Checklist

### Lake Setup
- [ ] Create lakes aligned with organizational boundaries
- [ ] Define zones for different data stages (raw, curated, etc.)
- [ ] Implement consistent naming conventions
- [ ] Document lake and zone purposes
- [ ] Configure appropriate IAM roles

### Asset Management
- [ ] Add all data sources as assets
- [ ] Enable automatic discovery
- [ ] Set appropriate discovery schedules
- [ ] Configure include/exclude patterns
- [ ] Monitor discovery job status

### Data Quality
- [ ] Define data quality rules for critical datasets
- [ ] Schedule regular quality scans
- [ ] Monitor quality scores and trends
- [ ] Set up alerts for quality issues
- [ ] Document quality requirements

### Security
- [ ] Implement least privilege access
- [ ] Apply data classification tags
- [ ] Enable audit logging
- [ ] Review access permissions regularly
- [ ] Comply with regulatory requirements

### Cost Management
- [ ] Implement lifecycle policies
- [ ] Monitor storage usage
- [ ] Archive or delete old data
- [ ] Use appropriate storage classes
- [ ] Review costs regularly

---

*Best Practices for Google Cloud Data Engineer Certification*
