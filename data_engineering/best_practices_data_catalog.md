# Data Catalog Best Practices

*Last Updated: December 25, 2025*

## Overview

Data Catalog is a fully managed and scalable metadata management service that enables organizations to quickly discover, understand, and manage their data across Google Cloud. It provides a unified view of data assets with automatic metadata harvesting and powerful search capabilities.

---

## 1. Catalog Organization

### Entry Groups

**Best Practices:**
- Organize entries by system, department, or data domain
- Use consistent naming conventions
- Document entry group purposes
- Implement logical hierarchies

```python
from google.cloud import datacatalog_v1

def create_entry_group(project_id, location, entry_group_id, display_name):
    """Create a Data Catalog entry group."""
    
    client = datacatalog_v1.DataCatalogClient()
    parent = f"projects/{project_id}/locations/{location}"
    
    entry_group = datacatalog_v1.EntryGroup(
        display_name=display_name,
        description=f"Entry group for {display_name}",
    )
    
    created_group = client.create_entry_group(
        parent=parent,
        entry_group_id=entry_group_id,
        entry_group=entry_group
    )
    
    print(f"Created entry group: {created_group.name}")
    return created_group

# Example: Create entry groups by domain
domains = [
    ("finance", "Finance Data Assets"),
    ("marketing", "Marketing Data Assets"),
    ("sales", "Sales Data Assets"),
    ("customer", "Customer Data Assets"),
]

for domain_id, display_name in domains:
    create_entry_group(
        project_id="my-project",
        location="us-central1",
        entry_group_id=domain_id,
        display_name=display_name
    )
```

---

## 2. Entry Creation and Management

### Create Custom Entries

**Best Practices:**
- Register all data assets (not just GCP resources)
- Include comprehensive descriptions
- Add relevant schema information
- Maintain entry accuracy

```python
def create_custom_entry(
    project_id,
    location,
    entry_group_id,
    entry_id,
    display_name,
    entry_type,
    linked_resource=None
):
    """Create a custom Data Catalog entry."""
    
    client = datacatalog_v1.DataCatalogClient()
    parent = f"projects/{project_id}/locations/{location}/entryGroups/{entry_group_id}"
    
    entry = datacatalog_v1.Entry(
        display_name=display_name,
        description="Detailed description of the data asset",
        type_=entry_type,
        user_specified_system="custom-system",
        user_specified_type="custom-type",
    )
    
    # Link to external resource if applicable
    if linked_resource:
        entry.linked_resource = linked_resource
    
    # Add schema if applicable
    entry.schema = datacatalog_v1.Schema(
        columns=[
            datacatalog_v1.ColumnSchema(
                column="customer_id",
                type_="STRING",
                description="Unique customer identifier",
                mode="REQUIRED"
            ),
            datacatalog_v1.ColumnSchema(
                column="email",
                type_="STRING",
                description="Customer email address",
                mode="NULLABLE"
            ),
            datacatalog_v1.ColumnSchema(
                column="created_at",
                type_="TIMESTAMP",
                description="Account creation timestamp",
                mode="REQUIRED"
            ),
        ]
    )
    
    created_entry = client.create_entry(
        parent=parent,
        entry_id=entry_id,
        entry=entry
    )
    
    print(f"Created entry: {created_entry.name}")
    return created_entry

# Example: Register external database
create_custom_entry(
    project_id="my-project",
    location="us-central1",
    entry_group_id="finance",
    entry_id="mysql-customer-db",
    display_name="MySQL Customer Database",
    entry_type=datacatalog_v1.EntryType.DATABASE,
    linked_resource="mysql://prod-server:3306/customers"
)
```

### Lookup Existing Entries

**Find GCP Resources:**
```python
def lookup_entry_by_resource(resource_name):
    """Lookup Data Catalog entry for a GCP resource."""
    
    client = datacatalog_v1.DataCatalogClient()
    
    # Lookup BigQuery table
    request = datacatalog_v1.LookupEntryRequest(
        linked_resource=resource_name
    )
    
    entry = client.lookup_entry(request=request)
    
    print(f"Entry name: {entry.name}")
    print(f"Display name: {entry.display_name}")
    print(f"Type: {entry.type_.name}")
    print(f"Description: {entry.description}")
    
    if entry.schema:
        print("\nSchema:")
        for column in entry.schema.columns:
            print(f"  {column.column} ({column.type_}): {column.description}")
    
    return entry

# Examples
# BigQuery table
lookup_entry_by_resource(
    "//bigquery.googleapis.com/projects/my-project/datasets/analytics/tables/customers"
)

# Pub/Sub topic
lookup_entry_by_resource(
    "//pubsub.googleapis.com/projects/my-project/topics/order-events"
)

# Cloud Storage bucket
lookup_entry_by_resource(
    "//storage.googleapis.com/buckets/my-data-bucket"
)
```

---

## 3. Tagging Strategy

### Tag Templates

**Best Practices:**
- Create reusable tag templates
- Define clear field types and constraints
- Use enumerations for controlled vocabularies
- Document template purposes

```python
def create_tag_template(project_id, location, template_id):
    """Create comprehensive tag template."""
    
    client = datacatalog_v1.DataCatalogClient()
    parent = f"projects/{project_id}/locations/{location}"
    
    tag_template = datacatalog_v1.TagTemplate(
        display_name="Data Asset Metadata",
        fields={
            # Data Owner
            "data_owner": datacatalog_v1.TagTemplateField(
                display_name="Data Owner",
                description="Email of the data owner",
                type_=datacatalog_v1.FieldType(
                    primitive_type=datacatalog_v1.FieldType.PrimitiveType.STRING
                ),
                is_required=True
            ),
            # Business Unit
            "business_unit": datacatalog_v1.TagTemplateField(
                display_name="Business Unit",
                description="Organizational unit responsible for the data",
                type_=datacatalog_v1.FieldType(
                    enum_type=datacatalog_v1.FieldType.EnumType(
                        allowed_values=[
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Finance"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Marketing"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Sales"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Operations"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="IT"),
                        ]
                    )
                ),
                is_required=True
            ),
            # Data Classification
            "data_classification": datacatalog_v1.TagTemplateField(
                display_name="Data Classification",
                description="Sensitivity level of the data",
                type_=datacatalog_v1.FieldType(
                    enum_type=datacatalog_v1.FieldType.EnumType(
                        allowed_values=[
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Public"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Internal"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Confidential"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Restricted"),
                        ]
                    )
                ),
                is_required=True
            ),
            # Update Frequency
            "update_frequency": datacatalog_v1.TagTemplateField(
                display_name="Update Frequency",
                description="How often the data is updated",
                type_=datacatalog_v1.FieldType(
                    enum_type=datacatalog_v1.FieldType.EnumType(
                        allowed_values=[
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Real-time"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Hourly"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Daily"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Weekly"),
                            datacatalog_v1.FieldType.EnumType.EnumValue(display_name="Monthly"),
                        ]
                    )
                )
            ),
            # Contains PII
            "contains_pii": datacatalog_v1.TagTemplateField(
                display_name="Contains PII",
                description="Whether the data contains personally identifiable information",
                type_=datacatalog_v1.FieldType(
                    primitive_type=datacatalog_v1.FieldType.PrimitiveType.BOOL
                ),
                is_required=True
            ),
            # Retention Period (days)
            "retention_days": datacatalog_v1.TagTemplateField(
                display_name="Retention Period (Days)",
                description="Number of days to retain the data",
                type_=datacatalog_v1.FieldType(
                    primitive_type=datacatalog_v1.FieldType.PrimitiveType.DOUBLE
                )
            ),
            # Data Quality Score
            "data_quality_score": datacatalog_v1.TagTemplateField(
                display_name="Data Quality Score",
                description="Data quality score (0-100)",
                type_=datacatalog_v1.FieldType(
                    primitive_type=datacatalog_v1.FieldType.PrimitiveType.DOUBLE
                )
            ),
            # Documentation URL
            "documentation_url": datacatalog_v1.TagTemplateField(
                display_name="Documentation URL",
                description="Link to detailed documentation",
                type_=datacatalog_v1.FieldType(
                    primitive_type=datacatalog_v1.FieldType.PrimitiveType.STRING
                )
            ),
            # Last Validated
            "last_validated": datacatalog_v1.TagTemplateField(
                display_name="Last Validated",
                description="Date when data was last validated",
                type_=datacatalog_v1.FieldType(
                    primitive_type=datacatalog_v1.FieldType.PrimitiveType.TIMESTAMP
                )
            ),
        }
    )
    
    created_template = client.create_tag_template(
        parent=parent,
        tag_template_id=template_id,
        tag_template=tag_template
    )
    
    print(f"Created tag template: {created_template.name}")
    return created_template
```

### Apply Tags to Entries

**Best Practices:**
- Tag all critical data assets
- Keep tags up-to-date
- Use tags for access control
- Automate tagging where possible

```python
def create_and_attach_tag(entry_name, template_name, tag_values):
    """Create and attach a tag to an entry."""
    
    client = datacatalog_v1.DataCatalogClient()
    
    tag = datacatalog_v1.Tag(
        template=template_name,
        fields={
            "data_owner": datacatalog_v1.TagField(
                string_value=tag_values.get("data_owner")
            ),
            "business_unit": datacatalog_v1.TagField(
                enum_value=datacatalog_v1.TagField.EnumValue(
                    display_name=tag_values.get("business_unit")
                )
            ),
            "data_classification": datacatalog_v1.TagField(
                enum_value=datacatalog_v1.TagField.EnumValue(
                    display_name=tag_values.get("data_classification")
                )
            ),
            "update_frequency": datacatalog_v1.TagField(
                enum_value=datacatalog_v1.TagField.EnumValue(
                    display_name=tag_values.get("update_frequency")
                )
            ),
            "contains_pii": datacatalog_v1.TagField(
                bool_value=tag_values.get("contains_pii", False)
            ),
            "retention_days": datacatalog_v1.TagField(
                double_value=tag_values.get("retention_days", 365.0)
            ),
            "data_quality_score": datacatalog_v1.TagField(
                double_value=tag_values.get("data_quality_score", 0.0)
            ),
            "documentation_url": datacatalog_v1.TagField(
                string_value=tag_values.get("documentation_url", "")
            ),
        }
    )
    
    created_tag = client.create_tag(parent=entry_name, tag=tag)
    
    print(f"Created tag on {entry_name}")
    return created_tag

# Example: Tag a BigQuery table
entry_name = "projects/my-project/locations/us-central1/entryGroups/@bigquery/entries/..."
template_name = "projects/my-project/locations/us-central1/tagTemplates/data_asset_metadata"

create_and_attach_tag(
    entry_name=entry_name,
    template_name=template_name,
    tag_values={
        "data_owner": "data-team@example.com",
        "business_unit": "Finance",
        "data_classification": "Confidential",
        "update_frequency": "Daily",
        "contains_pii": True,
        "retention_days": 2555.0,  # 7 years
        "data_quality_score": 95.0,
        "documentation_url": "https://wiki.example.com/data/customers"
    }
)
```

### Bulk Tagging

**Automate Tag Application:**
```python
def bulk_tag_tables(project_id, dataset_id, template_name, common_tags):
    """Apply tags to all tables in a BigQuery dataset."""
    
    from google.cloud import bigquery
    
    bq_client = bigquery.Client(project=project_id)
    dc_client = datacatalog_v1.DataCatalogClient()
    
    # List all tables in dataset
    dataset_ref = f"{project_id}.{dataset_id}"
    tables = bq_client.list_tables(dataset_ref)
    
    for table in tables:
        table_resource = f"//bigquery.googleapis.com/projects/{project_id}/datasets/{dataset_id}/tables/{table.table_id}"
        
        try:
            # Lookup entry
            entry = dc_client.lookup_entry(
                request=datacatalog_v1.LookupEntryRequest(
                    linked_resource=table_resource
                )
            )
            
            # Create tag
            create_and_attach_tag(
                entry_name=entry.name,
                template_name=template_name,
                tag_values=common_tags
            )
            
            print(f"Tagged table: {table.table_id}")
            
        except Exception as e:
            print(f"Error tagging {table.table_id}: {e}")
    
    print(f"Bulk tagging complete for dataset {dataset_id}")

# Example: Tag all tables in a dataset
bulk_tag_tables(
    project_id="my-project",
    dataset_id="analytics",
    template_name="projects/my-project/locations/us-central1/tagTemplates/data_asset_metadata",
    common_tags={
        "data_owner": "analytics-team@example.com",
        "business_unit": "Marketing",
        "data_classification": "Internal",
        "update_frequency": "Daily",
        "contains_pii": False,
        "retention_days": 365.0,
    }
)
```

---

## 4. Search and Discovery

### Search Catalog

**Best Practices:**
- Use specific search queries
- Leverage faceted search
- Search by tags and properties
- Use wildcards appropriately

```python
def search_catalog(project_id, query, scope=None):
    """Search Data Catalog with advanced filters."""
    
    client = datacatalog_v1.DataCatalogClient()
    
    # Define search scope
    if scope is None:
        scope = datacatalog_v1.SearchCatalogRequest.Scope(
            include_project_ids=[project_id]
        )
    
    request = datacatalog_v1.SearchCatalogRequest(
        scope=scope,
        query=query,
        page_size=100,
        order_by="relevance"
    )
    
    results = client.search_catalog(request=request)
    
    for result in results:
        print(f"\nResource: {result.relative_resource_name}")
        print(f"  Type: {result.search_result_type.name}")
        print(f"  Subtype: {result.search_result_subtype}")
        print(f"  System: {result.user_specified_system}")
        print(f"  Description: {result.description[:100]}...")
        
        if result.linked_resource:
            print(f"  Linked resource: {result.linked_resource}")
    
    return results

# Example searches

# Find all tables containing customer data
search_catalog(
    project_id="my-project",
    query="type=TABLE name:customer*"
)

# Find all resources with PII
search_catalog(
    project_id="my-project",
    query="tag:contains_pii=true"
)

# Find resources by owner
search_catalog(
    project_id="my-project",
    query='tag:data_owner="data-team@example.com"'
)

# Find resources by classification
search_catalog(
    project_id="my-project",
    query="tag:data_classification=Confidential"
)

# Find BigQuery tables in specific dataset
search_catalog(
    project_id="my-project",
    query="system=bigquery dataset:analytics type=TABLE"
)

# Complex search with multiple criteria
search_catalog(
    project_id="my-project",
    query='type=TABLE tag:business_unit=Finance tag:contains_pii=true'
)
```

### Advanced Search Patterns

**Search by Schema:**
```python
def search_by_column(project_id, column_name):
    """Search for tables containing a specific column."""
    
    client = datacatalog_v1.DataCatalogClient()
    
    query = f"column:{column_name}"
    
    scope = datacatalog_v1.SearchCatalogRequest.Scope(
        include_project_ids=[project_id]
    )
    
    request = datacatalog_v1.SearchCatalogRequest(
        scope=scope,
        query=query,
        page_size=100
    )
    
    results = client.search_catalog(request=request)
    
    print(f"Tables containing column '{column_name}':")
    for result in results:
        print(f"  - {result.relative_resource_name}")
    
    return results

# Find all tables with customer_id column
search_by_column("my-project", "customer_id")

# Find all tables with email column
search_by_column("my-project", "email")
```

---

## 5. Policy Tags and Access Control

### Create Policy Taxonomies

**Best Practices:**
- Create taxonomies for data classification
- Use policy tags for column-level security
- Implement fine-grained access control
- Document taxonomy structure

```python
from google.cloud.datacatalog_v1 import PolicyTagManagerClient

def create_data_classification_taxonomy(project_id, location):
    """Create taxonomy for data classification."""
    
    client = PolicyTagManagerClient()
    parent = f"projects/{project_id}/locations/{location}"
    
    # Create taxonomy
    taxonomy = {
        "display_name": "Data Sensitivity Classification",
        "description": "Classification levels for data sensitivity",
        "activated_policy_types": ["FINE_GRAINED_ACCESS_CONTROL"]
    }
    
    created_taxonomy = client.create_taxonomy(
        parent=parent,
        taxonomy=taxonomy
    )
    
    print(f"Created taxonomy: {created_taxonomy.name}")
    
    # Create policy tags
    policy_tags = [
        {
            "display_name": "Public",
            "description": "Data that can be publicly shared"
        },
        {
            "display_name": "Internal",
            "description": "Data for internal use only"
        },
        {
            "display_name": "Confidential",
            "description": "Sensitive business data requiring protection"
        },
        {
            "display_name": "Restricted",
            "description": "Highly sensitive data (PII, PHI, financial)"
        }
    ]
    
    created_policy_tags = []
    for tag_config in policy_tags:
        policy_tag = client.create_policy_tag(
            parent=created_taxonomy.name,
            policy_tag=tag_config
        )
        created_policy_tags.append(policy_tag)
        print(f"Created policy tag: {policy_tag.display_name}")
    
    return created_taxonomy, created_policy_tags

# Create taxonomy
taxonomy, policy_tags = create_data_classification_taxonomy(
    project_id="my-project",
    location="us-central1"
)
```

### Apply Policy Tags to Columns

**Column-Level Security:**
```python
def apply_policy_tag_to_column(
    project_id,
    dataset_id,
    table_id,
    column_name,
    policy_tag_name
):
    """Apply policy tag to BigQuery column for fine-grained access control."""
    
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    table = client.get_table(table_ref)
    
    # Update schema with policy tag
    new_schema = []
    for field in table.schema:
        if field.name == column_name:
            # Add policy tag to field
            new_field = bigquery.SchemaField(
                name=field.name,
                field_type=field.field_type,
                mode=field.mode,
                description=field.description,
                fields=field.fields,
                policy_tags=bigquery.PolicyTagList([policy_tag_name])
            )
            new_schema.append(new_field)
        else:
            new_schema.append(field)
    
    table.schema = new_schema
    updated_table = client.update_table(table, ["schema"])
    
    print(f"Applied policy tag to {table_id}.{column_name}")
    return updated_table

# Example: Mark email column as restricted
apply_policy_tag_to_column(
    project_id="my-project",
    dataset_id="customers",
    table_id="profiles",
    column_name="email",
    policy_tag_name="projects/my-project/locations/us-central1/taxonomies/123/policyTags/456"
)
```

### Manage Policy Tag Access

**Grant Access to Policy Tags:**
```bash
# Grant access to view data with specific policy tag
gcloud data-catalog taxonomies policy-tags set-iam-policy \
    projects/my-project/locations/us-central1/taxonomies/123/policyTags/456 \
    --member=user:analyst@example.com \
    --role=roles/datacatalog.categoryFineGrainedReader

# Grant access to sensitive data
gcloud data-catalog taxonomies policy-tags set-iam-policy \
    projects/my-project/locations/us-central1/taxonomies/123/policyTags/restricted \
    --member=group:data-governance@example.com \
    --role=roles/datacatalog.categoryFineGrainedReader
```

---

## 6. Business Glossary

### Create Business Terms

**Best Practices:**
- Define common business terminology
- Link terms to data assets
- Maintain term definitions
- Implement governance approval workflow

```python
def create_business_glossary(project_id, location):
    """Create business glossary with common terms."""
    
    client = datacatalog_v1.DataCatalogClient()
    parent = f"projects/{project_id}/locations/{location}"
    
    # Create entry group for business glossary
    glossary_group = datacatalog_v1.EntryGroup(
        display_name="Business Glossary",
        description="Standard business terminology and definitions"
    )
    
    created_group = client.create_entry_group(
        parent=parent,
        entry_group_id="business-glossary",
        entry_group=glossary_group
    )
    
    # Define business terms
    terms = [
        {
            "term_id": "customer",
            "display_name": "Customer",
            "description": "An individual or organization that purchases goods or services",
            "examples": ["B2C customer, B2B client, subscriber"]
        },
        {
            "term_id": "revenue",
            "display_name": "Revenue",
            "description": "Total income generated from business operations",
            "examples": ["Product sales, subscription fees, service charges"]
        },
        {
            "term_id": "churn",
            "display_name": "Customer Churn",
            "description": "The rate at which customers stop doing business with an entity",
            "formula": "(Customers Lost / Total Customers at Start) * 100"
        },
        {
            "term_id": "lifetime_value",
            "display_name": "Customer Lifetime Value (CLV)",
            "description": "Total revenue a business can expect from a single customer account",
            "formula": "Average Purchase Value × Purchase Frequency × Customer Lifespan"
        }
    ]
    
    # Create entries for each term
    for term in terms:
        entry = datacatalog_v1.Entry(
            display_name=term["display_name"],
            description=term["description"],
            type_=datacatalog_v1.EntryType.DATA_STREAM,  # Use appropriate type
            user_specified_system="business-glossary",
            user_specified_type="business-term"
        )
        
        created_entry = client.create_entry(
            parent=created_group.name,
            entry_id=term["term_id"],
            entry=entry
        )
        
        print(f"Created business term: {term['display_name']}")
    
    return created_group

# Create glossary
create_business_glossary("my-project", "us-central1")
```

---

## 7. Data Lineage

### Track Data Lineage

**Best Practices:**
- Document data transformations
- Track upstream and downstream dependencies
- Visualize data flow
- Maintain lineage accuracy

```python
def create_lineage_events(project_id):
    """Create data lineage events."""
    
    from google.cloud import lineage_v1
    
    client = lineage_v1.LineageClient()
    parent = f"projects/{project_id}/locations/us-central1"
    
    # Example: ETL process lineage
    process = lineage_v1.Process(
        display_name="Daily Customer ETL",
        attributes={
            "schedule": "0 2 * * *",
            "owner": "data-team@example.com"
        }
    )
    
    # Create process (ETL job, Dataflow pipeline, etc.)
    created_process = client.create_process(
        parent=parent,
        process=process
    )
    
    # Create run (execution instance)
    run = lineage_v1.Run(
        display_name="ETL Run 2025-01-01",
        attributes={
            "execution_date": "2025-01-01",
            "status": "SUCCESS"
        }
    )
    
    created_run = client.create_run(
        parent=created_process.name,
        run=run
    )
    
    # Create lineage events (source → process → target)
    event = lineage_v1.LineageEvent(
        start_time={"seconds": 1704096000},  # 2025-01-01 00:00:00
        end_time={"seconds": 1704099600},    # 2025-01-01 01:00:00
        links=[
            # Source tables
            lineage_v1.EventLink(
                source=lineage_v1.EntityReference(
                    fully_qualified_name="bigquery:project.raw_data.customers"
                ),
                target=lineage_v1.EntityReference(
                    fully_qualified_name=f"process:{created_process.name}"
                )
            ),
            # Target table
            lineage_v1.EventLink(
                source=lineage_v1.EntityReference(
                    fully_qualified_name=f"process:{created_process.name}"
                ),
                target=lineage_v1.EntityReference(
                    fully_qualified_name="bigquery:project.analytics.customer_summary"
                )
            )
        ]
    )
    
    created_event = client.create_lineage_event(
        parent=created_run.name,
        lineage_event=event
    )
    
    print(f"Created lineage event: {created_event.name}")
    return created_event

def query_lineage(project_id, table_resource):
    """Query lineage for a table."""
    
    from google.cloud import lineage_v1
    
    client = lineage_v1.LineageClient()
    
    # Search for lineage links
    request = lineage_v1.SearchLinksRequest(
        parent=f"projects/{project_id}/locations/us-central1",
        target=lineage_v1.EntityReference(
            fully_qualified_name=table_resource
        )
    )
    
    links = client.search_links(request=request)
    
    print(f"Lineage for {table_resource}:")
    for link in links:
        print(f"  Source: {link.source.fully_qualified_name}")
        print(f"  Target: {link.target.fully_qualified_name}")
        print("---")
    
    return links
```

---

## 8. Integration with Other Services

### BigQuery Integration

**Automatic Metadata Sync:**
```python
def enrich_bigquery_metadata(project_id, dataset_id, table_id):
    """Enrich BigQuery table with Data Catalog metadata."""
    
    from google.cloud import bigquery
    
    bq_client = bigquery.Client(project=project_id)
    dc_client = datacatalog_v1.DataCatalogClient()
    
    # Get BigQuery table
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    table = bq_client.get_table(table_ref)
    
    # Update table description
    table.description = "Customer profiles with contact information and preferences"
    
    # Update column descriptions
    new_schema = []
    column_descriptions = {
        "customer_id": "Unique identifier for the customer",
        "email": "Customer email address (PII)",
        "first_name": "Customer first name (PII)",
        "last_name": "Customer last name (PII)",
        "created_at": "Account creation timestamp",
        "last_login": "Most recent login timestamp"
    }
    
    for field in table.schema:
        new_field = bigquery.SchemaField(
            name=field.name,
            field_type=field.field_type,
            mode=field.mode,
            description=column_descriptions.get(field.name, field.description)
        )
        new_schema.append(new_field)
    
    table.schema = new_schema
    updated_table = bq_client.update_table(table, ["description", "schema"])
    
    # Lookup Data Catalog entry
    table_resource = f"//bigquery.googleapis.com/projects/{project_id}/datasets/{dataset_id}/tables/{table_id}"
    entry = dc_client.lookup_entry(
        request=datacatalog_v1.LookupEntryRequest(linked_resource=table_resource)
    )
    
    print(f"Enriched metadata for {table_id}")
    print(f"  Data Catalog entry: {entry.name}")
    
    return updated_table, entry
```

### Pub/Sub Integration

**Catalog Pub/Sub Topics:**
```python
def catalog_pubsub_topics(project_id):
    """Add Pub/Sub topics to Data Catalog."""
    
    from google.cloud import pubsub_v1
    
    ps_client = pubsub_v1.PublisherClient()
    dc_client = datacatalog_v1.DataCatalogClient()
    
    # List Pub/Sub topics
    project_path = f"projects/{project_id}"
    topics = ps_client.list_topics(request={"project": project_path})
    
    for topic in topics:
        # Lookup or create entry
        topic_resource = f"//pubsub.googleapis.com/{topic.name}"
        
        try:
            entry = dc_client.lookup_entry(
                request=datacatalog_v1.LookupEntryRequest(
                    linked_resource=topic_resource
                )
            )
            
            # Update entry description
            entry.description = f"Pub/Sub topic for event streaming: {topic.name}"
            
            dc_client.update_entry(entry=entry)
            print(f"Updated entry for topic: {topic.name}")
            
        except Exception as e:
            print(f"Topic not in catalog: {topic.name}")
    
    print("Pub/Sub topics cataloged")
```

---

## 9. Monitoring and Audit

### Audit Logging

**Best Practices:**
- Enable Data Catalog audit logs
- Monitor metadata changes
- Track search patterns
- Review access patterns

```python
from google.cloud import logging

def query_catalog_audit_logs(project_id):
    """Query Data Catalog audit logs."""
    
    client = logging.Client(project=project_id)
    
    filter_str = """
    resource.type="datacatalog.googleapis.com/Entry"
    protoPayload.methodName=~"google.cloud.datacatalog"
    """
    
    for entry in client.list_entries(filter_=filter_str, page_size=100):
        print(f"Timestamp: {entry.timestamp}")
        print(f"User: {entry.payload.get('authenticationInfo', {}).get('principalEmail')}")
        print(f"Method: {entry.payload.get('methodName')}")
        print(f"Resource: {entry.payload.get('resourceName')}")
        print("---")

def create_catalog_alerts(project_id):
    """Create alerts for Data Catalog operations."""
    
    from google.cloud import monitoring_v3
    
    client = monitoring_v3.AlertPolicyServiceClient()
    project_name = f"projects/{project_id}"
    
    alert_policy = monitoring_v3.AlertPolicy(
        display_name="Unauthorized Data Catalog Access",
        conditions=[
            monitoring_v3.AlertPolicy.Condition(
                display_name="Failed access attempts",
                condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                    filter='resource.type="datacatalog.googleapis.com/Entry" '
                           'protoPayload.status.code!=0',
                    comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                    threshold_value=5,
                    duration={"seconds": 300},
                ),
            )
        ],
    )
    
    created_policy = client.create_alert_policy(
        name=project_name,
        alert_policy=alert_policy
    )
    
    return created_policy
```

---

## 10. Common Anti-Patterns

### ❌ Anti-Pattern 1: Incomplete Metadata
**Problem:** Missing descriptions and tags
**Solution:** Enforce metadata standards and validation

### ❌ Anti-Pattern 2: Inconsistent Tagging
**Problem:** Adhoc tagging without standards
**Solution:** Create tag templates and governance policies

### ❌ Anti-Pattern 3: No Access Control
**Problem:** All data visible to everyone
**Solution:** Implement policy tags and fine-grained access

### ❌ Anti-Pattern 4: Stale Metadata
**Problem:** Outdated descriptions and tags
**Solution:** Regular metadata reviews and updates

### ❌ Anti-Pattern 5: Ignoring Lineage
**Problem:** Unknown data dependencies
**Solution:** Document and maintain data lineage

---

## 11. Quick Reference Checklist

### Setup
- [ ] Create entry groups for organization
- [ ] Register all data assets
- [ ] Define tag templates
- [ ] Create policy taxonomies
- [ ] Set up business glossary

### Metadata Management
- [ ] Add descriptions to all assets
- [ ] Document column-level metadata
- [ ] Apply consistent tags
- [ ] Update metadata regularly
- [ ] Validate metadata completeness

### Security
- [ ] Create data classification taxonomy
- [ ] Apply policy tags to sensitive columns
- [ ] Implement fine-grained access control
- [ ] Audit access patterns
- [ ] Review permissions regularly

### Discovery
- [ ] Test search functionality
- [ ] Create search documentation
- [ ] Train users on search syntax
- [ ] Monitor search patterns
- [ ] Improve findability

### Governance
- [ ] Define metadata standards
- [ ] Implement approval workflows
- [ ] Schedule metadata reviews
- [ ] Monitor compliance
- [ ] Document policies

---

*Best Practices for Google Cloud Data Engineer Certification*
