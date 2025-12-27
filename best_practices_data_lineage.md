# Data Lineage Best Practices

## Overview
Data lineage tracks the flow of data from its origin through various transformations to its final destination. It provides visibility into data dependencies, helps with impact analysis, and supports compliance requirements. This document covers best practices for implementing and managing data lineage in GCP.

---

## 1. Data Lineage Strategy

### 1.1 Define Lineage Objectives
- **Compliance and governance**: Track data for regulatory requirements
- **Impact analysis**: Understand downstream effects of changes
- **Root cause analysis**: Trace data quality issues to source
- **Data discovery**: Help users find and understand data
- **Change management**: Plan migrations and updates safely
- **Data documentation**: Auto-generate data flow documentation

### 1.2 Determine Lineage Scope
- **System boundaries**: Which systems to include in lineage
- **Granularity level**: Table, column, or field-level tracking
- **Historical depth**: How far back to track lineage
- **Real-time vs. batch**: Frequency of lineage updates
- **Cross-platform**: Include non-GCP systems

### 1.3 Identify Stakeholders
- **Data governance teams**: Policy enforcement
- **Data engineers**: Pipeline development and maintenance
- **Data analysts**: Understanding data sources
- **Compliance officers**: Audit and regulatory requirements
- **Business users**: Data discovery and trust

---

## 2. Lineage Capture Methods

### 2.1 Automatic Lineage Capture
- **Data Catalog automatic lineage**:
  - BigQuery queries automatically tracked
  - Cloud Storage to BigQuery loads
  - BigQuery ML model training
  - Pub/Sub to BigQuery streaming

- **Dataplex automatic discovery**:
  - Scans data assets across lakes
  - Captures table and column relationships
  - Tracks transformation logic
  - Updates lineage in near real-time

- **Enable automatic lineage** in supported services:
  ```bash
  gcloud data-catalog entries update \
    --lookup-entry='bigquery:project_id.dataset.table' \
    --update-mask=schema
  ```

### 2.2 Manual Lineage Registration
- **Use Data Catalog API** for custom integrations:
  ```python
  from google.cloud import datacatalog_v1
  
  client = datacatalog_v1.LineageClient()
  
  # Create lineage event
  lineage_event = {
      "source": source_resource,
      "target": target_resource,
      "operation_type": "TRANSFORM"
  }
  ```

- **Document ETL/ELT processes**:
  - Register transformations manually
  - Link source and target datasets
  - Capture transformation logic

### 2.3 API-Based Integration
- **Data Lineage API** for custom workflows:
  - Create process objects for pipelines
  - Link runs to processes
  - Capture lineage events programmatically

- **Integrate with orchestration tools**:
  - Cloud Composer/Airflow operators
  - Dataflow pipeline metadata
  - Custom pipeline frameworks

---

## 3. Lineage Granularity

### 3.1 Table-Level Lineage
- **Track dataset relationships**:
  - Source tables feeding target tables
  - Dependencies between datasets
  - Cross-project data flows

- **Use cases**:
  - Impact analysis for table changes
  - Understanding data pipelines at high level
  - Compliance reporting

- **Best practices**:
  - Start with table-level for broad coverage
  - Easier to implement and maintain
  - Lower storage and processing costs

### 3.2 Column-Level Lineage
- **Track field transformations**:
  - Which source columns feed which target columns
  - Transformations applied to each field
  - Data type conversions

- **Use cases**:
  - Detailed impact analysis
  - PII tracking and compliance
  - Understanding complex transformations
  - Data quality root cause analysis

- **Implementation approaches**:
  - BigQuery column-level lineage (automatically captured)
  - SQL parsing for custom pipelines
  - Metadata annotations in transformations

- **Best practices**:
  - Implement for critical and sensitive data first
  - Balance detail with performance overhead
  - Use for compliance-critical columns (PII, financial)

### 3.3 Field-Level Lineage
- **Finest granularity tracking**:
  - Individual field values through transformations
  - Complex nested structure lineage
  - Array and struct field tracking

- **Use cases**:
  - Regulatory compliance requiring fine-grained tracking
  - Debugging complex data transformations
  - Advanced data quality analysis

- **Challenges**:
  - High storage and compute requirements
  - Complex to implement and maintain
  - May not be necessary for most use cases

---

## 4. Lineage in Data Catalog

### 4.1 Data Catalog Setup
- **Enable Data Catalog API**:
  ```bash
  gcloud services enable datacatalog.googleapis.com
  ```

- **Configure automatic metadata capture**:
  - BigQuery metadata automatically synced
  - Cloud Storage buckets can be registered
  - Pub/Sub topics and subscriptions supported

- **Set up entry groups** for organization:
  - Group related data assets
  - Apply consistent tagging
  - Manage access control

### 4.2 Lineage Visualization
- **Use Data Catalog UI** to view lineage:
  - Navigate to table in Data Catalog
  - View "Lineage" tab
  - Explore upstream and downstream dependencies

- **Customize lineage views**:
  - Filter by date range
  - Show/hide specific relationship types
  - Expand/collapse lineage depth

### 4.3 Searching with Lineage
- **Search by lineage relationships**:
  ```
  system=bigquery AND hasLineage:true
  ```

- **Find upstream sources**:
  ```
  ancestor:project_id.dataset.source_table
  ```

- **Find downstream consumers**:
  ```
  descendant:project_id.dataset.target_table
  ```

---

## 5. Lineage in Dataplex

### 5.1 Dataplex Lineage Features
- **Unified lineage across data lakes**:
  - Track data across multiple zones
  - Cross-project lineage visibility
  - Integration with BigQuery and Cloud Storage

- **Process-based lineage**:
  - Capture pipeline execution metadata
  - Link runs to data assets
  - Track transformation processes

### 5.2 Dataplex Configuration
- **Create data lakes and zones**:
  ```bash
  gcloud dataplex lakes create my-lake \
    --location=us-central1 \
    --project=my-project
  ```

- **Enable lineage tracking**:
  - Automatic discovery of assets
  - Integration with Data Catalog
  - Process metadata capture

### 5.3 Dataplex Lineage API
- **Create lineage events**:
  ```python
  from google.cloud import dataplex_v1
  
  client = dataplex_v1.DataLineageServiceClient()
  
  event = dataplex_v1.LineageEvent(
      source=source_entity,
      target=target_entity,
      operation=operation_type
  )
  
  response = client.create_lineage_event(parent=parent, event=event)
  ```

---

## 6. Lineage for Specific Services

### 6.1 BigQuery Lineage
- **Automatic capture**:
  - Query-based lineage automatically tracked
  - Table-to-table and column-to-column lineage
  - View and materialized view lineage
  - Scheduled query lineage

- **Access lineage information**:
  ```sql
  -- Query INFORMATION_SCHEMA for lineage
  SELECT * FROM `region-us`.INFORMATION_SCHEMA.TABLE_LINEAGE
  WHERE table_catalog = 'project_id'
    AND table_schema = 'dataset_id'
    AND table_name = 'table_name'
  ```

- **Best practices**:
  - Use descriptive query labels for tracking
  - Document complex transformations in comments
  - Regularly review lineage for accuracy

### 6.2 Dataflow Lineage
- **Capture pipeline metadata**:
  ```python
  from apache_beam.options.pipeline_options import PipelineOptions
  
  options = PipelineOptions([
      '--project=my-project',
      '--labels={"purpose":"etl","team":"data-eng"}',
      '--dataflow_service_options=enable_google_cloud_profiler'
  ])
  ```

- **Custom lineage tracking**:
  - Log source and target datasets in pipeline
  - Use Dataflow metrics for tracking
  - Integrate with Data Catalog API

### 6.3 Cloud Composer/Airflow Lineage
- **Use Airflow lineage backend**:
  ```python
  from airflow.lineage.backend import LineageBackend
  from airflow.operators.bash import BashOperator
  
  # Define inlets and outlets
  task = BashOperator(
      task_id='process_data',
      bash_command='process.sh',
      inlets={'datasets': ['/input/data']},
      outlets={'datasets': ['/output/data']}
  )
  ```

- **Integrate with Data Catalog**:
  - Use custom operators to register lineage
  - Capture DAG-level lineage
  - Track dataset dependencies

### 6.4 Dataproc Lineage
- **Capture Spark job lineage**:
  - Use Spark listeners for lineage events
  - Integrate with Data Catalog
  - Track Hive metastore lineage

- **Custom lineage capture**:
  ```python
  # In Spark job
  spark.conf.set("spark.sql.queryExecutionListeners", 
                 "com.example.LineageListener")
  ```

### 6.5 Data Fusion Lineage
- **Built-in lineage tracking**:
  - Automatic capture in pipeline designer
  - Visual lineage in UI
  - Export to Data Catalog

- **Best practices**:
  - Use descriptive plugin names
  - Document transformation logic
  - Review lineage after pipeline changes

---

## 7. Impact Analysis

### 7.1 Downstream Impact Analysis
- **Identify affected assets** before changes:
  1. Query lineage for target table
  2. Find all downstream consumers
  3. Assess impact on reports, dashboards, downstream pipelines
  4. Notify stakeholders of potential impact

- **Use Data Catalog API** for programmatic analysis:
  ```python
  def get_downstream_tables(source_table):
      # Query Data Catalog for lineage
      descendants = catalog_client.search_catalog(
          scope=scope,
          query=f'descendant:{source_table}'
      )
      return descendants
  ```

### 7.2 Upstream Impact Analysis
- **Trace data quality issues to source**:
  1. Identify problematic table/column
  2. Query upstream lineage
  3. Find transformation logic
  4. Identify root cause in source data

- **Change propagation analysis**:
  - Understand how source changes affect targets
  - Plan schema evolution
  - Coordinate updates across pipelines

### 7.3 Impact Assessment Workflow
1. **Identify change scope**: What's being modified
2. **Query lineage**: Find affected assets
3. **Classify impact**: Critical, high, medium, low
4. **Notify stakeholders**: Alert data owners and consumers
5. **Plan changes**: Schedule updates, coordinate teams
6. **Test changes**: Validate in dev/staging
7. **Monitor post-change**: Verify no unexpected impacts

---

## 8. Compliance and Governance

### 8.1 Regulatory Requirements
- **GDPR compliance**:
  - Track personal data flow
  - Document data processing activities
  - Support right to erasure (delete data lineage)
  - Provide data subject access reports

- **HIPAA compliance**:
  - Track Protected Health Information (PHI)
  - Audit access to PHI
  - Document data transformations
  - Maintain lineage for audit trails

- **SOX compliance**:
  - Track financial data lineage
  - Audit data transformations
  - Ensure data integrity
  - Maintain historical lineage

### 8.2 Data Sovereignty
- **Track cross-region data movement**:
  - Identify data residency violations
  - Document data localization
  - Support compliance audits

- **Implement controls**:
  - Alert on unauthorized cross-region transfers
  - Enforce regional data policies
  - Audit lineage for compliance

### 8.3 Audit Trail
- **Maintain lineage history**:
  - Keep historical lineage records
  - Track lineage changes over time
  - Support forensic analysis
  - Retain for compliance periods

- **Audit lineage access**:
  - Log who views lineage information
  - Track lineage modifications
  - Report on lineage usage

---

## 9. Data Discovery and Understanding

### 9.1 Self-Service Data Discovery
- **Enable users to find data sources**:
  - Search by upstream data sources
  - Find related datasets
  - Understand data provenance

- **Provide context through lineage**:
  - Show data origins
  - Display transformation logic
  - Link to documentation

### 9.2 Data Trust and Quality
- **Use lineage to build trust**:
  - Transparent data sources
  - Clear transformation logic
  - Visible data quality checks

- **Track quality through pipeline**:
  - Link quality metrics to lineage
  - Show where quality checks occur
  - Trace quality issues to source

### 9.3 Documentation Generation
- **Auto-generate data flow diagrams**:
  - Extract lineage relationships
  - Create visual representations
  - Keep documentation current

- **Generate data dictionaries**:
  - Link column lineage to descriptions
  - Show field origins and transformations
  - Maintain centralized documentation

---

## 10. Lineage Maintenance

### 10.1 Keeping Lineage Current
- **Automated updates**:
  - Enable automatic lineage capture
  - Schedule regular lineage scans
  - Update on pipeline changes

- **Manual updates**:
  - Register new data sources
  - Update transformation logic
  - Document external dependencies

### 10.2 Lineage Validation
- **Regular lineage audits**:
  - Verify accuracy of lineage relationships
  - Check for missing connections
  - Validate transformation logic

- **Automated validation**:
  ```python
  def validate_lineage(table):
      # Check if lineage exists
      # Verify expected sources present
      # Validate transformation logic documented
      pass
  ```

### 10.3 Cleanup and Archival
- **Remove obsolete lineage**:
  - Clean up after table deletion
  - Archive historical lineage
  - Maintain retention policies

- **Optimize lineage storage**:
  - Compress old lineage data
  - Move to cheaper storage tiers
  - Balance detail with cost

---

## 11. Visualization and Reporting

### 11.1 Lineage Diagrams
- **Create visual representations**:
  - Use Data Catalog UI for exploration
  - Export lineage to visualization tools
  - Generate custom diagrams

- **Diagram best practices**:
  - Show appropriate level of detail
  - Use color coding for asset types
  - Highlight critical paths
  - Include transformation descriptions

### 11.2 Lineage Reports
- **Generate regular reports**:
  - Data asset inventory
  - Lineage coverage metrics
  - Missing lineage identification
  - Compliance reports

- **Custom reporting**:
  ```python
  def generate_lineage_report(project_id):
      # Query all tables
      # Extract lineage for each
      # Generate report showing:
      # - Tables with lineage
      # - Tables without lineage
      # - Lineage depth and complexity
      pass
  ```

### 11.3 Dashboards
- **Create lineage dashboards**:
  - Lineage coverage metrics
  - Most connected datasets
  - Lineage update frequency
  - Compliance status

- **Use Looker Studio** for visualization:
  - Connect to lineage metadata in BigQuery
  - Create interactive lineage explorers
  - Share with stakeholders

---

## 12. Integration with Development Workflow

### 12.1 CI/CD Integration
- **Validate lineage in CI/CD**:
  ```yaml
  - name: Validate Lineage
    run: |
      python scripts/validate_lineage.py
      # Check that lineage is registered for new tables
      # Verify transformation documentation
  ```

- **Automated lineage registration**:
  - Register lineage during deployment
  - Update lineage on pipeline changes
  - Validate lineage completeness

### 12.2 Infrastructure as Code
- **Include lineage in Terraform**:
  ```hcl
  resource "google_data_catalog_entry" "my_table" {
    entry_group = google_data_catalog_entry_group.my_group.id
    entry_id    = "my_table"
    
    linked_resource = "//bigquery.googleapis.com/projects/${var.project}/datasets/${var.dataset}/tables/my_table"
    
    # Define lineage relationships
    schema {
      column {
        column = "customer_id"
        description = "Derived from source_db.customers.id"
      }
    }
  }
  ```

### 12.3 Code Review Integration
- **Review lineage changes**:
  - Include lineage impact in PR descriptions
  - Validate lineage updates in code reviews
  - Require lineage documentation for new pipelines

---

## 13. Performance and Scalability

### 13.1 Lineage Query Optimization
- **Optimize lineage searches**:
  - Use specific queries instead of broad searches
  - Limit lineage depth when appropriate
  - Cache frequently accessed lineage

- **Index lineage metadata**:
  - Create indexes on key fields
  - Optimize for common query patterns

### 13.2 Handling Large-Scale Lineage
- **Strategies for scale**:
  - Partition lineage by date or project
  - Summarize deep lineage chains
  - Use sampling for visualization
  - Implement pagination for large results

### 13.3 Lineage Storage Optimization
- **Optimize storage costs**:
  - Compress lineage metadata
  - Archive old lineage to Cloud Storage
  - Use lifecycle policies for retention
  - Balance detail with cost

---

## 14. Advanced Lineage Patterns

### 14.1 Cross-Platform Lineage
- **Track lineage across systems**:
  - GCP to on-premise systems
  - Multi-cloud lineage
  - External SaaS applications

- **Integration approaches**:
  - Custom connectors to Data Catalog
  - API-based lineage registration
  - Periodic sync from external systems

### 14.2 Real-Time Lineage
- **Streaming lineage updates**:
  - Capture lineage during stream processing
  - Update lineage in near real-time
  - Track evolving data flows

- **Implementation**:
  ```python
  def publish_lineage_event(source, target, operation):
      event = {
          'source': source,
          'target': target,
          'operation': operation,
          'timestamp': datetime.now()
      }
      pubsub_client.publish(lineage_topic, event)
  ```

### 14.3 ML Pipeline Lineage
- **Track ML model lineage**:
  - Training data sources
  - Feature engineering pipelines
  - Model versions and deployments
  - Prediction outputs

- **Vertex AI integration**:
  - Automatic ML lineage capture
  - Link models to training data
  - Track feature store lineage

---

## 15. Security and Access Control

### 15.1 Lineage Access Control
- **IAM roles for lineage**:
  - `roles/datacatalog.viewer`: View lineage
  - `roles/datacatalog.editor`: Edit lineage
  - `roles/datacatalog.admin`: Full control

- **Fine-grained access**:
  - Control who can view lineage for sensitive data
  - Separate read and write permissions
  - Audit lineage access

### 15.2 Sensitive Data Lineage
- **Track PII and sensitive data**:
  - Tag sensitive columns in lineage
  - Track transformations that handle PII
  - Support data anonymization tracking

- **Redact sensitive information**:
  - Don't expose data values in lineage
  - Mask sensitive field names if needed
  - Control access to sensitive lineage

### 15.3 Encryption and Privacy
- **Encrypt lineage metadata** at rest
- **Use VPC Service Controls** for lineage APIs
- **Implement data loss prevention** for lineage metadata

---

## 16. Troubleshooting

### 16.1 Common Issues

**Missing lineage:**
- Verify automatic lineage is enabled
- Check IAM permissions for lineage capture
- Ensure supported services and operations
- Manually register lineage if needed

**Incomplete lineage:**
- Review lineage granularity settings
- Check for unsupported transformations
- Validate custom lineage registration
- Verify lineage API calls succeed

**Performance issues:**
- Optimize lineage query scope
- Reduce lineage depth displayed
- Implement caching for common queries
- Use pagination for large results

**Lineage not updating:**
- Check automatic sync settings
- Verify pipeline execution completes successfully
- Review lineage API quotas and limits
- Check for errors in lineage capture logs

---

## 17. Migration and Adoption

### 17.1 Starting with Lineage
1. **Assess current state**: What lineage exists today?
2. **Identify priorities**: Which systems need lineage first?
3. **Enable automatic capture**: Start with built-in lineage
4. **Fill gaps manually**: Register custom lineage as needed
5. **Establish processes**: Define lineage maintenance workflows

### 17.2 Scaling Lineage Adoption
- **Start with critical pipelines**: Focus on high-value data flows
- **Expand gradually**: Add more systems over time
- **Train teams**: Educate on lineage tools and practices
- **Measure progress**: Track lineage coverage metrics
- **Iterate and improve**: Refine based on feedback

### 17.3 Change Management
- **Communicate value**: Show benefits of lineage to stakeholders
- **Provide tools and training**: Enable self-service lineage
- **Lead by example**: Use lineage in daily workflows
- **Celebrate successes**: Share positive impacts
- **Address concerns**: Be responsive to feedback

---

## Quick Reference Checklist

### Initial Setup
- [ ] Enable Data Catalog and Dataplex APIs
- [ ] Configure automatic lineage capture for BigQuery
- [ ] Set up entry groups for data organization
- [ ] Define lineage granularity requirements
- [ ] Establish lineage governance policies
- [ ] Configure IAM roles and permissions
- [ ] Set up lineage visualization tools
- [ ] Create documentation and training materials
- [ ] Implement lineage validation processes
- [ ] Plan for lineage maintenance

### Regular Operations
- [ ] Monitor lineage coverage and completeness
- [ ] Validate lineage accuracy periodically
- [ ] Update lineage for pipeline changes
- [ ] Review and clean up obsolete lineage
- [ ] Generate lineage reports for stakeholders
- [ ] Conduct impact analysis before major changes
- [ ] Audit lineage for compliance requirements
- [ ] Optimize lineage query performance
- [ ] Train new team members on lineage tools
- [ ] Continuously improve lineage processes

### Before Making Changes
- [ ] Query lineage for affected assets
- [ ] Identify all downstream consumers
- [ ] Assess impact on reports and dashboards
- [ ] Notify stakeholders of planned changes
- [ ] Document changes in lineage metadata
- [ ] Test changes in dev/staging environment
- [ ] Update lineage documentation
- [ ] Monitor post-change for issues
- [ ] Verify lineage accuracy after changes
- [ ] Communicate completion to stakeholders

---

## Resources

### Official Documentation
- [Data Catalog Lineage](https://cloud.google.com/data-catalog/docs/concepts/about-data-lineage)
- [Dataplex Lineage](https://cloud.google.com/dataplex/docs/data-lineage)
- [BigQuery Lineage](https://cloud.google.com/bigquery/docs/lineage-overview)
- [Data Lineage API](https://cloud.google.com/data-catalog/docs/reference/data-lineage/rest)

### Tools and Integrations
- Data Catalog for metadata management
- Dataplex for unified data governance
- BigQuery for automatic query lineage
- Looker Studio for lineage visualization
- Cloud Composer for workflow lineage

---

*Last Updated: December 26, 2025*
*Version: 1.0*
