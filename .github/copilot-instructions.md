# GitHub Copilot Instructions - GCP Data Engineering Best Practices

## Repository Overview
This is a comprehensive documentation repository for **Google Cloud Platform (GCP) Data Engineering** certification preparation. It contains best practices documentation for 30+ GCP data engineering services.

**Repository Structure:**
- `gcp_data_engineering_services.md` - Master index of all GCP data engineering services with brief descriptions
- `best_practices_*.md` - Individual best practices documents (one per service)

## Architecture & Content Structure

### Document Organization Pattern
Each best practices document follows a **standardized comprehensive structure**:

1. **Overview** - Service description and scope
2. **Numbered Sections (10-20)** - Core best practices organized by topic
3. **Anti-Patterns Section** - Common mistakes to avoid (marked with ❌)
4. **Quick Reference Checklist** - Actionable items for implementation
5. **Resources Section** - Official documentation links and tools
6. **Metadata Footer** - Last updated date and version

**Example from `best_practices_bigquery.md`:**
```markdown
## 1. Table Design & Schema Best Practices
### Partitioning
✅ Use time-series data partitioning
❌ Don't partition small tables (<1GB)

## 15. Quick Reference Checklist
- [ ] Enable clustering on filter columns
```

### Service Categories
Documents are organized by these GCP service categories:
- **Storage**: Cloud Storage, Cloud SQL, Cloud Spanner, Bigtable, Firestore, Memorystore
- **Processing**: BigQuery, Dataflow, Dataproc, Data Fusion, Dataprep
- **Ingestion**: Pub/Sub, Datastream, Transfer Service, BigQuery Data Transfer
- **Orchestration**: Cloud Composer (Airflow)
- **Governance**: Dataplex, Data Catalog, DLP, Data Lineage
- **ML/AI**: Vertex AI, BigQuery ML
- **Visualization**: Looker, Looker Studio
- **Transformation**: Dataform, Dataprep

## Content Creation Guidelines

### When Generating New Best Practices Documents

1. **Always follow the established structure** from existing documents:
   - Start with `# [Service Name] Best Practices`
   - Include `## Overview` section explaining the service
   - Create 10-20 numbered main sections covering different aspects
   - Include practical code examples in SQL, Python, or bash
   - Add anti-patterns section with ❌ markers
   - Include Quick Reference Checklist
   - End with Resources section and metadata footer

2. **Use consistent formatting conventions**:
   - ✅ for recommended practices
   - ❌ for anti-patterns
   - Code blocks with language identifiers (```sql, ```python, ```bash, ```json, ```yaml)
   - Nested subsections using ### and #### headings
   - Bullet points for lists
   - `backticks` for service names, parameters, and technical terms

3. **Content depth expectations**:
   - Documents should be **comprehensive** (300-800 lines typically)
   - Include real-world examples and use cases
   - Cover: design patterns, performance optimization, cost optimization, security, monitoring, troubleshooting
   - Provide specific parameter values and configuration examples
   - Include integration patterns with other GCP services

4. **Code example patterns**:
   ```sql
   -- BigQuery: Show complete DDL with options
   CREATE TABLE dataset.table (columns)
   PARTITION BY DATE(timestamp_col)
   CLUSTER BY (col1, col2)
   OPTIONS(partition_expiration_days=365);
   ```
   
   ```python
   # Dataflow: Show full pipeline configuration
   options = PipelineOptions([
       '--project=my-project',
       '--region=us-central1',
       '--temp_location=gs://bucket/temp'
   ])
   ```

### Document Naming Convention
- Use snake_case: `best_practices_[service_name].md`
- Service names match the official GCP naming (lowercase, underscores for spaces)
- Examples: `best_practices_cloud_storage.md`, `best_practices_bigquery_ml.md`

### Cross-Service Integration Patterns
When documenting a service, always consider integration with:
- **BigQuery** (most services integrate with it for analytics)
- **Cloud Storage** (common data lake foundation)
- **Pub/Sub** (event-driven architectures)
- **Cloud Composer** (orchestration)
- **Dataflow** (data processing pipelines)

Example: In `best_practices_pubsub.md`, include sections on:
- Pub/Sub → Dataflow integration
- Pub/Sub → BigQuery streaming
- Pub/Sub → Cloud Functions triggers

## Common Data Engineering Patterns in This Repository

### Pattern 1: Batch Processing Pipeline
```
Cloud Storage → Dataproc/Dataflow → BigQuery → Looker Studio
```

### Pattern 2: Streaming Pipeline
```
Pub/Sub → Dataflow → BigQuery → Real-time Dashboard
```

### Pattern 3: Data Lake Architecture
```
Cloud Storage (Raw) → Processing → Cloud Storage (Curated) → BigQuery
```

### Pattern 4: CDC (Change Data Capture)
```
Source DB → Datastream → BigQuery/Cloud Storage
```

## Important Conventions

### Best Practices Sections to Always Include
1. **Performance Optimization** - Query/pipeline tuning, resource sizing
2. **Cost Optimization** - Pricing considerations, cost reduction strategies
3. **Security Best Practices** - IAM, encryption, VPC, compliance
4. **Monitoring & Observability** - Metrics, logging, alerting
5. **Troubleshooting Guide** - Common issues and solutions
6. **Anti-Patterns** - What NOT to do

### Terminology Consistency
- Use "BigQuery" not "BQ" or "Big Query"
- Use "Cloud Storage" not "GCS" (except in URIs like `gs://`)
- Use "Pub/Sub" not "PubSub" or "Pub-Sub"
- Use "Dataflow" not "Cloud Dataflow" (unless referring to the service formally)
- Use "Cloud Composer" not "Composer" alone

### Version Control Best Practices
- Update the `*Last Updated: [Date]*` footer when modifying documents
- Increment version numbers for significant changes: `*Version: 1.1*`
- Keep `gcp_data_engineering_services.md` as the single source of truth for the service index

## Workflow for User Requests

### When asked to "generate best practices for [service]":
1. Check if document already exists in directory listing
2. Reference `gcp_data_engineering_services.md` for service description
3. Generate comprehensive 300-800 line document following established structure
4. Include 15-20 major sections with subsections
5. Add practical code examples (3-5 per major section)
6. Include Quick Reference Checklist at end
7. Add proper metadata footer with current date

### When asked to generate multiple service documents:
- Create all requested documents in parallel when possible
- Maintain consistency across all generated documents
- Use the same section numbering and structural patterns

### When updating existing documents:
- Read the entire existing document first
- Preserve the established structure and formatting
- Maintain consistency with other documents in the repository
- Update the version number and date in footer

## Key Files to Reference

- **`gcp_data_engineering_services.md`** - Master service index, reference for service descriptions
- **`best_practices_bigquery.md`** - Excellent example of comprehensive structure (814 lines)
- **`best_practices_dataflow_streaming.md`** - Shows specialized service focus
- **`best_practices_dataplex_data_quality.md`** - Example of governance service documentation

## Quality Standards

### Documentation Quality Checklist
- [ ] Document is 300+ lines with comprehensive coverage
- [ ] Follows standardized structure from existing documents
- [ ] Includes 10+ practical code examples with proper syntax highlighting
- [ ] Contains Quick Reference Checklist section
- [ ] Has anti-patterns section with ❌ markers
- [ ] Includes official documentation links in Resources section
- [ ] Footer has current date and version number
- [ ] Uses consistent terminology matching GCP documentation
- [ ] Covers all key aspects: design, performance, cost, security, monitoring, troubleshooting

---

**For AI Agents:** When working in this repository, prioritize consistency with existing documentation patterns. Always check multiple existing best practices documents to understand the expected depth, structure, and formatting before creating new content.
