# Dataprep by Trifacta Best Practices

## Overview
Dataprep by Trifacta is an intelligent data preparation service that provides a visual interface for cleaning, transforming, and preparing data for analysis. It uses machine learning to suggest transformations and automatically detect data patterns.

---

## 1. Getting Started with Dataprep

### Key Features
✅ **Visual interface**: No coding required (but supports formulas)
✅ **Intelligent suggestions**: ML-powered transformation recommendations
✅ **Automatic schema detection**: Infers data types and patterns
✅ **Data profiling**: Statistical analysis and quality metrics
✅ **Integration with Dataflow**: Executes transformations at scale
✅ **Sampling**: Work with subsets for faster iteration

### Dataprep vs Other Tools

| Feature | Dataprep | Wrangler (Data Fusion) | Python/Pandas |
|---------|----------|----------------------|---------------|
| Interface | Visual GUI | Visual GUI | Code |
| Learning Curve | Low | Low-Medium | High |
| Scalability | High (via Dataflow) | High | Limited |
| ML Suggestions | Yes | No | No |
| Cost | Usage-based | Instance-based | Compute-based |
| Best For | Data analysts | Data engineers | Data scientists |

---

## 2. Data Source Configuration

### Supported Sources
✅ **Cloud Storage**: CSV, JSON, Excel, Avro, Parquet
✅ **BigQuery**: Tables and query results
✅ **Cloud SQL**: MySQL, PostgreSQL
✅ **Sheets**: Google Sheets
✅ **Local Upload**: Small files for testing

### Connecting to Data Sources
```javascript
// Cloud Storage
gs://my-bucket/data/file.csv

// BigQuery
project-id.dataset_id.table_name

// Cloud SQL (requires connection setup)
cloudsql://connection-name/database/table
```

### Data Source Best Practices
✅ Use **Cloud Storage** for large files (> 100 MB)
✅ Use **BigQuery** for structured data and joins
✅ Use **sampling** for initial exploration (faster iteration)
✅ Store source data in **optimal formats** (Parquet > CSV)
✅ Use **partitioned tables** in BigQuery for better performance

---

## 3. Data Profiling & Quality Assessment

### Automatic Profiling
Dataprep automatically analyzes your data and provides:
- **Data types**: String, Integer, Float, Date, Boolean
- **Value distribution**: Histograms and frequency charts
- **Data quality**: Missing values, outliers, anomalies
- **Patterns**: Common formats, unique values
- **Statistics**: Min, max, mean, median, standard deviation

### Quality Indicators
✅ **Green bar**: High quality data
✅ **Yellow bar**: Potential issues (e.g., some nulls)
✅ **Red bar**: Significant issues (e.g., many mismatched types)

### Data Quality Checks
```
# Check for missing values
- Click column → View histogram
- Red section shows missing/null values

# Check for outliers
- View statistical summary
- Identify values outside expected range

# Check data type consistency
- Dataprep highlights type mismatches
- Review mismatched column suggestions
```

### Profiling Best Practices
✅ Review **all columns** for quality issues
✅ Check **value distributions** for anomalies
✅ Identify **high-cardinality columns** (performance impact)
✅ Look for **patterns** in unstructured data
✅ Document **data quality findings**

---

## 4. Data Transformation

### Common Transformations

**Remove Duplicates**
```
1. Select column(s) for deduplication
2. Click suggestion: "Keep unique rows"
3. Or manually: Recipe → Delete rows → Duplicate values
```

**Handle Missing Values**
```
# Fill with default value
1. Select column with nulls
2. Click suggestion: "Fill empty or missing values with [value]"

# Fill with average/median
1. Select numeric column
2. Recipe → Set → Replace missing with average

# Remove rows with nulls
1. Select column
2. Click suggestion: "Delete rows where [column] is null"
```

**Change Data Types**
```
# Convert to date
1. Select column
2. Click suggestion: "Change type to Date"
3. Specify date format

# Convert to integer
1. Select column
2. Recipe → Change type → Integer
```

**Split Column**
```
# Split by delimiter
1. Select column (e.g., "John Doe")
2. Click suggestion: "Split column by space"
3. Name new columns: first_name, last_name

# Split at position
1. Select column
2. Recipe → Split → At positions
3. Specify character positions
```

**Merge Columns**
```
1. Select multiple columns
2. Recipe → Merge → Columns
3. Specify delimiter (e.g., space, comma)
```

**Filter Rows**
```
# Filter by condition
1. Select column
2. Click suggestion: "Keep rows where [condition]"
3. Or Recipe → Filter rows → Custom condition

# Examples:
- Keep rows where age > 18
- Keep rows where status == 'active'
- Keep rows where email contains '@gmail.com'
```

**Extract Patterns**
```
# Extract email domains
1. Select email column
2. Recipe → Extract → Pattern
3. Pattern: domain after @

# Extract dates from text
1. Select text column
2. Recipe → Extract → Date
3. Dataprep suggests date patterns
```

**Join Data**
```
1. Import second dataset
2. Recipe → Join
3. Select join type: Inner, Left, Right, Full Outer
4. Select join keys
5. Choose columns to include from each dataset
```

### Formula Language
```javascript
// String functions
UPPER(column)                    // Convert to uppercase
LOWER(column)                    // Convert to lowercase
SUBSTRING(column, start, length) // Extract substring
CONCAT(col1, ' ', col2)         // Concatenate strings
TRIM(column)                     // Remove whitespace

// Numeric functions
ROUND(column, 2)                 // Round to 2 decimals
ABS(column)                      // Absolute value
column1 + column2                // Add columns
column1 / column2                // Divide columns

// Date functions
DATEFORMAT(column, 'yyyy-MM-dd') // Format date
DATEDIF(date1, date2, 'days')   // Date difference
NOW()                            // Current timestamp

// Conditional logic
IF(column > 100, 'High', 'Low') // If-then-else
CASE(
  column == 'A', 'Category 1',
  column == 'B', 'Category 2',
  'Other'
)

// Aggregations (in window functions)
AVERAGE(column)
SUM(column)
COUNT(column)
MAX(column)
MIN(column)
```

---

## 5. Sampling Strategies

### Sampling Types

**Automatic Sampling** (Default)
- Dataprep automatically samples data for faster interaction
- Smart sampling ensures representative data
- Default: 10,000 rows

**Random Sampling**
```
Settings → Sampling → Random
- Specify sample size or percentage
- Good for: Exploratory analysis
```

**Stratified Sampling**
```
Settings → Sampling → Stratified
- Sample proportionally from groups
- Good for: Ensuring representation of categories
```

**Cluster Sampling**
```
Settings → Sampling → Cluster
- Sample entire clusters/groups
- Good for: Hierarchical data
```

**Filter-Based Sampling**
```
Settings → Sampling → Filter
- Apply custom filter condition
- Good for: Testing specific subsets
```

### Sampling Best Practices
✅ Use **automatic sampling** for initial exploration
✅ Use **stratified sampling** for categorical data
✅ Use **full scan** only when ready to run at scale
✅ Test recipe on **small sample** before full run
✅ Verify transformations on **different samples**

---

## 6. Recipe Development

### Recipe Best Practices
✅ Start with **data profiling** to understand data
✅ Apply **transformations incrementally** (easier to debug)
✅ Use **descriptive step names** for complex recipes
✅ **Test frequently** on sample data
✅ **Document complex logic** in step descriptions
✅ **Reuse recipes** for similar datasets
✅ Use **parameters** for dynamic values

### Recipe Organization
```
1. Data Quality
   - Remove duplicates
   - Handle missing values
   - Fix data types

2. Data Cleaning
   - Trim whitespace
   - Standardize formats
   - Remove invalid characters

3. Data Transformation
   - Split/merge columns
   - Extract patterns
   - Calculate new fields

4. Data Enrichment
   - Join with reference data
   - Add derived columns
   - Categorize values

5. Final Formatting
   - Select required columns
   - Rename columns
   - Arrange column order
```

### Recipe Sharing & Reuse
✅ **Export recipes** for version control
✅ **Share recipes** with team members
✅ **Import recipes** to reuse on similar datasets
✅ Create **recipe templates** for common patterns

---

## 7. Output Configuration

### Output Destinations

**Cloud Storage**
```
Output Settings:
- Location: gs://my-bucket/output/
- Format: CSV, JSON, Avro, Parquet
- Compression: None, GZIP, SNAPPY
- Single/Multiple files: Based on size
```

**BigQuery**
```
Output Settings:
- Project: my-project
- Dataset: my_dataset
- Table: output_table
- Write mode: Append, Truncate, Create
- Partitioning: Optional (date column)
```

### Output Best Practices
✅ Use **Parquet** for Cloud Storage (compressed, columnar)
✅ Use **BigQuery** for structured analytics data
✅ Enable **compression** to reduce storage costs
✅ Use **partitioning** in BigQuery for better query performance
✅ Choose **append mode** for incremental updates
✅ Use **truncate mode** for full refreshes

---

## 8. Job Execution

### Execution via Dataflow
Dataprep uses **Cloud Dataflow** to execute transformations at scale:

```
Flow → Run Job
- Dataprep converts recipe to Dataflow pipeline
- Dataflow executes transformations in parallel
- Results written to output destination
```

### Job Configuration
```
Job Settings:
- Runner: Cloud Dataflow (default)
- Region: us-central1
- Max workers: 10
- Machine type: n1-standard-1
- Disk size: 30 GB
```

### Execution Best Practices
✅ Start with **small dataset** to validate recipe
✅ Use **appropriate machine types** based on data size
✅ Set **max workers** to control costs
✅ Monitor **job progress** in Dataflow console
✅ Review **job metrics** for optimization opportunities
✅ Schedule jobs during **off-peak hours** for cost savings

---

## 9. Scheduling & Automation

### Scheduled Runs
```
Flow → Schedule
- Frequency: Daily, Weekly, Monthly, Custom
- Time: Specify execution time
- Timezone: Set appropriate timezone
```

### Parameterization
```
# Use parameters for dynamic inputs
Input: gs://bucket/data-${date}.csv
Output: project.dataset.table_${date}

# Set parameters at schedule time
date: $CURRENT_DATE
```

### Integration with Cloud Composer
```python
# Airflow DAG for Dataprep job
from airflow.providers.google.cloud.operators.dataprep import \
    DataprepRunFlowOperator

run_dataprep = DataprepRunFlowOperator(
    task_id='run_dataprep_flow',
    flow_id='12345',
    region='us-central1'
)
```

### Scheduling Best Practices
✅ Schedule jobs based on **data availability**
✅ Use **appropriate frequency** (don't over-schedule)
✅ Set **retry policies** for failed jobs
✅ Use **Cloud Composer** for complex dependencies
✅ Monitor **scheduled job success rates**

---

## 10. Performance Optimization

### Data Preparation Performance
✅ **Sample data** during development (faster iteration)
✅ **Filter early** to reduce data volume
✅ Avoid **complex nested transformations**
✅ Use **native Dataprep functions** instead of custom formulas
✅ **Partition large files** before processing

### Dataflow Execution Performance
✅ Use **appropriate machine types**
✅ Increase **max workers** for large datasets
✅ Use **larger disk sizes** for shuffle-heavy operations
✅ **Optimize transformations** (remove unnecessary steps)
✅ Use **columnar formats** (Parquet, Avro) for better performance

### Cost-Performance Trade-offs
```
Fast + Expensive:
- n1-standard-4 machines
- Max workers: 50
- Execution time: 10 minutes

Slow + Cheaper:
- n1-standard-1 machines
- Max workers: 10
- Execution time: 30 minutes
```

---

## 11. Monitoring & Troubleshooting

### Job Monitoring
✅ Monitor in **Dataprep console** (job status)
✅ View details in **Dataflow console** (detailed metrics)
✅ Check **Cloud Logging** for errors

### Common Issues & Solutions

**Job Fails Immediately**
- Check source data availability
- Verify permissions (service account)
- Review recipe for invalid transformations

**Job Takes Too Long**
- Increase worker count
- Use larger machine types
- Optimize recipe (remove unnecessary steps)
- Check for data skew

**Out of Memory Errors**
- Increase machine memory (highmem types)
- Reduce sample size
- Break into smaller jobs

**Incorrect Output**
- Test recipe on sample data
- Verify transformation logic
- Check for data type mismatches
- Review join conditions

### Monitoring Best Practices
✅ Set up **alerts** for job failures
✅ Monitor **job duration** trends
✅ Track **cost per job** execution
✅ Review **error logs** regularly
✅ Use **Cloud Monitoring** dashboards

---

## 12. Security & Access Control

### IAM Permissions
✅ Use **dedicated service accounts** for Dataprep
✅ Grant **minimum required permissions**

**Required Roles**
- `roles/dataprep.user`: Run Dataprep flows
- `roles/storage.objectViewer`: Read from Cloud Storage
- `roles/storage.objectCreator`: Write to Cloud Storage
- `roles/bigquery.dataEditor`: BigQuery I/O

### Data Security
✅ Use **VPC peering** for private connectivity
✅ **Encrypt data** in transit and at rest (automatic)
✅ Use **CMEK** for customer-managed encryption
✅ Don't store **sensitive data** in recipe parameters
✅ Use **Secret Manager** for credentials

### Access Control
✅ Control **flow sharing** permissions
✅ Separate **dev/prod environments**
✅ Implement **least privilege** access
✅ Regular **access reviews**

---

## 13. Cost Optimization

### Usage-Based Pricing
Dataprep charges based on:
- **Dataflow execution**: Compute resources used
- **Data processed**: Volume of data transformed

### Cost Reduction Strategies
✅ **Filter data early** to reduce processing volume
✅ Use **smaller machine types** when possible
✅ Optimize **recipe efficiency** (fewer transformations)
✅ Use **sampling** during development (not full scans)
✅ Schedule jobs during **off-peak hours**
✅ Delete **unused flows** and datasets
✅ Monitor **cost per flow** execution

### Budget Controls
✅ Set **max workers** limit
✅ Use **committed use discounts** for Dataflow
✅ Monitor costs with **Cloud Billing**
✅ Set up **budget alerts**

---

## 14. Common Use Cases

### Data Cleaning
```
Use Case: Clean customer data
Steps:
1. Remove duplicates
2. Standardize phone numbers
3. Validate email formats
4. Fill missing addresses
5. Categorize by region
```

### Data Transformation
```
Use Case: Transform logs for analysis
Steps:
1. Parse JSON log entries
2. Extract timestamp and level
3. Filter by error level
4. Aggregate by hour
5. Output to BigQuery
```

### Data Enrichment
```
Use Case: Enrich sales data
Steps:
1. Load sales transactions
2. Join with product catalog
3. Join with customer info
4. Calculate total revenue
5. Add date dimensions
```

### Data Migration
```
Use Case: Migrate CSV to BigQuery
Steps:
1. Load CSV from Cloud Storage
2. Detect and fix schema issues
3. Transform to target schema
4. Validate data quality
5. Load to BigQuery with partitioning
```

---

## 15. Common Anti-Patterns to Avoid

❌ **Not using sampling**: Slow development iteration
❌ **Complex nested transformations**: Hard to debug, slow execution
❌ **Not handling nulls**: Unexpected results
❌ **Not testing on full dataset**: Issues only appear at scale
❌ **Hardcoded values**: Not reusable for different datasets
❌ **Not monitoring job costs**: Budget surprises
❌ **Ignoring data profiling**: Miss data quality issues
❌ **Not using parameters**: Inflexible schedules
❌ **Over-processing data**: Applying unnecessary transformations
❌ **Not versioning recipes**: Cannot rollback changes
❌ **Using CSV for large files**: Slow parsing, no schema
❌ **Not documenting complex logic**: Hard to maintain

---

## Quick Reference Checklist

- [ ] Start with data profiling to understand data
- [ ] Use sampling during recipe development
- [ ] Apply transformations incrementally
- [ ] Test recipe on small sample before full run
- [ ] Handle missing values and nulls
- [ ] Use appropriate data types
- [ ] Filter data early to reduce volume
- [ ] Use Parquet format for output
- [ ] Configure appropriate Dataflow resources
- [ ] Set up job scheduling if needed
- [ ] Monitor job execution and costs
- [ ] Use parameters for reusable flows
- [ ] Document complex transformation logic
- [ ] Version control recipe exports
- [ ] Set up alerts for job failures
- [ ] Use dedicated service accounts
- [ ] Optimize recipe for performance
- [ ] Review data quality indicators

---

## Additional Resources

- [Dataprep Documentation](https://cloud.google.com/dataprep/docs)
- [Transformation Functions Reference](https://cloud.google.com/dataprep/docs/html/Functions-Page_57344795)
- [Recipe Development Guide](https://cloud.google.com/dataprep/docs/html/Recipe-Basics_145278985)
- [Integration with Dataflow](https://cloud.google.com/dataprep/docs/concepts/dataflow-integration)
- [Pricing Information](https://cloud.google.com/dataprep/pricing)

---

*Last Updated: December 25, 2025*
