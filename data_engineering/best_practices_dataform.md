# Dataform Best Practices

## Overview
Dataform is a service for managing SQL-based data transformation workflows in BigQuery. It provides version control, dependency management, testing, and documentation for your data pipelines. This document covers best practices for building robust and maintainable data transformation pipelines with Dataform.

---

## 1. Project Structure and Organization

### 1.1 Directory Structure
- **Recommended structure**:
  ```
  dataform-project/
  ├── definitions/
  │   ├── sources/
  │   │   ├── raw_orders.sqlx
  │   │   └── raw_customers.sqlx
  │   ├── staging/
  │   │   ├── stg_orders.sqlx
  │   │   └── stg_customers.sqlx
  │   ├── intermediate/
  │   │   ├── int_customer_orders.sqlx
  │   │   └── int_order_metrics.sqlx
  │   ├── marts/
  │   │   ├── fct_orders.sqlx
  │   │   └── dim_customers.sqlx
  │   └── tests/
  │       ├── schema_tests.js
  │       └── data_quality_tests.js
  ├── includes/
  │   ├── constants.js
  │   └── macros.js
  ├── dataform.json
  └── package.json
  ```

### 1.2 Naming Conventions
- **Use consistent prefixes**:
  - `raw_`: Raw source data (sources)
  - `stg_`: Staging tables (cleaned, typed, renamed)
  - `int_`: Intermediate models (business logic, joins)
  - `fct_`: Fact tables (events, transactions)
  - `dim_`: Dimension tables (entities, attributes)
  - `rpt_`: Reports or aggregated views

- **Use descriptive names**:
  ```sql
  -- Good
  stg_orders
  dim_customers
  fct_daily_sales
  
  -- Bad
  table1
  temp
  final
  ```

### 1.3 Layer Separation
- **Organize by data layer**:
  1. **Source**: Declare raw data sources
  2. **Staging**: Clean and standardize
  3. **Intermediate**: Apply business logic
  4. **Marts**: Final consumption layer

---

## 2. Source Declaration

### 2.1 Declaring Sources
- **Always declare sources explicitly**:
  ```javascript
  // definitions/sources/raw_orders.sqlx
  config {
    type: "declaration",
    database: "raw_data",
    schema: "ecommerce",
    name: "orders"
  }
  ```

- **Benefits**:
  - Documents data lineage
  - Enables dependency tracking
  - Supports impact analysis
  - Provides clear data catalog

### 2.2 Source Documentation
- **Document all sources**:
  ```javascript
  config {
    type: "declaration",
    database: "raw_data",
    schema: "ecommerce",
    name: "orders",
    description: "Raw order data from production database, synced hourly",
    columns: {
      order_id: "Unique identifier for each order",
      customer_id: "Reference to customer table",
      order_date: "Timestamp when order was placed",
      total_amount: "Total order amount in USD"
    }
  }
  ```

### 2.3 Source Assertions
- **Validate source data**:
  ```javascript
  config {
    type: "declaration",
    database: "raw_data",
    schema: "ecommerce",
    name: "orders",
    assertions: {
      uniqueKey: ["order_id"],
      nonNull: ["order_id", "customer_id", "order_date"]
    }
  }
  ```

---

## 3. Transformation Best Practices

### 3.1 Staging Layer
- **Purpose**: Clean, standardize, and type cast raw data
- **Best practices**:
  - One staging model per source
  - Minimal transformations
  - Consistent naming and typing
  - No joins or aggregations

- **Example staging model**:
  ```sql
  -- definitions/staging/stg_orders.sqlx
  config {
    type: "table",
    schema: "staging",
    description: "Cleaned and standardized orders from raw source"
  }
  
  SELECT
      -- Cast and rename for consistency
      CAST(order_id AS STRING) AS order_id,
      CAST(customer_id AS STRING) AS customer_id,
      
      -- Parse timestamps consistently
      PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', order_date) AS order_timestamp,
      
      -- Standardize currency
      CAST(total_amount AS NUMERIC) AS order_amount_usd,
      
      -- Standardize status values
      LOWER(TRIM(status)) AS order_status,
      
      -- Add metadata
      CURRENT_TIMESTAMP() AS loaded_at
  
  FROM ${ref("raw_orders")}
  
  -- Filter out test data
  WHERE customer_id IS NOT NULL
    AND order_id NOT LIKE 'TEST%'
  ```

### 3.2 Intermediate Layer
- **Purpose**: Apply business logic, create reusable components
- **Best practices**:
  - Complex joins and unions
  - Business logic and calculations
  - Reusable components
  - Modular transformations

- **Example intermediate model**:
  ```sql
  -- definitions/intermediate/int_customer_orders.sqlx
  config {
    type: "table",
    schema: "intermediate",
    description: "Customer orders with enriched customer information"
  }
  
  WITH orders AS (
      SELECT * FROM ${ref("stg_orders")}
  ),
  
  customers AS (
      SELECT * FROM ${ref("stg_customers")}
  ),
  
  order_metrics AS (
      SELECT
          customer_id,
          COUNT(*) AS total_orders,
          SUM(order_amount_usd) AS lifetime_value,
          MIN(order_timestamp) AS first_order_date,
          MAX(order_timestamp) AS last_order_date
      FROM orders
      GROUP BY customer_id
  )
  
  SELECT
      o.order_id,
      o.customer_id,
      o.order_timestamp,
      o.order_amount_usd,
      o.order_status,
      
      c.customer_name,
      c.customer_email,
      c.customer_segment,
      
      m.total_orders,
      m.lifetime_value,
      m.first_order_date,
      m.last_order_date,
      
      -- Business logic
      CASE
          WHEN m.total_orders = 1 THEN 'New'
          WHEN m.total_orders BETWEEN 2 AND 5 THEN 'Returning'
          ELSE 'Loyal'
      END AS customer_type
  
  FROM orders o
  LEFT JOIN customers c USING (customer_id)
  LEFT JOIN order_metrics m USING (customer_id)
  ```

### 3.3 Marts Layer
- **Purpose**: Final models for consumption
- **Best practices**:
  - Wide, denormalized tables
  - Business-friendly naming
  - Comprehensive documentation
  - Optimized for query performance

- **Example fact table**:
  ```sql
  -- definitions/marts/fct_orders.sqlx
  config {
    type: "table",
    schema: "marts",
    description: "Order fact table for analytics and reporting",
    bigquery: {
      partitionBy: "DATE(order_timestamp)",
      clusterBy: ["customer_id", "order_status"]
    },
    assertions: {
      uniqueKey: ["order_id"],
      nonNull: ["order_id", "customer_id", "order_timestamp"]
    }
  }
  
  SELECT
      order_id,
      customer_id,
      order_timestamp,
      DATE(order_timestamp) AS order_date,
      order_amount_usd,
      order_status,
      customer_name,
      customer_email,
      customer_segment,
      customer_type,
      total_orders AS customer_total_orders,
      lifetime_value AS customer_lifetime_value
  
  FROM ${ref("int_customer_orders")}
  ```

---

## 4. Dependency Management

### 4.1 Using ref() Function
- **Always use ref() for dependencies**:
  ```sql
  -- Good
  FROM ${ref("stg_orders")}
  
  -- Bad
  FROM `project.schema.stg_orders`
  ```

- **Benefits**:
  - Automatic dependency resolution
  - Correct execution order
  - Lineage tracking
  - Easy refactoring

### 4.2 Dependency Graph
- **Design clear dependency flows**:
  - Sources → Staging → Intermediate → Marts
  - Avoid circular dependencies
  - Minimize cross-layer dependencies
  - Keep dependencies explicit

### 4.3 Modular Design
- **Create reusable components**:
  ```sql
  -- definitions/intermediate/int_order_metrics.sqlx
  config {
    type: "table",
    schema: "intermediate"
  }
  
  SELECT
      customer_id,
      COUNT(*) AS order_count,
      SUM(order_amount_usd) AS total_spent,
      AVG(order_amount_usd) AS avg_order_value
  FROM ${ref("stg_orders")}
  GROUP BY customer_id
  ```

---

## 5. Incremental Models

### 5.1 When to Use Incremental
- **Use incremental for**:
  - Large fact tables (millions+ rows)
  - Event/log data
  - Append-only data
  - Time-series data

- **Don't use incremental for**:
  - Small dimension tables
  - Full refresh snapshots
  - Complex aggregations across all data

### 5.2 Incremental Configuration
- **Basic incremental model**:
  ```sql
  -- definitions/marts/fct_events.sqlx
  config {
    type: "incremental",
    schema: "marts",
    bigquery: {
      partitionBy: "DATE(event_timestamp)",
      updatePartitionFilter: "DATE(event_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY)"
    }
  }
  
  SELECT
      event_id,
      user_id,
      event_type,
      event_timestamp,
      event_properties
  
  FROM ${ref("stg_events")}
  
  WHERE TRUE
      -- Only process new data in incremental runs
      ${ when(incremental(), `AND event_timestamp > (SELECT MAX(event_timestamp) FROM ${self()})`) }
  ```

### 5.3 Merge Strategy
- **Use merge for updates**:
  ```sql
  config {
    type: "incremental",
    schema: "marts",
    uniqueKey: ["order_id"],
    bigquery: {
      updatePartitionFilter: "DATE(updated_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)"
    }
  }
  
  SELECT
      order_id,
      customer_id,
      order_status,
      order_amount,
      updated_at
  
  FROM ${ref("stg_orders")}
  
  WHERE TRUE
      ${ when(incremental(), `AND updated_at > (SELECT MAX(updated_at) FROM ${self()})`) }
  ```

---

## 6. Testing and Data Quality

### 6.1 Assertions
- **Built-in assertions**:
  ```javascript
  config {
    type: "table",
    schema: "marts",
    assertions: {
      uniqueKey: ["order_id"],
      nonNull: ["order_id", "customer_id", "order_timestamp"],
      rowConditions: [
          "order_amount_usd >= 0",
          "order_status IN ('pending', 'completed', 'cancelled')"
      ]
    }
  }
  ```

### 6.2 Custom Tests
- **Create custom test queries**:
  ```javascript
  // definitions/tests/order_tests.sqlx
  config {
    type: "test",
    name: "assert_positive_order_amounts"
  }
  
  SELECT
      order_id,
      order_amount_usd
  FROM ${ref("fct_orders")}
  WHERE order_amount_usd < 0
  ```

### 6.3 Data Quality Checks
- **Implement comprehensive checks**:
  ```javascript
  // definitions/tests/data_quality_tests.sqlx
  
  // Test for duplicates
  config { type: "test", name: "check_order_duplicates" }
  SELECT
      order_id,
      COUNT(*) AS duplicate_count
  FROM ${ref("fct_orders")}
  GROUP BY order_id
  HAVING COUNT(*) > 1
  
  ---
  
  // Test for referential integrity
  config { type: "test", name: "check_customer_references" }
  SELECT
      o.order_id,
      o.customer_id
  FROM ${ref("fct_orders")} o
  LEFT JOIN ${ref("dim_customers")} c USING (customer_id)
  WHERE c.customer_id IS NULL
  
  ---
  
  // Test for data freshness
  config { type: "test", name: "check_data_freshness" }
  SELECT
      'orders' AS table_name,
      MAX(order_timestamp) AS latest_timestamp,
      CURRENT_TIMESTAMP() AS check_timestamp,
      TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(order_timestamp), HOUR) AS hours_old
  FROM ${ref("fct_orders")}
  HAVING hours_old > 24
  ```

---

## 7. Documentation

### 7.1 Model Documentation
- **Document all models**:
  ```javascript
  config {
    type: "table",
    schema: "marts",
    description: `
      ## Order Fact Table
      
      This table contains all orders with denormalized customer information.
      Updated daily at 2 AM UTC.
      
      ### Usage
      - Primary table for order analytics
      - Use for customer segmentation analysis
      - Source for daily order reports
      
      ### SLA
      - Freshness: < 24 hours
      - Completeness: 99.9%
    `,
    columns: {
      order_id: "Unique identifier for each order (primary key)",
      customer_id: "Foreign key to dim_customers",
      order_timestamp: "When the order was placed (UTC)",
      order_amount_usd: "Total order amount in USD (always positive)"
    }
  }
  ```

### 7.2 Code Comments
- **Add inline comments for complex logic**:
  ```sql
  SELECT
      order_id,
      customer_id,
      
      -- Calculate customer lifetime value at time of order
      -- Uses window function to avoid self-join
      SUM(order_amount_usd) OVER (
          PARTITION BY customer_id
          ORDER BY order_timestamp
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS ltv_at_order,
      
      -- Categorize orders by size
      CASE
          WHEN order_amount_usd < 50 THEN 'Small'
          WHEN order_amount_usd < 200 THEN 'Medium'
          WHEN order_amount_usd < 500 THEN 'Large'
          ELSE 'Extra Large'
      END AS order_size_category
  
  FROM ${ref("stg_orders")}
  ```

### 7.3 README and Guides
- **Create project documentation**:
  ```markdown
  # Dataform Project: E-commerce Analytics
  
  ## Overview
  This project transforms raw e-commerce data into analytics-ready tables.
  
  ## Data Flow
  1. Raw data ingested from production DB
  2. Staging layer cleans and standardizes
  3. Intermediate layer applies business logic
  4. Marts layer provides final analytics tables
  
  ## Key Tables
  - `fct_orders`: Order fact table
  - `dim_customers`: Customer dimension
  - `fct_daily_sales`: Daily aggregated sales
  
  ## Running the Project
  ```bash
  # Full refresh
  dataform run
  
  # Run specific model
  dataform run --tags=orders
  
  # Run with assertions
  dataform run --include-deps --run-tests
  ```
  
  ## Schedules
  - Staging: Hourly
  - Marts: Daily at 2 AM UTC
  ```

---

## 8. Configuration Management

### 8.1 Environment Configuration
- **Use workflow_settings.yaml**:
  ```yaml
  # workflow_settings.yaml
  defaultProject: production-project
  defaultLocation: US
  defaultDataset: analytics
  
  dataformCoreVersion: 2.9.0
  
  vars:
    environment: production
    days_to_lookback: 7
  ```

### 8.2 Using Variables
- **Define constants**:
  ```javascript
  // includes/constants.js
  const PROJECT_ID = dataform.projectConfig.vars.environment === 'production' 
      ? 'prod-project' 
      : 'dev-project';
  
  const LOOKBACK_DAYS = dataform.projectConfig.vars.days_to_lookback || 7;
  
  module.exports = { PROJECT_ID, LOOKBACK_DAYS };
  ```

- **Use in models**:
  ```sql
  ${const { LOOKBACK_DAYS } = require("../includes/constants");}
  
  SELECT *
  FROM ${ref("stg_orders")}
  WHERE order_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL ${LOOKBACK_DAYS} DAY)
  ```

### 8.3 Macros and Functions
- **Create reusable SQL**:
  ```javascript
  // includes/macros.js
  function generateDateSpine(startDate, endDate) {
      return `
          SELECT date
          FROM UNNEST(
              GENERATE_DATE_ARRAY('${startDate}', '${endDate}', INTERVAL 1 DAY)
          ) AS date
      `;
  }
  
  function surrogatKey(columns) {
      return `FARM_FINGERPRINT(CONCAT(${columns.join(', ')}))`;
  }
  
  module.exports = { generateDateSpine, surrogatKey };
  ```

---

## 9. Performance Optimization

### 9.1 Table Configuration
- **Optimize BigQuery tables**:
  ```javascript
  config {
    type: "table",
    schema: "marts",
    bigquery: {
      partitionBy: "DATE(order_timestamp)",
      clusterBy: ["customer_id", "order_status"],
      requirePartitionFilter: true
    }
  }
  ```

### 9.2 Query Optimization
- **Follow SQL best practices**:
  ```sql
  -- Use CTEs for readability
  WITH recent_orders AS (
      SELECT *
      FROM ${ref("stg_orders")}
      WHERE DATE(order_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
  ),
  
  -- Filter early to reduce data processed
  active_customers AS (
      SELECT DISTINCT customer_id
      FROM recent_orders
  )
  
  SELECT
      c.*,
      COALESCE(o.order_count, 0) AS order_count
  FROM ${ref("dim_customers")} c
  -- Join only with active customers
  INNER JOIN active_customers ac USING (customer_id)
  LEFT JOIN (
      SELECT customer_id, COUNT(*) AS order_count
      FROM recent_orders
      GROUP BY customer_id
  ) o USING (customer_id)
  ```

### 9.3 Materialization Strategy
- **Choose appropriate types**:
  - **Table**: Default, best for most cases
  - **Incremental**: Large tables, append-only data
  - **View**: Small, frequently changing logic
  - **Inline**: Reusable CTEs

### 9.4 Avoiding Full Scans
- **Use partition filters**:
  ```sql
  config {
    bigquery: {
      partitionBy: "DATE(order_timestamp)",
      requirePartitionFilter: true
    }
  }
  
  SELECT *
  FROM ${ref("fct_orders")}
  -- Always filter on partition column
  WHERE DATE(order_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
  ```

---

## 10. Version Control and CI/CD

### 10.1 Git Best Practices
- **Branch strategy**:
  - `main`: Production code
  - `develop`: Development branch
  - `feature/*`: Feature branches
  - `hotfix/*`: Emergency fixes

- **.gitignore**:
  ```
  # Dataform
  .df-credentials.json
  node_modules/
  
  # IDE
  .vscode/
  .idea/
  
  # OS
  .DS_Store
  ```

### 10.2 Code Review Process
- **Review checklist**:
  - [ ] Code follows naming conventions
  - [ ] Models are documented
  - [ ] Dependencies are correct
  - [ ] Tests are included
  - [ ] Performance optimizations applied
  - [ ] No hardcoded values

### 10.3 Automated Testing
- **GitHub Actions example**:
  ```yaml
  name: Dataform CI
  
  on: [pull_request]
  
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v2
        
        - name: Setup Node.js
          uses: actions/setup-node@v2
          with:
            node-version: '16'
        
        - name: Install Dataform
          run: npm install -g @dataform/cli
        
        - name: Install dependencies
          run: npm install
        
        - name: Compile project
          run: dataform compile
        
        - name: Run tests
          run: dataform test
  ```

### 10.4 Deployment Strategy
- **Automated deployment**:
  ```yaml
  name: Deploy to Production
  
  on:
    push:
      branches: [main]
  
  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v2
        
        - name: Setup Cloud SDK
          uses: google-github-actions/setup-gcloud@v0
          with:
            service_account_key: ${{ secrets.GCP_SA_KEY }}
        
        - name: Deploy to Dataform
          run: |
            gcloud dataform repositories commit-workspace-changes \
              --repository=my-repo \
              --location=us-central1
            
            gcloud dataform workflow-invocations create \
              --repository=my-repo \
              --location=us-central1
  ```

---

## 11. Monitoring and Observability

### 11.1 Execution Monitoring
- **Monitor workflow executions**:
  - Execution duration trends
  - Failure rates
  - Data freshness
  - Row counts

### 11.2 Alerting
- **Set up alerts for**:
  - Workflow failures
  - Test failures
  - Data freshness SLA violations
  - Significant row count changes

### 11.3 Logging Best Practices
- **Add logging to complex models**:
  ```sql
  -- Log execution metadata
  SELECT
      '${self()}' AS model_name,
      CURRENT_TIMESTAMP() AS execution_time,
      COUNT(*) AS row_count,
      MIN(order_timestamp) AS min_timestamp,
      MAX(order_timestamp) AS max_timestamp
  FROM ${self()}
  ```

---

## 12. Common Anti-Patterns to Avoid

- ❌ **No source declarations**: Can't track lineage
- ❌ **Hardcoded table references**: Breaks dependency management
- ❌ **Complex logic in staging**: Keep staging simple
- ❌ **No tests**: Can't catch data quality issues
- ❌ **Poor documentation**: Hard to understand and maintain
- ❌ **No partitioning**: Poor performance on large tables
- ❌ **Circular dependencies**: Execution fails
- ❌ **Mixing layers**: Confusing data flow
- ❌ **No incremental models**: Slow and expensive
- ❌ **Inconsistent naming**: Hard to navigate project

---

## Quick Reference Checklist

### New Model Checklist
- [ ] Model type configured (table/view/incremental)
- [ ] Dependencies use ref() function
- [ ] Model is documented with description
- [ ] Key columns documented
- [ ] Assertions added (uniqueKey, nonNull)
- [ ] Naming follows conventions
- [ ] Placed in correct directory/layer
- [ ] BigQuery optimizations applied (partition, cluster)
- [ ] Tests created for data quality
- [ ] Code reviewed and approved

### Project Maintenance
- [ ] Review and update documentation monthly
- [ ] Monitor execution performance weekly
- [ ] Review and optimize slow models
- [ ] Update dependencies quarterly
- [ ] Archive unused models
- [ ] Review test coverage
- [ ] Validate data lineage accuracy
- [ ] Check for circular dependencies

---

## Resources

### Official Documentation
- [Dataform Documentation](https://cloud.google.com/dataform/docs)
- [Dataform Core Reference](https://docs.dataform.co)
- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)

### Community
- Dataform Slack Community
- GitHub Discussions
- Stack Overflow

---

*Last Updated: December 26, 2025*
*Version: 1.0*
