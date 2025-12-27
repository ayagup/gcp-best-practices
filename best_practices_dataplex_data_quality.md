# Dataplex Data Quality Best Practices

## Overview
Dataplex Data Quality is a managed service that helps you measure, monitor, and manage the quality of your data across BigQuery, Cloud Storage, and other GCP services. This document provides best practices for implementing effective data quality management.

---

## 1. Data Quality Strategy

### 1.1 Define Quality Dimensions
- **Accuracy**: Data correctly represents real-world values
- **Completeness**: All required data is present
- **Consistency**: Data is uniform across systems and time
- **Timeliness**: Data is available when needed and up-to-date
- **Validity**: Data conforms to defined formats and constraints
- **Uniqueness**: No unintended duplicate records

### 1.2 Establish Quality Metrics
- **Define measurable KPIs** for each quality dimension
- **Set baseline measurements** before implementing checks
- **Establish acceptable thresholds** for pass/fail criteria
- **Track quality trends** over time
- **Align metrics with business impact**

### 1.3 Prioritize Quality Checks
- **Critical data first**: Focus on data used in key business decisions
- **High-impact fields**: Prioritize columns that affect downstream systems
- **Regulatory requirements**: Ensure compliance-related checks
- **Cost vs. benefit**: Balance thoroughness with execution costs

---

## 2. Rule Design Best Practices

### 2.1 Completeness Rules
- **Null checks**:
  ```
  Check that required_field IS NOT NULL
  ```
- **Empty string validation**:
  ```
  Check that TRIM(text_field) != ''
  ```
- **Required field combinations**:
  ```
  Check that (field_a IS NOT NULL AND field_b IS NOT NULL)
  ```

### 2.2 Validity Rules
- **Data type validation**:
  ```
  Check that SAFE_CAST(field AS INT64) IS NOT NULL
  ```
- **Format validation** (regex patterns):
  ```
  Check that REGEXP_CONTAINS(email, r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
  ```
- **Enumeration checks**:
  ```
  Check that status IN ('active', 'pending', 'completed', 'cancelled')
  ```

### 2.3 Consistency Rules
- **Cross-field validation**:
  ```
  Check that end_date >= start_date
  ```
- **Referential integrity**:
  ```
  Check that customer_id IN (SELECT id FROM customers)
  ```
- **Aggregate consistency**:
  ```
  Check that total_amount = SUM(line_items.amount)
  ```

### 2.4 Accuracy Rules
- **Range validation**:
  ```
  Check that age BETWEEN 0 AND 120
  Check that price > 0
  ```
- **Statistical validation**:
  ```
  Check that value BETWEEN (mean - 3*stddev) AND (mean + 3*stddev)
  ```
- **Business rule validation**:
  ```
  Check that discount_amount <= total_amount
  ```

### 2.5 Uniqueness Rules
- **Primary key uniqueness**:
  ```
  Check that COUNT(DISTINCT id) = COUNT(id)
  ```
- **Composite key uniqueness**:
  ```
  Check that COUNT(DISTINCT CONCAT(field1, field2)) = COUNT(*)
  ```
- **Duplicate detection**:
  ```
  Check that no duplicate records exist within time window
  ```

### 2.6 Timeliness Rules
- **Freshness checks**:
  ```
  Check that MAX(updated_timestamp) >= CURRENT_TIMESTAMP() - INTERVAL 24 HOUR
  ```
- **Latency validation**:
  ```
  Check that processing_time - event_time < INTERVAL 1 HOUR
  ```
- **Completeness within SLA**:
  ```
  Check that record_count >= expected_count
  ```

---

## 3. Rule Configuration and Organization

### 3.1 Rule Naming Conventions
- **Use descriptive names**: `customer_email_format_validation`
- **Include scope**: `orders_2024_completeness_check`
- **Indicate severity**: `critical_payment_amount_validation`
- **Version rules**: `email_validation_v2`

### 3.2 Rule Categorization
- **Group by data domain**: Customer, Product, Transaction
- **Group by quality dimension**: Accuracy, Completeness, Validity
- **Group by criticality**: Critical, High, Medium, Low
- **Tag for filtering**: compliance, pii, financial

### 3.3 Rule Parameters
- **Use parameterized rules** for reusability:
  ```yaml
  rule:
    dimension: COMPLETENESS
    params:
      column_name: ${COLUMN}
      threshold: ${THRESHOLD}
  ```
- **Externalize thresholds** for easy adjustment
- **Document parameter meanings** clearly
- **Version control rule definitions**

---

## 4. Data Quality Scans

### 4.1 Scan Configuration
- **Choose appropriate scan scope**:
  - Full table scans for comprehensive checks
  - Incremental scans for large datasets
  - Partition-based scans for efficiency

- **Set scan schedules**:
  - Critical data: Hourly or real-time
  - Important data: Daily
  - Reference data: Weekly or on-change
  - Historical data: Monthly

- **Configure sampling** for large datasets:
  - Random sampling for statistical validity
  - Stratified sampling for diverse data
  - Balance between coverage and cost

### 4.2 Scan Performance Optimization
- **Use partition pruning**:
  ```sql
  WHERE partition_date = CURRENT_DATE()
  ```
- **Limit scan scope** to recent data when appropriate
- **Batch similar rules** for efficiency
- **Use incremental validation** for large tables
- **Schedule scans during off-peak hours**

### 4.3 Scan Execution Best Practices
- **Start with subset of rules** and expand gradually
- **Test rules on sample data** before full deployment
- **Monitor scan execution time** and costs
- **Set appropriate timeouts** for long-running scans
- **Enable parallel execution** when possible

---

## 5. Threshold Management

### 5.1 Setting Thresholds
- **Start with baseline measurements**:
  - Run initial scans without failures
  - Analyze historical quality metrics
  - Understand current state before enforcing

- **Use statistical approaches**:
  - Set thresholds at 2-3 standard deviations
  - Account for seasonal variations
  - Consider business context

- **Implement graduated thresholds**:
  - Warning level: 90-95% quality
  - Error level: <90% quality
  - Critical level: <80% quality

### 5.2 Threshold Adjustment
- **Review thresholds regularly** (monthly or quarterly)
- **Adjust based on business requirements** changes
- **Account for data growth** and scale
- **Document threshold changes** and rationale
- **Use A/B testing** for threshold optimization

### 5.3 Dynamic Thresholds
- **Implement time-based thresholds**:
  - Stricter during business hours
  - Relaxed during batch processing windows
  
- **Context-aware thresholds**:
  - Different thresholds per data source
  - Varying by criticality level
  - Adjusted for known data patterns

---

## 6. Integration with Data Pipelines

### 6.1 Pipeline Integration Patterns
- **Pre-processing validation**:
  - Validate source data before ingestion
  - Reject or quarantine bad data early
  - Prevent downstream contamination

- **Post-processing validation**:
  - Verify transformation accuracy
  - Validate aggregations and calculations
  - Ensure output data quality

- **Continuous monitoring**:
  - Real-time quality checks during streaming
  - Incremental validation for batch pipelines
  - Alert on quality degradation

### 6.2 Dataflow Integration
- **Embed quality checks in Dataflow pipelines**:
  ```python
  def validate_record(element):
      if not is_valid(element):
          yield pvalue.TaggedOutput('invalid', element)
      else:
          yield element
  ```
- **Route failed records** to dead-letter queues
- **Log quality metrics** to Cloud Monitoring
- **Implement circuit breakers** for severe quality issues

### 6.3 Cloud Composer Integration
- **Schedule quality scans as Airflow tasks**:
  ```python
  quality_check = DataplexDataQualityOperator(
      task_id='check_data_quality',
      data_scan_id='my_quality_scan',
      project_id=project_id
  )
  ```
- **Add quality gates** between pipeline stages
- **Trigger downstream tasks** only on quality pass
- **Send notifications** on quality failures

### 6.4 BigQuery Integration
- **Use BigQuery scheduled queries** for custom checks
- **Integrate with dbt tests** for transformation validation
- **Create quality summary tables** for reporting
- **Leverage table snapshots** for point-in-time validation

---

## 7. Monitoring and Alerting

### 7.1 Quality Metrics to Monitor
- **Pass/fail rates** per rule and scan
- **Quality score trends** over time
- **Rule execution time** and performance
- **Data volume processed**
- **Failed record counts** and percentages
- **SLA compliance** for quality checks

### 7.2 Alert Configuration
- **Critical alerts** (immediate action required):
  - Quality drops below critical threshold
  - Scan failures or timeouts
  - Complete data absence
  - Regulatory compliance violations

- **Warning alerts** (investigation needed):
  - Quality trending downward
  - New anomaly patterns detected
  - Threshold approaching
  - Scan latency increasing

- **Informational alerts**:
  - Quality improvements
  - Scan completion status
  - Summary reports

### 7.3 Alert Channels
- **Email notifications** for scheduled reports
- **Slack/Teams webhooks** for team notifications
- **PagerDuty/Opsgenie** for on-call incidents
- **Cloud Monitoring alerts** for infrastructure integration
- **Custom webhooks** for automation

### 7.4 Alert Best Practices
- **Avoid alert fatigue**: Set appropriate thresholds
- **Provide context**: Include affected data, timeframe, impact
- **Enable quick action**: Include links to dashboards and logs
- **Implement alert deduplication**: Group similar alerts
- **Set up escalation paths**: Define who gets alerted when

---

## 8. Reporting and Visualization

### 8.1 Quality Dashboards
- **Executive dashboard**:
  - Overall quality score
  - Trend analysis
  - Critical issues summary
  - Business impact metrics

- **Operational dashboard**:
  - Real-time quality status
  - Active issues and alerts
  - Scan execution status
  - Failed records queue

- **Technical dashboard**:
  - Rule performance metrics
  - Scan execution details
  - Resource utilization
  - Error logs and debugging info

### 8.2 Reporting Best Practices
- **Use Looker Studio** for interactive dashboards
- **Export results to BigQuery** for custom analysis
- **Create data quality scorecards** by domain/owner
- **Generate automated reports** on schedule
- **Track quality KPIs** against SLAs

### 8.3 Stakeholder Communication
- **Tailor reports to audience**:
  - Business stakeholders: Impact and trends
  - Data engineers: Technical details
  - Data owners: Domain-specific quality

- **Establish reporting cadence**:
  - Daily: Critical quality metrics
  - Weekly: Detailed quality reviews
  - Monthly: Trend analysis and improvements
  - Quarterly: Strategic quality assessment

---

## 9. Remediation and Incident Response

### 9.1 Incident Response Plan
- **Define severity levels**:
  - P0: Critical data quality issue affecting production
  - P1: Major issue affecting multiple systems
  - P2: Moderate issue with workarounds available
  - P3: Minor issue with low impact

- **Establish response procedures**:
  1. Detect and alert
  2. Assess impact and severity
  3. Contain and isolate bad data
  4. Investigate root cause
  5. Implement fix
  6. Verify resolution
  7. Document and prevent recurrence

### 9.2 Remediation Strategies
- **Data correction**:
  - Fix at source when possible
  - Apply transformations to correct errors
  - Backfill corrected data
  - Validate corrections

- **Quarantine bad data**:
  - Move to separate tables/buckets
  - Prevent downstream propagation
  - Enable investigation and recovery
  - Track remediation status

- **Pipeline adjustments**:
  - Add validation logic
  - Implement data cleansing
  - Update transformation rules
  - Enhance error handling

### 9.3 Root Cause Analysis
- **Ask the five whys** to find root cause
- **Analyze patterns** in quality issues
- **Review upstream data sources**
- **Examine recent changes** to pipelines
- **Document findings** in incident reports

### 9.4 Preventive Actions
- **Implement additional validation** at failure points
- **Add monitoring** for early detection
- **Update data quality rules** based on learnings
- **Improve data producer quality** at source
- **Conduct quality training** for data owners

---

## 10. Data Profiling

### 10.1 Profiling Strategy
- **Initial profiling** for new data sources:
  - Understand data structure and types
  - Identify patterns and distributions
  - Detect anomalies and outliers
  - Establish baseline metrics

- **Continuous profiling**:
  - Monitor schema evolution
  - Detect data drift
  - Track distribution changes
  - Identify emerging quality issues

### 10.2 Profile Metrics to Collect
- **Column-level statistics**:
  - Null percentages
  - Distinct value counts
  - Min/max values
  - Mean, median, standard deviation
  - Top value frequencies

- **Table-level statistics**:
  - Row counts
  - Growth rates
  - Partition statistics
  - Schema changes

### 10.3 Using Profile Results
- **Define appropriate quality rules** based on patterns
- **Set realistic thresholds** using statistical data
- **Identify candidate columns** for PII scanning
- **Optimize storage and partitioning** strategies
- **Plan data retention policies**

---

## 11. Automation and Orchestration

### 11.1 Automated Quality Checks
- **Trigger scans automatically**:
  - On data arrival (Cloud Storage events)
  - On table update (BigQuery triggers)
  - On schedule (Cloud Scheduler)
  - On pipeline completion (Airflow DAGs)

- **Implement quality gates**:
  - Block downstream processing on failures
  - Require manual approval for warnings
  - Auto-retry transient failures
  - Route to exception handling

### 11.2 Self-Healing Mechanisms
- **Automatic data correction** for known patterns
- **Retry mechanisms** for transient issues
- **Fallback to backup sources** on quality failure
- **Graceful degradation** with partial data

### 11.3 CI/CD for Quality Rules
- **Version control rule definitions** in Git
- **Test rules in dev/staging** environments
- **Automated deployment** via CI/CD pipelines
- **Rollback capability** for problematic rules
- **Change management** and approval workflows

---

## 12. Security and Compliance

### 12.1 Access Control
- **Use least privilege principle** for IAM roles
- **Separate permissions** for:
  - Rule definition
  - Scan execution
  - Results viewing
  - Data remediation

- **Implement role-based access**:
  - Data quality admins
  - Data owners
  - Read-only viewers
  - Auditors

### 12.2 Data Privacy
- **Avoid logging sensitive data** in quality results
- **Mask PII** in error messages and samples
- **Comply with data residency** requirements
- **Implement data retention policies** for quality results

### 12.3 Audit and Compliance
- **Enable audit logging** for quality operations
- **Track rule changes** and approvals
- **Document quality standards** for compliance
- **Generate compliance reports** regularly
- **Maintain quality evidence** for audits

---

## 13. Cost Optimization

### 13.1 Scan Cost Management
- **Use sampling** for large datasets when appropriate
- **Schedule scans efficiently** to avoid duplication
- **Partition-based scanning** to reduce data processed
- **Incremental validation** instead of full scans
- **Combine related rules** in single scans

### 13.2 Storage Cost Optimization
- **Set retention policies** for quality results
- **Archive historical scan data** to cheaper storage
- **Clean up temporary tables** used in validation
- **Use appropriate storage classes** for quality data

### 13.3 Compute Cost Optimization
- **Right-size scan resources** based on workload
- **Use committed use discounts** for regular scans
- **Optimize rule complexity** to reduce execution time
- **Batch rule execution** when possible
- **Monitor and eliminate redundant rules**

---

## 14. Advanced Patterns

### 14.1 Machine Learning Integration
- **Use ML for anomaly detection**:
  - Detect outliers in numeric distributions
  - Identify unusual patterns
  - Predict quality degradation

- **Automated rule generation**:
  - Learn patterns from historical data
  - Suggest new validation rules
  - Optimize threshold settings

### 14.2 Cross-System Quality Checks
- **Validate data consistency** across systems
- **Compare source and target** after replication
- **Reconciliation checks** for data migration
- **End-to-end quality tracking** across pipelines

### 14.3 Real-Time Quality Scoring
- **Calculate live quality scores** during streaming
- **Dynamic routing** based on quality
- **Real-time quality dashboards**
- **Immediate alerting** on quality drops

---

## 15. Organizational Best Practices

### 15.1 Governance Structure
- **Establish data quality council**:
  - Define quality standards
  - Approve quality policies
  - Review quality metrics
  - Drive quality initiatives

- **Define clear ownership**:
  - Data owners responsible for quality
  - Quality team provides tools and guidance
  - Engineers implement checks
  - Business validates requirements

### 15.2 Quality Culture
- **Make quality visible**: Dashboards and reports
- **Celebrate improvements**: Recognize quality wins
- **Share best practices**: Cross-team learning
- **Provide training**: Quality tools and techniques
- **Include quality in KPIs**: Team performance metrics

### 15.3 Documentation
- **Document quality standards** and policies
- **Maintain rule catalog** with descriptions
- **Create runbooks** for common issues
- **Record architectural decisions** for quality design
- **Keep troubleshooting guides** updated

---

## 16. Common Anti-Patterns to Avoid

### 16.1 Rule Design Anti-Patterns
- ❌ **Too many rules**: Causes alert fatigue and high costs
- ❌ **Overly complex rules**: Hard to maintain and understand
- ❌ **Redundant checks**: Duplicate validation across rules
- ❌ **Unrealistic thresholds**: Constant false positives
- ❌ **Vague rule names**: Unclear purpose and scope

### 16.2 Implementation Anti-Patterns
- ❌ **No incremental validation**: Scanning entire tables repeatedly
- ❌ **Synchronous blocking**: Stopping pipelines for all quality checks
- ❌ **No dead-letter queues**: Losing invalid data
- ❌ **Hard-coded values**: Rules not configurable
- ❌ **No version control**: Can't track rule changes

### 16.3 Operational Anti-Patterns
- ❌ **Set and forget**: Not reviewing or updating rules
- ❌ **Ignoring alerts**: Alert fatigue leads to missed issues
- ❌ **No root cause analysis**: Fixing symptoms not causes
- ❌ **Quality as afterthought**: Adding validation too late
- ❌ **Siloed quality teams**: No integration with data engineering

---

## 17. Migration and Adoption

### 17.1 Getting Started
1. **Identify critical data assets** to prioritize
2. **Start with basic rules** (nulls, data types)
3. **Establish baselines** before enforcing
4. **Pilot with one domain** before expanding
5. **Gather feedback** and iterate

### 17.2 Scaling Quality Practices
- **Expand gradually** to more datasets
- **Standardize rule templates** for consistency
- **Build rule library** for reuse
- **Train data owners** on self-service tools
- **Automate routine tasks**

### 17.3 Change Management
- **Communicate benefits** clearly to stakeholders
- **Provide training** and documentation
- **Offer support** during transition
- **Measure and show improvements**
- **Address concerns** proactively

---

## 18. Troubleshooting Guide

### 18.1 Common Issues

**Scan failures:**
- Check IAM permissions for data access
- Verify rule syntax is correct
- Ensure BigQuery quotas not exceeded
- Review execution logs for errors

**False positives:**
- Review and adjust thresholds
- Check if rule logic matches requirements
- Verify data sampling is representative
- Consider edge cases in rule design

**Performance issues:**
- Optimize rule complexity
- Use sampling for large datasets
- Implement partition pruning
- Schedule during off-peak hours

**Missing alerts:**
- Verify alert configuration is correct
- Check notification channels are active
- Review alert threshold settings
- Ensure monitoring is enabled

---

## Quick Reference Checklist

### Initial Setup
- [ ] Identify critical data assets for quality monitoring
- [ ] Define quality dimensions and KPIs
- [ ] Establish baseline quality metrics
- [ ] Design initial set of validation rules
- [ ] Configure scan schedules
- [ ] Set up monitoring and alerting
- [ ] Create quality dashboards
- [ ] Document quality standards
- [ ] Train data owners and engineers
- [ ] Implement incident response procedures

### Regular Operations
- [ ] Review quality metrics weekly
- [ ] Analyze quality trends monthly
- [ ] Update rules based on data changes
- [ ] Adjust thresholds as needed
- [ ] Conduct root cause analysis for incidents
- [ ] Optimize scan performance and costs
- [ ] Archive historical quality results
- [ ] Update documentation
- [ ] Provide quality reports to stakeholders
- [ ] Continuously improve quality processes

### Quality Rule Development
- [ ] Define clear rule purpose
- [ ] Use descriptive naming conventions
- [ ] Document rule logic and thresholds
- [ ] Test on sample data first
- [ ] Set appropriate severity levels
- [ ] Configure alerting appropriately
- [ ] Version control rule definitions
- [ ] Review and approve changes
- [ ] Deploy to staging before production
- [ ] Monitor impact after deployment

---

## Resources

### Official Documentation
- [Dataplex Data Quality Documentation](https://cloud.google.com/dataplex/docs/data-quality-overview)
- [BigQuery Data Quality Best Practices](https://cloud.google.com/bigquery/docs/best-practices-data-quality)
- [Data Quality Rules Reference](https://cloud.google.com/dataplex/docs/data-quality-reference)

### Tools and Integrations
- Cloud Monitoring for metrics and alerts
- Looker Studio for quality dashboards
- Cloud Composer for orchestration
- BigQuery for results analysis
- Data Catalog for metadata integration

---

*Last Updated: December 26, 2025*
*Version: 1.0*
