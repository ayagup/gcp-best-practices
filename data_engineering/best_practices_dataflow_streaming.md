# Dataflow Streaming Best Practices

## Overview
Google Cloud Dataflow is a fully managed service for stream and batch data processing based on Apache Beam. This document focuses on best practices for real-time streaming pipelines.

---

## 1. Pipeline Design Best Practices

### 1.1 Windowing Strategies
- **Choose the right window type** for your use case:
  - **Fixed Windows**: Regular time intervals (e.g., 1-minute, 5-minute windows)
  - **Sliding Windows**: Overlapping time periods for moving averages
  - **Session Windows**: User activity tracking with gaps
  - **Global Windows**: Use with custom triggers for specialized needs

- **Set appropriate window sizes**:
  - Balance between latency and completeness
  - Smaller windows = lower latency but more processing overhead
  - Larger windows = better throughput but higher latency

- **Consider allowed lateness**:
  - Define how long to wait for late-arriving data
  - Balance between data completeness and resource usage
  - Use `.withAllowedLateness()` to specify late data tolerance

### 1.2 Triggering Configuration
- **Use appropriate triggers**:
  - **Default trigger**: Fires when watermark passes window end
  - **Early triggers**: Provide speculative results before window closes
  - **Late triggers**: Handle late-arriving data after window closes
  - **Composite triggers**: Combine multiple trigger conditions

- **Implement early and late firing triggers**:
  ```
  AfterWatermark.pastEndOfWindow()
    .withEarlyFirings(AfterProcessingTime.pastFirstElementInPane().plusDelayOf(Duration.standardMinutes(1)))
    .withLateFirings(AfterPane.elementCountAtLeast(1))
  ```

- **Use accumulation modes wisely**:
  - **Accumulating**: Include previous pane results (for cumulative metrics)
  - **Discarding**: Only new data (for additive operations)
  - **Accumulating and Retracting**: Correct previous results

### 1.3 Watermark Management
- **Understand watermark semantics**:
  - Watermark represents progress through event time
  - Indicates when system believes all data up to that time has arrived
  - Critical for determining when to close windows

- **Handle monotonicity**:
  - Watermarks must be monotonically increasing
  - Account for out-of-order data
  - Use custom watermark estimators when needed

- **Monitor watermark lag**:
  - Track difference between event time and processing time
  - High lag indicates pipeline falling behind
  - Set up alerts for watermark lag thresholds

---

## 2. Exactly-Once Processing

### 2.1 State and Timers
- **Use stateful processing for deduplication**:
  - Implement idempotent operations
  - Use state to track processed elements
  - Consider state TTL to prevent unbounded growth

- **Leverage timers for delayed processing**:
  - Event-time timers for time-based logic
  - Processing-time timers for timeouts
  - Clean up state using timer callbacks

### 2.2 Side Inputs and Outputs
- **Handle side inputs properly**:
  - Use for slowly changing reference data
  - Refresh side inputs periodically
  - Consider memory implications

- **Ensure idempotent writes**:
  - Use unique identifiers for records
  - Implement proper error handling and retries
  - Leverage sink-specific exactly-once semantics (e.g., BigQuery streaming inserts)

### 2.3 Checkpointing and Snapshots
- **Enable automatic checkpointing**:
  - Dataflow automatically manages checkpoints
  - Provides fault tolerance and recovery
  - No manual configuration needed in most cases

- **Monitor checkpoint completion**:
  - Track checkpoint duration
  - Alert on failed checkpoints
  - Investigate long checkpoint times

---

## 3. Performance Optimization

### 3.1 Autoscaling Configuration
- **Enable autoscaling**:
  - Set appropriate min and max worker counts
  - Allow Dataflow to scale based on backlog
  - Use `--autoscalingAlgorithm=THROUGHPUT_BASED`

- **Configure worker machine types**:
  - Choose appropriate CPU and memory based on workload
  - Use custom machine types for specific requirements
  - Consider streaming engine for reduced resource usage

- **Set appropriate disk sizes**:
  - Ensure sufficient disk for shuffles and state
  - Monitor disk usage metrics
  - Use persistent disk for stateful operations

### 3.2 Streaming Engine
- **Enable Streaming Engine** for improved performance:
  - Offloads shuffle and state management to service backend
  - Reduces worker resource requirements
  - Faster autoscaling and better resource utilization
  - Use `--experiments=enable_streaming_engine`

### 3.3 Fusion Optimization
- **Understand fusion optimization**:
  - Dataflow fuses multiple transforms into single operations
  - Reduces serialization overhead
  - Improves pipeline efficiency

- **Avoid fusion breaks when possible**:
  - Minimize use of `.withOutputTags()` unnecessarily
  - Be cautious with side inputs that force materialization
  - Profile pipeline to identify fusion boundaries

### 3.4 Batch Processing in Streaming
- **Use bundling for efficiency**:
  - Group multiple elements for batch operations
  - Reduce API call overhead (e.g., database writes)
  - Implement custom bundling logic with `@StartBundle` and `@FinishBundle`

- **Configure bundle sizes**:
  - Balance between latency and throughput
  - Larger bundles = better throughput
  - Smaller bundles = lower latency

---

## 4. Data Quality and Late Data Handling

### 4.1 Late Data Strategy
- **Set allowed lateness appropriately**:
  - Consider business requirements for data freshness vs. completeness
  - Typical values: 1 hour to 24 hours
  - Monitor late data metrics

- **Handle data beyond allowed lateness**:
  - Log or route to dead-letter queue
  - Implement custom logic for critical late data
  - Alert on excessive late data

### 4.2 Out-of-Order Data
- **Design for out-of-order arrival**:
  - Use event timestamps, not processing time
  - Implement proper windowing and watermarking
  - Test with artificially delayed data

### 4.3 Data Validation
- **Implement data quality checks**:
  - Validate schema and data types
  - Check for required fields
  - Implement business rule validation

- **Use dead-letter queues**:
  - Route invalid data to separate output
  - Enable investigation and reprocessing
  - Monitor dead-letter queue volume

---

## 5. Monitoring and Observability

### 5.1 Key Metrics to Monitor
- **System metrics**:
  - Data freshness (system lag)
  - Elements added per second
  - Backlog elements
  - Worker CPU and memory utilization
  - Throughput (elements/sec)

- **Pipeline-specific metrics**:
  - Watermark lag
  - Late data dropped
  - Window pane counts
  - Custom counters and distributions

### 5.2 Logging Best Practices
- **Use structured logging**:
  - Log at appropriate levels (INFO, WARNING, ERROR)
  - Include context (window, timestamp, keys)
  - Avoid excessive logging in production

- **Export logs to Cloud Logging**:
  - Centralized log management
  - Set up log-based metrics
  - Create alerts based on log patterns

### 5.3 Alerting
- **Set up critical alerts**:
  - Watermark lag exceeds threshold
  - Pipeline stuck or making no progress
  - High error rates
  - Worker failures or autoscaling issues
  - Backlog growing continuously

### 5.4 Custom Metrics
- **Implement custom counters**:
  - Track business metrics (records processed, errors, etc.)
  - Use `Metrics.counter()` in your DoFns
  - Export to Cloud Monitoring

- **Use distributions for latency tracking**:
  - Measure processing time distributions
  - Track custom latency metrics
  - Analyze percentiles (p50, p95, p99)

---

## 6. Resource Management

### 6.1 Worker Configuration
- **Choose appropriate machine types**:
  - n1-standard-1/2/4 for light workloads
  - n1-highmem for memory-intensive operations
  - Custom machine types for specific requirements

- **Configure worker pool**:
  - Set minimum workers to handle baseline load
  - Set maximum workers to prevent cost overruns
  - Allow headroom for traffic spikes

### 6.2 Memory Management
- **Avoid memory leaks**:
  - Clean up resources in `@Teardown` methods
  - Use weak references for large caches
  - Monitor heap usage

- **Optimize state size**:
  - Implement state expiration
  - Use efficient data structures
  - Consider state size when designing keys

### 6.3 Network Optimization
- **Minimize cross-region traffic**:
  - Deploy pipeline in same region as data sources/sinks
  - Use regional endpoints
  - Consider data locality

- **Optimize serialization**:
  - Use efficient coders (Avro, Protocol Buffers)
  - Avoid Java serialization when possible
  - Implement custom coders for complex types

---

## 7. Error Handling and Reliability

### 7.1 Retry Logic
- **Implement exponential backoff**:
  - Handle transient failures gracefully
  - Use appropriate retry limits
  - Log retry attempts

- **Distinguish retryable vs. non-retryable errors**:
  - Retry transient network issues
  - Don't retry invalid data or authorization errors
  - Route non-retryable errors to dead-letter queue

### 7.2 Pipeline Updates
- **Use update mode for pipeline changes**:
  - `--update` flag preserves pipeline state
  - Compatible changes only (add transforms, modify non-stateful logic)
  - Test updates in dev environment first

- **Use drain for graceful shutdown**:
  - Processes in-flight data before stopping
  - Prevents data loss
  - Use `gcloud dataflow jobs drain`

### 7.3 Fault Tolerance
- **Design for failures**:
  - Assume workers can fail at any time
  - Implement idempotent operations
  - Use checkpointing for recovery

- **Test failure scenarios**:
  - Simulate worker failures
  - Test recovery from checkpoints
  - Validate exactly-once semantics

---

## 8. Cost Optimization

### 8.1 Resource Efficiency
- **Enable Streaming Engine**:
  - Reduces worker resource requirements
  - Can lower costs by 50% or more
  - Better autoscaling efficiency

- **Use appropriate worker types**:
  - Don't over-provision CPU/memory
  - Use preemptible VMs for cost savings (with caution)
  - Monitor resource utilization

### 8.2 Data Processing Efficiency
- **Minimize data shuffles**:
  - Use `Combine.perKey()` instead of `GroupByKey()` when possible
  - Optimize key design to reduce hot keys
  - Profile pipeline to identify bottlenecks

- **Filter early**:
  - Remove unnecessary data as early as possible
  - Reduce downstream processing
  - Lower resource consumption

### 8.3 Autoscaling Tuning
- **Set appropriate scaling parameters**:
  - Tune `--maxNumWorkers` based on budget
  - Use metrics to determine optimal worker count
  - Consider business hours vs. off-hours scaling

---

## 9. Security Best Practices

### 9.1 Access Control
- **Use least privilege principle**:
  - Grant minimal IAM roles required
  - Use service accounts for pipeline execution
  - Separate dev/staging/prod environments

- **Secure data access**:
  - Use VPC Service Controls
  - Enable Private Google Access
  - Encrypt data in transit and at rest

### 9.2 Data Protection
- **Handle sensitive data**:
  - Use Cloud DLP for PII detection/redaction
  - Implement field-level encryption when needed
  - Comply with data residency requirements

- **Audit logging**:
  - Enable Data Access audit logs
  - Monitor pipeline access and modifications
  - Track data lineage

---

## 10. Testing and Deployment

### 10.1 Pipeline Testing
- **Unit test transforms**:
  - Use `TestPipeline` and `PAssert`
  - Test DoFns independently
  - Mock external dependencies

- **Integration testing**:
  - Test with realistic data volumes
  - Validate end-to-end pipeline
  - Test error handling paths

### 10.2 Deployment Strategy
- **Use CI/CD pipelines**:
  - Automate pipeline deployment
  - Version control pipeline code
  - Implement automated testing

- **Environment progression**:
  - Dev → Staging → Production
  - Test thoroughly in each environment
  - Use feature flags for gradual rollouts

### 10.3 Monitoring After Deployment
- **Watch for issues**:
  - Monitor metrics for first 24-48 hours
  - Compare with baseline performance
  - Have rollback plan ready

---

## 11. Common Anti-Patterns to Avoid

### 11.1 Design Anti-Patterns
- ❌ **Using processing time instead of event time** for windowing
- ❌ **Ignoring late data** without proper handling strategy
- ❌ **Creating hot keys** that cause skewed processing
- ❌ **Not using Streaming Engine** for streaming pipelines
- ❌ **Over-complicated pipeline logic** that's hard to maintain

### 11.2 Performance Anti-Patterns
- ❌ **Excessive state usage** without cleanup
- ❌ **Synchronous external calls** in high-throughput transforms
- ❌ **Not leveraging batching** for sink operations
- ❌ **Inefficient serialization** (Java serialization)
- ❌ **Fixed parallelism** without autoscaling

### 11.3 Operations Anti-Patterns
- ❌ **No monitoring or alerting** set up
- ❌ **Canceling instead of draining** pipelines
- ❌ **Not testing pipeline updates** before production
- ❌ **Ignoring watermark lag** warnings
- ❌ **No dead-letter queue** for bad data

---

## 12. Advanced Patterns

### 12.1 Stateful Processing
- **Use state for aggregations**:
  - Maintain running totals per key
  - Implement session tracking
  - Build custom windowing logic

### 12.2 Side Input Patterns
- **Slowly changing dimensions**:
  - Use side inputs for reference data
  - Refresh periodically via triggers
  - Handle updates gracefully

### 12.3 Dynamic Destinations
- **Route data dynamically**:
  - Write to different sinks based on content
  - Implement dynamic partitioning
  - Use for multi-tenancy patterns

---

## 13. Integration Best Practices

### 13.1 Pub/Sub Integration
- **Configure subscriptions properly**:
  - Use appropriate acknowledgment deadlines
  - Enable dead-letter topics
  - Monitor subscription backlog

- **Handle duplicates**:
  - Pub/Sub provides at-least-once delivery
  - Implement deduplication logic
  - Use message IDs for tracking

### 13.2 BigQuery Integration
- **Use streaming inserts efficiently**:
  - Batch records when possible
  - Monitor streaming insert errors
  - Use Storage Write API for better performance

- **Schema management**:
  - Use schema auto-detection cautiously
  - Define schemas explicitly
  - Handle schema evolution

### 13.3 Bigtable Integration
- **Optimize for Bigtable**:
  - Design row keys to avoid hotspots
  - Use appropriate column families
  - Batch writes when possible

---

## 14. Troubleshooting Guide

### 14.1 Common Issues

**Pipeline not processing data:**
- Check Pub/Sub subscription has messages
- Verify IAM permissions
- Check for errors in worker logs
- Validate pipeline code for exceptions

**High watermark lag:**
- Insufficient worker capacity
- Slow external I/O operations
- Hot keys causing skew
- Increase max workers or optimize transforms

**High system lag:**
- Pipeline falling behind input rate
- Need more workers or better optimization
- Check for bottlenecks in transforms
- Consider Streaming Engine

**Out of memory errors:**
- Reduce state size
- Implement state cleanup
- Increase worker memory
- Use Streaming Engine to offload state

### 14.2 Debugging Techniques
- Enable verbose logging temporarily
- Use Dataflow UI to inspect pipeline graph
- Check step execution times
- Review autoscaling events
- Analyze custom metrics and counters

---

## 15. Resources and Tools

### 15.1 Official Documentation
- [Apache Beam Programming Guide](https://beam.apache.org/documentation/programming-guide/)
- [Dataflow Documentation](https://cloud.google.com/dataflow/docs)
- [Streaming Engine](https://cloud.google.com/dataflow/docs/guides/deploying-a-pipeline#streaming-engine)

### 15.2 Monitoring Tools
- Cloud Monitoring for metrics and alerting
- Cloud Logging for centralized logs
- Dataflow UI for pipeline visualization
- BigQuery for log analysis

### 15.3 Development Tools
- Apache Beam SDK (Java, Python, Go)
- Beam Playground for testing
- Template creation tools
- Local runner for development

---

## Quick Reference Checklist

### Before Deploying to Production
- [ ] Enable Streaming Engine
- [ ] Configure autoscaling (min/max workers)
- [ ] Set appropriate window and trigger strategies
- [ ] Implement exactly-once semantics where needed
- [ ] Configure allowed lateness for windows
- [ ] Set up dead-letter queue for bad data
- [ ] Implement custom metrics for business logic
- [ ] Configure monitoring and alerting
- [ ] Test with realistic data volumes
- [ ] Review IAM permissions
- [ ] Document pipeline behavior and dependencies
- [ ] Create runbook for common issues
- [ ] Set up log exports to Cloud Logging
- [ ] Verify cost estimates and budgets
- [ ] Test pipeline update and drain procedures

### Regular Monitoring Tasks
- [ ] Check watermark lag daily
- [ ] Review system lag trends
- [ ] Monitor worker autoscaling behavior
- [ ] Check for late data patterns
- [ ] Review error rates and logs
- [ ] Verify cost trends
- [ ] Update dependencies and SDKs
- [ ] Review and update alerts

---

*Last Updated: December 26, 2025*
*Version: 1.0*
