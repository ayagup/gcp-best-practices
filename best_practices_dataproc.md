# Dataproc Best Practices

## Overview
Cloud Dataproc is Google Cloud's fully managed Apache Spark and Hadoop service. This guide covers best practices for cluster configuration, job optimization, cost management, and integration with other GCP services.

---

## 1. Cluster Configuration

### Cluster Creation Best Practices

**Standard Cluster**
```bash
gcloud dataproc clusters create my-cluster \
  --region=us-central1 \
  --zone=us-central1-a \
  --master-machine-type=n1-standard-4 \
  --master-boot-disk-size=100 \
  --num-workers=2 \
  --worker-machine-type=n1-standard-4 \
  --worker-boot-disk-size=100 \
  --image-version=2.1-debian11 \
  --project=my-project
```

**High Availability Cluster**
```bash
gcloud dataproc clusters create my-ha-cluster \
  --region=us-central1 \
  --num-masters=3 \
  --master-machine-type=n1-standard-4 \
  --num-workers=4 \
  --worker-machine-type=n1-standard-4 \
  --enable-component-gateway \
  --properties=yarn:yarn.resourcemanager.recovery.enabled=true
```

### Machine Type Selection

**Master Node**
✅ **n1-standard-4**: Typical for most workloads (4 vCPUs, 15 GB RAM)
✅ **n1-standard-8**: Large clusters or complex jobs
✅ **n1-highmem-4**: Memory-intensive driver operations

**Worker Nodes**
✅ **n1-standard-4**: General purpose (4 vCPUs, 15 GB RAM)
✅ **n1-highmem-4**: Memory-intensive workloads (Spark caching)
✅ **n1-highcpu-4**: CPU-intensive workloads

**Preemptible Workers**
✅ Use for **cost savings** (70-80% cheaper)
✅ Suitable for **fault-tolerant workloads**
✅ Recommended ratio: 50% preemptible, 50% standard
✅ Not suitable for **streaming** or **critical jobs**

```bash
gcloud dataproc clusters create my-cluster \
  --num-workers=4 \
  --num-preemptible-workers=4 \
  --preemptible-worker-boot-disk-size=100
```

### Autoscaling

**Enable Autoscaling**
```bash
gcloud dataproc clusters create my-cluster \
  --enable-component-gateway \
  --autoscaling-policy=my-autoscaling-policy \
  --region=us-central1
```

**Create Autoscaling Policy**
```bash
gcloud dataproc autoscaling-policies create my-autoscaling-policy \
  --region=us-central1 \
  --worker-min-instances=2 \
  --worker-max-instances=20 \
  --cooldown-period=4m \
  --scale-up-factor=0.5 \
  --scale-down-factor=0.5
```

### Autoscaling Best Practices
✅ Set **minimum workers** to handle baseline load
✅ Set **maximum workers** to control costs
✅ Use **scale-up factor** of 0.5-1.0 for responsive scaling
✅ Use **scale-down factor** of 0.25-0.5 for gradual scale-down
✅ Set **cooldown period** (2-5 minutes) to avoid flapping

---

## 2. Ephemeral vs Persistent Clusters

### Ephemeral Clusters (Recommended)
✅ **Create per job**, delete after completion
✅ **Lower cost**: Pay only for job duration
✅ **Latest versions**: Always use latest Dataproc image
✅ **No maintenance**: No idle cluster management
✅ **Fast creation**: 90 seconds or less

```bash
# Submit job to ephemeral cluster
gcloud dataproc jobs submit spark \
  --cluster-name=temp-cluster-${JOB_ID} \
  --region=us-central1 \
  --class=com.example.MySparkJob \
  --jars=gs://my-bucket/my-job.jar \
  --cluster-create-flags="--num-workers=4" \
  --cluster-delete-delay=10m
```

### Persistent Clusters
✅ Use for **interactive development** (notebooks, shells)
✅ Use for **streaming jobs** (long-running)
✅ Use for **shared team environments**
✅ Consider **idle cluster cost**

### Workflow Templates (Best Practice)
```bash
# Create workflow template (manages cluster lifecycle)
gcloud dataproc workflow-templates create my-workflow \
  --region=us-central1

# Set managed cluster
gcloud dataproc workflow-templates set-managed-cluster my-workflow \
  --region=us-central1 \
  --cluster-name=temp-cluster \
  --num-workers=4 \
  --master-machine-type=n1-standard-4 \
  --worker-machine-type=n1-standard-4

# Add job to workflow
gcloud dataproc workflow-templates add-job spark \
  --workflow-template=my-workflow \
  --region=us-central1 \
  --step-id=process-data \
  --class=com.example.MySparkJob \
  --jars=gs://my-bucket/my-job.jar

# Instantiate workflow (creates cluster, runs job, deletes cluster)
gcloud dataproc workflow-templates instantiate my-workflow \
  --region=us-central1
```

---

## 3. Spark Optimization

### Spark Configuration

**Resource Allocation**
```bash
gcloud dataproc jobs submit spark \
  --cluster=my-cluster \
  --region=us-central1 \
  --class=com.example.MyJob \
  --jars=gs://bucket/job.jar \
  --properties="\
spark.executor.memory=4g,\
spark.executor.cores=2,\
spark.executor.instances=10,\
spark.default.parallelism=200,\
spark.sql.shuffle.partitions=200"
```

**Memory Configuration**
```python
# Spark submit with optimized memory settings
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 4g \
  --executor-memory 4g \
  --executor-cores 2 \
  --num-executors 10 \
  --conf spark.memory.fraction=0.8 \
  --conf spark.memory.storageFraction=0.3 \
  my_job.py
```

### Spark Best Practices

**Use DataFrames over RDDs**
```python
# ❌ Bad: RDD operations (slower)
rdd = sc.textFile('gs://bucket/data.txt')
filtered = rdd.filter(lambda x: x.startswith('ERROR'))

# ✅ Good: DataFrame operations (optimized)
df = spark.read.text('gs://bucket/data.txt')
filtered = df.filter(df.value.startswith('ERROR'))
```

**Partition Data Appropriately**
```python
# Read with appropriate partitioning
df = spark.read \
    .option('partition', 200) \
    .parquet('gs://bucket/data/')

# Repartition before expensive operations
df = df.repartition(200)

# Coalesce to reduce partitions (no shuffle)
df = df.coalesce(50)
```

**Use Broadcast Joins for Small Tables**
```python
from pyspark.sql.functions import broadcast

# Broadcast small table (< 10 MB)
result = large_df.join(
    broadcast(small_df),
    large_df.id == small_df.id
)
```

**Cache Strategically**
```python
# Cache DataFrames used multiple times
df = spark.read.parquet('gs://bucket/data/')
df.cache()

# Use multiple times
result1 = df.filter(df.status == 'active')
result2 = df.filter(df.status == 'inactive')

# Unpersist when done
df.unpersist()
```

**Avoid Shuffle Operations**
```python
# ❌ Bad: Multiple shuffles
df.groupBy('key').count() \
  .sort('count') \
  .repartition(10)

# ✅ Good: Minimize shuffles
df.groupBy('key').count() \
  .sort('count', ascending=False) \
  .limit(100)
```

### File Format Optimization
```python
# ✅ Use Parquet (columnar, compressed, fast)
df.write.mode('overwrite') \
    .partitionBy('date') \
    .parquet('gs://bucket/output/')

# ✅ Use Avro for row-oriented data
df.write.format('avro') \
    .mode('overwrite') \
    .save('gs://bucket/output/')

# ❌ Avoid CSV for large datasets (slow, no schema)
```

---

## 4. Integration with GCP Services

### BigQuery Integration
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('BigQueryApp').getOrCreate()

# Read from BigQuery
df = spark.read.format('bigquery') \
    .option('table', 'project.dataset.table') \
    .option('filter', 'date = "2025-12-25"') \
    .load()

# Write to BigQuery
df.write.format('bigquery') \
    .option('table', 'project.dataset.output_table') \
    .option('temporaryGcsBucket', 'my-bucket') \
    .mode('overwrite') \
    .save()
```

### Cloud Storage Integration
```python
# Read from Cloud Storage
df = spark.read.parquet('gs://my-bucket/data/*.parquet')

# Write to Cloud Storage
df.write.mode('overwrite') \
    .partitionBy('date', 'country') \
    .parquet('gs://my-bucket/output/')
```

### Cloud Bigtable Integration
```python
# Configure Bigtable connector
conf = {
    'hbase.zookeeper.quorum': 'bigtable-instance',
    'google.bigtable.instance.id': 'my-instance',
    'google.bigtable.project.id': 'my-project'
}

# Read from Bigtable
hbase_rdd = sc.newAPIHadoopRDD(
    'org.apache.hadoop.hbase.mapreduce.TableInputFormat',
    conf=conf
)
```

### Pub/Sub Integration
```python
# Spark Streaming with Pub/Sub
from pyspark.streaming.pubsub import PubsubUtils

ssc = StreamingContext(sc, 10)  # 10 second batches
pubsub_stream = PubsubUtils.createStream(
    ssc,
    project='my-project',
    subscription='my-subscription'
)

pubsub_stream.foreachRDD(lambda rdd: rdd.foreachPartition(process_partition))
ssc.start()
ssc.awaitTermination()
```

---

## 5. Initialization Actions & Custom Images

### Initialization Actions
```bash
# Create cluster with initialization actions
gcloud dataproc clusters create my-cluster \
  --region=us-central1 \
  --initialization-actions=gs://bucket/init-script.sh \
  --metadata='PIP_PACKAGES=pandas numpy scikit-learn'
```

**Sample Initialization Script**
```bash
#!/bin/bash
# install-python-packages.sh

# Install Python packages on all nodes
pip install pandas numpy scikit-learn

# Install on Conda environment (Dataproc 2.0+)
conda install -y -c conda-forge pandas numpy
```

### Custom Images (Recommended for Production)
```bash
# Create custom image
gcloud dataproc clusters create build-cluster \
  --region=us-central1 \
  --num-workers=0 \
  --image-version=2.1-debian11

# Install customizations on cluster
gcloud compute ssh build-cluster-m --command="sudo pip install pandas numpy"

# Create custom image
gcloud dataproc clusters export build-cluster \
  --destination=gs://bucket/my-custom-image.tar.gz

# Use custom image
gcloud dataproc clusters create my-cluster \
  --region=us-central1 \
  --image=my-custom-image-uri
```

### Custom Images Best Practices
✅ Use for **consistent environments** across clusters
✅ **Faster cluster creation** (no initialization actions)
✅ Include **frequently used libraries**
✅ Version control your custom images
✅ Test images in development first

---

## 6. Monitoring & Logging

### Cloud Monitoring
```bash
# View cluster metrics in Cloud Monitoring
# - YARN metrics (memory, CPU)
# - HDFS metrics (storage)
# - Spark metrics (jobs, stages, tasks)
```

### Key Metrics to Monitor
✅ **YARN memory utilization**: < 80% recommended
✅ **YARN CPU utilization**: < 75% recommended
✅ **HDFS storage**: < 70% of capacity
✅ **Job duration trends**: Detect performance degradation
✅ **Failed jobs**: Alert on failures
✅ **Autoscaling events**: Ensure proper scaling

### Spark UI & History Server
```bash
# Enable component gateway for web UIs
gcloud dataproc clusters create my-cluster \
  --enable-component-gateway \
  --region=us-central1

# Access Spark UI through component gateway
# Navigate to Cloud Console > Dataproc > Clusters > Web Interfaces
```

### Cloud Logging
```bash
# View cluster logs
gcloud logging read "resource.type=cloud_dataproc_cluster AND \
  resource.labels.cluster_name=my-cluster" \
  --limit=50 \
  --format=json

# Stream logs in real-time
gcloud logging tail "resource.type=cloud_dataproc_cluster"
```

---

## 7. Job Submission Best Practices

### Spark Jobs
```bash
# Submit Spark job
gcloud dataproc jobs submit spark \
  --cluster=my-cluster \
  --region=us-central1 \
  --class=com.example.MySparkJob \
  --jars=gs://bucket/my-job.jar \
  --properties=spark.executor.memory=4g \
  --max-failures-per-hour=5 \
  --max-failures-total=10
```

### PySpark Jobs
```bash
# Submit PySpark job
gcloud dataproc jobs submit pyspark \
  --cluster=my-cluster \
  --region=us-central1 \
  --py-files=gs://bucket/dependencies.zip \
  gs://bucket/my_job.py \
  -- arg1 arg2
```

### Hive Jobs
```bash
# Submit Hive query
gcloud dataproc jobs submit hive \
  --cluster=my-cluster \
  --region=us-central1 \
  --execute="SELECT COUNT(*) FROM my_table"
```

### Job Submission Best Practices
✅ Store job artifacts in **Cloud Storage**
✅ Use **workflow templates** for multi-job workflows
✅ Set **max-failures** limits
✅ Use **job labels** for cost tracking
✅ Pass secrets via **Secret Manager** (not command line)
✅ Use **service accounts** with minimum permissions

---

## 8. Cost Optimization

### Cluster Optimization
✅ Use **ephemeral clusters** (create per job)
✅ Use **preemptible workers** (50-70% cost savings)
✅ Enable **autoscaling** to match workload
✅ Right-size **machine types** based on monitoring
✅ Delete **idle clusters** immediately
✅ Use **Dataproc Serverless** for simpler workloads

### Preemptible Workers Strategy
```bash
# 50% preemptible, 50% standard workers
gcloud dataproc clusters create my-cluster \
  --num-workers=4 \
  --num-preemptible-workers=4 \
  --preemptible-worker-boot-disk-size=50
```

**When to Use Preemptible Workers**
✅ Batch processing jobs
✅ Fault-tolerant workloads
✅ Non-time-critical jobs
❌ Streaming jobs
❌ Interactive notebooks
❌ Mission-critical jobs

### Storage Optimization
✅ Delete **temporary data** in HDFS after job completion
✅ Use **Cloud Storage** instead of HDFS for long-term storage
✅ Use **compressed formats** (Parquet, Avro)
✅ Set appropriate **TTL** on Cloud Storage

### Regional Selection
✅ Choose **region close to data** (Cloud Storage, BigQuery)
✅ Use **multi-region** Cloud Storage buckets when possible
✅ Minimize **cross-region transfers**

### Committed Use Discounts
✅ Use **committed use discounts** for predictable workloads
✅ Save up to **57%** with 3-year commitments
✅ Analyze usage patterns before committing

---

## 9. Security Best Practices

### IAM & Service Accounts
✅ Use **dedicated service accounts** for clusters
✅ Grant **minimum required permissions**
✅ Don't use **default Compute Engine service account**

**Required Permissions**
- `roles/dataproc.worker`: For cluster VMs
- `roles/storage.objectViewer`: Read from Cloud Storage
- `roles/storage.objectCreator`: Write to Cloud Storage
- `roles/bigquery.dataEditor`: BigQuery integration

### Network Security
✅ Use **VPC** for cluster networking
✅ Enable **Private Google Access**
✅ Configure **firewall rules** appropriately
✅ Use **VPC Service Controls** for perimeter security

```bash
# Create cluster in VPC with private IPs
gcloud dataproc clusters create my-cluster \
  --region=us-central1 \
  --subnet=my-subnet \
  --no-address \
  --enable-private-google-access
```

### Encryption
✅ **Data at rest**: CMEK for boot disks and HDFS
✅ **Data in transit**: TLS for cluster communication
✅ **Metastore encryption**: Secure Hive metastore

```bash
# Create cluster with CMEK
gcloud dataproc clusters create my-cluster \
  --region=us-central1 \
  --gce-pd-kms-key=projects/PROJECT/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY
```

### Kerberos Authentication
```bash
# Create secure cluster with Kerberos
gcloud dataproc clusters create secure-cluster \
  --region=us-central1 \
  --kerberos-root-principal-password-uri=gs://bucket/password.encrypted \
  --kerberos-kms-key=projects/PROJECT/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY
```

---

## 10. Dataproc Serverless (Alternative)

### When to Use Dataproc Serverless
✅ **Simpler workloads**: No cluster management needed
✅ **Variable workloads**: Automatic scaling
✅ **Faster time to value**: No cluster configuration
✅ **Cost-effective**: Pay per request

```bash
# Submit Spark batch job (Serverless)
gcloud dataproc batches submit spark \
  --region=us-central1 \
  --class=com.example.MyJob \
  --jars=gs://bucket/my-job.jar \
  --subnet=my-subnet
```

### Serverless vs Clusters

| Feature | Serverless | Clusters |
|---------|-----------|----------|
| Setup time | Immediate | 90+ seconds |
| Management | Fully managed | User-managed |
| Customization | Limited | Full control |
| Cost | Per request | Per cluster-hour |
| Best for | Simple jobs | Complex workflows |

---

## 11. Common Anti-Patterns to Avoid

❌ **Long-running idle clusters**: Delete when not in use
❌ **Using default service account**: Security risk
❌ **Not using preemptible workers**: Wasting money
❌ **Over-provisioning clusters**: Start small, scale up
❌ **Storing data in HDFS long-term**: Use Cloud Storage
❌ **Not monitoring cluster metrics**: Miss optimization opportunities
❌ **Using CSV for large datasets**: Use Parquet/Avro
❌ **Multiple small jobs per cluster**: Use workflow templates
❌ **Not caching frequently used DataFrames**: Repeated computation
❌ **Ignoring data skew**: Causes straggler tasks
❌ **Using public IPs unnecessarily**: Higher egress costs
❌ **Not testing with small datasets**: Wasted time on failures

---

## 12. Migration from On-Premises Hadoop

### Assessment Phase
✅ Analyze **current workloads** and resource usage
✅ Identify **dependencies** (Hive, Pig, Spark, etc.)
✅ Review **custom configurations** and scripts
✅ Plan **data migration strategy**

### Migration Strategies

**Lift and Shift**
- Move existing Hadoop/Spark jobs to Dataproc
- Minimal code changes
- Use initialization actions for custom setup

**Modernize**
- Refactor jobs for cloud-native architecture
- Use Cloud Storage instead of HDFS
- Leverage BigQuery for data warehousing
- Use Dataflow for streaming

### Data Migration
```bash
# Use Dataproc cluster to copy data from on-prem HDFS
hadoop distcp \
  hdfs://on-prem-namenode:8020/data/* \
  gs://my-bucket/migrated-data/

# Or use Storage Transfer Service for large-scale migration
```

---

## Quick Reference Checklist

- [ ] Use ephemeral clusters (workflow templates)
- [ ] Enable autoscaling with appropriate limits
- [ ] Use preemptible workers (50% of cluster)
- [ ] Enable component gateway for web UIs
- [ ] Configure appropriate machine types
- [ ] Use Cloud Storage for input/output data
- [ ] Store job artifacts in Cloud Storage
- [ ] Use dedicated service accounts
- [ ] Enable Private Google Access
- [ ] Monitor cluster metrics regularly
- [ ] Use Parquet/Avro file formats
- [ ] Optimize Spark configurations
- [ ] Use broadcast joins for small tables
- [ ] Cache frequently used DataFrames
- [ ] Delete clusters when not in use
- [ ] Use labels for cost tracking
- [ ] Test jobs on small datasets first
- [ ] Set max-failures limits on jobs

---

## Additional Resources

- [Dataproc Documentation](https://cloud.google.com/dataproc/docs)
- [Dataproc Best Practices](https://cloud.google.com/dataproc/docs/concepts/best-practices)
- [Spark Performance Tuning](https://spark.apache.org/docs/latest/tuning.html)
- [Workflow Templates Guide](https://cloud.google.com/dataproc/docs/concepts/workflows/overview)
- [Pricing Calculator](https://cloud.google.com/products/calculator)

---

*Last Updated: December 25, 2025*
