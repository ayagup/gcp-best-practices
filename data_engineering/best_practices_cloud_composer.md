# Cloud Composer Best Practices

*Last Updated: December 25, 2025*

## Overview

Cloud Composer is a fully managed workflow orchestration service built on Apache Airflow. It enables you to author, schedule, and monitor pipelines that span across clouds and on-premises data centers.

---

## 1. Environment Configuration

### Environment Sizing

**Best Practices:**
- Start with small environments and scale up
- Use Composer 2 for better performance and cost
- Choose appropriate machine types based on workload

```bash
# Create Composer 2 environment (recommended)
gcloud composer environments create my-composer-env \
    --location=us-central1 \
    --environment-size=small \
    --image-version=composer-2-airflow-2.5.1 \
    --node-count=3 \
    --machine-type=n1-standard-4 \
    --disk-size=30 \
    --scheduler-count=2
```

**Environment Size Comparison:**

| Size | Scheduler CPU | Scheduler Memory | Worker CPU | Worker Memory | Use Case |
|------|---------------|------------------|------------|---------------|----------|
| Small | 0.5 | 1.875 GB | 0.5 | 1.875 GB | Development/Testing |
| Medium | 1 | 3.75 GB | 1 | 3.75 GB | Production (Light) |
| Large | 2 | 7.5 GB | 2 | 7.5 GB | Production (Heavy) |

### High Availability Configuration

```bash
# Create HA environment
gcloud composer environments create ha-composer-env \
    --location=us-central1 \
    --environment-size=medium \
    --scheduler-count=2 \
    --enable-high-resilience \
    --enable-scheduled-snapshot-creation \
    --snapshot-schedule-timezone="America/New_York" \
    --snapshot-creation-schedule="0 2 * * *"
```

---

## 2. DAG Development Best Practices

### DAG Structure

**Best Practices:**
- Keep DAGs modular and reusable
- Use dynamic task generation when appropriate
- Avoid complex logic in DAG definition

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime, timedelta

# Default arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

# DAG definition
with DAG(
    dag_id='etl_pipeline_example',
    default_args=default_args,
    description='ETL pipeline with best practices',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['production', 'etl', 'bigquery'],
    max_active_runs=1,
) as dag:
    
    # Task definitions follow
    pass
```

### Dynamic DAG Generation

**Pattern for Processing Multiple Tables:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Configuration
TABLES = ['orders', 'customers', 'products', 'inventory']

def process_table(table_name, **context):
    """Process individual table."""
    print(f"Processing table: {table_name}")
    # Processing logic here

default_args = {
    'owner': 'data-engineering',
    'retries': 2,
}

with DAG(
    'dynamic_table_processing',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    
    # Dynamically create tasks for each table
    for table in TABLES:
        task = PythonOperator(
            task_id=f'process_{table}',
            python_callable=process_table,
            op_kwargs={'table_name': table},
        )
```

### Task Dependencies

**Clear Dependency Management:**
```python
from airflow.operators.dummy import DummyOperator

with DAG('dependency_patterns', ...) as dag:
    
    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')
    
    # Extract tasks
    extract_orders = PythonOperator(...)
    extract_customers = PythonOperator(...)
    
    # Transform tasks
    transform_orders = PythonOperator(...)
    transform_customers = PythonOperator(...)
    
    # Load task
    load_to_warehouse = PythonOperator(...)
    
    # Define dependencies
    start >> [extract_orders, extract_customers]
    extract_orders >> transform_orders
    extract_customers >> transform_customers
    [transform_orders, transform_customers] >> load_to_warehouse >> end
```

---

## 3. Task Configuration

### Operator Selection

**Best Practices:**
- Use provider packages for cloud services
- Prefer operators over sensors when possible
- Use appropriate operators for the task

```python
# BigQuery operator example
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateEmptyTableOperator,
    BigQueryInsertJobOperator,
    BigQueryCheckOperator,
)

create_table = BigQueryCreateEmptyTableOperator(
    task_id='create_staging_table',
    dataset_id='staging',
    table_id='orders_{{ ds_nodash }}',
    schema_fields=[
        {'name': 'order_id', 'type': 'STRING', 'mode': 'REQUIRED'},
        {'name': 'amount', 'type': 'FLOAT64', 'mode': 'REQUIRED'},
        {'name': 'order_date', 'type': 'DATE', 'mode': 'REQUIRED'},
    ],
)

run_query = BigQueryInsertJobOperator(
    task_id='transform_orders',
    configuration={
        'query': {
            'query': '''
                SELECT
                    order_id,
                    SUM(amount) as total_amount,
                    order_date
                FROM `{{ params.project }}.staging.orders_{{ ds_nodash }}`
                GROUP BY order_id, order_date
            ''',
            'destinationTable': {
                'projectId': '{{ params.project }}',
                'datasetId': 'analytics',
                'tableId': 'orders_summary_{{ ds_nodash }}'
            },
            'writeDisposition': 'WRITE_TRUNCATE',
            'useLegacySql': False,
        }
    },
    params={'project': 'my-project'},
)

# Data quality check
check_results = BigQueryCheckOperator(
    task_id='check_data_quality',
    sql='''
        SELECT COUNT(*) > 0
        FROM `my-project.analytics.orders_summary_{{ ds_nodash }}`
    ''',
    use_legacy_sql=False,
)

create_table >> run_query >> check_results
```

### Sensor Configuration

**Best Practices:**
- Use mode='reschedule' for long-polling sensors
- Set appropriate timeouts and poke intervals
- Consider using deferrable operators in Airflow 2.2+

```python
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
from airflow.sensors.time_delta import TimeDeltaSensor
from datetime import timedelta

# Wait for file with reschedule mode
wait_for_file = GCSObjectExistenceSensor(
    task_id='wait_for_source_file',
    bucket='my-data-bucket',
    object='data/orders/{{ ds }}/orders.csv',
    mode='reschedule',  # Releases worker slot while waiting
    poke_interval=60,  # Check every 60 seconds
    timeout=3600,  # Timeout after 1 hour
)

# Wait for specific time
wait_until_noon = TimeDeltaSensor(
    task_id='wait_until_noon',
    delta=timedelta(hours=12),
)
```

---

## 4. Variables and Connections

### Variable Management

**Best Practices:**
- Use Secret Manager for sensitive data
- Store configuration in variables, not DAG code
- Use environment variables for environment-specific configs

```python
from airflow.models import Variable
from airflow.providers.google.cloud.secrets.secret_manager import CloudSecretManagerBackend

# Access variables
project_id = Variable.get("gcp_project_id")
dataset = Variable.get("bigquery_dataset")

# Access secrets from Secret Manager
secrets_backend = CloudSecretManagerBackend()
db_password = secrets_backend.get_secret(secret_id="postgres_password")

# JSON variables
config = Variable.get("pipeline_config", deserialize_json=True)
batch_size = config.get('batch_size', 1000)
```

**Setting Variables:**
```bash
# Via CLI
gcloud composer environments run my-composer-env \
    --location=us-central1 \
    variables set -- \
    gcp_project_id my-project-id

# JSON variable
gcloud composer environments run my-composer-env \
    --location=us-central1 \
    variables set -- \
    pipeline_config '{"batch_size": 1000, "timeout": 3600}'
```

### Connection Management

**Best Practices:**
- Store connections in Secret Manager backend
- Use service accounts for GCP connections
- Rotate credentials regularly

```python
from airflow.hooks.base import BaseHook

# Get connection
postgres_conn = BaseHook.get_connection('postgres_prod')

# Use in operator
from airflow.providers.postgres.operators.postgres import PostgresOperator

query_task = PostgresOperator(
    task_id='extract_from_postgres',
    postgres_conn_id='postgres_prod',
    sql='SELECT * FROM orders WHERE date = {{ ds }}',
)
```

---

## 5. XCom and Task Communication

### XCom Best Practices

**Appropriate Usage:**
- Use for small metadata (IDs, counts, status)
- Avoid passing large datasets
- Clean up old XCom data regularly

```python
from airflow.operators.python import PythonOperator

def extract_data(**context):
    """Extract data and return metadata."""
    # Process data
    records_processed = 1500
    
    # Push to XCom
    context['task_instance'].xcom_push(
        key='records_processed',
        value=records_processed
    )
    
    return records_processed

def load_data(**context):
    """Load data using metadata from previous task."""
    # Pull from XCom
    records = context['task_instance'].xcom_pull(
        task_ids='extract_data',
        key='records_processed'
    )
    
    print(f"Processing {records} records")

extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
)

load = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
)

extract >> load
```

### Alternative to XCom for Large Data

**Use Cloud Storage:**
```python
from google.cloud import storage

def extract_and_save(**context):
    """Extract data and save to GCS."""
    # Extract data
    data = extract_large_dataset()
    
    # Save to GCS
    client = storage.Client()
    bucket = client.bucket('temp-data-bucket')
    blob = bucket.blob(f"temp/{context['dag_run'].run_id}/data.parquet")
    blob.upload_from_string(data)
    
    # Push location to XCom
    return f"gs://temp-data-bucket/temp/{context['dag_run'].run_id}/data.parquet"

def load_from_gcs(**context):
    """Load data from GCS."""
    # Pull location from XCom
    gcs_path = context['task_instance'].xcom_pull(task_ids='extract_and_save')
    
    # Load data
    client = storage.Client()
    blob = storage.Blob.from_string(gcs_path, client=client)
    data = blob.download_as_string()
    
    # Process data
    process_data(data)
```

---

## 6. Error Handling and Monitoring

### Retry Configuration

**Best Practices:**
- Configure retries at task level
- Use exponential backoff
- Set appropriate retry delays

```python
from airflow.operators.python import PythonOperator
from datetime import timedelta

task_with_retries = PythonOperator(
    task_id='task_with_custom_retries',
    python_callable=my_function,
    retries=5,
    retry_delay=timedelta(minutes=2),
    retry_exponential_backoff=True,
    max_retry_delay=timedelta(minutes=30),
)
```

### Callback Functions

**Implement Alerting:**
```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

def task_failure_alert(context):
    """Send alert on task failure."""
    slack_msg = f"""
    :red_circle: Task Failed
    *Task*: {context.get('task_instance').task_id}
    *DAG*: {context.get('task_instance').dag_id}
    *Execution Time*: {context.get('execution_date')}
    *Log URL*: {context.get('task_instance').log_url}
    """
    
    failed_alert = SlackWebhookOperator(
        task_id='slack_alert',
        http_conn_id='slack_webhook',
        message=slack_msg,
    )
    
    return failed_alert.execute(context=context)

# Apply to DAG
default_args = {
    'on_failure_callback': task_failure_alert,
}

with DAG('my_dag', default_args=default_args, ...) as dag:
    pass
```

### SLA Monitoring

**Set Service Level Agreements:**
```python
from datetime import timedelta

task_with_sla = PythonOperator(
    task_id='time_sensitive_task',
    python_callable=process_data,
    sla=timedelta(hours=2),  # Alert if task takes longer than 2 hours
)

# DAG-level SLA
with DAG(
    'my_dag',
    dagrun_timeout=timedelta(hours=4),  # Fail entire DAG run after 4 hours
    sla_miss_callback=sla_miss_alert,
    ...
) as dag:
    pass
```

---

## 7. Resource Management

### Task Concurrency

**Best Practices:**
- Limit concurrent task instances
- Use pools for resource management
- Configure appropriate parallelism

```python
# Task-level concurrency
task = PythonOperator(
    task_id='resource_intensive_task',
    python_callable=heavy_processing,
    max_active_tis_per_dag=3,  # Max 3 instances across all DAG runs
    pool='heavy_processing_pool',  # Use dedicated pool
    pool_slots=2,  # Consumes 2 slots from pool
)
```

**Creating Pools:**
```bash
# Create pool via CLI
gcloud composer environments run my-composer-env \
    --location=us-central1 \
    pools set -- \
    heavy_processing_pool 10 "Pool for resource-intensive tasks"
```

### Memory Management

**Optimize Memory Usage:**
```python
from airflow.operators.python import PythonOperator

def process_in_chunks(**context):
    """Process data in chunks to manage memory."""
    chunk_size = 10000
    
    for chunk in read_data_in_chunks(chunk_size):
        process_chunk(chunk)
        # Memory is released after each chunk

task = PythonOperator(
    task_id='chunked_processing',
    python_callable=process_in_chunks,
)
```

---

## 8. Testing and Validation

### DAG Testing

**Unit Tests:**
```python
import pytest
from airflow.models import DagBag

def test_dag_loaded():
    """Test that DAG is loaded without errors."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"

def test_dag_structure():
    """Test DAG structure and properties."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('etl_pipeline_example')
    
    assert dag is not None
    assert len(dag.tasks) > 0
    assert dag.schedule_interval == '@daily'
    assert dag.catchup == False

def test_task_dependencies():
    """Test that task dependencies are correctly set."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('etl_pipeline_example')
    
    extract_task = dag.get_task('extract_data')
    transform_task = dag.get_task('transform_data')
    
    assert transform_task in extract_task.downstream_list
```

### Data Quality Checks

**Implement Validation:**
```python
from airflow.providers.google.cloud.operators.bigquery import BigQueryCheckOperator
from airflow.operators.python import BranchPythonOperator

# Row count check
check_row_count = BigQueryCheckOperator(
    task_id='check_row_count',
    sql='''
        SELECT COUNT(*) >= 1000
        FROM `project.dataset.table_{{ ds_nodash }}`
    ''',
    use_legacy_sql=False,
)

# Data freshness check
check_freshness = BigQueryCheckOperator(
    task_id='check_data_freshness',
    sql='''
        SELECT MAX(updated_at) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
        FROM `project.dataset.table_{{ ds_nodash }}`
    ''',
    use_legacy_sql=False,
)

# Null check
check_nulls = BigQueryCheckOperator(
    task_id='check_for_nulls',
    sql='''
        SELECT COUNT(*) = 0
        FROM `project.dataset.table_{{ ds_nodash }}`
        WHERE customer_id IS NULL
    ''',
    use_legacy_sql=False,
)
```

---

## 9. Performance Optimization

### DAG Parsing Optimization

**Best Practices:**
- Minimize expensive operations in DAG definition
- Use dynamic task generation judiciously
- Avoid database calls during DAG parsing

```python
# ❌ BAD: Database call during DAG parsing
from airflow import DAG
import database_module

# This runs every time DAG is parsed!
table_list = database_module.get_tables()  

with DAG('bad_dag', ...) as dag:
    for table in table_list:
        create_task(table)

# ✅ GOOD: Load configuration from Variable
from airflow.models import Variable

# Cached and efficient
table_list = Variable.get("table_list", deserialize_json=True)

with DAG('good_dag', ...) as dag:
    for table in table_list:
        create_task(table)
```

### Scheduler Performance

**Configuration:**
```bash
# Set scheduler performance configs
gcloud composer environments update my-composer-env \
    --location=us-central1 \
    --update-airflow-configs \
        scheduler-parsing_processes=4,\
        scheduler-max_threads=2,\
        core-parallelism=32,\
        core-dag_concurrency=16
```

---

## 10. Security Best Practices

### Service Account Configuration

**Principle of Least Privilege:**
```bash
# Create dedicated service account for Composer
gcloud iam service-accounts create composer-worker-sa \
    --display-name="Composer Worker Service Account"

# Grant specific permissions
gcloud projects add-iam-policy-binding my-project \
    --member=serviceAccount:composer-worker-sa@my-project.iam.gserviceaccount.com \
    --role=roles/bigquery.dataEditor

gcloud projects add-iam-policy-binding my-project \
    --member=serviceAccount:composer-worker-sa@my-project.iam.gserviceaccount.com \
    --role=roles/storage.objectAdmin
```

### Secrets Management

**Use Secret Manager:**
```python
from airflow.providers.google.cloud.secrets.secret_manager import CloudSecretManagerBackend

# Configure in airflow.cfg
secrets_backend = CloudSecretManagerBackend(
    project_id='my-project',
    connections_prefix='airflow-connections',
    variables_prefix='airflow-variables',
    sep='-'
)

# Access in DAG
from airflow.hooks.base import BaseHook

# Connection stored in Secret Manager as 'airflow-connections-postgres-prod'
conn = BaseHook.get_connection('postgres_prod')
```

---

## 11. Cost Optimization

### Environment Right-Sizing

**Best Practices:**
- Use Composer 2 (more cost-effective)
- Start with small environments
- Scale horizontally before vertically
- Use scheduled scaling for predictable workloads

```bash
# Update environment size
gcloud composer environments update my-composer-env \
    --location=us-central1 \
    --environment-size=medium

# Update worker configuration
gcloud composer environments update my-composer-env \
    --location=us-central1 \
    --min-workers=2 \
    --max-workers=10 \
    --scheduler-count=2
```

### Resource Optimization

**Efficient Task Design:**
```python
# ✅ Use external operators instead of custom Python
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator

# This runs in BigQuery, not on worker
run_query = BigQueryInsertJobOperator(
    task_id='efficient_query',
    configuration={'query': {...}},
)

# ❌ Avoid: Running queries in Python operator
def run_query_in_python():
    # This consumes worker resources
    client = bigquery.Client()
    client.query(sql).result()

inefficient_task = PythonOperator(
    task_id='inefficient_query',
    python_callable=run_query_in_python,
)
```

---

## 12. Common Anti-Patterns

### ❌ Anti-Pattern 1: Top-Level Code Execution
**Problem:** Expensive operations in DAG definition code
**Solution:** Move logic into tasks

### ❌ Anti-Pattern 2: Overly Complex DAGs
**Problem:** Single DAG doing too much
**Solution:** Break into multiple focused DAGs

### ❌ Anti-Pattern 3: Not Using Catchup Properly
**Problem:** Backfills when not needed
**Solution:** Set `catchup=False` for most DAGs

### ❌ Anti-Pattern 4: Large XCom Values
**Problem:** Passing large data through XCom
**Solution:** Use Cloud Storage for large datasets

### ❌ Anti-Pattern 5: Hardcoded Values
**Problem:** Environment-specific values in DAG code
**Solution:** Use Variables and Connections

---

## 13. Quick Reference Checklist

### DAG Development
- [ ] Use descriptive DAG and task IDs
- [ ] Set appropriate schedule_interval
- [ ] Configure catchup correctly
- [ ] Add tags for organization
- [ ] Implement retry logic
- [ ] Set up failure callbacks
- [ ] Add documentation strings

### Performance
- [ ] Minimize DAG parsing time
- [ ] Use appropriate operators
- [ ] Configure task concurrency
- [ ] Implement resource pools
- [ ] Optimize scheduler settings

### Security
- [ ] Use Secret Manager for secrets
- [ ] Apply least privilege IAM
- [ ] Rotate credentials regularly
- [ ] Enable audit logging
- [ ] Use VPC-native environments

### Monitoring
- [ ] Set up SLA monitoring
- [ ] Configure alerting
- [ ] Monitor task duration
- [ ] Track DAG run success rate
- [ ] Monitor resource utilization

### Cost Optimization
- [ ] Right-size environment
- [ ] Use Composer 2
- [ ] Delete old DAG runs
- [ ] Optimize task design
- [ ] Monitor environment costs

---

*Best Practices for Google Cloud Data Engineer Certification*
