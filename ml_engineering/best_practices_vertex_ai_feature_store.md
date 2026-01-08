# Vertex AI Feature Store Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Feature Store provides centralized storage and serving of ML features with support for batch and streaming ingestion, online and offline serving, point-in-time lookups, and feature sharing across teams.

---

## 1. Feature Store Setup

### Initialize Feature Store

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import featurestore

class FeatureStoreManager:
    """Manage Vertex AI Feature Store operations."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        
        aiplatform.init(project=project_id, location=location)
    
    def create_feature_store(
        self,
        featurestore_id,
        online_serving_enabled=True,
        max_age_seconds=86400  # 24 hours
    ):
        """Create a new Feature Store."""
        
        fs = aiplatform.Featurestore.create(
            featurestore_id=featurestore_id,
            online_store_fixed_node_count=1,
            online_serving_config_max_age_seconds=max_age_seconds
        )
        
        print(f"✓ Created Feature Store: {fs.name}")
        print(f"  Resource name: {fs.resource_name}")
        print(f"  Online serving: {online_serving_enabled}")
        
        return fs
    
    def create_entity_type(
        self,
        featurestore_id,
        entity_type_id,
        description=''
    ):
        """Create entity type (e.g., 'user', 'product')."""
        
        fs = aiplatform.Featurestore(featurestore_id)
        
        entity_type = fs.create_entity_type(
            entity_type_id=entity_type_id,
            description=description
        )
        
        print(f"✓ Created entity type: {entity_type_id}")
        print(f"  Resource name: {entity_type.resource_name}")
        
        return entity_type
    
    def create_features(
        self,
        featurestore_id,
        entity_type_id,
        features_config
    ):
        """Create multiple features for an entity type.
        
        Args:
            features_config: List of dicts with 'feature_id', 'value_type', 'description'
        """
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        created_features = []
        
        for config in features_config:
            feature = entity_type.create_feature(
                feature_id=config['feature_id'],
                value_type=config['value_type'],
                description=config.get('description', '')
            )
            
            created_features.append(feature)
            print(f"✓ Created feature: {config['feature_id']} ({config['value_type']})")
        
        return created_features

# Example usage
fs_manager = FeatureStoreManager(project_id='my-project')

# Create Feature Store
# fs = fs_manager.create_feature_store(
#     featurestore_id='ecommerce_features',
#     online_serving_enabled=True,
#     max_age_seconds=3600
# )

# Create entity type
# user_entity = fs_manager.create_entity_type(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     description='User features for recommendations'
# )

# Create features
# features_config = [
#     {
#         'feature_id': 'age',
#         'value_type': 'INT64',
#         'description': 'User age'
#     },
#     {
#         'feature_id': 'total_purchases',
#         'value_type': 'INT64',
#         'description': 'Total purchase count'
#     },
#     {
#         'feature_id': 'avg_order_value',
#         'value_type': 'DOUBLE',
#         'description': 'Average order value'
#     },
#     {
#         'feature_id': 'preferred_category',
#         'value_type': 'STRING',
#         'description': 'Most purchased category'
#     }
# ]

# features = fs_manager.create_features(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     features_config=features_config
# )
```

---

## 2. Batch Feature Ingestion

### Ingest Features from BigQuery

```python
class BatchFeatureIngestion:
    """Batch ingestion from BigQuery and Cloud Storage."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def ingest_from_bigquery(
        self,
        featurestore_id,
        entity_type_id,
        bq_source_uri,
        entity_id_field,
        feature_time_field,
        worker_count=10
    ):
        """Ingest features from BigQuery table."""
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        # Configure import
        import_config = {
            'bigquery_source': {
                'input_uri': bq_source_uri
            },
            'entity_id_field': entity_id_field,
            'feature_time_field': feature_time_field,
            'worker_count': worker_count
        }
        
        # Start import
        import_operation = entity_type.ingest_from_bq(
            feature_ids=['*'],  # Import all features
            feature_time=feature_time_field,
            entity_id_field=entity_id_field,
            bq_source_uri=bq_source_uri,
            worker_count=worker_count
        )
        
        print(f"✓ Started batch ingestion from BigQuery")
        print(f"  Source: {bq_source_uri}")
        print(f"  Entity field: {entity_id_field}")
        print(f"  Time field: {feature_time_field}")
        
        # Wait for completion
        import_operation.wait()
        
        print(f"✓ Batch ingestion completed successfully")
        
        return import_operation
    
    def ingest_from_gcs(
        self,
        featurestore_id,
        entity_type_id,
        gcs_source_uris,
        entity_id_field,
        feature_time_field,
        file_format='csv'
    ):
        """Ingest features from Cloud Storage files."""
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        import_operation = entity_type.ingest_from_gcs(
            feature_ids=['*'],
            feature_time=feature_time_field,
            entity_id_field=entity_id_field,
            gcs_source_uris=gcs_source_uris,
            gcs_source_type=file_format.upper()
        )
        
        print(f"✓ Started batch ingestion from GCS")
        print(f"  Sources: {len(gcs_source_uris)} files")
        print(f"  Format: {file_format}")
        
        import_operation.wait()
        
        print(f"✓ Batch ingestion completed")
        
        return import_operation
    
    def schedule_batch_ingestion(
        self,
        featurestore_id,
        entity_type_id,
        bq_source_uri,
        entity_id_field,
        feature_time_field,
        schedule='0 2 * * *'  # Daily at 2 AM
    ):
        """Schedule recurring batch ingestion with Cloud Scheduler."""
        
        from google.cloud import scheduler_v1
        
        client = scheduler_v1.CloudSchedulerClient()
        
        parent = f"projects/{self.project_id}/locations/{self.location}"
        
        # Create ingestion job
        job = {
            'name': f"{parent}/jobs/feature-ingestion-{entity_type_id}",
            'schedule': schedule,
            'time_zone': 'UTC',
            'http_target': {
                'uri': f'https://{self.location}-aiplatform.googleapis.com/v1/...',
                'http_method': 'POST',
                'headers': {
                    'Content-Type': 'application/json'
                }
            }
        }
        
        response = client.create_job(parent=parent, job=job)
        
        print(f"✓ Scheduled batch ingestion")
        print(f"  Schedule: {schedule}")
        print(f"  Job: {response.name}")
        
        return response

# Example usage
batch_ingestion = BatchFeatureIngestion(project_id='my-project')

# Ingest from BigQuery
# batch_ingestion.ingest_from_bigquery(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     bq_source_uri='bq://my-project.features.user_features',
#     entity_id_field='user_id',
#     feature_time_field='feature_timestamp',
#     worker_count=10
# )
```

---

## 3. Streaming Feature Ingestion

### Stream Features with WriteFeatureValues

```python
class StreamingFeatureIngestion:
    """Real-time streaming feature ingestion."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def write_feature_values(
        self,
        featurestore_id,
        entity_type_id,
        entity_id,
        feature_values,
        feature_time=None
    ):
        """Write individual feature values."""
        
        import datetime
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        if feature_time is None:
            feature_time = datetime.datetime.now()
        
        # Write features
        entity_type.write_feature_values(
            instances=[{
                'entity_id': entity_id,
                'feature_values': feature_values,
                'feature_time': feature_time
            }]
        )
        
        print(f"✓ Wrote feature values for entity: {entity_id}")
        
        return True
    
    def stream_from_pubsub(
        self,
        featurestore_id,
        entity_type_id,
        subscription_id,
        batch_size=100
    ):
        """Stream features from Pub/Sub."""
        
        from google.cloud import pubsub_v1
        import json
        
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(
            self.project_id, 
            subscription_id
        )
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        batch = []
        
        def callback(message):
            """Process incoming message."""
            
            data = json.loads(message.data.decode('utf-8'))
            
            batch.append({
                'entity_id': data['entity_id'],
                'feature_values': data['features'],
                'feature_time': data.get('timestamp')
            })
            
            # Write batch
            if len(batch) >= batch_size:
                entity_type.write_feature_values(instances=batch)
                print(f"✓ Wrote batch of {len(batch)} feature updates")
                batch.clear()
            
            message.ack()
        
        streaming_pull_future = subscriber.subscribe(
            subscription_path, 
            callback=callback
        )
        
        print(f"✓ Started streaming from Pub/Sub: {subscription_id}")
        
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            
            # Write remaining batch
            if batch:
                entity_type.write_feature_values(instances=batch)
                print(f"✓ Wrote final batch of {len(batch)} updates")
    
    def batch_write_feature_values(
        self,
        featurestore_id,
        entity_type_id,
        instances
    ):
        """Write multiple feature values in batch."""
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        entity_type.write_feature_values(instances=instances)
        
        print(f"✓ Wrote {len(instances)} feature value sets")
        
        return True

# Example usage
streaming_ingestion = StreamingFeatureIngestion(project_id='my-project')

# Write single entity features
# streaming_ingestion.write_feature_values(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     entity_id='user_12345',
#     feature_values={
#         'total_purchases': 42,
#         'avg_order_value': 125.50,
#         'preferred_category': 'electronics'
#     }
# )

# Batch write
# instances = [
#     {
#         'entity_id': 'user_12345',
#         'feature_values': {'total_purchases': 42, 'avg_order_value': 125.50}
#     },
#     {
#         'entity_id': 'user_67890',
#         'feature_values': {'total_purchases': 18, 'avg_order_value': 89.99}
#     }
# ]
# streaming_ingestion.batch_write_feature_values(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     instances=instances
# )
```

---

## 4. Online Feature Serving

### Serve Features for Real-Time Predictions

```python
class OnlineFeatureServing:
    """Serve features for online predictions."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def read_feature_values(
        self,
        featurestore_id,
        entity_type_id,
        entity_ids,
        feature_ids
    ):
        """Read online feature values for entities."""
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        # Read features
        feature_values = entity_type.read(
            entity_ids=entity_ids,
            feature_ids=feature_ids
        )
        
        print(f"✓ Read features for {len(entity_ids)} entities")
        
        return feature_values
    
    def batch_read_feature_values(
        self,
        featurestore_id,
        entity_type_id,
        entity_ids,
        feature_ids
    ):
        """Batch read for multiple entities."""
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        results = {}
        
        # Read in batches of 100
        batch_size = 100
        
        for i in range(0, len(entity_ids), batch_size):
            batch_ids = entity_ids[i:i+batch_size]
            
            batch_values = entity_type.read(
                entity_ids=batch_ids,
                feature_ids=feature_ids
            )
            
            results.update(batch_values)
        
        print(f"✓ Read features for {len(entity_ids)} entities in batches")
        
        return results
    
    def get_features_for_prediction(
        self,
        featurestore_id,
        entity_type_id,
        entity_id,
        feature_ids=None
    ):
        """Get all features for an entity for prediction."""
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        if feature_ids is None:
            # Get all features
            features = entity_type.list_features()
            feature_ids = [f.name.split('/')[-1] for f in features]
        
        feature_values = entity_type.read(
            entity_ids=[entity_id],
            feature_ids=feature_ids
        )
        
        # Convert to dict
        entity_features = {}
        if entity_id in feature_values:
            entity_features = feature_values[entity_id]
        
        print(f"✓ Retrieved {len(entity_features)} features for prediction")
        
        return entity_features
    
    def serve_features_for_batch(
        self,
        featurestore_id,
        entity_type_configs,
        entity_ids_per_type
    ):
        """Serve features from multiple entity types.
        
        Args:
            entity_type_configs: Dict mapping entity_type_id to feature_ids
            entity_ids_per_type: Dict mapping entity_type_id to entity_ids
        """
        
        fs = aiplatform.Featurestore(featurestore_id)
        
        all_features = {}
        
        for entity_type_id, feature_ids in entity_type_configs.items():
            entity_type = fs.get_entity_type(entity_type_id)
            entity_ids = entity_ids_per_type[entity_type_id]
            
            feature_values = entity_type.read(
                entity_ids=entity_ids,
                feature_ids=feature_ids
            )
            
            all_features[entity_type_id] = feature_values
        
        print(f"✓ Served features from {len(entity_type_configs)} entity types")
        
        return all_features

# Example usage
online_serving = OnlineFeatureServing(project_id='my-project')

# Read features for single entity
# features = online_serving.read_feature_values(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     entity_ids=['user_12345'],
#     feature_ids=['age', 'total_purchases', 'avg_order_value']
# )

# Get all features for prediction
# prediction_features = online_serving.get_features_for_prediction(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     entity_id='user_12345'
# )
```

---

## 5. Offline Feature Serving & Point-in-Time Lookups

### Batch Read for Training

```python
class OfflineFeatureServing:
    """Offline feature serving for training datasets."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def batch_read_to_bigquery(
        self,
        featurestore_id,
        entity_type_id,
        destination_table,
        feature_ids=None,
        entity_ids_file=None
    ):
        """Read features to BigQuery for training."""
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        # Start batch read
        batch_read_operation = entity_type.batch_serve_to_bq(
            bq_destination_output_uri=f"bq://{destination_table}",
            feature_ids=feature_ids or ['*'],
            serving_feature_ids=entity_ids_file
        )
        
        print(f"✓ Started batch read to BigQuery")
        print(f"  Destination: {destination_table}")
        
        batch_read_operation.wait()
        
        print(f"✓ Batch read completed")
        
        return batch_read_operation
    
    def point_in_time_lookup(
        self,
        featurestore_id,
        entity_type_id,
        entity_timestamp_pairs,
        feature_ids
    ):
        """Perform point-in-time feature lookup.
        
        Args:
            entity_timestamp_pairs: List of (entity_id, timestamp) tuples
        """
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        results = []
        
        for entity_id, timestamp in entity_timestamp_pairs:
            # Read feature values as of timestamp
            feature_values = entity_type.read(
                entity_ids=[entity_id],
                feature_ids=feature_ids,
                read_time=timestamp
            )
            
            results.append({
                'entity_id': entity_id,
                'timestamp': timestamp,
                'features': feature_values.get(entity_id, {})
            })
        
        print(f"✓ Performed point-in-time lookup for {len(results)} entities")
        
        return results
    
    def create_training_dataset(
        self,
        featurestore_id,
        entity_type_id,
        label_query,
        feature_ids,
        output_table
    ):
        """Create training dataset with features and labels.
        
        Args:
            label_query: BigQuery SQL to get entity_id, timestamp, label
        """
        
        from google.cloud import bigquery
        
        bq_client = bigquery.Client(project=self.project_id)
        
        # Step 1: Get labels from BigQuery
        labels_df = bq_client.query(label_query).to_dataframe()
        
        # Step 2: Point-in-time feature lookup
        entity_timestamp_pairs = [
            (row['entity_id'], row['timestamp'])
            for _, row in labels_df.iterrows()
        ]
        
        feature_results = self.point_in_time_lookup(
            featurestore_id=featurestore_id,
            entity_type_id=entity_type_id,
            entity_timestamp_pairs=entity_timestamp_pairs,
            feature_ids=feature_ids
        )
        
        # Step 3: Merge features with labels
        import pandas as pd
        
        features_df = pd.DataFrame(feature_results)
        training_df = labels_df.merge(
            features_df,
            on=['entity_id', 'timestamp']
        )
        
        # Step 4: Write to BigQuery
        training_df.to_gbq(
            destination_table=output_table,
            project_id=self.project_id,
            if_exists='replace'
        )
        
        print(f"✓ Created training dataset: {output_table}")
        print(f"  Rows: {len(training_df)}")
        print(f"  Features: {len(feature_ids)}")
        
        return training_df

# Example usage
offline_serving = OfflineFeatureServing(project_id='my-project')

# Batch read to BigQuery
# offline_serving.batch_read_to_bigquery(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     destination_table='my-project.training.user_features',
#     feature_ids=['age', 'total_purchases', 'avg_order_value']
# )

# Create training dataset with point-in-time lookup
# label_query = """
#     SELECT 
#         user_id as entity_id,
#         event_timestamp as timestamp,
#         churned as label
#     FROM `my-project.labels.churn_labels`
#     WHERE event_timestamp BETWEEN '2024-01-01' AND '2024-12-31'
# """

# training_df = offline_serving.create_training_dataset(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     label_query=label_query,
#     feature_ids=['age', 'total_purchases', 'avg_order_value'],
#     output_table='my-project.training.churn_training_data'
# )
```

---

## 6. Feature Monitoring

```python
class FeatureMonitoring:
    """Monitor feature quality and freshness."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def check_feature_freshness(
        self,
        featurestore_id,
        entity_type_id,
        max_age_hours=24
    ):
        """Check feature freshness."""
        
        import datetime
        
        fs = aiplatform.Featurestore(featurestore_id)
        entity_type = fs.get_entity_type(entity_type_id)
        
        # Get feature metadata
        features = entity_type.list_features()
        
        now = datetime.datetime.now(datetime.timezone.utc)
        stale_features = []
        
        for feature in features:
            if feature.update_time:
                age = now - feature.update_time
                age_hours = age.total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    stale_features.append({
                        'feature_id': feature.name.split('/')[-1],
                        'age_hours': age_hours,
                        'last_updated': feature.update_time
                    })
        
        if stale_features:
            print(f"⚠ Found {len(stale_features)} stale features:")
            for feature in stale_features:
                print(f"  - {feature['feature_id']}: {feature['age_hours']:.1f}h old")
        else:
            print(f"✓ All features are fresh (< {max_age_hours}h)")
        
        return stale_features
    
    def monitor_feature_statistics(
        self,
        featurestore_id,
        entity_type_id,
        feature_id,
        sample_size=1000
    ):
        """Monitor feature statistics."""
        
        # Implementation would query feature values
        # and compute statistics
        
        print(f"Monitoring statistics for feature: {feature_id}")
        
        return {
            'feature_id': feature_id,
            'sample_size': sample_size,
            'statistics': {
                # Compute mean, std, min, max, etc.
            }
        }

# Example
monitoring = FeatureMonitoring(project_id='my-project')

# Check freshness
# stale = monitoring.check_feature_freshness(
#     featurestore_id='ecommerce_features',
#     entity_type_id='user',
#     max_age_hours=24
# )
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Create Feature Store with appropriate configuration
- [ ] Define entity types (user, product, etc.)
- [ ] Create features with correct value types
- [ ] Set up IAM permissions

### Ingestion
- [ ] Implement batch ingestion from BigQuery
- [ ] Set up streaming ingestion for real-time features
- [ ] Schedule regular batch updates
- [ ] Monitor ingestion latency
- [ ] Handle ingestion failures

### Serving
- [ ] Configure online serving for low-latency reads
- [ ] Implement offline serving for training
- [ ] Use point-in-time lookups for training datasets
- [ ] Optimize batch read performance
- [ ] Cache frequently accessed features

### Monitoring
- [ ] Monitor feature freshness
- [ ] Track feature statistics
- [ ] Alert on stale features
- [ ] Monitor serving latency
- [ ] Track ingestion errors

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
