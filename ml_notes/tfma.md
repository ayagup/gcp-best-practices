# TFMA (TensorFlow Model Analysis) in TFX Pipelines

## Overview

**TFMA (TensorFlow Model Analysis)** is a library for evaluating TensorFlow models in production ML pipelines. It's a core component of TFX that provides scalable, comprehensive model evaluation and analysis.

## Key Features

### 1. **Model Evaluation at Scale**
- Computes metrics on large datasets using Apache Beam
- Distributed computation for big data
- Evaluates entire validation/test datasets (not just small samples)
- Supports multiple runners: DirectRunner, DataflowRunner, SparkRunner

### 2. **Sliced Metrics Analysis**
- Evaluate model performance on data **slices** (subgroups)
- Identify bias and fairness issues
- Analyze performance across different feature values

### 3. **Model Comparison**
- Compare multiple models side-by-side
- Compare current model vs. baseline/previous versions
- A/B testing support

### 4. **Visualization**
- Interactive notebooks for analysis
- Metric visualization across slices
- Performance over time tracking

## TFMA in TFX Pipeline

```python
from tfx import v1 as tfx
import tensorflow_model_analysis as tfma

# Define evaluation configuration
eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(
            signature_name='serving_default',
            label_key='label',
            prediction_key='predictions'
        )
    ],
    slicing_specs=[
        # Overall metrics
        tfma.SlicingSpec(),
        
        # Metrics by feature values
        tfma.SlicingSpec(feature_keys=['product_category']),
        tfma.SlicingSpec(feature_keys=['age_group']),
        tfma.SlicingSpec(feature_keys=['product_category', 'age_group']),
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='BinaryAccuracy'),
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='Precision'),
                tfma.MetricConfig(class_name='Recall'),
                tfma.MetricConfig(
                    class_name='FairnessIndicators',
                    config='{"thresholds": [0.5]}'
                ),
            ]
        )
    ]
)

# Add Evaluator component to TFX pipeline
evaluator = tfx.components.Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],  # Optional: for comparison
    eval_config=eval_config
)

# Complete pipeline
pipeline = tfx.Pipeline(
    pipeline_name='my-tfx-pipeline',
    components=[
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,  # TFMA component
        pusher
    ]
)
```

## Apache Beam Runners for TFMA

TFMA uses Apache Beam for distributed computation. You can choose different runners based on your scale and infrastructure:

### 1. **DirectRunner** (Local)
- Runs on local machine
- Good for development and small datasets
- No additional infrastructure needed

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Default - runs locally
eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=model_path,
    add_metrics_callbacks=[...]
)

with beam.Pipeline(options=PipelineOptions()) as pipeline:
    _ = (
        pipeline
        | 'ReadData' >> beam.io.ReadFromTFRecord(data_path)
        | 'ExtractEvaluateAndWriteResults' >>
           tfma.ExtractEvaluateAndWriteResults(
               eval_shared_model=eval_shared_model,
               output_path=output_path,
               eval_config=eval_config
           )
    )
```

### 2. **DataflowRunner** (GCP - Recommended for Production)
- Fully managed distributed processing on Google Cloud
- Auto-scales based on data volume
- Best for large-scale production workloads
- Integrated with Vertex AI Pipelines

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Dataflow runner options
pipeline_options = PipelineOptions([
    '--runner=DataflowRunner',
    '--project=my-gcp-project',
    '--region=us-central1',
    '--temp_location=gs://my-bucket/temp',
    '--staging_location=gs://my-bucket/staging',
    '--job_name=tfma-evaluation-job',
    '--max_num_workers=10',
    '--machine_type=n1-standard-4',
    '--disk_size_gb=50'
])

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=model_path
)

with beam.Pipeline(options=pipeline_options) as pipeline:
    _ = (
        pipeline
        | 'ReadData' >> beam.io.ReadFromTFRecord('gs://my-bucket/data/*')
        | 'ExtractEvaluateAndWriteResults' >>
           tfma.ExtractEvaluateAndWriteResults(
               eval_shared_model=eval_shared_model,
               output_path='gs://my-bucket/output',
               eval_config=eval_config
           )
    )
```

### 3. **SparkRunner** (Apache Spark)
- For organizations with existing Spark infrastructure
- Can run on Dataproc (GCP's managed Spark)
- Good for hybrid cloud scenarios

```python
from apache_beam.options.pipeline_options import PipelineOptions

# Spark runner options
pipeline_options = PipelineOptions([
    '--runner=SparkRunner',
    '--spark_master_url=spark://spark-master:7077',
    '--spark_submit_uber_jar=gs://my-bucket/beam-spark.jar'
])

# Or use Dataproc
pipeline_options = PipelineOptions([
    '--runner=DataflowRunner',  # Dataproc supports both
    '--project=my-gcp-project',
    '--region=us-central1',
    '--temp_location=gs://my-bucket/temp',
    '--dataproc_cluster=my-dataproc-cluster'
])
```

### 4. **FlinkRunner** (Apache Flink)
- Alternative distributed processing framework
- Good for streaming evaluation scenarios
- Can run on GCP with self-managed Flink clusters

```python
pipeline_options = PipelineOptions([
    '--runner=FlinkRunner',
    '--flink_master=localhost:8081',
    '--parallelism=4'
])
```

## Runner Comparison

| Runner | Scale | Cost | Setup | Use Case |
|--------|-------|------|-------|----------|
| **DirectRunner** | Small | Free | None | Development, testing, small datasets (<1GB) |
| **DataflowRunner** | Large | Pay-per-use | Minimal | Production on GCP, auto-scaling needed |
| **SparkRunner** | Large | Cluster cost | Medium | Existing Spark infrastructure |
| **FlinkRunner** | Large | Cluster cost | Complex | Streaming evaluation, Flink expertise |

## TFX Pipeline with DataflowRunner

```python
from tfx import v1 as tfx
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
import tensorflow_model_analysis as tfma

# Define evaluation configuration
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='Precision'),
            ]
        )
    ]
)

# Configure Evaluator to use Dataflow
evaluator = tfx.components.Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    eval_config=eval_config
)

# Create pipeline with Beam runner configuration
pipeline = tfx.Pipeline(
    pipeline_name='my-tfx-pipeline',
    pipeline_root='gs://my-bucket/pipeline-root',
    components=[
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        evaluator,  # Will use Dataflow for large-scale evaluation
        pusher
    ]
)

# Run on Dataflow
beam_pipeline_args = [
    '--runner=DataflowRunner',
    '--project=my-gcp-project',
    '--region=us-central1',
    '--temp_location=gs://my-bucket/temp',
    '--max_num_workers=20'
]

BeamDagRunner(beam_pipeline_args=beam_pipeline_args).run(pipeline)
```

## Vertex AI Pipelines (Managed Runner)

When using Vertex AI Pipelines, Google manages the runner automatically:

```python
from google.cloud import aiplatform
from tfx import v1 as tfx

# Vertex AI automatically handles Dataflow backend
aiplatform.init(
    project='my-gcp-project',
    location='us-central1',
    staging_bucket='gs://my-bucket'
)

# TFX pipeline definition
pipeline = tfx.Pipeline(...)

# Compile and run - Vertex AI manages execution
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner

runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
    config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(),
    output_filename='pipeline.json'
)

runner.run(pipeline)

# Submit to Vertex AI
job = aiplatform.PipelineJob(
    display_name='my-tfx-pipeline',
    template_path='pipeline.json',
    enable_caching=True
)

job.run()
```

## Best Practices for Runners

### For Development
- Use **DirectRunner** for rapid iteration
- Test with small data samples locally
- Validate evaluation logic before scaling

### For Production
- Use **DataflowRunner** on GCP for automatic scaling
- Set appropriate `max_num_workers` based on data size
- Use `machine_type` with sufficient memory for model loading
- Enable `autoscaling_algorithm=THROUGHPUT_BASED`

### Cost Optimization
```python
# Cost-effective Dataflow configuration
pipeline_options = PipelineOptions([
    '--runner=DataflowRunner',
    '--project=my-gcp-project',
    '--region=us-central1',
    '--temp_location=gs://my-bucket/temp',
    '--autoscaling_algorithm=THROUGHPUT_BASED',
    '--max_num_workers=10',  # Set reasonable limit
    '--machine_type=n1-standard-2',  # Start small
    '--disk_size_gb=30',  # Minimize disk
    '--use_public_ips=false',  # Save IP costs
    '--subnetwork=regions/us-central1/subnetworks/my-subnet'
])
```

## Slicing Examples

### Basic Slicing

```python
slicing_specs = [
    # Overall metrics (no slicing)
    tfma.SlicingSpec(),
    
    # Single feature slicing
    tfma.SlicingSpec(feature_keys=['country']),
    tfma.SlicingSpec(feature_keys=['age_bucket']),
    
    # Specific feature values
    tfma.SlicingSpec(
        feature_keys=['country'],
        feature_values={'country': 'US'}
    ),
    
    # Multiple features (cross-slicing)
    tfma.SlicingSpec(feature_keys=['country', 'gender']),
    
    # Numeric ranges
    tfma.SlicingSpec(
        feature_keys=['age'],
        feature_values={'age': {'min': 18, 'max': 25}}
    )
]
```

## Key Metrics You Can Compute

### Classification Metrics
```python
metrics = [
    tfma.MetricConfig(class_name='BinaryAccuracy'),
    tfma.MetricConfig(class_name='AUC'),
    tfma.MetricConfig(class_name='Precision'),
    tfma.MetricConfig(class_name='Recall'),
    tfma.MetricConfig(class_name='TruePositives'),
    tfma.MetricConfig(class_name='FalsePositives'),
    tfma.MetricConfig(class_name='TrueNegatives'),
    tfma.MetricConfig(class_name='FalseNegatives'),
    tfma.MetricConfig(class_name='ConfusionMatrixPlot'),
]
```

### Regression Metrics
```python
metrics = [
    tfma.MetricConfig(class_name='MeanSquaredError'),
    tfma.MetricConfig(class_name='MeanAbsoluteError'),
    tfma.MetricConfig(class_name='RootMeanSquaredError'),
    tfma.MetricConfig(class_name='R2Score'),
]
```

### Fairness Metrics
```python
tfma.MetricConfig(
    class_name='FairnessIndicators',
    config='{"thresholds": [0.1, 0.3, 0.5, 0.7, 0.9]}'
)
```

## Model Validation (Blessing)

TFMA can automatically validate models against thresholds:

```python
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='Precision'),
            ],
            # Define thresholds for model validation
            thresholds={
                'AUC': tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.7}  # Must be >= 0.7
                    ),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': -0.05}  # Can't drop more than 5%
                    )
                ),
                'Precision': tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.8}
                    )
                )
            }
        )
    ]
)

# Evaluator will output a 'blessing' based on thresholds
evaluator = tfx.components.Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config
)

# Pusher only deploys if model is "blessed"
pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],  # Gates deployment
    push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=serving_model_dir
        )
    )
)
```

## Analyzing Results in Jupyter

```python
import tensorflow_model_analysis as tfma

# Load evaluation results
eval_result = tfma.load_eval_result(output_path=evaluator_output_path)

# View overall metrics
tfma.view.render_slicing_metrics(eval_result)

# View metrics by slice
tfma.view.render_slicing_metrics(
    eval_result,
    slicing_spec=tfma.SlicingSpec(feature_keys=['product_category'])
)

# Compare models
eval_results = [
    tfma.load_eval_result(baseline_model_path),
    tfma.load_eval_result(candidate_model_path)
]

tfma.view.render_slicing_metrics(
    eval_results,
    slicing_spec=tfma.SlicingSpec()
)

# Time series analysis
tfma.view.render_time_series(eval_results)
```

## TFMA vs Standard model.evaluate()

| Feature | TFMA | model.evaluate() |
|---------|------|------------------|
| **Scale** | Distributed (Beam) | Single machine |
| **Data Size** | Entire dataset | Sample/batch |
| **Slicing** | ✅ Advanced slicing | ❌ Manual |
| **Comparison** | ✅ Multi-model | ❌ Single model |
| **Fairness** | ✅ Built-in | ❌ Manual |
| **Thresholds** | ✅ Automated validation | ❌ Manual checks |
| **Production** | ✅ Pipeline integrated | ❌ Ad-hoc |
| **Visualization** | ✅ Rich UI | ❌ Basic |

## Use Cases for TFMA

1. **Bias Detection**: Identify performance disparities across demographic groups
2. **Model Validation**: Automated quality gates before deployment
3. **A/B Testing**: Compare challenger vs. champion models
4. **Regulatory Compliance**: Document model fairness metrics
5. **Performance Monitoring**: Track metrics over time in production
6. **Slice-based Analysis**: Find underperforming segments

## TFMA in Vertex AI Pipelines

```python
from google_cloud_pipeline_components.v1.model_evaluation import ModelEvaluationOp

# Use TFMA within Vertex AI Pipeline
model_evaluation = ModelEvaluationOp(
    project=project_id,
    location=region,
    model=model_resource_name,
    target_field_name='label',
    prediction_type='classification',
    class_labels=['0', '1'],
    slicing_specs=[
        {'feature_keys': ['age_group']},
        {'feature_keys': ['gender']},
    ]
)
```

## Best Practices

1. **Always use slicing** - Don't just look at overall metrics
2. **Set validation thresholds** - Automate model blessing/rejection
3. **Compare to baseline** - Track if new models improve
4. **Monitor fairness** - Use FairnessIndicators for sensitive features
5. **Evaluate on full dataset** - Don't rely on small validation sets
6. **Version control eval_config** - Track changes to evaluation criteria

## Key Takeaways for GCP Data Engineer Exam

- TFMA evaluates models **at scale** using Apache Beam
- **Slicing** is the killer feature - analyze subgroup performance
- Integrated with TFX **Evaluator** component
- Provides automated **model validation** (blessing/no blessing)
- Essential for **production ML pipelines** and **fairness analysis**
- Works with Vertex AI Pipelines for managed ML workflows
- **DataflowRunner** is the recommended runner for production workloads on GCP
- **DirectRunner** for local development and testing
- Beam runners enable distributed processing across billions of examples
- Vertex AI Pipelines automatically manages runner infrastructure