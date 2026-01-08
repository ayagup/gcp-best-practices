# Vertex AI Experiments Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Experiments provides experiment tracking, parameter and metric logging, run comparison, and artifact management for ML experimentation across TensorFlow, PyTorch, scikit-learn, and XGBoost frameworks.

---

## 1. Experiment Setup

### Initialize and Create Experiments

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import metadata

class ExperimentManager:
    """Manage Vertex AI Experiments."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        
        aiplatform.init(
            project=project_id,
            location=location,
            experiment='default-experiment'
        )
    
    def create_experiment(self, experiment_name, description=''):
        """Create a new experiment."""
        
        aiplatform.init(
            project=self.project_id,
            location=self.location,
            experiment=experiment_name
        )
        
        print(f"✓ Created experiment: {experiment_name}")
        print(f"  Description: {description}")
        
        return experiment_name
    
    def start_run(self, run_name, description=''):
        """Start an experiment run."""
        
        aiplatform.start_run(run=run_name)
        
        print(f"✓ Started run: {run_name}")
        
        return run_name
    
    def end_run(self):
        """End current experiment run."""
        
        aiplatform.end_run()
        
        print(f"✓ Ended current run")
    
    def list_experiments(self):
        """List all experiments."""
        
        experiments = aiplatform.Experiment.list()
        
        print(f"\nExperiments ({len(experiments)}):\n")
        
        for exp in experiments:
            print(f"{exp.name}")
            print(f"  Description: {exp.description or 'N/A'}")
            print(f"  Created: {exp.create_time}")
            print()
        
        return experiments

# Example usage
exp_manager = ExperimentManager(project_id='my-project')

# Create experiment
# exp = exp_manager.create_experiment(
#     experiment_name='fraud-detection-experiments',
#     description='Fraud detection model experiments'
# )

# Start run
# run = exp_manager.start_run(
#     run_name='xgboost-baseline-v1',
#     description='XGBoost baseline model'
# )
```

---

## 2. Parameter Logging

### Log Hyperparameters and Configuration

```python
class ParameterLogger:
    """Log experiment parameters."""
    
    def __init__(self):
        pass
    
    def log_params(self, params):
        """Log multiple parameters."""
        
        aiplatform.log_params(params)
        
        print(f"✓ Logged {len(params)} parameters")
        
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    def log_model_hyperparameters(
        self,
        learning_rate,
        batch_size,
        epochs,
        optimizer,
        **kwargs
    ):
        """Log common model hyperparameters."""
        
        params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'optimizer': optimizer
        }
        
        # Add additional parameters
        params.update(kwargs)
        
        aiplatform.log_params(params)
        
        print(f"✓ Logged model hyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    def log_data_configuration(
        self,
        dataset_uri,
        train_size,
        val_size,
        test_size,
        features
    ):
        """Log dataset configuration."""
        
        data_config = {
            'dataset_uri': dataset_uri,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'num_features': len(features) if isinstance(features, list) else features
        }
        
        aiplatform.log_params(data_config)
        
        print(f"✓ Logged data configuration")
    
    def log_framework_info(self, framework, version):
        """Log ML framework information."""
        
        framework_info = {
            'framework': framework,
            'framework_version': version
        }
        
        aiplatform.log_params(framework_info)
        
        print(f"✓ Logged framework: {framework} {version}")

# Example usage
param_logger = ParameterLogger()

# Log hyperparameters
# param_logger.log_model_hyperparameters(
#     learning_rate=0.001,
#     batch_size=32,
#     epochs=50,
#     optimizer='adam',
#     dropout=0.2,
#     hidden_layers=[128, 64, 32]
# )

# Log data config
# param_logger.log_data_configuration(
#     dataset_uri='gs://my-bucket/data/train.csv',
#     train_size=80000,
#     val_size=10000,
#     test_size=10000,
#     features=['age', 'income', 'transaction_amount']
# )
```

---

## 3. Metric Logging

### Track Training and Evaluation Metrics

```python
class MetricLogger:
    """Log experiment metrics."""
    
    def __init__(self):
        pass
    
    def log_metrics(self, metrics, step=None):
        """Log multiple metrics."""
        
        aiplatform.log_metrics(metrics, step=step)
        
        step_info = f" at step {step}" if step else ""
        print(f"✓ Logged {len(metrics)} metrics{step_info}")
        
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    def log_training_metrics(
        self,
        epoch,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy
    ):
        """Log training metrics per epoch."""
        
        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
        
        aiplatform.log_metrics(metrics, step=epoch)
        
        print(f"✓ Epoch {epoch} metrics logged")
    
    def log_evaluation_metrics(
        self,
        accuracy,
        precision,
        recall,
        f1_score,
        auc_roc=None
    ):
        """Log evaluation metrics."""
        
        metrics = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1_score': f1_score
        }
        
        if auc_roc is not None:
            metrics['test_auc_roc'] = auc_roc
        
        aiplatform.log_metrics(metrics)
        
        print(f"✓ Logged evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    def log_time_series_metrics(self, metric_name, values, steps):
        """Log time series of metrics."""
        
        for step, value in zip(steps, values):
            aiplatform.log_metrics({metric_name: value}, step=step)
        
        print(f"✓ Logged {len(values)} {metric_name} values")
    
    def log_confusion_matrix(self, cm, class_names):
        """Log confusion matrix."""
        
        # Flatten confusion matrix for logging
        for i, true_class in enumerate(class_names):
            for j, pred_class in enumerate(class_names):
                metric_name = f'confusion_matrix_{true_class}_as_{pred_class}'
                aiplatform.log_metrics({metric_name: int(cm[i][j])})
        
        print(f"✓ Logged confusion matrix ({len(class_names)}x{len(class_names)})")

# Example usage with TensorFlow
import tensorflow as tf

metric_logger = MetricLogger()

# During training loop
# for epoch in range(num_epochs):
#     # Training...
#     train_loss, train_acc = train_one_epoch()
#     val_loss, val_acc = validate()
#     
#     metric_logger.log_training_metrics(
#         epoch=epoch,
#         train_loss=train_loss,
#         train_accuracy=train_acc,
#         val_loss=val_loss,
#         val_accuracy=val_acc
#     )

# After training
# metric_logger.log_evaluation_metrics(
#     accuracy=0.95,
#     precision=0.93,
#     recall=0.94,
#     f1_score=0.935,
#     auc_roc=0.98
# )
```

---

## 4. Artifact Logging

### Log Models and Files

```python
class ArtifactLogger:
    """Log experiment artifacts."""
    
    def __init__(self):
        pass
    
    def log_model_artifact(self, model_path, artifact_id='model'):
        """Log trained model as artifact."""
        
        aiplatform.log_model(
            model=model_path,
            artifact_id=artifact_id
        )
        
        print(f"✓ Logged model artifact: {artifact_id}")
        print(f"  Path: {model_path}")
    
    def log_file_artifact(self, file_path, artifact_id):
        """Log file as artifact."""
        
        aiplatform.log_artifact(
            artifact=file_path,
            artifact_id=artifact_id
        )
        
        print(f"✓ Logged file artifact: {artifact_id}")
    
    def save_and_log_sklearn_model(self, model, model_name='sklearn_model'):
        """Save and log scikit-learn model."""
        
        import joblib
        import tempfile
        import os
        
        # Save model to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, f'{model_name}.joblib')
            joblib.dump(model, model_path)
            
            # Log artifact
            aiplatform.log_model(
                model=model_path,
                artifact_id=model_name
            )
        
        print(f"✓ Saved and logged sklearn model: {model_name}")
    
    def save_and_log_tf_model(self, model, model_name='tf_model'):
        """Save and log TensorFlow model."""
        
        import tempfile
        import os
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, model_name)
            model.save(model_path)
            
            # Log artifact
            aiplatform.log_model(
                model=model_path,
                artifact_id=model_name
            )
        
        print(f"✓ Saved and logged TensorFlow model: {model_name}")
    
    def log_visualization(self, figure, artifact_id='plot'):
        """Log matplotlib figure."""
        
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, f'{artifact_id}.png')
            figure.savefig(plot_path)
            
            aiplatform.log_artifact(
                artifact=plot_path,
                artifact_id=artifact_id
            )
        
        print(f"✓ Logged visualization: {artifact_id}")
    
    def log_dataset(self, dataset_uri, artifact_id='dataset'):
        """Log dataset reference."""
        
        metadata_store = metadata.MetadataStore()
        
        dataset_artifact = metadata.Artifact.create(
            schema_title='system.Dataset',
            uri=dataset_uri,
            display_name=artifact_id
        )
        
        print(f"✓ Logged dataset artifact: {artifact_id}")
        print(f"  URI: {dataset_uri}")

# Example usage
artifact_logger = ArtifactLogger()

# Log sklearn model
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100)
# model.fit(X_train, y_train)
# artifact_logger.save_and_log_sklearn_model(model, 'random_forest_v1')

# Log visualization
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(history['loss'])
# ax.plot(history['val_loss'])
# artifact_logger.log_visualization(fig, 'training_curves')
```

---

## 5. Run Comparison

### Compare Experiment Runs

```python
class RunComparator:
    """Compare experiment runs."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def get_experiment_runs(self, experiment_name):
        """Get all runs from an experiment."""
        
        experiment = aiplatform.Experiment(experiment_name)
        runs_df = experiment.get_data_frame()
        
        print(f"\nExperiment: {experiment_name}")
        print(f"Total runs: {len(runs_df)}\n")
        
        return runs_df
    
    def compare_runs(self, experiment_name, metric_columns=None):
        """Compare runs by metrics."""
        
        runs_df = self.get_experiment_runs(experiment_name)
        
        if metric_columns is None:
            # Get all metric columns
            metric_columns = [
                col for col in runs_df.columns 
                if col.startswith('metric.')
            ]
        
        # Sort by first metric
        if metric_columns:
            runs_df_sorted = runs_df.sort_values(
                by=metric_columns[0],
                ascending=False
            )
            
            print("Top 10 Runs:\n")
            print(runs_df_sorted[['run_name'] + metric_columns].head(10))
        
        return runs_df
    
    def find_best_run(self, experiment_name, metric_name, maximize=True):
        """Find best run by metric."""
        
        runs_df = self.get_experiment_runs(experiment_name)
        
        metric_col = f'metric.{metric_name}'
        
        if metric_col not in runs_df.columns:
            print(f"Metric '{metric_name}' not found")
            return None
        
        if maximize:
            best_run = runs_df.loc[runs_df[metric_col].idxmax()]
        else:
            best_run = runs_df.loc[runs_df[metric_col].idxmin()]
        
        print(f"\n{'Maximizing' if maximize else 'Minimizing'} {metric_name}:")
        print(f"\nBest Run: {best_run['run_name']}")
        print(f"  {metric_name}: {best_run[metric_col]}")
        
        # Print all metrics
        metric_cols = [col for col in runs_df.columns if col.startswith('metric.')]
        for col in metric_cols:
            print(f"  {col.replace('metric.', '')}: {best_run[col]}")
        
        return best_run
    
    def compare_specific_runs(self, experiment_name, run_names):
        """Compare specific runs."""
        
        runs_df = self.get_experiment_runs(experiment_name)
        
        # Filter to specific runs
        comparison_df = runs_df[runs_df['run_name'].isin(run_names)]
        
        print(f"\nComparing {len(run_names)} runs:\n")
        
        # Show parameters and metrics
        param_cols = [col for col in comparison_df.columns if col.startswith('param.')]
        metric_cols = [col for col in comparison_df.columns if col.startswith('metric.')]
        
        display_cols = ['run_name'] + param_cols + metric_cols
        
        print(comparison_df[display_cols].to_string(index=False))
        
        return comparison_df
    
    def visualize_runs(self, experiment_name, x_param, y_metric):
        """Visualize runs with scatter plot."""
        
        import matplotlib.pyplot as plt
        
        runs_df = self.get_experiment_runs(experiment_name)
        
        param_col = f'param.{x_param}'
        metric_col = f'metric.{y_metric}'
        
        if param_col not in runs_df.columns or metric_col not in runs_df.columns:
            print("Parameter or metric not found")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(runs_df[param_col], runs_df[metric_col])
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_metric)
        ax.set_title(f'{y_metric} vs {x_param}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        print(f"✓ Created visualization: {y_metric} vs {x_param}")
        
        return fig

# Example usage
comparator = RunComparator(project_id='my-project')

# Get all runs
# runs = comparator.get_experiment_runs('fraud-detection-experiments')

# Find best run
# best = comparator.find_best_run(
#     experiment_name='fraud-detection-experiments',
#     metric_name='test_f1_score',
#     maximize=True
# )

# Compare specific runs
# comparison = comparator.compare_specific_runs(
#     experiment_name='fraud-detection-experiments',
#     run_names=['xgboost-v1', 'random-forest-v1', 'neural-net-v1']
# )
```

---

## 6. Framework Integration

### TensorFlow Integration

```python
class TensorFlowExperiment:
    """Vertex AI Experiments with TensorFlow."""
    
    def __init__(self, experiment_name, run_name):
        self.experiment_name = experiment_name
        self.run_name = run_name
        
        aiplatform.init(experiment=experiment_name)
        aiplatform.start_run(run=run_name)
    
    def train_with_logging(
        self,
        model,
        train_dataset,
        val_dataset,
        epochs,
        batch_size
    ):
        """Train TensorFlow model with automatic logging."""
        
        import tensorflow as tf
        
        # Log hyperparameters
        aiplatform.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': model.optimizer.__class__.__name__,
            'learning_rate': float(tf.keras.backend.get_value(model.optimizer.lr))
        })
        
        # Custom callback for logging
        class VertexAICallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                aiplatform.log_metrics(logs, step=epoch)
        
        # Train model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[VertexAICallback()]
        )
        
        # Log final model
        artifact_logger = ArtifactLogger()
        artifact_logger.save_and_log_tf_model(model, 'trained_model')
        
        aiplatform.end_run()
        
        print(f"✓ Training completed and logged")
        
        return history

# Example with PyTorch
class PyTorchExperiment:
    """Vertex AI Experiments with PyTorch."""
    
    def __init__(self, experiment_name, run_name):
        self.experiment_name = experiment_name
        self.run_name = run_name
        
        aiplatform.init(experiment=experiment_name)
        aiplatform.start_run(run=run_name)
    
    def train_with_logging(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs
    ):
        """Train PyTorch model with logging."""
        
        import torch
        
        # Log hyperparameters
        aiplatform.log_params({
            'epochs': epochs,
            'batch_size': train_loader.batch_size,
            'optimizer': optimizer.__class__.__name__,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Training loop
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_loss += criterion(output, target).item()
            
            # Log metrics
            aiplatform.log_metrics({
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader)
            }, step=epoch)
        
        # Save model
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pth')
            torch.save(model.state_dict(), model_path)
            aiplatform.log_model(model=model_path, artifact_id='pytorch_model')
        
        aiplatform.end_run()
        
        print(f"✓ Training completed and logged")

# Example usage
# tf_exp = TensorFlowExperiment(
#     experiment_name='image-classification',
#     run_name='resnet50-v1'
# )
# history = tf_exp.train_with_logging(model, train_ds, val_ds, epochs=50, batch_size=32)
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Initialize Vertex AI with project and location
- [ ] Create experiment with descriptive name
- [ ] Start run before training
- [ ] Install SDK: `pip install google-cloud-aiplatform`

### Logging
- [ ] Log all hyperparameters
- [ ] Log training metrics per epoch
- [ ] Log evaluation metrics
- [ ] Log trained model artifacts
- [ ] Log visualizations and plots
- [ ] End run after completion

### Comparison
- [ ] Compare runs by key metrics
- [ ] Identify best performing run
- [ ] Analyze parameter impact
- [ ] Document findings
- [ ] Export comparison results

### Best Practices
- [ ] Use consistent naming conventions
- [ ] Log comprehensive metadata
- [ ] Track data versions
- [ ] Compare multiple runs
- [ ] Clean up old experiments

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
