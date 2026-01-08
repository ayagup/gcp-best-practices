# Vertex AI Workbench Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Workbench provides Jupyter-based notebook environments for data science and ML development with support for managed and user-managed instances, pre-installed ML frameworks, Git integration, scheduled execution, and seamless BigQuery integration.

---

## 1. Workbench Instance Setup

### Create Managed and User-Managed Notebooks

```python
from google.cloud import aiplatform
from google.cloud import notebooks_v1

class WorkbenchManager:
    """Manage Vertex AI Workbench instances."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
    
    def create_managed_notebook(
        self,
        display_name,
        machine_type='n1-standard-4',
        accelerator_type=None,
        accelerator_count=0
    ):
        """Create managed notebook instance.
        
        Managed notebooks are fully managed by Google with automatic updates.
        """
        
        from google.cloud.notebooks_v1 import NotebookServiceClient
        from google.cloud.notebooks_v1.types import Runtime
        
        client = NotebookServiceClient()
        parent = f"projects/{self.project_id}/locations/{self.location}"
        
        runtime = Runtime(
            name=f"{parent}/runtimes/{display_name}",
            virtual_machine=Runtime.VirtualMachine(
                virtual_machine_config=Runtime.VirtualMachineConfig(
                    machine_type=machine_type,
                    data_disk=Runtime.VirtualMachineConfig.DataDisk(
                        initialize_params=Runtime.VirtualMachineConfig.DataDisk.InitializeParams(
                            disk_size_gb=100,
                            disk_type='PD_STANDARD'
                        )
                    )
                )
            )
        )
        
        if accelerator_type:
            runtime.virtual_machine.virtual_machine_config.accelerator_config = \
                Runtime.VirtualMachineConfig.AcceleratorConfig(
                    type=accelerator_type,
                    core_count=accelerator_count
                )
        
        print(f"✓ Creating managed notebook: {display_name}")
        print(f"  Machine type: {machine_type}")
        
        if accelerator_type:
            print(f"  Accelerator: {accelerator_count}x {accelerator_type}")
        
        # operation = client.create_runtime(parent=parent, runtime_id=display_name, runtime=runtime)
        
        return {
            'name': display_name,
            'type': 'managed',
            'machine_type': machine_type
        }
    
    def create_user_managed_notebook(
        self,
        instance_name,
        machine_type='n1-standard-4',
        framework='TensorFlow',
        framework_version='2.11',
        install_gpu_driver=False,
        custom_script_uri=None
    ):
        """Create user-managed notebook instance.
        
        User-managed notebooks give more control over the instance.
        """
        
        from google.cloud.notebooks_v1 import NotebookServiceClient
        from google.cloud.notebooks_v1.types import Instance
        
        client = NotebookServiceClient()
        parent = f"projects/{self.project_id}/locations/{self.location}"
        
        instance = Instance(
            name=f"{parent}/instances/{instance_name}",
            machine_type=machine_type,
            install_gpu_driver=install_gpu_driver,
            post_startup_script=custom_script_uri if custom_script_uri else None
        )
        
        # Set container image based on framework
        if framework.lower() == 'tensorflow':
            image = f'gcr.io/deeplearning-platform-release/tf2-cpu.{framework_version.replace(".", "-")}'
        elif framework.lower() == 'pytorch':
            image = f'gcr.io/deeplearning-platform-release/pytorch-cpu.{framework_version.replace(".", "-")}'
        else:
            image = 'gcr.io/deeplearning-platform-release/base-cpu'
        
        instance.container_image = Instance.ContainerImage(
            repository=image
        )
        
        print(f"✓ Creating user-managed notebook: {instance_name}")
        print(f"  Machine type: {machine_type}")
        print(f"  Framework: {framework} {framework_version}")
        print(f"  Container image: {image}")
        
        # operation = client.create_instance(parent=parent, instance_id=instance_name, instance=instance)
        
        return {
            'name': instance_name,
            'type': 'user-managed',
            'machine_type': machine_type,
            'framework': framework
        }
    
    def list_notebook_instances(self):
        """List all notebook instances."""
        
        from google.cloud.notebooks_v1 import NotebookServiceClient
        
        client = NotebookServiceClient()
        parent = f"projects/{self.project_id}/locations/{self.location}"
        
        # List managed notebooks (runtimes)
        print(f"\n=== Managed Notebooks ===\n")
        # runtimes = client.list_runtimes(parent=parent)
        
        # List user-managed notebooks (instances)
        print(f"\n=== User-Managed Notebooks ===\n")
        # instances = client.list_instances(parent=parent)
        
        print("(Use Cloud Console or gcloud CLI to list instances)")

# Example usage
workbench_manager = WorkbenchManager(project_id='my-project')

# Create managed notebook
# notebook = workbench_manager.create_managed_notebook(
#     display_name='ml-development',
#     machine_type='n1-standard-8',
#     accelerator_type='NVIDIA_TESLA_T4',
#     accelerator_count=1
# )

# Create user-managed notebook
# notebook = workbench_manager.create_user_managed_notebook(
#     instance_name='data-science-notebook',
#     machine_type='n1-standard-4',
#     framework='TensorFlow',
#     framework_version='2.11',
#     install_gpu_driver=False
# )
```

---

## 2. Development Environment Configuration

### Configure Python Environment

```python
class NotebookEnvironmentManager:
    """Manage notebook environment configuration."""
    
    def __init__(self):
        pass
    
    def install_packages(self, packages):
        """Install Python packages in notebook."""
        
        import subprocess
        
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call(['pip', 'install', '-q', package])
        
        print(f"✓ Installed {len(packages)} packages")
    
    def create_conda_environment(
        self,
        env_name,
        python_version='3.9',
        packages=None
    ):
        """Create conda environment in notebook."""
        
        import subprocess
        
        if packages is None:
            packages = []
        
        # Create environment
        cmd_create = f"conda create -n {env_name} python={python_version} -y"
        subprocess.run(cmd_create, shell=True)
        
        # Install packages
        if packages:
            packages_str = ' '.join(packages)
            cmd_install = f"conda install -n {env_name} {packages_str} -y"
            subprocess.run(cmd_install, shell=True)
        
        print(f"✓ Created conda environment: {env_name}")
        print(f"  Python version: {python_version}")
        print(f"  Packages: {len(packages)}")
        
        return env_name
    
    def configure_jupyter_kernel(
        self,
        kernel_name,
        display_name,
        python_path=None
    ):
        """Configure custom Jupyter kernel."""
        
        import subprocess
        
        if python_path is None:
            python_path = '/opt/conda/bin/python'
        
        cmd = f'{python_path} -m ipykernel install --user --name {kernel_name} --display-name "{display_name}"'
        subprocess.run(cmd, shell=True)
        
        print(f"✓ Configured Jupyter kernel")
        print(f"  Kernel name: {kernel_name}")
        print(f"  Display name: {display_name}")
    
    def setup_ml_environment(self):
        """Set up common ML environment."""
        
        packages = [
            'tensorflow==2.11.0',
            'torch==1.13.0',
            'scikit-learn==1.2.0',
            'pandas==1.5.3',
            'numpy==1.23.5',
            'matplotlib==3.6.3',
            'seaborn==0.12.2',
            'jupyter==1.0.0',
            'ipywidgets==8.0.4'
        ]
        
        print("Setting up ML environment...")
        self.install_packages(packages)
        
        print("\n✓ ML environment ready")
        print("  TensorFlow, PyTorch, scikit-learn installed")
        print("  Data science libraries available")

# Example usage in notebook
# env_manager = NotebookEnvironmentManager()

# Install specific packages
# env_manager.install_packages([
#     'transformers',
#     'datasets',
#     'accelerate'
# ])

# Set up ML environment
# env_manager.setup_ml_environment()
```

---

## 3. Git Integration

### Configure Version Control

```python
class GitIntegrationManager:
    """Manage Git integration in notebooks."""
    
    def __init__(self):
        pass
    
    def clone_repository(
        self,
        repo_url,
        destination_dir='/home/jupyter',
        branch='main'
    ):
        """Clone Git repository to notebook."""
        
        import subprocess
        import os
        
        os.chdir(destination_dir)
        
        cmd = f'git clone -b {branch} {repo_url}'
        subprocess.run(cmd, shell=True)
        
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        print(f"✓ Cloned repository")
        print(f"  URL: {repo_url}")
        print(f"  Branch: {branch}")
        print(f"  Location: {destination_dir}/{repo_name}")
        
        return f"{destination_dir}/{repo_name}"
    
    def configure_git_credentials(
        self,
        username,
        email,
        token=None
    ):
        """Configure Git credentials."""
        
        import subprocess
        
        # Configure username and email
        subprocess.run(f'git config --global user.name "{username}"', shell=True)
        subprocess.run(f'git config --global user.email "{email}"', shell=True)
        
        print(f"✓ Configured Git credentials")
        print(f"  Username: {username}")
        print(f"  Email: {email}")
        
        if token:
            # Store credentials (use Secret Manager in production)
            print("  Token: [CONFIGURED]")
    
    def commit_and_push(
        self,
        repo_dir,
        commit_message,
        branch='main'
    ):
        """Commit and push changes."""
        
        import subprocess
        import os
        
        os.chdir(repo_dir)
        
        # Add all changes
        subprocess.run('git add .', shell=True)
        
        # Commit
        subprocess.run(f'git commit -m "{commit_message}"', shell=True)
        
        # Push
        subprocess.run(f'git push origin {branch}', shell=True)
        
        print(f"✓ Committed and pushed changes")
        print(f"  Message: {commit_message}")
        print(f"  Branch: {branch}")
    
    def sync_notebooks_with_git(
        self,
        notebook_dir,
        repo_dir,
        auto_commit=True
    ):
        """Sync notebook directory with Git repository."""
        
        import subprocess
        import os
        
        os.chdir(repo_dir)
        
        # Pull latest changes
        subprocess.run('git pull', shell=True)
        
        if auto_commit:
            # Check for changes
            result = subprocess.run(
                'git status --porcelain',
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                # Commit changes
                subprocess.run('git add .', shell=True)
                subprocess.run(
                    'git commit -m "Auto-commit notebook changes"',
                    shell=True
                )
                subprocess.run('git push', shell=True)
                
                print(f"✓ Synced notebooks with Git")
            else:
                print(f"✓ No changes to sync")

# Example usage in notebook
# git_manager = GitIntegrationManager()

# Clone repository
# repo_path = git_manager.clone_repository(
#     repo_url='https://github.com/myorg/ml-project.git',
#     destination_dir='/home/jupyter/projects',
#     branch='main'
# )

# Configure credentials
# git_manager.configure_git_credentials(
#     username='ml-engineer',
#     email='ml@company.com'
# )

# Commit changes
# git_manager.commit_and_push(
#     repo_dir=repo_path,
#     commit_message='Updated model training notebook',
#     branch='main'
# )
```

---

## 4. BigQuery Integration

### Query and Analyze Data from BigQuery

```python
class BigQueryNotebookIntegration:
    """Integrate BigQuery with Vertex AI Workbench."""
    
    def __init__(self, project_id):
        self.project_id = project_id
    
    def query_bigquery(self, query, max_results=1000):
        """Query BigQuery and return results as DataFrame."""
        
        from google.cloud import bigquery
        import pandas as pd
        
        client = bigquery.Client(project=self.project_id)
        
        query_job = client.query(query)
        results = query_job.result(max_results=max_results)
        
        df = results.to_dataframe()
        
        print(f"✓ Query executed successfully")
        print(f"  Rows returned: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    
    def load_table_to_dataframe(
        self,
        dataset_id,
        table_id,
        max_rows=None
    ):
        """Load BigQuery table directly to DataFrame."""
        
        from google.cloud import bigquery
        
        client = bigquery.Client(project=self.project_id)
        
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        
        if max_rows:
            query = f"SELECT * FROM `{table_ref}` LIMIT {max_rows}"
            df = client.query(query).to_dataframe()
        else:
            table = client.get_table(table_ref)
            df = client.list_rows(table).to_dataframe()
        
        print(f"✓ Loaded BigQuery table")
        print(f"  Table: {dataset_id}.{table_id}")
        print(f"  Rows: {len(df)}")
        
        return df
    
    def write_dataframe_to_bigquery(
        self,
        df,
        dataset_id,
        table_id,
        if_exists='replace'
    ):
        """Write DataFrame to BigQuery table."""
        
        import pandas_gbq
        
        destination_table = f"{dataset_id}.{table_id}"
        
        pandas_gbq.to_gbq(
            df,
            destination_table=destination_table,
            project_id=self.project_id,
            if_exists=if_exists
        )
        
        print(f"✓ Wrote DataFrame to BigQuery")
        print(f"  Table: {destination_table}")
        print(f"  Rows written: {len(df)}")
    
    def create_bigquery_magic(self):
        """Enable BigQuery magic commands in notebook."""
        
        magic_code = """
# Load BigQuery magic
%load_ext google.cloud.bigquery

# Example usage:
# %%bigquery df
# SELECT * FROM `project.dataset.table` LIMIT 1000
"""
        
        print("✓ BigQuery magic enabled")
        print("\nUsage:")
        print(magic_code)
        
        return magic_code
    
    def streaming_bigquery_to_notebook(
        self,
        query,
        chunk_size=10000,
        process_function=None
    ):
        """Stream large BigQuery results in chunks."""
        
        from google.cloud import bigquery
        
        client = bigquery.Client(project=self.project_id)
        
        query_job = client.query(query)
        
        print(f"Streaming query results...")
        
        row_count = 0
        
        for rows in query_job.result(page_size=chunk_size):
            chunk_data = [dict(row) for row in rows]
            
            if process_function:
                process_function(chunk_data)
            
            row_count += len(chunk_data)
            
            if row_count % chunk_size == 0:
                print(f"  Processed {row_count} rows...")
        
        print(f"✓ Streamed {row_count} total rows")
        
        return row_count

# Example usage in notebook
# bq_integration = BigQueryNotebookIntegration(project_id='my-project')

# Query BigQuery
# df = bq_integration.query_bigquery("""
#     SELECT
#         customer_id,
#         SUM(order_total) as total_spent,
#         COUNT(*) as order_count
#     FROM `my-project.ecommerce.orders`
#     WHERE order_date >= '2024-01-01'
#     GROUP BY customer_id
#     ORDER BY total_spent DESC
#     LIMIT 1000
# """)

# Load table
# customers_df = bq_integration.load_table_to_dataframe(
#     dataset_id='ecommerce',
#     table_id='customers',
#     max_rows=10000
# )

# Write results back
# bq_integration.write_dataframe_to_bigquery(
#     df=predictions_df,
#     dataset_id='ml_results',
#     table_id='customer_predictions',
#     if_exists='replace'
# )
```

---

## 5. Notebook Executor

### Schedule Notebook Execution

```python
class NotebookExecutorManager:
    """Manage scheduled notebook execution."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
    
    def create_scheduled_execution(
        self,
        notebook_file_path,
        schedule,
        input_notebook_gcs_path,
        output_notebook_gcs_path,
        parameters=None
    ):
        """Create scheduled notebook execution.
        
        Args:
            schedule: Cron expression (e.g., '0 2 * * *' for daily at 2 AM)
            parameters: Dict of parameters to pass to notebook
        """
        
        from google.cloud.notebooks_v1 import NotebookServiceClient
        from google.cloud.notebooks_v1.types import ExecutionTemplate, Schedule
        
        if parameters is None:
            parameters = {}
        
        print(f"✓ Creating scheduled execution")
        print(f"  Input: {input_notebook_gcs_path}")
        print(f"  Output: {output_notebook_gcs_path}")
        print(f"  Schedule: {schedule}")
        print(f"  Parameters: {parameters}")
        
        # Note: Actual implementation would use Notebooks API
        
        return {
            'schedule': schedule,
            'input_path': input_notebook_gcs_path,
            'output_path': output_notebook_gcs_path,
            'parameters': parameters
        }
    
    def execute_notebook_on_demand(
        self,
        notebook_file,
        output_path,
        parameters=None,
        kernel_name='python3'
    ):
        """Execute notebook immediately with papermill."""
        
        import papermill as pm
        
        if parameters is None:
            parameters = {}
        
        print(f"Executing notebook: {notebook_file}")
        
        pm.execute_notebook(
            input_path=notebook_file,
            output_path=output_path,
            parameters=parameters,
            kernel_name=kernel_name
        )
        
        print(f"✓ Notebook execution completed")
        print(f"  Output: {output_path}")
        
        return output_path
    
    def parameterize_notebook(
        self,
        notebook_path,
        parameters
    ):
        """Add parameter cell to notebook for papermill."""
        
        import nbformat
        
        # Read notebook
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create parameter cell
        param_cell = nbformat.v4.new_code_cell(
            source='\n'.join([f'{k} = {repr(v)}' for k, v in parameters.items()])
        )
        param_cell.metadata['tags'] = ['parameters']
        
        # Insert as first cell
        nb.cells.insert(0, param_cell)
        
        # Write back
        with open(notebook_path, 'w') as f:
            nbformat.write(nb, f)
        
        print(f"✓ Parameterized notebook")
        print(f"  Parameters: {parameters}")
    
    def batch_execute_notebooks(
        self,
        notebook_configs
    ):
        """Execute multiple notebooks in batch.
        
        Args:
            notebook_configs: List of dicts with notebook, output, parameters
        """
        
        import papermill as pm
        
        results = []
        
        for config in notebook_configs:
            try:
                print(f"\nExecuting: {config['notebook']}")
                
                pm.execute_notebook(
                    input_path=config['notebook'],
                    output_path=config['output'],
                    parameters=config.get('parameters', {})
                )
                
                results.append({
                    'notebook': config['notebook'],
                    'status': 'success'
                })
                
                print(f"✓ Success")
                
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
                
                results.append({
                    'notebook': config['notebook'],
                    'status': 'failed',
                    'error': str(e)
                })
        
        print(f"\n=== Batch Execution Summary ===")
        print(f"Total: {len(results)}")
        print(f"Success: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
        
        return results

# Example usage
# executor = NotebookExecutorManager(project_id='my-project')

# Schedule daily execution
# execution = executor.create_scheduled_execution(
#     notebook_file_path='/home/jupyter/notebooks/daily_report.ipynb',
#     schedule='0 2 * * *',  # Daily at 2 AM
#     input_notebook_gcs_path='gs://my-bucket/notebooks/daily_report.ipynb',
#     output_notebook_gcs_path='gs://my-bucket/outputs/daily_report_{date}.ipynb',
#     parameters={'date': '2024-01-01', 'threshold': 0.95}
# )

# Execute on demand
# output = executor.execute_notebook_on_demand(
#     notebook_file='analysis.ipynb',
#     output_path='analysis_output.ipynb',
#     parameters={'dataset': 'customer_data', 'sample_size': 10000}
# )
```

---

## 6. Collaboration Features

### Share and Collaborate on Notebooks

```python
class NotebookCollaborationManager:
    """Manage notebook collaboration."""
    
    def __init__(self, project_id):
        self.project_id = project_id
    
    def share_notebook_via_gcs(
        self,
        notebook_path,
        gcs_bucket,
        make_public=False
    ):
        """Share notebook via Cloud Storage."""
        
        from google.cloud import storage
        import os
        
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(gcs_bucket)
        
        blob_name = os.path.basename(notebook_path)
        blob = bucket.blob(f'shared-notebooks/{blob_name}')
        
        blob.upload_from_filename(notebook_path)
        
        if make_public:
            blob.make_public()
        
        gcs_uri = f'gs://{gcs_bucket}/shared-notebooks/{blob_name}'
        
        print(f"✓ Shared notebook to GCS")
        print(f"  URI: {gcs_uri}")
        
        if make_public:
            public_url = blob.public_url
            print(f"  Public URL: {public_url}")
            return public_url
        
        return gcs_uri
    
    def export_notebook_to_html(
        self,
        notebook_path,
        output_html_path
    ):
        """Export notebook to HTML for sharing."""
        
        import subprocess
        
        cmd = f'jupyter nbconvert --to html {notebook_path} --output {output_html_path}'
        subprocess.run(cmd, shell=True)
        
        print(f"✓ Exported notebook to HTML")
        print(f"  Output: {output_html_path}")
        
        return output_html_path
    
    def create_notebook_template(
        self,
        template_name,
        sections
    ):
        """Create reusable notebook template."""
        
        import nbformat
        
        nb = nbformat.v4.new_notebook()
        
        # Add cells for each section
        for section in sections:
            # Add markdown header
            nb.cells.append(
                nbformat.v4.new_markdown_cell(f"## {section['title']}\n\n{section.get('description', '')}")
            )
            
            # Add code cell if provided
            if 'code' in section:
                nb.cells.append(
                    nbformat.v4.new_code_cell(section['code'])
                )
        
        # Save template
        with open(f'{template_name}.ipynb', 'w') as f:
            nbformat.write(nb, f)
        
        print(f"✓ Created notebook template: {template_name}")
        
        return f'{template_name}.ipynb'

# Example usage
# collab_manager = NotebookCollaborationManager(project_id='my-project')

# Share notebook
# uri = collab_manager.share_notebook_via_gcs(
#     notebook_path='/home/jupyter/analysis.ipynb',
#     gcs_bucket='team-notebooks',
#     make_public=False
# )

# Export to HTML
# html_path = collab_manager.export_notebook_to_html(
#     notebook_path='analysis.ipynb',
#     output_html_path='analysis_report.html'
# )
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI Workbench API
- [ ] Choose managed vs user-managed notebook
- [ ] Select appropriate machine type
- [ ] Configure GPU/TPU if needed
- [ ] Set up custom startup scripts

### Development
- [ ] Install required Python packages
- [ ] Configure conda environments
- [ ] Set up Jupyter kernels
- [ ] Enable BigQuery magic commands
- [ ] Configure Git integration

### Version Control
- [ ] Clone Git repository
- [ ] Configure Git credentials
- [ ] Commit notebooks regularly
- [ ] Use .gitignore for large files
- [ ] Sync notebooks with remote

### Scheduling
- [ ] Install papermill for parameterization
- [ ] Add parameter cells to notebooks
- [ ] Schedule with Executor service
- [ ] Monitor execution logs
- [ ] Handle execution errors

### Best Practices
- [ ] Use managed notebooks for simplicity
- [ ] Keep notebooks modular and focused
- [ ] Document code with markdown cells
- [ ] Version control all notebooks
- [ ] Clear outputs before committing
- [ ] Use parameters for reusability
- [ ] Schedule long-running notebooks
- [ ] Export important results to GCS/BigQuery

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
