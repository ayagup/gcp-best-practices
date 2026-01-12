# GCP Vertex AI Deep Learning VM (DLVM) Images

## Overview
**Deep Learning VM Images** are pre-configured virtual machine images on Google Cloud that come with popular ML frameworks and tools pre-installed, optimized for AI/ML workloads.

## Key Features

### **Pre-installed Frameworks**
- TensorFlow
- PyTorch
- JAX
- scikit-learn
- XGBoost
- Rapids (GPU-accelerated data science)

### **Development Tools**
- JupyterLab
- VS Code (code-server)
- NVIDIA CUDA & cuDNN (GPU images)
- Conda/pip package managers
- Git, Docker

### **Optimized Libraries**
- **Intel MKL (Math Kernel Library)**: Optimized math routines for CPUs
- **Intel MKL-DNN**: Deep neural network primitives optimized for Intel architecture
- **NVIDIA cuBLAS**: GPU-accelerated linear algebra
- **NVIDIA cuDNN**: GPU-accelerated deep learning primitives
- **NCCL (NVIDIA Collective Communications Library)**: Multi-GPU and multi-node communication
- **TensorRT**: High-performance deep learning inference optimizer
- **OpenMPI**: Message Passing Interface for distributed computing
- **Horovod**: Distributed training framework (on some images)

### **Hardware Support**
- CPU-only instances
- NVIDIA GPU instances (T4, V100, A100, L4)
- TPU instances (Tensor Processing Units)

## Image Types

### 1. **Framework-Specific Images**
```bash
# TensorFlow Enterprise
gcloud compute instances create my-tf-vm \
    --zone=us-central1-a \
    --image-family=tf-ent-latest-gpu \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB
```

### 2. **PyTorch Images**
```bash
# PyTorch with GPU
gcloud compute instances create my-pytorch-vm \
    --zone=us-central1-a \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1"
```

### 3. **Common Images**
```bash
# Common framework (TF + PyTorch + JAX)
gcloud compute instances create my-ml-vm \
    --zone=us-central1-a \
    --image-family=common-cu113 \
    --image-project=deeplearning-platform-release
```

## Available Image Families

| Image Family | Description | GPU Support |
|--------------|-------------|-------------|
| `tf-ent-latest-gpu` | TensorFlow Enterprise (latest) | ✅ |
| `tf-ent-latest-cpu` | TensorFlow Enterprise (CPU) | ❌ |
| `pytorch-latest-gpu` | PyTorch (latest) | ✅ |
| `pytorch-latest-cpu` | PyTorch (CPU) | ❌ |
| `common-cu113` | Multi-framework (CUDA 11.3) | ✅ |
| `common-cpu` | Multi-framework (CPU) | ❌ |
| `rapids-latest-gpu` | RAPIDS (GPU data science) | ✅ |

## Common Use Cases

### **1. Interactive Development**
- Launch JupyterLab for notebook-based development
- Pre-configured with popular libraries
- Easy data access to Cloud Storage

### **2. Training Jobs**
- Run long-running training jobs
- Scale up/down with different machine types
- Use preemptible instances for cost savings

### **3. Model Development**
- Experiment with different frameworks
- Quick environment setup (minutes vs hours)
- Consistent development environments

## Optimized Libraries Deep Dive

### **CPU Optimization**
DLVMs include Intel-optimized libraries for maximum CPU performance:

- **Intel MKL (Math Kernel Library)**
  - Optimized BLAS, LAPACK, FFT operations
  - Automatic thread scaling
  - 3-10x speedup on Intel CPUs for numerical operations

- **Intel oneDNN (formerly MKL-DNN)**
  - Optimized convolution, pooling, normalization layers
  - Integrated with TensorFlow and PyTorch
  - Significant speedup for inference on CPUs

### **GPU Optimization**
For GPU-enabled images, NVIDIA-optimized libraries provide maximum throughput:

- **CUDA Toolkit**
  - Parallel computing platform and API
  - Pre-installed with compatible drivers
  - Versions: CUDA 11.3, 11.8, 12.0+ (depending on image)

- **cuDNN (CUDA Deep Neural Network library)**
  - GPU-accelerated primitives for DNNs
  - Optimized convolution, pooling, normalization
  - 5-10x faster than CPU equivalents

- **cuBLAS & cuSPARSE**
  - GPU-accelerated linear algebra operations
  - Dense and sparse matrix operations
  - Used by all major ML frameworks

- **TensorRT**
  - Inference optimizer and runtime
  - FP16/INT8 quantization support
  - Up to 6x faster inference than native frameworks

- **NCCL (NVIDIA Collective Communications Library)**
  - Multi-GPU communication primitives
  - Optimized all-reduce, broadcast operations
  - Essential for distributed training

### **Distributed Training**
- **Horovod** (select images)
  - Distributed deep learning training framework
  - Works with TensorFlow, PyTorch, MXNet
  - Easy scaling to multiple GPUs/nodes

- **OpenMPI**
  - Message Passing Interface implementation
  - Required for multi-node distributed training
  - Pre-configured for GCP networking

### **Performance Benefits**
| Operation Type | CPU (MKL) | GPU (cuDNN) | Speedup |
|----------------|-----------|-------------|---------|
| Matrix Multiplication | ✅ 5-10x | ✅ 50-100x | vs unoptimized |
| Convolution | ✅ 3-8x | ✅ 20-50x | vs unoptimized |
| Inference (TensorRT) | - | ✅ 3-6x | vs native GPU |
| Multi-GPU Training (NCCL) | - | ✅ 0.9x per GPU | near-linear scaling |



## Best Practices

### **Cost Optimization**
```bash
# Use preemptible instances for experimentation
gcloud compute instances create my-preemptible-vm \
    --zone=us-central1-a \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --preemptible \
    --accelerator="type=nvidia-tesla-t4,count=1"
```

### **Auto-shutdown for Idle VMs**
```bash
# Set idle shutdown (stops after 60 min idle)
gcloud compute instances create my-auto-shutdown-vm \
    --zone=us-central1-a \
    --image-family=tf-ent-latest-gpu \
    --image-project=deeplearning-platform-release \
    --metadata="idle-timeout-seconds=3600"
```

### **Persistent Disk for Data**
```bash
# Create with larger persistent disk
gcloud compute instances create my-ml-vm \
    --zone=us-central1-a \
    --image-family=common-cu113 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd
```

## Integration with Vertex AI

### **Vertex AI Workbench (Managed Notebooks)**
- Serverless Jupyter environment
- Automatically managed infrastructure
- Built-in version control and security
- Based on DLVM images

### **Vertex AI Training**
- Use DLVM containers for custom training
- Pre-built training containers available
- Scales to multiple GPUs/TPUs

## Access Methods

### **1. SSH**
```bash
gcloud compute ssh my-ml-vm --zone=us-central1-a
```

### **2. JupyterLab (Port Forwarding)**
```bash
gcloud compute ssh my-ml-vm --zone=us-central1-a \
    -- -L 8080:localhost:8080
# Access: http://localhost:8080
```

### **3. VS Code Remote**
- Use Remote-SSH extension
- Direct browser access via IAP

## Advantages

✅ **Quick Setup**: Minutes instead of hours  
✅ **Optimized**: Pre-tuned for ML workloads with Intel MKL, NVIDIA cuDNN, TensorRT  
✅ **Performance**: 5-10x speedup with optimized libraries vs vanilla installations  
✅ **Updated**: Regular security and framework updates  
✅ **Consistent**: Same environment across team  
✅ **Integrated**: Works with GCS, BigQuery, Vertex AI  
✅ **Pre-configured**: CUDA, drivers, and GPU libraries already installed and tested  

## Disadvantages

❌ **Cost**: Running 24/7 can be expensive  
❌ **Customization**: Less flexible than custom images  
❌ **Overhead**: Includes tools you may not need  

## Pricing Considerations

- Charged for compute (VM), GPU/TPU, and storage separately
- Use committed use discounts for long-term workloads
- Consider preemptible VMs for batch processing (up to 80% discount)

---

**Pro Tip**: For production workloads, consider **Vertex AI Workbench** or **Vertex AI Training** instead of managing VMs directly.