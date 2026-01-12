# Pytorch

PyTorch provides several sophisticated methods for distributed training, each designed to solve specific bottlenecks such as dataset size or model memory constraints.

Distributed training in PyTorch is generally categorized into **Data Parallelism** (replicating the model to process more data) and **Model Parallelism** (splitting the model when it's too big to fit on one GPU).

### 1. Data Parallelism Methods

These methods are used when your model fits on a single GPU, but you want to speed up training by using multiple GPUs to process different parts of your dataset simultaneously.

* **DataParallel (DP):** A simple, single-process, multi-threaded approach. It replicates the model on each GPU and shards the data. However, it suffers from a "master-GPU bottleneck" because one GPU must aggregate all gradients, which slows down scaling.
* **DistributedDataParallel (DDP):** The recommended industry standard. It runs a separate process for each GPU, avoiding Python's Global Interpreter Lock (GIL). It uses an efficient **All-Reduce** algorithm to synchronize gradients across GPUs, making it much faster and more scalable than DP.
* **Fully Sharded Data Parallel (FSDP):** A more advanced version of data parallelism. Instead of keeping a full copy of the model on every GPU, it "shards" (splits) the model parameters, gradients, and optimizer states across all GPUs. It only reconstructs the full layers "on-the-fly" during the forward and backward pass, allowing you to train models that are far larger than a single GPU's memory.

---

### 2. Model Parallelism Methods

These methods are necessary when the model itself is too large for one GPU's memory.

* **Tensor Parallelism (TP):** This shards individual tensors (like a large weight matrix in a Linear layer) across multiple GPUs. Each GPU computes a portion of the matrix multiplication.
* **Pipeline Parallelism (PP):** This splits the model "vertically" by placing different layers on different GPUs. To keep the GPUs busy, it uses "micro-batches" so that while the second GPU is working on the first micro-batch's 2nd layer, the first GPU can start on the second micro-batch's 1st layer.
* **RPC-based Training:** A flexible framework for distributed training where you can manually define how parts of your model are distributed and how they communicate across different nodes using Remote Procedure Calls.

---

### Comparison Table: PyTorch Distribution Strategies

| Method | Primary Goal | **Data Parallel?** | **Model Parallel?** | Hardware Support |
| --- | --- | --- | --- | --- |
| **DataParallel (DP)** | Simple multi-GPU speedup | **Yes** (Splits batches) | **No** (Full replica) | Single Machine |
| **DistributedDataParallel (DDP)** | Scalable multi-GPU/Node training | **Yes** (Splits batches) | **No** (Full replica) | Multi-Node / Cluster |
| **Fully Sharded (FSDP)** | Training massive models (7B+ params) | **Yes** (Splits batches) | **Yes** (Shards parameters) | Multi-Node / Cluster |
| **Tensor Parallel (TP)** | Splitting giant layers | **No** | **Yes** (Shards tensors) | High-speed Interconnect |
| **Pipeline Parallel (PP)** | Splitting model by layers | **No** | **Yes** (Shards by layers) | Multi-GPU |
| **RPC Framework** | General distributed logic | Flexible | **Yes** (Shards arbitrarily) | Multi-Node |

---

### Hybrid Strategy: 3D Parallelism

For the largest models (like GPT-4 or Llama-3), researchers often use a combination of these strategies, known as **3D Parallelism**. This involves:

1. **Pipeline Parallelism** to split layers across nodes.
2. **Tensor Parallelism** to split large layers within a node.
3. **Data Parallelism (DDP/FSDP)** to replicate the entire pipeline across multiple sets of GPUs.

**Would you like to see a code example of how to implement `DistributedDataParallel` for a standard multi-GPU training job?**




# Tensorflow

To understand how data and models are handled, we need to distinguish between **Data Parallelism** (splitting the data) and **Model Parallelism** (splitting the model). In almost all standard TensorFlow strategies, **Data Parallelism** is the default behavior.

Here is the expanded comparison table:

### Distribution Strategy Comparison

| Strategy | Hardware | Sync/Async | **Data Split?** | **Model Split?** |
| --- | --- | --- | --- | --- |
| **MirroredStrategy** | 1 Machine, Multi-GPU | Synchronous | **Yes** (Each GPU gets a batch slice) | **No** (Full model replicated on every GPU) |
| **MultiWorkerMirrored** | Multi-Machine, Multi-GPU | Synchronous | **Yes** (Data split across all workers) | **No** (Full model replicated on every worker) |
| **TPUStrategy** | TPU Pods / Slices | Synchronous | **Yes** (Data split across TPU cores) | **No** (Standard use replicates model) |
| **ParameterServer** | Multi-Machine | Asynchronous | **Yes** (Workers process data slices) | **Partial** (Variables reside on PS; compute on Workers) |
| **CentralStorage** | 1 Machine, Multi-GPU | Synchronous | **Yes** (Data split across GPUs) | **No** (Model variables stored on CPU) |

---

### Key Concepts: Data vs. Model Splitting

#### 1. Data Parallelism (The "Default")

This is what happens in `MirroredStrategy`. If you have a global batch size of **128** and **4 GPUs**, TensorFlow automatically splits the data so that each GPU processes **32** samples.

* **How it works:** Each GPU has a complete copy of the model. They compute gradients on their specific slice of data and then sync up to update the weights.
* **When to use:** When your model fits on a single GPU's memory, but your dataset is large.

#### 2. Model Parallelism (Sharding)

TensorFlow does not use the standard `Strategy` API for model splitting. Instead, you would use **DTensor** or **Mesh TensorFlow**.

* **How it works:** The model itself is too big for one GPU (e.g., a massive LLM). Different layers or different parts of a large weight matrix are placed on different GPUs.
* **When to use:** When the model is so large (billions of parameters) that it cannot fit into the VRAM of a single accelerator.

#### 3. The Parameter Server Exception

`ParameterServerStrategy` is a hybrid. While it is technically data-parallel (workers handle different data), the **model variables** are moved off the compute nodes and onto dedicated "Parameter Server" nodes. This allows you to have a model with massive embedding tables that might exceed the memory of a single worker node.

**Would you like me to explain how to handle "Model Parallelism" specifically, or are you focusing on scaling standard models for the exam?**