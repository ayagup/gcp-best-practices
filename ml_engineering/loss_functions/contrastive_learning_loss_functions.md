# Contrastive Learning Loss Functions

## 1. InfoNCE Loss

**Formula:** L = -log[exp(sim(q,k₊)/τ) / Σ exp(sim(q,kᵢ)/τ)]

**Pros:**
- Foundation of many contrastive methods
- Maximizes mutual information
- Works with large negative sets
- Scalable to large datasets
- Theoretically grounded in information theory
- Proven effective for representation learning

**Cons:**
- Requires large batch sizes or memory bank
- Sensitive to temperature parameter τ
- Computational cost with many negatives
- Needs careful negative sampling
- Can be biased by false negatives
- Memory intensive

**When to Use:**
- Self-supervised learning
- Contrastive Predictive Coding (CPC)
- When you have positive and negative pairs
- Representation learning
- Vision and language pretraining
- Foundation for other contrastive methods

**When NOT to Use:**
- With very small batch sizes (needs many negatives)
- Without careful temperature tuning (τ is critical)
- When memory is severely limited (memory bank or large batches required)
- For datasets with many false negatives (biases learning)
- Without computational resources for many negative samples

---

## 2. SimCLR Loss

**Formula:** NT-Xent (Normalized Temperature-scaled Cross Entropy)

**Pros:**
- Simple and effective framework
- Strong augmentation strategy
- Large batch training
- State-of-the-art self-supervised results
- No memory bank needed (uses batch)
- Well-documented and reproducible

**Cons:**
- Requires very large batch sizes (thousands)
- Computationally expensive
- Heavy augmentation dependence
- Needs substantial GPU memory
- Training time is long
- May not work well with small batches

**When to Use:**
- Self-supervised visual representation learning
- When you have computational resources for large batches
- Image classification pretraining
- Transfer learning
- When you want strong baseline for self-supervised learning
- Research and production CV systems

**When NOT to Use:**
- With limited GPU memory (requires batches of 256-8192)
- For quick experiments without large compute (expensive training)
- When strong augmentation is not available or appropriate
- With small datasets where large batches aren't possible
- For real-time or resource-constrained applications

---

## 3. MoCo Loss

**Formula:** Similar to InfoNCE with momentum encoder and queue

**Pros:**
- Efficient with momentum encoder
- Uses queue for large negative set
- Doesn't require huge batches
- Memory efficient compared to SimCLR
- Consistent representations via momentum
- Good for limited GPU memory

**Cons:**
- More complex implementation (queue + momentum)
- Momentum hyperparameter needs tuning
- Queue size is a hyperparameter
- Can lag behind in performance vs SimCLR (depends on version)
- Requires careful momentum update strategy

**When to Use:**
- Self-supervised learning with limited GPU memory
- When large batches are infeasible
- Object detection pretraining
- Instance discrimination tasks
- As memory-efficient alternative to SimCLR
- Video representation learning

**When NOT to Use:**
- When you have resources for large batches (SimCLR may be simpler/better)
- Without proper momentum and queue size tuning (critical hyperparameters)
- For tasks where SimCLR has proven superior with available resources
- When implementation simplicity is paramount (queue + momentum adds complexity)
- For very small-scale experiments (overhead may not be worth it)

---

## 4. Supervised Contrastive Loss

**Formula:** L = Σᵢ -1/|P(i)| Σₚ∈P(i) log[exp(zᵢ·zₚ/τ) / Σₐ∈A(i) exp(zᵢ·zₐ/τ)]

**Pros:**
- Extends contrastive learning to supervised setting
- Leverages label information
- Better than cross-entropy alone
- More robust representations
- Multiple positives per sample (same class)
- Improves generalization

**Cons:**
- Requires labels (not self-supervised)
- Still needs large batches
- More complex than standard classification
- Computational overhead
- Hyperparameter sensitivity
- May need careful batch composition

**When to Use:**
- Supervised representation learning
- When you have labels and want better features
- Combined with standard classification
- Few-shot learning
- Transfer learning with labeled data
- As alternative to standard cross-entropy

**When NOT to Use:**
- When labels are unavailable (use self-supervised methods instead)
- With small batch sizes (needs sufficient samples per class in batch)
- For tasks where standard cross-entropy works perfectly well
- When computational budget doesn't allow for contrastive overhead
- Without careful batch sampling to ensure class diversity

---

## 5. Self-Supervised Contrastive Loss

**Formula:** Various formulations (SimCLR, MoCo, BYOL, etc.)

**Pros:**
- No labels required
- Learns from data structure
- Transferable representations
- Reduces annotation cost
- Works with unlabeled data
- Increasingly competitive with supervised learning

**Cons:**
- "Self-supervised" is a broad category
- Requires careful augmentation design
- May need large datasets
- Pretraining can be expensive
- Performance depends on augmentation quality
- Different methods have different trade-offs

**When to Use:**
- When labels are scarce or expensive
- Pretraining for downstream tasks
- Learning universal representations
- Medical imaging (limited labels)
- Domain-specific representation learning
- Transfer learning scenarios

**When NOT to Use:**
- When abundant high-quality labels exist (supervised may be simpler/better)
- Without good augmentation strategies (augmentation quality is critical)
- For very small datasets (may not learn meaningful representations)
- When pretraining cost outweighs benefits for your specific task
- Without sufficient computational resources for pretraining

---

## 6. NT-Xent Loss

**Formula:** L = -Σᵢ log[exp(sim(zᵢ, zⱼ₍ᵢ₎)/τ) / Σₖ₌₁²ᴺ 1[k≠i] exp(sim(zᵢ, zₖ)/τ)]

**Pros:**
- Normalized temperature-scaled cross-entropy
- Used in SimCLR (proven effective)
- Handles positive and negative pairs naturally
- Temperature scaling provides control
- Symmetric formulation
- Well-studied and understood

**Cons:**
- Same limitations as SimCLR
- Requires large batches
- Sensitive to temperature τ
- Can be dominated by hard negatives
- Computational cost scales with batch size
- May suffer from false negatives

**When to Use:**
- SimCLR and variants
- Contrastive learning with augmentations
- Self-supervised visual learning
- When you have large batch capability
- As standard contrastive objective
- Research baselines

**When NOT to Use:**
- With small batches (requires large batches for many negatives)
- Without temperature tuning (τ significantly affects performance)
- When hard negatives dominate and hurt learning
- For tasks with many false negatives in the batch
- Without sufficient computational resources for large-scale contrastive training
