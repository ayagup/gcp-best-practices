# Regularization Loss Functions

## 1. L1 Regularization (Lasso)

**Formula:** L = Loss + λ Σᵢ |wᵢ|

**Pros:**
- Induces sparsity (feature selection)
- Sets some weights to exactly zero
- Automatic feature selection
- Interpretable models
- Reduces model complexity
- Prevents overfitting

**Cons:**
- Not differentiable at zero
- Can be unstable when features are correlated
- May arbitrarily select one feature from correlated group
- Biased estimates for large coefficients
- Can lead to slower convergence
- Sensitive to λ parameter

**When to Use:**
- High-dimensional data with many irrelevant features
- When you want feature selection
- Sparse model requirements
- When interpretability is important
- Linear models with many features
- Compressing neural networks

**When NOT to Use:**
- When all features are relevant (L2 is better)
- With highly correlated features (unstable selection, use Elastic Net)
- For smooth optimization requirements (not differentiable at zero)
- When you need all features and just want to prevent large weights (use L2)
- For tasks where feature selection is not needed or desired

---

## 2. L2 Regularization (Ridge)

**Formula:** L = Loss + λ Σᵢ wᵢ²

**Pros:**
- Smooth and differentiable everywhere
- Prevents large weights
- More stable than L1
- Handles correlated features better
- Well-suited for gradient descent
- Reduces overfitting effectively

**Cons:**
- Doesn't induce sparsity
- Keeps all features (no selection)
- Can shrink important weights too much
- May not work well with irrelevant features
- Requires tuning of λ
- No automatic feature selection

**When to Use:**
- Default regularization choice for most models
- Neural network training
- When all features might be relevant
- Preventing overfitting without feature selection
- Ridge regression
- Deep learning (weight decay)

**When NOT to Use:**
- When you need feature selection (use L1 instead)
- For high-dimensional data with many irrelevant features (L1 or Elastic Net better)
- When sparse models are required
- If shrinking important weights is problematic for your task
- When interpretability via feature selection is crucial

---

## 3. Elastic Net

**Formula:** L = Loss + λ₁ Σᵢ |wᵢ| + λ₂ Σᵢ wᵢ²

**Pros:**
- Combines L1 and L2 benefits
- Feature selection with stability
- Handles correlated features better than L1
- Can select groups of correlated features
- More flexible than pure L1 or L2
- Good for high-dimensional data

**Cons:**
- Two hyperparameters to tune (λ₁, λ₂)
- More complex than L1 or L2 alone
- Computational overhead
- Requires careful hyperparameter selection
- Can be harder to interpret
- May be overkill for simple problems

**When to Use:**
- High-dimensional data with correlated features
- When you want both sparsity and stability
- Genomics and bioinformatics
- When L1 is too unstable
- Feature selection with grouped variables
- As compromise between L1 and L2

**When NOT to Use:**
- When simple L1 or L2 works well (unnecessary complexity)
- For tasks without correlated features (L1 or L2 alone may suffice)
- With limited hyperparameter tuning resources (two λ parameters to tune)
- For simple, low-dimensional problems (overkill)
- When computational efficiency is critical (more expensive than L1 or L2 alone)

---

## 4. Dropout Loss

**Formula:** L = Loss computed with randomly dropped units

**Pros:**
- Prevents co-adaptation of neurons
- Effective regularization for neural networks
- Acts as ensemble method
- Simple to implement
- Widely proven in deep learning
- Improves generalization significantly

**Cons:**
- Not applicable to all model types
- Increases training time (more epochs needed)
- Requires tuning dropout rate
- Can hurt performance if rate too high
- Only active during training
- May need different rates for different layers

**When to Use:**
- Neural network training
- Deep learning models
- When model is overfitting
- Computer vision and NLP models
- As standard practice in deep networks
- Large neural networks with many parameters

**When NOT to Use:**
- For shallow models or traditional ML (not designed for them)
- When training time is extremely limited (slows convergence)
- With very small networks that aren't overfitting
- For models that already generalize well without it
- When inference time consistency is critical (behavior differs between train/test)

---

## 5. Orthogonality Loss

**Formula:** L = Loss + λ ||W^T W - I||²

**Pros:**
- Encourages orthogonal weight matrices
- Prevents gradient vanishing/exploding
- Improves training stability
- Better conditioning of weight matrices
- Useful in RNNs and transformers
- Maintains information flow

**Cons:**
- Additional computational cost
- Not always necessary
- May conflict with primary objective
- Requires matrix operations
- Hyperparameter tuning needed
- Can slow down convergence

**When to Use:**
- Recurrent Neural Networks (RNNs)
- Deep networks with gradient issues
- When training stability is a problem
- Transformer models
- GANs (in discriminator)
- Preventing mode collapse in generators

**When NOT to Use:**
- For shallow networks without gradient issues (unnecessary overhead)
- When training is already stable (adds computational cost without benefit)
- With models that don't have vanishing/exploding gradient problems
- For simple feedforward networks (typically not needed)
- When computational budget is tight (matrix operations are expensive)

---

## 6. Diversity Loss

**Formula:** L = Loss - λ × Diversity(predictions/features)

**Pros:**
- Encourages diverse outputs
- Prevents mode collapse
- Useful in ensemble methods
- Improves exploration
- Better coverage of output space
- Enhances model robustness

**Cons:**
- Can conflict with accuracy objective
- Requires careful weighting
- Implementation varies by application
- May reduce individual prediction quality
- Hyperparameter sensitivity
- Not always well-defined

**When to Use:**
- GANs (preventing mode collapse)
- Ensemble learning
- Multi-agent systems
- Beam search in NLP
- When output diversity is important
- Recommendation systems
- Neural architecture search

**When NOT to Use:**
- When accuracy/precision is the sole objective (may hurt individual predictions)
- For tasks where diversity is not valued or measured
- Without careful tuning of diversity weight (can conflict with primary loss)
- When diversity metric is not well-defined for your problem
- For single-model predictions where diversity doesn't apply
