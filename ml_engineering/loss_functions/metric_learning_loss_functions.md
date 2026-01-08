# Metric Learning Loss Functions

## 1. Mahalanobis Distance Loss

**Formula:** d(x,y) = √[(x-y)ᵀM(x-y)] where M is learned metric

**Pros:**
- Learns task-specific distance metric
- Generalizes Euclidean distance
- Can capture feature correlations
- Theoretically grounded
- Flexible distance representation
- Works with feature weighting
- Interpretable metric matrix

**Cons:**
- Requires learning positive semi-definite matrix M
- Computationally expensive (matrix operations)
- Can be unstable during training
- Overfitting risk with high dimensions
- Requires careful regularization
- Scaling issues with large feature spaces
- May be overkill for simple tasks

**When to Use:**
- Metric learning tasks
- When Euclidean distance is insufficient
- Feature space has correlations
- Classification with distance-based methods
- k-NN improvement
- When you need interpretable metric
- Clustering with learned metrics

**When NOT to Use:**
- With very high-dimensional features (computational cost and overfitting risk)
- When Euclidean distance already works well (added complexity not justified)
- For simple tasks not requiring feature correlation modeling
- With limited training data (risk of overfitting the metric matrix)
- When computational efficiency is paramount (matrix operations are expensive)

---

## 2. Cosine Embedding Loss

**Formula:** L = 1 - cos(x,y) = 1 - (x·y)/(||x|| ||y||) for similar pairs

**Pros:**
- Measures angular similarity
- Normalized by vector magnitudes
- Good for high-dimensional data
- Widely used in NLP (word embeddings)
- Simple and efficient
- Invariant to scaling
- Works well for text and embeddings

**Cons:**
- Ignores magnitude information
- May not be suitable for all domains
- Only considers angle, not distance
- Can be less effective than learned metrics
- Binary similarity assumption
- May need margin tuning

**When to Use:**
- Text similarity
- Word embeddings (Word2Vec, etc.)
- Document similarity
- Face verification
- Siamese networks for similarity
- When magnitude is not important
- High-dimensional embedding spaces

**When NOT to Use:**
- When magnitude information is critical (cosine ignores it)
- For low-dimensional data where Euclidean works better
- When you need to distinguish between "close but different scale" vs "truly similar"
- If binary similarity assumption is too restrictive
- When angular similarity doesn't match your task semantics

---

## 3. Euclidean Distance Loss

**Formula:** L = ||x - y||² for similar pairs (with margin for dissimilar)

**Pros:**
- Simple and intuitive
- Computationally efficient
- Well-understood properties
- Works in many scenarios
- Easy to implement
- Standard distance metric
- No additional parameters

**Cons:**
- Assumes all dimensions equally important
- Not invariant to scaling
- Can be dominated by large-scale features
- May not capture complex relationships
- Sensitive to outliers
- Curse of dimensionality
- May need feature normalization

**When to Use:**
- Simple metric learning
- Low-dimensional feature spaces
- When features are normalized
- As baseline for comparison
- Contrastive learning
- Siamese networks (simple version)
- When computational efficiency is critical

**When NOT to Use:**
- With unnormalized features of different scales (can be dominated by large-scale features)
- In very high-dimensional spaces (curse of dimensionality)
- When feature correlations are important (assumes independence)
- If outliers are prevalent (sensitive to outliers)
- When learned metrics (Mahalanobis) provide significant improvement

---

## 4. Manhattan Distance Loss

**Formula:** L = Σᵢ |xᵢ - yᵢ|

**Pros:**
- More robust to outliers than Euclidean
- Computationally simple
- Works well in high dimensions sometimes
- Natural for grid-like spaces
- Differentiable (with smoothing)
- Sparse-friendly
- Less sensitive to large differences

**Cons:**
- Less commonly used than Euclidean
- May not be natural for all feature spaces
- Not as smooth as Euclidean (at zero)
- Less theoretical support in metric learning
- May need smoothing for gradient descent
- Can be less effective than learned metrics
- Not rotation-invariant

**When to Use:**
- Grid-based representations
- When robustness to outliers is important
- High-dimensional sparse data
- As alternative to Euclidean
- City-block distance applications
- When L1 properties are desired
- Taxi-cab geometry scenarios

**When NOT to Use:**
- When Euclidean distance is more natural for your feature space
- For tasks requiring rotation-invariant metrics
- When smoothness at zero is critical (requires smoothing for gradients)
- If Euclidean already provides good results (Manhattan is less standard)
- When theoretical justification for metric choice is important
