# Multi-Label Classification Loss Functions

## 1. Binary Cross-Entropy (per label)

**Formula:** BCE = -(1/n×L) × Σᵢ Σⱼ [yᵢⱼ log(ŷᵢⱼ) + (1-yᵢⱼ) log(1-ŷᵢⱼ)]

**Pros:**
- Natural extension of binary classification to multi-label
- Each label treated independently
- Works well with sigmoid activation per label
- Simple and well-understood
- Supports partial label scenarios
- Easy to implement and optimize

**Cons:**
- Ignores label correlations
- Assumes label independence (often violated)
- Sensitive to label imbalance per label
- Can be dominated by frequent labels
- Doesn't model label co-occurrence

**When to Use:**
- Default choice for multi-label classification
- When labels are relatively independent
- With sigmoid outputs for each label
- When you need probability estimates per label
- In text classification, image tagging tasks

**When NOT to Use:**
- When label correlations are strong and important (use structured prediction losses)
- With extreme label imbalance per class (consider Focal Loss or Asymmetric Loss)
- If computational efficiency is critical with thousands of labels (may need hierarchical approaches)
- When you need to model label dependencies explicitly (use conditional models)
- For ordinal multi-label problems where label relationships matter

---

## 2. Hamming Loss

**Formula:** Hamming = (1/n×L) × Σᵢ Σⱼ 1(yᵢⱼ ≠ ŷᵢⱼ)

**Pros:**
- Directly measures label prediction errors
- Simple and interpretable
- Treats all label errors equally
- Easy to understand for non-technical stakeholders
- Symmetric metric

**Cons:**
- Not differentiable (can't be used directly for training)
- Doesn't provide gradient information
- Treats all labels equally (no weighting)
- Ignores severity of errors
- Typically used as evaluation metric, not loss

**When to Use:**
- As an evaluation metric, not training loss
- When you want to count incorrect label predictions
- For reporting model performance
- When all labels are equally important
- In combination with differentiable training loss

**When NOT to Use:**
- For training neural networks (not differentiable - use BCE or other differentiable losses)
- When label importance varies (all errors weighted equally)
- If you need gradient-based optimization (no gradients available)
- When probability estimates are needed (only counts binary errors)
- As the sole metric when label imbalance exists (doesn't account for it)

---

## 3. Micro/Macro F1 Loss

**Formula:** 
- Micro F1: Aggregates TP, FP, FN across all labels
- Macro F1: Averages F1 scores per label

**Pros:**
- Balances precision and recall
- Handles class imbalance better than accuracy
- Macro version gives equal weight to all labels
- Micro version emphasizes common labels
- Comprehensive performance measure

**Cons:**
- Not inherently differentiable (requires approximations)
- Complex to optimize directly
- Can be unstable during training
- Requires threshold selection for binary predictions
- Computational overhead

**When to Use:**
- As an evaluation metric primarily
- With differentiable approximations for training
- When you need balanced precision-recall
- For imbalanced multi-label datasets
- When label performance matters equally (macro) or proportionally (micro)

**When NOT to Use:**
- For direct gradient-based training without approximations (not differentiable)
- When computational efficiency is critical (complex to compute)
- If you only care about ranking, not F1 specifically (use ranking losses)
- With very imbalanced labels where micro F1 might be misleading
- When simple BCE already gives good results (added complexity may not help)

---

## 4. Ranking Loss

**Formula:** L = Σᵢ Σⱼ₊ Σⱼ₋ max(0, margin - (ŷᵢⱼ₊ - ŷᵢⱼ₋))

**Pros:**
- Focuses on ranking relevant labels higher
- Handles label imbalance naturally
- Considers label ordering
- Good for retrieval-like scenarios
- Doesn't require exact probabilities

**Cons:**
- More complex than BCE
- Requires sampling positive/negative pairs
- Computational cost grows with label pairs
- Margin hyperparameter needs tuning
- May converge slower

**When to Use:**
- When label ranking is more important than exact predictions
- In information retrieval scenarios
- For recommendation systems with multi-label outputs
- When you care about ordering relevant labels
- In extreme multi-label classification

**When NOT to Use:**
- When you need calibrated probabilities (focuses on ranking, not calibration)
- For small-scale multi-label problems (BCE is simpler)
- If computational resources are limited (pair-wise comparisons are expensive)
- When absolute predictions matter more than relative ordering
- With datasets where all labels are equally frequent (no ranking needed)

---

## 5. Asymmetric Loss

**Formula:** L = Σ [L₊(positive) + L₋(negative)] with different weightings

**Pros:**
- Addresses positive-negative imbalance
- Focuses on hard negatives
- Reduces false positives
- Designed for multi-label with many negative labels
- Improves precision without sacrificing recall

**Cons:**
- Requires careful hyperparameter tuning
- More complex than standard BCE
- May need different weights per dataset
- Relatively new (less tested)
- Implementation varies

**When to Use:**
- When negative labels vastly outnumber positives
- In image tagging with sparse labels
- When false positives are costly
- For extreme multi-label problems (many possible labels)
- When standard BCE underperforms due to imbalance

**When NOT to Use:**
- When labels are relatively balanced (BCE is simpler)
- If you lack resources for extensive hyperparameter tuning
- For problems where false negatives are more critical than false positives
- When using pre-trained models not designed for asymmetric loss
- If your dataset is small and complex loss may cause overfitting
