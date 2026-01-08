# Binary Classification Loss Functions

## 1. Binary Cross-Entropy (Log Loss)

**Formula:** BCE = -(1/n) × Σ[yᵢ log(ŷᵢ) + (1 - yᵢ) log(1 - ŷᵢ)]

**Pros:**
- Standard loss for binary classification
- Probabilistic interpretation
- Well-behaved gradients
- Works well with sigmoid activation
- Penalizes confident wrong predictions heavily

**Cons:**
- Sensitive to class imbalance
- Can suffer from numerical instability (log of zero)
- Assumes independence between samples
- May converge slowly with imbalanced data

**When to Use:**
- Default choice for binary classification
- When you need probability estimates
- With neural networks using sigmoid output
- When classes are relatively balanced

**When NOT to Use:**
- With severe class imbalance (use Focal Loss or weighted BCE instead)
- When you need margin-based guarantees (use Hinge Loss)
- With noisy labels (consider robust alternatives)
- When computational efficiency is critical and you have massive datasets
- If model outputs are not properly normalized to [0,1] range

---

## 2. Hinge Loss

**Formula:** L(y, ŷ) = Σ max(0, 1 - yᵢ × ŷᵢ) where y ∈ {-1, 1}

**Pros:**
- Maximum margin principle (SVM)
- Encourages confident predictions
- Sparse solutions
- Robust to outliers (plateaus for large margins)
- Focuses on decision boundary

**Cons:**
- Not differentiable at y × ŷ = 1
- Doesn't provide probability estimates
- Requires labels in {-1, 1} format
- Zero gradient for correctly classified samples with margin > 1

**When to Use:**
- Training Support Vector Machines
- When you want maximum margin classification
- When you don't need probability estimates
- For linearly separable or nearly separable data

**When NOT to Use:**
- When you need probability estimates (use BCE instead)
- With highly imbalanced datasets (no built-in imbalance handling)
- When training deep neural networks (gradients vanish for large margins)
- If you need smooth gradients everywhere (use Squared Hinge instead)
- With noisy data containing many outliers close to the boundary

---

## 3. Squared Hinge Loss

**Formula:** L(y, ŷ) = Σ max(0, 1 - yᵢ × ŷᵢ)²

**Pros:**
- Differentiable everywhere
- Smoother than hinge loss
- Still encourages large margins
- Better for gradient-based optimization
- Penalizes violations more heavily

**Cons:**
- More sensitive to outliers than hinge loss
- Larger gradients for misclassified samples
- May over-penalize difficult samples
- Less robust than standard hinge

**When to Use:**
- When you need differentiable hinge loss
- For neural network training with margin-based objectives
- When using gradient descent optimization
- As an alternative to standard hinge in deep learning

**When NOT to Use:**
- With datasets containing many outliers (overly sensitive to them)
- When you need probability calibration (no probabilistic interpretation)
- For extremely imbalanced datasets without weighting
- If training stability is an issue (large gradients can cause instability)
- When standard hinge loss works well (simpler is often better)

---

## 4. Binary Focal Loss

**Formula:** FL = -(1/n) × Σ αᵢ(1 - pᵢ)^γ log(pᵢ)

**Pros:**
- Specifically designed for class imbalance
- Down-weights easy examples
- Focuses on hard-to-classify samples
- Reduces effect of well-classified examples
- Improves performance on minority class

**Cons:**
- Requires tuning of γ and α hyperparameters
- More complex than standard cross-entropy
- Can be harder to optimize
- May overfit on small datasets

**When to Use:**
- With highly imbalanced datasets
- In object detection tasks
- When easy examples dominate training
- When minority class performance is critical

**When NOT to Use:**
- With balanced datasets (adds unnecessary complexity)
- When you have limited data for hyperparameter tuning (γ and α need careful tuning)
- If computational resources are constrained (more expensive than BCE)
- With very noisy labels (over-focuses on hard/noisy examples)
- When standard BCE with class weights works adequately

---

## 5. Exponential Loss

**Formula:** L(y, ŷ) = Σ exp(-yᵢ × ŷᵢ) where y ∈ {-1, 1}

**Pros:**
- Theoretical foundation in AdaBoost
- Strongly penalizes misclassifications
- Encourages confident predictions
- Works well with ensemble methods

**Cons:**
- Extremely sensitive to outliers and label noise
- Can lead to overfitting
- Exponentially increasing penalty
- Not robust to mislabeled data

**When to Use:**
- In AdaBoost and boosting algorithms
- When data is clean and well-labeled
- For ensemble learning methods
- When you can afford to focus heavily on hard examples

**When NOT to Use:**
- With noisy or mislabeled data (extremely sensitive to label noise)
- When outliers are present (exponential penalty can dominate)
- For deep neural network training (gradients can explode)
- With imbalanced datasets without careful weighting
- When robustness is a priority over theoretical optimality

---

## 6. Logistic Loss

**Formula:** L(y, ŷ) = Σ log(1 + exp(-yᵢ × ŷᵢ)) where y ∈ {-1, 1}

**Pros:**
- Equivalent to binary cross-entropy with different formulation
- Probabilistic interpretation
- Smooth and convex
- Well-studied theoretical properties
- Standard for logistic regression

**Cons:**
- Sensitive to outliers
- Can be affected by class imbalance
- Never reaches zero loss
- May give too much weight to outliers

**When to Use:**
- In logistic regression models
- When you need probability estimates
- For convex optimization problems
- As an alternative formulation to BCE

**When NOT to Use:**
- With data containing significant outliers (not robust)
- When you need margin-based classification (use Hinge Loss)
- For highly imbalanced data without proper weighting
- If you prefer the [0,1] label format (use standard BCE instead)
- When working with modern deep learning frameworks (BCE is more standard)

---

## 7. Perceptron Loss

**Formula:** L(y, ŷ) = Σ max(0, -yᵢ × ŷᵢ) where y ∈ {-1, 1}

**Pros:**
- Simple and intuitive
- Computationally efficient
- Only penalizes misclassifications
- Historical significance (original neural network loss)
- Zero loss for correct classifications

**Cons:**
- No margin enforcement
- Not differentiable at zero
- Doesn't encourage confident predictions
- Can lead to solutions close to decision boundary
- Obsolete compared to modern losses

**When to Use:**
- Educational purposes (understanding perceptrons)
- Extremely simple classification tasks
- When computational resources are very limited
- Rarely used in modern practice

**When NOT to Use:**
- For any modern machine learning application (vastly superior alternatives exist)
- When you need probability estimates (no probabilistic interpretation)
- When you want margin enforcement (use Hinge Loss)
- For deep learning (insufficient gradient signal)
- With noisy or overlapping classes (no robustness properties)

---

## 8. Modified Huber Loss

**Formula:** Combines squared hinge for small errors and linear hinge for large errors

**Pros:**
- Robust to outliers
- Smooth variant of hinge loss
- Combines benefits of squared loss and hinge loss
- Less sensitive to far outliers than squared hinge
- Suitable for noisy data

**Cons:**
- More complex than standard hinge
- Less commonly used
- Requires understanding of both hinge and squared loss
- Not as well-supported in libraries

**When to Use:**
- When data contains outliers
- As a robust alternative to squared hinge
- In sklearn SGDClassifier with modified_huber loss
- When you want hinge-like behavior with outlier robustness

**When NOT to Use:**
- When you need probability estimates (designed for margin-based classification)
- If your data is clean without outliers (standard hinge or BCE is simpler)
- For deep learning frameworks where it may not be readily available
- When interpretability is important (complex hybrid behavior)
- With highly imbalanced data (no specific imbalance handling)

---

## 9. Sigmoid Focal Loss

**Formula:** Variant of focal loss specifically for sigmoid activations

**Pros:**
- Combines focal loss benefits with sigmoid stability
- Handles class imbalance well
- Reduces gradient from easy examples
- Better numerical stability than softmax version
- Directly works with binary outputs

**Cons:**
- Requires hyperparameter tuning (γ, α)
- More complex than BCE
- Computational overhead
- May need careful initialization

**When to Use:**
- In object detection with binary labels
- For severely imbalanced binary classification
- When using sigmoid activation in final layer
- As an improvement over standard focal loss for binary tasks

**When NOT to Use:**
- With balanced datasets (unnecessary complexity)
- When you lack resources for hyperparameter tuning (γ and α are critical)
- For small datasets (may overfit due to complexity)
- If standard weighted BCE solves your imbalance problem
- When training time is a constraint (slower than BCE)
