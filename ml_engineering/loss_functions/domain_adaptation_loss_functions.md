# Domain Adaptation Loss Functions

## 1. Maximum Mean Discrepancy (MMD)

**Formula:** MMD²(P,Q) = ||μₚ - μQ||²ₕ (distance in RKHS)

**Pros:**
- Measures distribution difference
- Non-parametric and flexible
- Works with kernel methods
- Theoretically grounded
- No adversarial training needed
- Can match high-order moments
- Smooth optimization

**Cons:**
- Requires kernel selection
- Computationally expensive for large datasets
- Can be sensitive to kernel bandwidth
- Quadratic complexity in sample size
- May not capture all distribution aspects
- Hyperparameter tuning needed

**When to Use:**
- Domain adaptation tasks
- Transfer learning
- Distribution matching
- When adversarial training is too unstable
- Covariate shift problems
- When you have source and target domain data
- Unsupervised domain adaptation

**When NOT to Use:**
- When you have very limited computational resources (quadratic complexity)
- For real-time applications requiring fast inference
- When kernel selection is poorly understood for your data
- With very high-dimensional data (computational cost becomes prohibitive)
- If simple moment matching (mean/variance) is sufficient for your domain shift

---

## 2. CORAL Loss

**Formula:** L = ||Cₛ - Cₜ||²F (Frobenius norm of covariance difference)

**Pros:**
- Simple and effective
- Aligns second-order statistics
- Computationally efficient
- No adversarial training
- Easy to implement
- Works well in practice
- Can be added to any network

**Cons:**
- Only matches second-order statistics
- May not capture complex distribution differences
- Assumes linear correlation is sufficient
- May not work for highly nonlinear shifts
- Limited to covariance alignment
- Doesn't consider higher moments

**When to Use:**
- Domain adaptation with covariate shift
- Transfer learning
- When computational efficiency is important
- Visual domain adaptation
- As simple baseline for domain adaptation
- When domains differ mainly in covariance
- Deep CORAL networks

**When NOT to Use:**
- When domain shift is highly nonlinear (only captures second-order statistics)
- For tasks requiring higher-order moment matching
- When domains differ in more than covariance structure
- If label distributions differ significantly (CORAL focuses on features, not labels)
- When adversarial approaches provide significantly better results

---

## 3. Domain Adversarial Loss

**Formula:** L = Ltask + λLdomain where Ldomain is adversarial

**Pros:**
- Learns domain-invariant features
- State-of-the-art domain adaptation
- Encourages feature confusion between domains
- Works with gradient reversal layer
- Flexible framework
- Proven effective in many applications
- Can handle complex domain shifts

**Cons:**
- Adversarial training can be unstable
- Requires careful balancing of λ
- May hurt task performance if too strong
- Training can be tricky
- Requires domain labels
- Can suffer from mode collapse
- Hyperparameter sensitive

**When to Use:**
- Domain Adversarial Neural Networks (DANN)
- When you have labeled source and unlabeled target
- Visual domain adaptation
- Sentiment analysis across domains
- When domains have significant shift
- Most modern domain adaptation scenarios
- Transfer learning with distribution shift

**When NOT to Use:**
- When training stability is critical (adversarial training can be unstable)
- For quick prototyping (requires careful hyperparameter tuning)
- When task performance degradation is unacceptable (strong domain confusion may hurt primary task)
- With limited data (adversarial training needs sufficient samples)
- If simpler methods like CORAL work well enough

---

## 4. Domain Confusion Loss

**Formula:** L = -H(D(f(x))) (maximize entropy of domain classifier)

**Pros:**
- Encourages domain confusion
- Simpler than full adversarial training
- No domain discriminator needed
- More stable than adversarial methods
- Easier to implement
- Can be combined with task loss directly
- Less hyperparameter sensitivity

**Cons:**
- May not be as effective as adversarial methods
- Indirect domain alignment
- May not fully align distributions
- Can hurt task performance
- Less theoretically motivated
- May need careful weighting

**When to Use:**
- As simpler alternative to domain adversarial
- When adversarial training is too unstable
- Domain adaptation with limited resources
- When you want stable training
- As auxiliary loss for domain adaptation
- Combined with other adaptation methods
- When full DANN is overkill

**When NOT to Use:**
- When maximum alignment performance is required (less effective than full adversarial)
- For complex domain shifts (indirect alignment may be insufficient)
- When you have sufficient resources for full adversarial training
- If task performance is very sensitive to confusion (can hurt primary task)
- When theoretical guarantees are important (less grounded than MMD or adversarial)
