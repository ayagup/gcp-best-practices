# Time Series Loss Functions

## 1. Temporal Difference Loss

**Formula:** L = Σₜ [rₜ + γV(sₜ₊₁) - V(sₜ)]²

**Pros:**
- Fundamental to reinforcement learning
- Learns value functions from experience
- No model of environment needed
- Can learn online (incremental)
- Bootstrap from current estimates
- Balances bias and variance

**Cons:**
- Can be unstable with function approximation
- Requires careful tuning of γ (discount factor)
- May have high variance
- Convergence not guaranteed in all cases
- Sensitive to learning rate
- Can diverge with off-policy learning

**When to Use:**
- Reinforcement learning value estimation
- Q-learning and SARSA algorithms
- Time series prediction with temporal dependencies
- When learning from sequential data
- Model-free RL
- Online learning scenarios

**When NOT to Use:**
- With function approximation without careful stabilization (can diverge)
- For non-sequential supervised learning (not designed for it)
- When off-policy learning causes instability (may need importance sampling)
- Without proper discount factor γ tuning (critical hyperparameter)
- For tasks where model-based methods are more sample-efficient

---

## 2. Dynamic Time Warping (DTW) Loss

**Formula:** DTW(X,Y) = minimum cumulative distance with alignment

**Pros:**
- Handles sequences of different lengths
- Allows non-linear alignment
- Robust to time shifts and distortions
- Works with temporal variations
- Natural for comparing time series
- No strict time correspondence needed

**Cons:**
- Computationally expensive O(n²)
- Not differentiable (requires soft-DTW approximation)
- Can overfit to training alignments
- Requires careful distance metric choice
- Memory intensive for long sequences
- Can create unrealistic alignments

**When to Use:**
- Speech recognition
- Gesture recognition
- Time series clustering
- When sequences have temporal variations
- Comparing time series of different lengths
- Activity recognition

**When NOT to Use:**
- For gradient-based optimization (not differentiable - use soft-DTW)
- With very long sequences (O(n²) complexity is prohibitive)
- When sequences are already aligned (simpler losses work)
- For real-time applications (computationally expensive)
- With strict memory constraints (memory intensive)

---

## 3. Dilate Loss

**Formula:** Combination of shape and temporal losses

**Pros:**
- Designed specifically for time series forecasting
- Considers both shape and timing
- Better than MSE for capturing temporal patterns
- Balances point-wise and shape accuracy
- Works with DTW-based components
- Proven in forecasting competitions

**Cons:**
- More complex than standard losses
- Computationally expensive
- Requires tuning of α (balance parameter)
- Implementation complexity
- Relatively new (less tested)
- May be overkill for simple forecasts

**When to Use:**
- Time series forecasting
- When timing and shape both matter
- Financial forecasting
- Energy consumption prediction
- When MSE fails to capture temporal structure
- Multi-step ahead forecasting

**When NOT to Use:**
- For simple forecasting where MSE works well (unnecessary complexity)
- When computational efficiency is critical (expensive computation)
- Without proper α tuning (balance between shape and temporal components)
- For single-step forecasting where timing is less critical
- When implementation resources are limited (complex to implement)

---

## 4. Shape Loss

**Formula:** Focuses on similarity of time series shapes

**Pros:**
- Emphasizes pattern similarity
- Robust to scaling and offsets
- Good for morphological matching
- Captures structural features
- Works with normalized data
- Useful for pattern recognition

**Cons:**
- May ignore absolute values
- Requires definition of "shape"
- Can be sensitive to noise
- May not care about magnitude
- Implementation varies
- Usually needs combination with other losses

**When to Use:**
- When pattern matters more than magnitude
- ECG signal analysis
- Shape-based time series retrieval
- When comparing normalized series
- Pattern recognition in signals
- Combined with magnitude-aware losses

**When NOT to Use:**
- When absolute values are critical (ignores magnitude)
- For tasks where scale matters (focuses only on shape)
- Without combining with magnitude losses (incomplete objective)
- When noise significantly affects shape (sensitive to noise)
- For well-defined problems where standard losses work

---

## 5. Temporal Loss

**Formula:** Penalizes temporal inconsistencies

**Pros:**
- Enforces temporal smoothness
- Reduces jittery predictions
- Maintains temporal coherence
- Works for video and sequence data
- Simple to implement
- Improves visual quality

**Cons:**
- Can over-smooth predictions
- May miss abrupt changes
- Hyperparameter tuning needed
- Can conflict with accuracy
- May reduce responsiveness
- Not suitable for all temporal data

**When to Use:**
- Video prediction
- Video generation
- When temporal smoothness is desired
- Optical flow estimation
- Frame interpolation
- Removing temporal artifacts
- Combined with per-frame losses

**When NOT to Use:**
- For time series with abrupt changes (over-smooths)
- When rapid temporal dynamics are important (reduces responsiveness)
- As standalone loss without per-frame accuracy (conflicts with accuracy)
- For non-visual temporal data where smoothness isn't desired
- When ground truth has natural jitter that should be preserved
