# Regression Loss Functions

## 1. Mean Squared Error (MSE)

**Formula:** MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

**Pros:**
- Differentiable everywhere, making it suitable for gradient-based optimization
- Heavily penalizes large errors due to squaring
- Simple to understand and implement
- Widely supported in all ML frameworks

**Cons:**
- Very sensitive to outliers (errors are squared)
- Not robust to extreme values
- Loss values can grow very large
- Not in the same unit as the target variable

**When to Use:**
- When you want to heavily penalize large errors
- When outliers are rare or already handled
- When your target distribution is approximately Gaussian
- As the default choice for most regression problems

**When NOT to Use:**
- When your data contains many outliers or extreme values
- When small and large errors should be treated more equally
- When the target variable spans multiple orders of magnitude
- When you need the loss in interpretable units
- For heavy-tailed or non-Gaussian distributions

---

## 2. Root Mean Squared Error (RMSE)

**Formula:** RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]

**Pros:**
- In the same units as the target variable
- Interpretable compared to MSE
- Still differentiable
- Penalizes large errors

**Cons:**
- Sensitive to outliers
- Slightly more computationally expensive than MSE
- Gradient can be unstable near zero

**When to Use:**
- When you need loss in the same units as predictions
- For interpretable error metrics
- When comparing models across different scales
- When you want to penalize large errors but need interpretability

**When NOT to Use:**
- When your data has significant outliers
- When computational efficiency is critical (use MSE instead)
- When predictions approach zero (gradient instability)
- For heavy-tailed error distributions
- When you need equal treatment of all error sizes

---

## 3. Mean Absolute Error (MAE)

**Formula:** MAE = (1/n) × Σ|yᵢ - ŷᵢ|

**Pros:**
- Robust to outliers (linear penalty)
- Easy to understand and interpret
- In the same units as the target variable
- Less sensitive to extreme values than MSE

**Cons:**
- Not differentiable at zero (can cause optimization issues)
- Treats all errors equally (doesn't emphasize large errors)
- Can lead to slower convergence
- Gradient is constant, which may not be ideal for optimization

**When to Use:**
- When your data contains many outliers
- When all errors should be treated equally
- When you need a more robust metric
- For datasets with extreme values

**When NOT to Use:**
- When you want to heavily penalize large errors
- With optimization algorithms sensitive to non-smooth gradients
- When faster convergence is critical
- When large errors are more problematic than small ones
- In deep learning without careful handling of zero gradients

---

## 4. Mean Absolute Percentage Error (MAPE)

**Formula:** MAPE = (100/n) × Σ|(yᵢ - ŷᵢ)/yᵢ|

**Pros:**
- Scale-independent (expressed as percentage)
- Easy to interpret and explain to non-technical stakeholders
- Good for comparing across different scales
- Intuitive metric

**Cons:**
- Undefined when actual values are zero
- Asymmetric (penalizes under-predictions more than over-predictions)
- Biased towards lower forecasts
- Not suitable when target values are close to zero

**When to Use:**
- When you need scale-independent metrics
- For business reporting and dashboards
- When comparing models across different datasets
- When actual values are always positive and away from zero

**When NOT to Use:**
- When target values can be zero or near zero
- When over-predictions and under-predictions should be treated equally
- For data with negative values
- When absolute error magnitude matters more than percentage
- In scenarios where asymmetric penalties are problematic

---

## 5. Mean Squared Logarithmic Error (MSLE)

**Formula:** MSLE = (1/n) × Σ[log(yᵢ + 1) - log(ŷᵢ + 1)]²

**Pros:**
- Penalizes under-predictions more than over-predictions
- Handles large value ranges well
- Robust when targets span several orders of magnitude
- Less sensitive to large outliers in the output

**Cons:**
- Only works for positive values
- Can be harder to interpret
- More sensitive to small values
- Asymmetric error penalization

**When to Use:**
- When target values span multiple orders of magnitude
- When under-prediction is more costly than over-prediction
- For count data or exponential growth predictions
- When you want to minimize relative errors rather than absolute

**When NOT to Use:**
- When data contains negative values or zeros
- When over-predictions and under-predictions should be penalized equally
- For data with uniform or small value ranges
- When interpretability is important
- When symmetric error treatment is required

---

## 6. Huber Loss

**Formula:** 
- L(y, ŷ) = 0.5(y - ŷ)² for |y - ŷ| ≤ δ
- L(y, ŷ) = δ|y - ŷ| - 0.5δ² for |y - ŷ| > δ

**Pros:**
- Combines benefits of MSE and MAE
- Robust to outliers while still differentiable
- Quadratic for small errors, linear for large errors
- Tunable threshold (δ parameter)

**Cons:**
- Requires tuning of the δ hyperparameter
- More complex than MSE or MAE
- Less interpretable
- Computational overhead

**When to Use:**
- When you have outliers but still want to penalize large errors
- When you need a differentiable loss that's robust
- For datasets with mixed clean and noisy data
- When MSE is too sensitive and MAE too insensitive

**When NOT to Use:**
- When you don't have time to tune δ hyperparameter
- For clean data without outliers (use MSE instead)
- When simplicity is preferred over robustness
- In applications where computational efficiency is critical
- When you need easy interpretability

---

## 7. Log-Cosh Loss

**Formula:** L(y, ŷ) = Σ log(cosh(ŷᵢ - yᵢ))

**Pros:**
- Smooth everywhere (twice differentiable)
- Approximately equal to (y - ŷ)²/2 for small errors
- Approximately equal to |y - ŷ| - log(2) for large errors
- Robust to outliers like MAE but smoother

**Cons:**
- Less intuitive than other losses
- Can suffer from numerical instability for very large values
- Not as commonly implemented
- Slightly more computational cost

**When to Use:**
- When you need smooth gradients everywhere
- For second-order optimization methods
- When you want MSE-like behavior for small errors and MAE-like for large
- In deep learning with gradient-based optimization

**When NOT to Use:**
- When computational simplicity is critical
- For very large prediction values (numerical stability issues)
- When standard losses are well-implemented in your framework
- If interpretability is more important than smoothness
- For applications where MAE or Huber loss suffice

---

## 8. Quantile Loss

**Formula:** L(y, ŷ) = Σ max[τ(yᵢ - ŷᵢ), (τ - 1)(yᵢ - ŷᵢ)]

**Pros:**
- Enables prediction of specific quantiles
- Asymmetric loss function
- Useful for uncertainty quantification
- Provides prediction intervals

**Cons:**
- Requires choosing quantile parameter (τ)
- Not symmetric
- May need multiple models for different quantiles
- Less common in standard implementations

**When to Use:**
- When you need prediction intervals
- For risk-sensitive applications
- When different types of errors have different costs
- For probabilistic forecasting

**When NOT to Use:**
- For simple point prediction tasks
- When you only need mean/median predictions
- If you don't have expertise in quantile regression
- When symmetric error treatment is required
- For single-valued predictions without uncertainty

---

## 9. Tukey's Biweight Loss

**Formula:** Complex piecewise function with tuning parameter c

**Pros:**
- Very robust to outliers
- Completely downweights extreme outliers
- Smooth and differentiable in the main region
- Good statistical properties

**Cons:**
- Complex to implement
- Requires tuning of threshold parameter
- Can completely ignore some data points
- Less common in ML libraries

**When to Use:**
- When you have severe outlier contamination
- In robust regression scenarios
- When some data points are completely unreliable
- For statistical robustness

**When NOT to Use:**
- For clean datasets without extreme outliers
- When all data points should contribute to the model
- If implementation complexity is a concern
- When using standard ML libraries without custom loss support
- For applications requiring all observations to influence the model

---

## 10. Cauchy Loss

**Formula:** L(y, ŷ) = Σ log(1 + ((yᵢ - ŷᵢ)/c)²)

**Pros:**
- Extremely robust to outliers
- Logarithmic growth for large errors
- Differentiable everywhere
- Heavy-tailed distribution assumption

**Cons:**
- Can be too robust (may ignore important large errors)
- Requires tuning of scale parameter
- Can lead to slower convergence
- Less emphasis on fitting the majority of data

**When to Use:**
- When data contains extreme outliers
- For heavy-tailed error distributions
- When robustness is more important than exact fit
- In contaminated data scenarios

**When NOT to Use:**
- For clean, well-behaved data
- When you need to fit the majority of data points well
- If large errors need strong penalization
- When convergence speed is critical
- For Gaussian or near-Gaussian error distributions

---

## 11. Arctan Loss

**Formula:** L(y, ŷ) = Σ arctan²(yᵢ - ŷᵢ)

**Pros:**
- Bounded output (unlike MSE)
- Robust to outliers
- Smooth and differentiable
- Provides more stable gradients

**Cons:**
- Less common in practice
- Requires careful scaling
- May converge slower
- Limited theoretical analysis

**When to Use:**
- When you need bounded loss values
- For stable gradient flow in deep networks
- When dealing with outliers
- In experimental robust regression

**When NOT to Use:**
- For mainstream applications (use established losses)
- When interpretability is important
- If standard implementations are lacking
- When convergence speed matters
- For well-understood, standard regression problems

---

## 12. Smooth L1 Loss

**Formula:**
- 0.5x² if |x| < 1
- |x| - 0.5 otherwise

**Pros:**
- Combines L1 and L2 loss benefits
- Less sensitive to outliers than L2
- Smooth at zero unlike pure L1
- Popular in object detection

**Cons:**
- Fixed transition point at 1
- Not as customizable as Huber loss
- May need input scaling
- Arbitrary threshold

**When to Use:**
- In object detection models (Faster R-CNN, etc.)
- When you want L1 robustness with L2 smoothness
- For bounding box regression
- As a simpler alternative to Huber loss

**When NOT to Use:**
- When you need tunable transition thresholds (use Huber instead)
- For general regression outside object detection
- When input scaling is problematic
- If IoU-based losses are available (for detection)
- For applications where the fixed threshold is suboptimal

---

## 13. Charbonnier Loss

**Formula:** L(y, ŷ) = Σ √((yᵢ - ŷᵢ)² + ε²)

**Pros:**
- Differentiable variant of L1 loss
- Smoother than L1 at zero
- Robust to outliers
- Used in image processing and computer vision

**Cons:**
- Requires epsilon parameter tuning
- Less intuitive than standard losses
- Limited to specific domains
- Not widely implemented

**When to Use:**
- In image super-resolution
- For optical flow estimation
- When you need differentiable L1-like behavior
- In computer vision applications

**When NOT to Use:**
- For general-purpose regression tasks
- When standard losses perform adequately
- Outside computer vision domains
- If epsilon tuning adds unwanted complexity
- When framework support is limited

---

## 14. Berhu Loss

**Formula:** Combines L1 and L2 in a different way than Smooth L1

**Pros:**
- Reverse Huber loss characteristics
- L1 for small errors, L2 for large errors
- Used in depth estimation
- Different error emphasis

**Cons:**
- Less common than Huber
- Requires threshold tuning
- Opposite behavior to Huber may be counterintuitive
- Limited use cases

**When to Use:**
- In depth estimation tasks
- When small errors should have linear penalty
- For specific computer vision applications
- When you want to emphasize large errors quadratically

**When NOT to Use:**
- For most standard regression problems
- When the reverse Huber behavior is counterintuitive for your task
- Outside specialized computer vision applications
- If Huber loss is more natural for your problem
- When simpler losses work adequately

---

## 15. Wing Loss

**Formula:** Complex piecewise function designed for facial landmark detection

**Pros:**
- Designed for pose estimation and landmark detection
- Handles small and medium errors well
- Amplifies attention to small errors
- Proven effective in face alignment

**Cons:**
- Very specialized use case
- Complex formulation
- Requires careful hyperparameter tuning
- Not general-purpose

**When to Use:**
- For facial landmark detection
- In pose estimation tasks
- When small localization errors are critical
- For coordinate regression in computer vision

**When NOT to Use:**
- For general regression tasks outside computer vision
- When you don't have landmark or keypoint detection problems
- If the specialized formulation is unnecessary complexity
- For tasks where standard losses perform well
- Outside facial analysis and pose estimation domains
