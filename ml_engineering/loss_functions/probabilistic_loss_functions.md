# Probabilistic Loss Functions

## 1. Negative Log-Likelihood

**Formula:** NLL = -Σᵢ log P(yᵢ|xᵢ; θ)

**Pros:**
- Principled probabilistic framework
- Maximum likelihood interpretation
- Flexible for various probability distributions
- Theoretical foundation in statistics
- Works with any parametric distribution
- Natural choice for probabilistic models

**Cons:**
- Requires correct specification of probability distribution
- Sensitive to outliers
- Can be numerically unstable (log of small numbers)
- Assumes data follows specified distribution
- May require log-space computation

**When to Use:**
- Training probabilistic models
- When you have a well-defined probability distribution
- Generative modeling
- Bayesian inference
- When maximum likelihood estimation is appropriate
- For probabilistic neural networks

**When NOT to Use:**
- When your distributional assumptions are incorrect (garbage in, garbage out)
- With heavy outliers (extremely sensitive)
- For tasks where simpler losses like MSE/MAE work well (unnecessary complexity)
- When numerical stability is a major concern (log of near-zero probabilities)
- If you don't need probability estimates (deterministic losses may suffice)

---

## 2. Poisson Loss

**Formula:** L = Σᵢ [ŷᵢ - yᵢ log(ŷᵢ)]

**Pros:**
- Designed for count data
- Appropriate for rate/event prediction
- Handles non-negative integer targets
- Natural for frequency modeling
- Theoretically sound for Poisson-distributed data
- Used in GLMs (Generalized Linear Models)

**Cons:**
- Only suitable for count data
- Assumes mean equals variance (equidispersion)
- Not appropriate for continuous targets
- Can struggle with over/under-dispersed data
- Requires positive predictions

**When to Use:**
- Predicting counts (number of events)
- Traffic flow prediction
- Customer arrival rates
- Call center volume forecasting
- Disease outbreak modeling
- Any Poisson-distributed target variable

**When NOT to Use:**
- For continuous non-count data (use MSE, MAE, or Gaussian NLL)
- When variance doesn't equal mean (over/under-dispersion - use Negative Binomial)
- With negative values (Poisson requires non-negative integers)
- For large counts where Poisson approximation breaks down
- When data exhibits excess zeros (use zero-inflated models)

---

## 3. Tweedie Loss

**Formula:** L depends on power parameter p; encompasses multiple distributions

**Pros:**
- Flexible family of distributions
- Covers Normal (p=0), Poisson (p=1), Gamma (p=2), Inverse Gaussian (p=3)
- Handles zero-inflated data well
- Good for insurance/actuarial applications
- Can model various data types with single framework
- Handles positive continuous and count data

**Cons:**
- Requires selecting power parameter p
- More complex than simple loss functions
- Less intuitive interpretation
- Not widely implemented in all frameworks
- Computational complexity
- Requires understanding of compound Poisson process

**When to Use:**
- Insurance claim modeling (zero-inflated)
- Rainfall prediction (many zeros, continuous positives)
- Actuarial science applications
- When data has exact zeros and positive continuous values
- Credit risk modeling
- When standard distributions don't fit well

**When NOT to Use:**
- For simple regression tasks (use MSE/MAE - much simpler)
- When you can't determine appropriate power parameter p (requires domain expertise)
- With data that doesn't have zero-inflation (use simpler distributions)
- In frameworks without Tweedie support (limited availability)
- When interpretability is crucial (complex compound Poisson process)

---

## 4. Gaussian Negative Log-Likelihood

**Formula:** L = Σᵢ [½log(2πσ²) + (yᵢ - ŷᵢ)²/(2σ²)]

**Pros:**
- Assumes Gaussian/Normal distribution
- Equivalent to MSE when variance is constant
- Provides principled probabilistic interpretation
- Can model both mean and variance
- Well-understood theoretical properties
- Natural for many real-world phenomena

**Cons:**
- Assumes Gaussian distribution (may not fit data)
- Sensitive to outliers (like MSE)
- Requires estimation of variance parameter
- Not robust to heavy-tailed distributions
- Can be dominated by variance estimation

**When to Use:**
- When target distribution is approximately Normal
- Regression with Gaussian assumptions
- When you need to model uncertainty (variance)
- Probabilistic regression
- Gaussian process models
- When MSE is appropriate but you want probabilistic interpretation

**When NOT to Use:**
- With heavy-tailed or skewed distributions (Gaussian assumption violated)
- When outliers are present (very sensitive like MSE)
- For non-continuous targets (use appropriate discrete distributions)
- If you don't need uncertainty estimates (MSE is simpler)
- When computational simplicity is paramount (MSE is computationally cheaper)

---

## 5. Beta Loss

**Formula:** L = -Σᵢ [log B(α, β) - (α-1)log(yᵢ) - (β-1)log(1-yᵢ)]

**Pros:**
- Designed for targets in [0, 1] interval
- Flexible shape (can model various distributions)
- Appropriate for proportions and rates
- Can model skewed distributions
- Better than logistic for bounded continuous targets
- Parameters control distribution shape

**Cons:**
- Requires targets strictly between 0 and 1
- Need to estimate α and β parameters
- More complex than standard losses
- Limited implementation availability
- Can be numerically unstable at boundaries
- Requires understanding of Beta distribution

**When to Use:**
- Predicting proportions (e.g., success rates)
- Modeling percentages
- Click-through rate prediction
- Conversion rate modeling
- When target is continuous in (0, 1)
- Market share prediction

**When NOT to Use:**
- When targets can be exactly 0 or 1 (Beta requires open interval (0,1))
- For unbounded continuous values (use Gaussian or other distributions)
- With limited implementation support in your framework
- When simpler logistic regression is adequate for binary outcomes
- If estimating α and β is impractical (additional complexity)

---

## 6. Gamma Loss

**Formula:** L = Σᵢ [yᵢ/ŷᵢ + log(ŷᵢ)]

**Pros:**
- Appropriate for positive continuous targets
- Natural for right-skewed data
- Common in survival analysis
- Good for duration/time modeling
- Handles multiplicative effects well
- Theoretically sound for Gamma-distributed data

**Cons:**
- Only works with positive targets
- Assumes Gamma distribution
- Less intuitive than MSE or MAE
- Requires positive predictions
- Can be sensitive to very small values
- May need careful scaling

**When to Use:**
- Survival time prediction
- Insurance claim amounts
- Rainfall amount (when positive)
- Income modeling
- Time-to-event data
- When data is positive and right-skewed

**When NOT to Use:**
- With zero or negative target values (Gamma requires positive values)
- For symmetric or left-skewed distributions (use Gaussian or other distributions)
- When data isn't right-skewed (MSE may be simpler and better)
- If Gamma assumption doesn't fit your data distribution
- For general regression where sign matters (strictly positive constraint)

---

## 7. Maximum Likelihood Estimation (MLE) Loss

**Formula:** L = -log P(D|θ) where D is data and θ are parameters

**Pros:**
- General framework for parameter estimation
- Theoretically optimal under correct model specification
- Asymptotic properties well-understood
- Basis for many statistical methods
- Flexible for any parametric model
- Provides consistent estimates

**Cons:**
- Requires correct model specification
- Can be biased for small samples
- Computationally intensive for complex models
- Sensitive to model misspecification
- May not have closed-form solution
- Can overfit without regularization

**When to Use:**
- Parameter estimation in statistical models
- Training probabilistic models
- When you have a parametric model of the data
- Generative modeling
- As theoretical foundation for loss design
- In classical statistics and machine learning

**When NOT to Use:**
- When model is misspecified (very sensitive to wrong assumptions)
- With very small sample sizes (can be biased)
- For complex non-parametric models (may not have closed-form)
- When overfitting is a concern without regularization
- If computational resources are limited for complex likelihood calculations
- When non-probabilistic losses work adequately and are simpler
