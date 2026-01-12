# Comprehensive BigQuery ML (BQML) Transformations Guide

## Overview

BigQuery ML provides built-in SQL functions for data preprocessing and feature engineering directly within BigQuery, eliminating the need for external data processing. This guide covers all BQML transformation capabilities with practical examples.

## Table of Contents

1. [Built-in TRANSFORM Clause](#1-built-in-transform-clause)
2. [Feature Preprocessing Functions](#2-feature-preprocessing-functions)
3. [Feature Engineering](#3-feature-engineering)
4. [Text and NLP Transformations](#4-text-and-nlp-transformations)
5. [Time Series Transformations](#5-time-series-transformations)
6. [Categorical Encoding](#6-categorical-encoding)
7. [Array and Struct Transformations](#7-array-and-struct-transformations)
8. [Advanced Transformation Patterns](#8-advanced-transformation-patterns)
9. [Best Practices](#9-best-practices)

---

## 1. Built-in TRANSFORM Clause

### 1.1 Basic TRANSFORM Syntax

```sql
-- Basic TRANSFORM usage
CREATE OR REPLACE MODEL `project.dataset.model_name`
TRANSFORM(
  -- Transformations here
  feature1,
  feature2 AS renamed_feature,
  ML.FEATURE_CROSS(STRUCT(feature3, feature4)) AS crossed_features
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.training_data`;
```

**Key Points:**
- ✅ **TRANSFORM is applied automatically** during prediction
- ✅ **Transformations saved with model** - no need to reapply
- ✅ **Consistent preprocessing** - same logic train and predict
- ⚠️ **Cannot use subqueries** in TRANSFORM clause
- ⚠️ **Cannot use aggregate functions** in TRANSFORM

### 1.2 Complete TRANSFORM Example

```sql
-- Comprehensive transformation example
CREATE OR REPLACE MODEL `project.dataset.customer_churn_model`
TRANSFORM(
  -- 1. Numerical scaling
  ML.STANDARD_SCALER(age) OVER() AS age_scaled,
  ML.MIN_MAX_SCALER(tenure) OVER() AS tenure_normalized,
  
  -- 2. Categorical encoding
  ML.BUCKETIZE(monthly_charges, [30, 60, 90]) AS charge_bucket,
  
  -- 3. Feature interactions
  ML.FEATURE_CROSS(STRUCT(contract_type, payment_method)) AS contract_payment,
  
  -- 4. Polynomial features
  ML.POLYNOMIAL_EXPAND(STRUCT(total_charges), 2) AS total_charges_poly,
  
  -- 5. Text features
  ML.NGRAMS(SPLIT(service_description, ' '), [1, 2]) AS service_ngrams,
  
  -- 6. Feature selection
  * EXCEPT(customer_id, signup_date),  -- Exclude ID and dates
  
  -- 7. Label
  churned AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label'],
  data_split_method='AUTO_SPLIT'
) AS
SELECT * FROM `project.dataset.customer_data`;
```

---

## 2. Feature Preprocessing Functions

### 2.1 Numerical Scaling

#### ML.STANDARD_SCALER
**Purpose**: Standardize features to mean=0, std=1 (Z-score normalization)

```sql
-- Standardize numerical features
CREATE OR REPLACE MODEL `project.dataset.standard_scaler_model`
TRANSFORM(
  -- Standard scaling: (x - mean) / std
  ML.STANDARD_SCALER(age) OVER() AS age_scaled,
  ML.STANDARD_SCALER(income) OVER() AS income_scaled,
  ML.STANDARD_SCALER(credit_score) OVER() AS credit_scaled,
  
  label
)
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['label']
) AS
SELECT 
  age,
  income,
  credit_score,
  loan_amount AS label
FROM `project.dataset.loan_applications`;

-- Prediction with automatic scaling
SELECT * FROM ML.PREDICT(
  MODEL `project.dataset.standard_scaler_model`,
  (SELECT 35 AS age, 75000 AS income, 720 AS credit_score)
);
```

**When to Use:**
- ✅ Features have different scales (age vs income)
- ✅ Using distance-based algorithms (K-means, KNN)
- ✅ Neural networks, gradient descent optimization
- ✅ Model interpretability (comparable coefficients)

**When NOT to Use:**
- ❌ Tree-based models (Random Forest, XGBoost) - scale invariant
- ❌ Data already normalized
- ❌ Features on same scale already

#### ML.MIN_MAX_SCALER
**Purpose**: Scale features to [0, 1] range

```sql
-- Min-Max scaling: (x - min) / (max - min)
CREATE OR REPLACE MODEL `project.dataset.minmax_model`
TRANSFORM(
  -- Scale to [0, 1]
  ML.MIN_MAX_SCALER(page_views) OVER() AS page_views_norm,
  ML.MIN_MAX_SCALER(session_duration) OVER() AS duration_norm,
  ML.MIN_MAX_SCALER(bounce_rate) OVER() AS bounce_norm,
  
  -- Can also specify custom range
  ML.MIN_MAX_SCALER(clicks, 0, 100) OVER() AS clicks_0_100,
  
  converted AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.web_analytics`;
```

**When to Use:**
- ✅ Features bounded by nature (percentages, probabilities)
- ✅ Neural networks with sigmoid/tanh activation
- ✅ Distance-based algorithms
- ✅ Want interpretable [0,1] scale

**When NOT to Use:**
- ❌ Outliers present (will compress majority to narrow range)
- ❌ Distribution highly skewed
- ❌ Tree-based models

#### ML.MAX_ABS_SCALER
**Purpose**: Scale by maximum absolute value to [-1, 1]

```sql
-- Max Abs Scaling: x / max(|x|)
CREATE OR REPLACE MODEL `project.dataset.maxabs_model`
TRANSFORM(
  -- Preserves sparsity (zeros stay zero)
  ML.MAX_ABS_SCALER(temperature_change) OVER() AS temp_change_scaled,
  ML.MAX_ABS_SCALER(price_change) OVER() AS price_change_scaled,
  
  label
)
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.time_series_data`;
```

**When to Use:**
- ✅ **Sparse data** - preserves zeros (important!)
- ✅ Data already centered around zero
- ✅ Features with positive and negative values

**When NOT to Use:**
- ❌ Data not centered (use standard scaler)
- ❌ Need bounded range (use min-max)

### 2.2 Numerical Transformations

#### ML.BUCKETIZE
**Purpose**: Discretize continuous features into bins

```sql
-- Bucketize continuous features
CREATE OR REPLACE MODEL `project.dataset.bucketize_model`
TRANSFORM(
  -- Create age groups
  ML.BUCKETIZE(age, [18, 25, 35, 45, 55, 65]) AS age_group,
  -- Returns: 0 (<18), 1 (18-24), 2 (25-34), 3 (35-44), etc.
  
  -- Create income brackets
  ML.BUCKETIZE(income, [30000, 50000, 75000, 100000, 150000]) AS income_bracket,
  
  -- Equal-width bins (alternative approach)
  CAST(FLOOR(credit_score / 50) AS INT64) AS credit_tier,
  
  purchased AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.customers`;

-- Custom bucketization with descriptive labels
CREATE OR REPLACE MODEL `project.dataset.custom_buckets_model`
TRANSFORM(
  CASE 
    WHEN age < 18 THEN 'minor'
    WHEN age BETWEEN 18 AND 24 THEN 'young_adult'
    WHEN age BETWEEN 25 AND 34 THEN 'adult'
    WHEN age BETWEEN 35 AND 54 THEN 'middle_age'
    ELSE 'senior'
  END AS age_category,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.customers`;
```

**When to Use:**
- ✅ Non-linear relationships (bucketing captures)
- ✅ Reduce impact of outliers
- ✅ Create interpretable categories
- ✅ Handle skewed distributions
- ✅ Interaction with categorical features

**When NOT to Use:**
- ❌ Linear relationships (loses information)
- ❌ Small dataset (too many bins = overfitting)
- ❌ Continuous relationships important

#### ML.QUANTILE_BUCKETIZE
**Purpose**: Create equal-frequency bins (each bin has ~same # samples)

```sql
-- Quantile bucketization
CREATE OR REPLACE MODEL `project.dataset.quantile_model`
TRANSFORM(
  -- Create 4 quartiles (Q1, Q2, Q3, Q4)
  ML.QUANTILE_BUCKETIZE(salary, 4) OVER() AS salary_quartile,
  
  -- Create deciles (10 bins)
  ML.QUANTILE_BUCKETIZE(transaction_amount, 10) OVER() AS amount_decile,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.transactions`;
```

**When to Use:**
- ✅ **Skewed distributions** - each bin has similar sample count
- ✅ Want balanced bins
- ✅ Don't know natural cutpoints
- ✅ Feature importance by percentile

**When NOT to Use:**
- ❌ Natural/business-defined cutpoints exist
- ❌ Need consistent bins across datasets
- ❌ Interpretability critical (percentiles less intuitive)

#### Mathematical Transformations

```sql
-- Log, square root, and polynomial transformations
CREATE OR REPLACE MODEL `project.dataset.math_transform_model`
TRANSFORM(
  -- Log transformation for skewed features
  LOG(revenue + 1) AS log_revenue,  -- +1 to handle zeros
  LN(page_views + 1) AS ln_page_views,
  
  -- Square root for count data
  SQRT(number_of_transactions) AS sqrt_transactions,
  
  -- Power transformations
  POW(age, 2) AS age_squared,
  POW(income, 0.5) AS income_sqrt,
  
  -- Inverse transformation
  1.0 / (wait_time + 0.01) AS inverse_wait,  -- Small constant to avoid division by zero
  
  label
)
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.data`;
```

**Log Transformation:**
- ✅ Right-skewed distributions (revenue, prices)
- ✅ Exponential growth data
- ✅ Multiplicative relationships → additive
- ❌ Negative or zero values (need +1 offset)

**Square Root:**
- ✅ Count data (Poisson-distributed)
- ✅ Moderate skewness
- ✅ Stabilize variance

**Polynomial:**
- ✅ Non-linear relationships
- ✅ Interaction terms
- ❌ High-degree = overfitting risk

---

## 3. Feature Engineering

### 3.1 Feature Crosses (Interactions)

#### ML.FEATURE_CROSS
**Purpose**: Create interaction features between categorical variables

```sql
-- Feature crosses for capturing interactions
CREATE OR REPLACE MODEL `project.dataset.feature_cross_model`
TRANSFORM(
  -- Single cross: location × product_category
  ML.FEATURE_CROSS(STRUCT(location, product_category)) AS location_product,
  
  -- Multiple crosses
  ML.FEATURE_CROSS(STRUCT(
    day_of_week,
    hour_of_day,
    weather_condition
  )) AS time_weather_cross,
  
  -- Cross numerical × categorical (bucket first)
  ML.FEATURE_CROSS(STRUCT(
    ML.BUCKETIZE(age, [25, 35, 50, 65]) AS age_group,
    income_bracket,
    education_level
  )) AS demographic_cross,
  
  -- Multiple individual crosses
  ML.FEATURE_CROSS(STRUCT(city, state)) AS city_state,
  ML.FEATURE_CROSS(STRUCT(device_type, browser)) AS device_browser,
  
  conversion AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.marketing_data`;

-- Real-world example: E-commerce
CREATE OR REPLACE MODEL `project.dataset.ecommerce_model`
TRANSFORM(
  -- User × Product interaction
  ML.FEATURE_CROSS(STRUCT(user_id, product_id)) AS user_product,
  
  -- Location × Time interaction
  ML.FEATURE_CROSS(STRUCT(
    country,
    EXTRACT(DAYOFWEEK FROM order_date) AS day,
    EXTRACT(HOUR FROM order_timestamp) AS hour
  )) AS location_time,
  
  -- Device × Marketing interaction
  ML.FEATURE_CROSS(STRUCT(
    device_category,
    traffic_source,
    campaign_type
  )) AS device_marketing,
  
  purchased AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.user_sessions`;
```

**When to Use:**
- ✅ **Categorical interactions matter** (city + product type)
- ✅ Linear models (trees capture interactions automatically)
- ✅ Recommendation systems (user × item)
- ✅ Marketing (channel × demographic)
- ✅ Geographic + temporal patterns

**When NOT to Use:**
- ❌ Tree-based models (handle interactions natively)
- ❌ Too many unique combinations (curse of dimensionality)
- ❌ Sparse data (most combinations have few samples)
- ❌ Deep neural networks (learn interactions)

**Best Practices:**
```sql
-- ❌ BAD: Too many combinations (10,000 × 50 × 100 = 50M features)
ML.FEATURE_CROSS(STRUCT(user_id, product_id, timestamp))

-- ✅ GOOD: Aggregate to manageable categories
ML.FEATURE_CROSS(STRUCT(
  user_segment,  -- 10 segments
  product_category,  -- 20 categories
  hour_of_day  -- 24 hours
))  -- = 4,800 features (manageable)
```

### 3.2 Polynomial Expansion

#### ML.POLYNOMIAL_EXPAND
**Purpose**: Create polynomial features (x², x³, x×y, etc.)

```sql
-- Polynomial feature expansion
CREATE OR REPLACE MODEL `project.dataset.polynomial_model`
TRANSFORM(
  -- Degree 2: includes x, x², and interactions
  ML.POLYNOMIAL_EXPAND(STRUCT(age, income), 2) AS demographic_poly,
  
  -- Degree 3: includes x, x², x³, and all interactions
  ML.POLYNOMIAL_EXPAND(STRUCT(temperature, humidity, pressure), 3) AS weather_poly,
  
  -- Single feature polynomial
  ML.POLYNOMIAL_EXPAND(STRUCT(distance), 2) AS distance_poly,
  -- Creates: distance, distance²
  
  sales AS label
)
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.data`;

-- Example: House price with polynomial features
CREATE OR REPLACE MODEL `project.dataset.house_price_poly`
TRANSFORM(
  -- Polynomial features for non-linear relationships
  ML.POLYNOMIAL_EXPAND(STRUCT(
    sqft,
    num_bedrooms,
    num_bathrooms
  ), 2) AS house_features_poly,
  -- Creates: sqft, sqft², bedrooms, bedrooms², bathrooms, bathrooms²,
  --          sqft×bedrooms, sqft×bathrooms, bedrooms×bathrooms
  
  price AS label
)
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.houses`;
```

**What Polynomial Expansion Creates:**

**Degree 2 (x, y):**
- 1 (intercept - automatically handled)
- x, y (original features)
- x², y² (squares)
- x×y (interaction)

**Degree 3 (x, y):**
- All from degree 2, plus:
- x³, y³ (cubes)
- x²×y, x×y² (higher interactions)

**When to Use:**
- ✅ Non-linear relationships in data
- ✅ Linear models need flexibility
- ✅ Small number of numerical features (2-4)
- ✅ Domain knowledge suggests polynomial relationships

**When NOT to Use:**
- ❌ Many features (exponential growth: n features → C(n+d,d) features)
- ❌ Tree-based models (capture non-linearity natively)
- ❌ Neural networks (learn non-linear transformations)
- ❌ Risk of overfitting (high-degree polynomials)

**Example: Feature Explosion**
```sql
-- ❌ DANGEROUS: 10 features, degree 3
ML.POLYNOMIAL_EXPAND(STRUCT(
  f1, f2, f3, f4, f5, f6, f7, f8, f9, f10
), 3)
-- Creates: C(10+3, 3) = 286 features!

-- ✅ SAFE: 3 features, degree 2
ML.POLYNOMIAL_EXPAND(STRUCT(age, income, credit_score), 2)
-- Creates: C(3+2, 2) = 10 features
```

### 3.3 Custom Feature Engineering

```sql
-- Complex custom feature engineering
CREATE OR REPLACE MODEL `project.dataset.custom_features_model`
TRANSFORM(
  -- 1. Ratio features
  total_purchases / NULLIF(total_visits, 0) AS purchase_rate,
  revenue / NULLIF(num_orders, 0) AS average_order_value,
  
  -- 2. Difference features
  current_balance - previous_balance AS balance_change,
  DATE_DIFF(CURRENT_DATE(), last_purchase_date, DAY) AS days_since_purchase,
  
  -- 3. Aggregated features (from window functions)
  AVG(transaction_amount) OVER (
    PARTITION BY customer_id 
    ORDER BY transaction_date 
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS moving_avg_7day,
  
  -- 4. Boolean flags
  IF(num_logins > 10, 1, 0) AS is_active_user,
  IF(total_spent > 1000, 1, 0) AS is_vip,
  
  -- 5. Binned + one-hot encoding
  CASE 
    WHEN tenure_months < 12 THEN 'new'
    WHEN tenure_months < 36 THEN 'established'
    ELSE 'loyal'
  END AS customer_segment,
  
  -- 6. Time-based features
  EXTRACT(DAYOFWEEK FROM event_date) AS day_of_week,
  EXTRACT(MONTH FROM event_date) AS month,
  EXTRACT(HOUR FROM event_timestamp) AS hour_of_day,
  IF(EXTRACT(DAYOFWEEK FROM event_date) IN (1, 7), 1, 0) AS is_weekend,
  
  -- 7. Text length features
  LENGTH(review_text) AS review_length,
  ARRAY_LENGTH(SPLIT(review_text, ' ')) AS word_count,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.data`;
```

---

## 4. Text and NLP Transformations

### 4.1 ML.NGRAMS

```sql
-- N-gram extraction for text features
CREATE OR REPLACE MODEL `project.dataset.text_ngram_model`
TRANSFORM(
  -- Unigrams (single words)
  ML.NGRAMS(SPLIT(review_text, ' '), [1, 1]) AS unigrams,
  
  -- Bigrams (2-word phrases)
  ML.NGRAMS(SPLIT(product_description, ' '), [2, 2]) AS bigrams,
  
  -- Unigrams + Bigrams
  ML.NGRAMS(SPLIT(comment, ' '), [1, 2]) AS text_features,
  
  -- Trigrams
  ML.NGRAMS(SPLIT(title, ' '), [3, 3]) AS trigrams,
  
  sentiment AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.reviews`;

-- Text preprocessing + N-grams
CREATE OR REPLACE MODEL `project.dataset.clean_text_model`
TRANSFORM(
  ML.NGRAMS(
    SPLIT(
      LOWER(  -- Convert to lowercase
        REGEXP_REPLACE(  -- Remove punctuation
          REGEXP_REPLACE(  -- Remove extra spaces
            review_text,
            r'\s+', ' '
          ),
          r'[^\w\s]', ''
        )
      ),
      ' '
    ),
    [1, 2]  -- Unigrams and bigrams
  ) AS text_features,
  
  rating AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.reviews`;
```

**ML.NGRAMS Parameters:**
- `[1, 1]`: Unigrams only ("great", "product", "quality")
- `[2, 2]`: Bigrams only ("great product", "high quality")
- `[1, 2]`: Unigrams + Bigrams
- `[1, 3]`: Unigrams + Bigrams + Trigrams

**When to Use:**
- ✅ Text classification (sentiment, topics)
- ✅ Short text (reviews, tweets, descriptions)
- ✅ Bag-of-words models
- ✅ Simple text features

**When NOT to Use:**
- ❌ Need semantic understanding (use pre-trained embeddings)
- ❌ Long documents (too many features)
- ❌ Word order critical (use RNN/Transformer)
- ❌ Multiple languages without preprocessing

### 4.2 ML.TF_IDF (Term Frequency-Inverse Document Frequency)

```sql
-- TF-IDF for text importance weighting
CREATE OR REPLACE MODEL `project.dataset.tfidf_model`
TRANSFORM(
  -- TF-IDF on document text
  ML.TF_IDF(
    SPLIT(document_text, ' ')
  ) OVER() AS tfidf_features,
  
  category AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.documents`;

-- Combined N-grams + TF-IDF
CREATE OR REPLACE MODEL `project.dataset.ngram_tfidf_model`
TRANSFORM(
  ML.TF_IDF(
    ML.NGRAMS(SPLIT(LOWER(article_text), ' '), [1, 2])
  ) OVER() AS text_features,
  
  topic AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.articles`;
```

**TF-IDF Formula:**
- **TF** (Term Frequency): # times term appears in document / total terms in document
- **IDF** (Inverse Document Frequency): log(total documents / documents containing term)
- **TF-IDF** = TF × IDF

**When to Use:**
- ✅ Downweight common words ("the", "is", "and")
- ✅ Emphasize rare/distinctive words
- ✅ Document classification
- ✅ Information retrieval

**When NOT to Use:**
- ❌ Short text (TF-IDF less meaningful)
- ❌ All documents similar (IDF ineffective)
- ❌ Need semantic similarity (use embeddings)

### 4.3 Text Preprocessing Best Practices

```sql
-- Complete text preprocessing pipeline
CREATE OR REPLACE MODEL `project.dataset.text_preprocess_model`
TRANSFORM(
  -- Step 1: Clean text
  ML.NGRAMS(
    ARRAY(
      SELECT word
      FROM UNNEST(
        SPLIT(
          LOWER(
            REGEXP_REPLACE(
              REGEXP_REPLACE(text, r'[^\w\s]', ''),  -- Remove punctuation
              r'\s+', ' '  -- Normalize whitespace
            )
          ),
          ' '
        )
      ) AS word
      -- Step 2: Remove stop words
      WHERE word NOT IN (
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
        'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'up', 'about'
      )
      -- Step 3: Filter short words
      AND LENGTH(word) > 2
    ),
    [1, 2]
  ) AS text_features,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.text_data`;

-- Alternative: Custom stop words from table
CREATE OR REPLACE TABLE `project.dataset.stop_words` AS
SELECT word FROM UNNEST([
  'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
  'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'up', 'about',
  'into', 'through', 'during', 'before', 'after', 'above', 'below'
]) AS word;

CREATE OR REPLACE MODEL `project.dataset.text_model_v2`
TRANSFORM(
  ML.NGRAMS(
    ARRAY(
      SELECT word
      FROM UNNEST(SPLIT(LOWER(text), ' ')) AS word
      WHERE word NOT IN (SELECT word FROM `project.dataset.stop_words`)
      AND LENGTH(word) > 2
    ),
    [1, 2]
  ) AS text_features,
  
  label
)
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['label'])
AS SELECT * FROM `project.dataset.text_data`;
```

---

## 5. Time Series Transformations

### 5.1 Temporal Feature Engineering

```sql
-- Time series feature engineering
CREATE OR REPLACE MODEL `project.dataset.time_series_model`
TRANSFORM(
  -- 1. Extract temporal components
  EXTRACT(YEAR FROM date) AS year,
  EXTRACT(MONTH FROM date) AS month,
  EXTRACT(DAY FROM date) AS day,
  EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
  EXTRACT(DAYOFYEAR FROM date) AS day_of_year,
  EXTRACT(WEEK FROM date) AS week_of_year,
  EXTRACT(QUARTER FROM date) AS quarter,
  
  -- 2. Cyclical encoding (sine/cosine for periodicity)
  SIN(2 * ACOS(-1) * EXTRACT(MONTH FROM date) / 12) AS month_sin,
  COS(2 * ACOS(-1) * EXTRACT(MONTH FROM date) / 12) AS month_cos,
  
  SIN(2 * ACOS(-1) * EXTRACT(DAYOFWEEK FROM date) / 7) AS day_sin,
  COS(2 * ACOS(-1) * EXTRACT(DAYOFWEEK FROM date) / 7) AS day_cos,
  
  -- 3. Boolean temporal flags
  IF(EXTRACT(DAYOFWEEK FROM date) IN (1, 7), 1, 0) AS is_weekend,
  IF(EXTRACT(MONTH FROM date) IN (11, 12), 1, 0) AS is_holiday_season,
  IF(EXTRACT(DAY FROM date) = 1, 1, 0) AS is_month_start,
  IF(EXTRACT(DAY FROM date) = EXTRACT(DAY FROM LAST_DAY(date)), 1, 0) AS is_month_end,
  
  -- 4. Lag features (requires window functions in SELECT)
  LAG(sales, 1) OVER (ORDER BY date) AS sales_lag_1,
  LAG(sales, 7) OVER (ORDER BY date) AS sales_lag_7,
  LAG(sales, 30) OVER (ORDER BY date) AS sales_lag_30,
  
  -- 5. Rolling statistics
  AVG(sales) OVER (
    ORDER BY date 
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS sales_rolling_avg_7day,
  
  STDDEV(sales) OVER (
    ORDER BY date 
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS sales_rolling_std_7day,
  
  -- 6. Date differences
  DATE_DIFF(date, LAG(date, 1) OVER (ORDER BY date), DAY) AS days_since_last,
  
  sales AS label
)
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.daily_sales`;
```

**Cyclical Encoding Explanation:**
```sql
-- ❌ BAD: Month as integer (December=12, January=1 are far apart)
EXTRACT(MONTH FROM date) AS month  -- 12 and 1 seem far, but they're adjacent!

-- ✅ GOOD: Sine/Cosine encoding (December and January are close)
SIN(2 * π * month / 12) AS month_sin,
COS(2 * π * month / 12) AS month_cos
-- December: sin(-0.5), cos(0.866)
-- January:  sin(-0.866), cos(0.5)  -- Close in 2D space!
```

### 5.2 Time Series-Specific Models

```sql
-- ARIMA_PLUS for time series forecasting
CREATE OR REPLACE MODEL `project.dataset.arima_model`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='date',
  time_series_data_col='sales',
  time_series_id_col='store_id',
  holiday_region='US'  -- Automatically include holidays
) AS
SELECT 
  date,
  store_id,
  sales
FROM `project.dataset.daily_sales`;

-- Forecasting with ARIMA_PLUS
SELECT * FROM ML.FORECAST(
  MODEL `project.dataset.arima_model`,
  STRUCT(
    30 AS horizon,  -- Forecast 30 days ahead
    0.95 AS confidence_level  -- 95% confidence intervals
  )
);

-- Time series with external regressors
CREATE OR REPLACE MODEL `project.dataset.arima_with_features`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='date',
  time_series_data_col='sales',
  time_series_id_col='product_id'
) AS
SELECT 
  date,
  product_id,
  sales,
  -- External features
  promotion_active,
  price,
  competitor_price,
  temperature,
  is_weekend
FROM `project.dataset.product_sales`;
```

---

## 6. Categorical Encoding

### 6.1 One-Hot Encoding (Automatic)

```sql
-- One-hot encoding is AUTOMATIC in BQML for STRING/CATEGORICAL features
CREATE OR REPLACE MODEL `project.dataset.onehot_model`
TRANSFORM(
  -- Categorical features are automatically one-hot encoded
  product_category,  -- "Electronics", "Clothing", "Food" → [1,0,0], [0,1,0], [0,0,1]
  region,
  customer_segment,
  
  -- Numerical features pass through
  age,
  income,
  
  purchased AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.data`;

-- Manual one-hot encoding (if needed for control)
CREATE OR REPLACE MODEL `project.dataset.manual_onehot`
TRANSFORM(
  -- Manual one-hot
  IF(category = 'Electronics', 1, 0) AS is_electronics,
  IF(category = 'Clothing', 1, 0) AS is_clothing,
  IF(category = 'Food', 1, 0) AS is_food,
  -- Note: Need to exclude one category to avoid multicollinearity
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.data`;
```

**Automatic One-Hot Encoding:**
- ✅ STRING columns → one-hot encoded
- ✅ No need for manual encoding
- ✅ Handles unseen categories (assigned to "other")
- ⚠️ High cardinality → many features → slow training

### 6.2 Label Encoding (for Tree Models)

```sql
-- Label encoding for categorical features (manual)
CREATE OR REPLACE MODEL `project.dataset.label_encoding_model`
TRANSFORM(
  -- Manual label encoding
  CASE category
    WHEN 'A' THEN 0
    WHEN 'B' THEN 1
    WHEN 'C' THEN 2
    ELSE 3  -- Unknown
  END AS category_encoded,
  
  -- Or use FARM_FINGERPRINT for automatic integer mapping
  FARM_FINGERPRINT(user_id) AS user_id_encoded,
  
  label
)
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.data`;
```

**When to Use Label Encoding:**
- ✅ Tree-based models (can handle ordinal encoding)
- ✅ High-cardinality features (avoid one-hot explosion)
- ✅ Ordinal categories (small < medium < large)

**When to Use One-Hot:**
- ✅ Linear models (need separate coefficients per category)
- ✅ Low-cardinality features (<50 categories)
- ✅ Nominal categories (no natural order)

### 6.3 Target Encoding

```sql
-- Target encoding (manual - compute mean target per category)
CREATE OR REPLACE TEMP TABLE target_encoding AS
SELECT
  category,
  AVG(IF(target = 'positive', 1, 0)) AS category_target_mean
FROM `project.dataset.training_data`
GROUP BY category;

CREATE OR REPLACE MODEL `project.dataset.target_encoding_model`
TRANSFORM(
  -- Join with target encoding
  te.category_target_mean AS category_encoded,
  
  -- Other features
  feature1,
  feature2,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT 
  t.*,
  te.category_target_mean,
  t.target AS label
FROM `project.dataset.training_data` t
LEFT JOIN target_encoding te USING(category);
```

**Target Encoding:**
- ✅ High-cardinality categorical features
- ✅ Strong relationship between category and target
- ⚠️ **Risk of target leakage** - use cross-validation encoding
- ⚠️ **Overfitting** - smooth with prior (Bayesian approach)

---

## 7. Array and Struct Transformations

### 7.1 Array Features

```sql
-- Working with array features
CREATE OR REPLACE MODEL `project.dataset.array_features_model`
TRANSFORM(
  -- Array length
  ARRAY_LENGTH(product_views) AS num_products_viewed,
  ARRAY_LENGTH(search_queries) AS num_searches,
  
  -- Array aggregations
  (SELECT AVG(rating) FROM UNNEST(ratings) AS rating) AS avg_rating,
  (SELECT MAX(price) FROM UNNEST(prices) AS price) AS max_price,
  (SELECT SUM(quantity) FROM UNNEST(quantities) AS quantity) AS total_quantity,
  
  -- Array membership
  IF('premium' IN UNNEST(user_tags), 1, 0) AS is_premium,
  IF('mobile' IN UNNEST(device_types), 1, 0) AS used_mobile,
  
  -- First/last array element
  product_views[OFFSET(0)] AS first_product_viewed,
  product_views[OFFSET(ARRAY_LENGTH(product_views) - 1)] AS last_product_viewed,
  
  -- Array to string
  ARRAY_TO_STRING(categories, ',') AS categories_concat,
  
  converted AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.user_sessions`;

-- Multi-hot encoding from arrays
CREATE OR REPLACE MODEL `project.dataset.multihot_model`
TRANSFORM(
  -- Multi-hot encoding for array of categories
  IF('sports' IN UNNEST(interests), 1, 0) AS interest_sports,
  IF('music' IN UNNEST(interests), 1, 0) AS interest_music,
  IF('travel' IN UNNEST(interests), 1, 0) AS interest_travel,
  IF('tech' IN UNNEST(interests), 1, 0) AS interest_tech,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.users`;
```

### 7.2 Struct Features

```sql
-- Working with struct features
CREATE OR REPLACE MODEL `project.dataset.struct_features_model`
TRANSFORM(
  -- Extract struct fields
  address.city AS city,
  address.state AS state,
  address.zip_code AS zip,
  
  -- Struct in feature cross
  ML.FEATURE_CROSS(STRUCT(
    address.state,
    address.city
  )) AS location_cross,
  
  -- Nested struct access
  user_profile.demographics.age AS age,
  user_profile.demographics.income AS income,
  user_profile.preferences.category AS preferred_category,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.users`;
```

---

## 8. Advanced Transformation Patterns

### 8.1 Missing Value Handling

```sql
-- Handling missing values in TRANSFORM
CREATE OR REPLACE MODEL `project.dataset.missing_values_model`
TRANSFORM(
  -- 1. Replace NULL with default value
  IFNULL(age, 0) AS age_filled,
  COALESCE(income, 0) AS income_filled,
  
  -- 2. Replace NULL with mean (use constant from separate query)
  IFNULL(credit_score, 650) AS credit_score_filled,  -- 650 = mean from training
  
  -- 3. Replace NULL with median
  IFNULL(transaction_amount, 50.0) AS amount_filled,
  
  -- 4. Indicator for missing values
  IF(age IS NULL, 1, 0) AS age_was_missing,
  IF(income IS NULL, 1, 0) AS income_was_missing,
  
  -- 5. Forward fill (requires window function in SELECT)
  LAST_VALUE(price IGNORE NULLS) OVER (
    PARTITION BY product_id 
    ORDER BY date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS price_filled,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.data`;

-- Compute imputation values beforehand
CREATE OR REPLACE TEMP TABLE imputation_values AS
SELECT
  AVG(age) AS age_mean,
  APPROX_QUANTILES(income, 2)[OFFSET(1)] AS income_median,
  AVG(credit_score) AS credit_mean
FROM `project.dataset.training_data`
WHERE age IS NOT NULL AND income IS NOT NULL AND credit_score IS NOT NULL;

CREATE OR REPLACE MODEL `project.dataset.smart_imputation_model`
TRANSFORM(
  IFNULL(age, (SELECT age_mean FROM imputation_values)) AS age,
  IFNULL(income, (SELECT income_median FROM imputation_values)) AS income,
  IFNULL(credit_score, (SELECT credit_mean FROM imputation_values)) AS credit_score,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.training_data`;
```

### 8.2 Outlier Handling

```sql
-- Outlier handling in TRANSFORM
CREATE OR REPLACE MODEL `project.dataset.outlier_model`
TRANSFORM(
  -- 1. Winsorization (cap at percentiles)
  CASE
    WHEN age < 18 THEN 18
    WHEN age > 80 THEN 80
    ELSE age
  END AS age_winsorized,
  
  -- 2. Log transformation for skewed data
  LOG(income + 1) AS log_income,
  
  -- 3. Clip to IQR-based bounds
  CASE
    WHEN transaction_amount < 10 THEN 10   -- Q1 - 1.5*IQR
    WHEN transaction_amount > 500 THEN 500 -- Q3 + 1.5*IQR
    ELSE transaction_amount
  END AS amount_clipped,
  
  -- 4. Z-score outlier indicator
  IF(ABS((credit_score - 650) / 100) > 3, 1, 0) AS is_credit_outlier,
  
  label
)
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.data`;
```

### 8.3 Feature Selection in TRANSFORM

```sql
-- Feature selection patterns
CREATE OR REPLACE MODEL `project.dataset.feature_selection_model`
TRANSFORM(
  -- 1. Exclude unwanted features
  * EXCEPT(customer_id, created_at, updated_at, internal_id),
  
  -- 2. Select specific features
  age,
  income,
  credit_score,
  -- Instead of: SELECT age, income, credit_score, ...
  
  -- 3. Conditional feature inclusion
  IF(data_quality_flag = 'high', transaction_amount, NULL) AS amount,
  
  label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT * FROM `project.dataset.data`;
```

### 8.4 Complex Transformation Pipeline

```sql
-- Complete end-to-end transformation pipeline
CREATE OR REPLACE MODEL `project.dataset.complete_pipeline_model`
TRANSFORM(
  -- === NUMERICAL FEATURES ===
  
  -- Scaling
  ML.STANDARD_SCALER(age) OVER() AS age_scaled,
  ML.MIN_MAX_SCALER(income) OVER() AS income_norm,
  
  -- Missing value handling + outlier clipping
  CASE
    WHEN IFNULL(credit_score, 0) < 300 THEN 300
    WHEN IFNULL(credit_score, 0) > 850 THEN 850
    ELSE IFNULL(credit_score, 650)
  END AS credit_score_clean,
  
  -- Log transformation for skewed features
  LOG(IFNULL(annual_revenue, 1) + 1) AS log_revenue,
  
  -- Polynomial features
  ML.POLYNOMIAL_EXPAND(STRUCT(
    tenure_months,
    total_purchases
  ), 2) AS customer_behavior_poly,
  
  -- === CATEGORICAL FEATURES ===
  
  -- Automatic one-hot (just include the column)
  product_category,
  region,
  
  -- Manual bucketization
  ML.BUCKETIZE(age, [25, 35, 50, 65]) AS age_group,
  
  -- Feature crosses
  ML.FEATURE_CROSS(STRUCT(
    region,
    product_category,
    ML.BUCKETIZE(income, [50000, 100000, 150000]) AS income_tier
  )) AS demographic_product,
  
  -- === TEMPORAL FEATURES ===
  
  EXTRACT(DAYOFWEEK FROM signup_date) AS signup_day,
  EXTRACT(MONTH FROM signup_date) AS signup_month,
  
  -- Cyclical encoding
  SIN(2 * ACOS(-1) * EXTRACT(MONTH FROM signup_date) / 12) AS month_sin,
  COS(2 * ACOS(-1) * EXTRACT(MONTH FROM signup_date) / 12) AS month_cos,
  
  -- Date differences
  DATE_DIFF(CURRENT_DATE(), last_purchase_date, DAY) AS days_since_purchase,
  
  -- === TEXT FEATURES ===
  
  ML.NGRAMS(
    SPLIT(LOWER(REGEXP_REPLACE(review_text, r'[^\w\s]', '')), ' '),
    [1, 2]
  ) AS review_ngrams,
  
  -- === DERIVED FEATURES ===
  
  -- Ratios
  total_purchases / NULLIF(total_visits, 0) AS conversion_rate,
  revenue / NULLIF(num_orders, 0) AS avg_order_value,
  
  -- Boolean flags
  IF(lifetime_value > 1000, 1, 0) AS is_high_value,
  IF(num_complaints > 0, 1, 0) AS has_complaints,
  
  -- Array features
  ARRAY_LENGTH(past_purchases) AS num_past_purchases,
  
  -- === LABEL ===
  churned AS label
)
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label'],
  l1_reg=0.01,
  l2_reg=0.01,
  max_iterations=50,
  enable_global_explain=TRUE,
  data_split_method='AUTO_SPLIT'
) AS
SELECT * FROM `project.dataset.customer_data`;
```

---

## 9. Best Practices

### 9.1 Transformation Best Practices

#### ✅ DO

```sql
-- 1. Always apply same transformations to train and predict
-- BQML TRANSFORM handles this automatically

-- 2. Handle missing values explicitly
TRANSFORM(
  IFNULL(age, 0) AS age,
  IF(income IS NULL, 1, 0) AS income_missing_flag,
  label
)

-- 3. Scale numerical features for linear models
TRANSFORM(
  ML.STANDARD_SCALER(age) OVER() AS age_scaled,
  ML.MIN_MAX_SCALER(income) OVER() AS income_norm,
  label
)

-- 4. Use feature crosses for interactions
TRANSFORM(
  ML.FEATURE_CROSS(STRUCT(city, product_category)) AS city_product,
  label
)

-- 5. Bucketize for non-linear relationships
TRANSFORM(
  ML.BUCKETIZE(age, [25, 35, 50, 65]) AS age_group,
  label
)

-- 6. Use cyclical encoding for periodic features
TRANSFORM(
  SIN(2 * ACOS(-1) * EXTRACT(MONTH FROM date) / 12) AS month_sin,
  COS(2 * ACOS(-1) * EXTRACT(MONTH FROM date) / 12) AS month_cos,
  label
)

-- 7. Clean text before N-grams
TRANSFORM(
  ML.NGRAMS(
    SPLIT(LOWER(REGEXP_REPLACE(text, r'[^\w\s]', '')), ' '),
    [1, 2]
  ) AS text_features,
  label
)
```

#### ❌ DON'T

```sql
-- 1. Don't use different transformations for train and predict
-- TRANSFORM clause prevents this!

-- 2. Don't ignore missing values
-- ❌ BAD: age  -- NULL values will cause issues
-- ✅ GOOD: IFNULL(age, 0) AS age

-- 3. Don't use aggregate functions in TRANSFORM
-- ❌ BAD: AVG(price) AS avg_price  -- Not allowed!
-- ✅ GOOD: Compute in SELECT, then use in TRANSFORM

-- 4. Don't create too many features
-- ❌ BAD: ML.FEATURE_CROSS with high cardinality (millions of features)
-- ✅ GOOD: Aggregate to reasonable categories first

-- 5. Don't use raw high-cardinality categoricals
-- ❌ BAD: user_id (millions of unique values)
-- ✅ GOOD: user_segment (10-20 segments)

-- 6. Don't over-polynomial
-- ❌ BAD: ML.POLYNOMIAL_EXPAND(STRUCT(f1, f2, f3, f4, f5), 3)
-- ✅ GOOD: ML.POLYNOMIAL_EXPAND(STRUCT(f1, f2), 2)

-- 7. Don't forget to exclude ID columns
-- ❌ BAD: Including customer_id, order_id in training
-- ✅ GOOD: * EXCEPT(customer_id, order_id)
```

### 9.2 Performance Optimization

```sql
-- 1. Reduce feature dimensionality
TRANSFORM(
  -- Instead of raw high-cardinality categorical
  -- ❌ product_id (10,000 unique values)
  
  -- Use bucketed version
  -- ✅ product_category (50 categories)
  product_category,
  
  label
)

-- 2. Use AUTO_CLASS_WEIGHTS for imbalanced data
OPTIONS(
  model_type='LOGISTIC_REG',
  auto_class_weights=TRUE  -- Automatically balance classes
)

-- 3. Sample large datasets for faster iteration
CREATE OR REPLACE MODEL `project.dataset.quick_model`
TRANSFORM(...)
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['label'])
AS
SELECT * FROM `project.dataset.data`
WHERE RAND() < 0.1;  -- 10% sample for quick testing

-- 4. Use data_split_method='AUTO_SPLIT'
OPTIONS(
  model_type='LOGISTIC_REG',
  data_split_method='AUTO_SPLIT'  -- Automatic train/val/test split
)

-- 5. Cache intermediate transformations
CREATE OR REPLACE TABLE `project.dataset.transformed_features` AS
SELECT
  customer_id,
  ML.STANDARD_SCALER(age) OVER() AS age_scaled,
  ML.MIN_MAX_SCALER(income) OVER() AS income_norm,
  -- ... other expensive transformations
FROM `project.dataset.raw_data`;

-- Then train on cached data
CREATE OR REPLACE MODEL `project.dataset.model`
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['label'])
AS SELECT * FROM `project.dataset.transformed_features`;
```

### 9.3 Model Evaluation with Transformations

```sql
-- Evaluate model (transformations applied automatically)
SELECT * FROM ML.EVALUATE(
  MODEL `project.dataset.model`,
  (SELECT * FROM `project.dataset.test_data`)
);

-- Predict with transformations applied automatically
SELECT * FROM ML.PREDICT(
  MODEL `project.dataset.model`,
  (SELECT * FROM `project.dataset.new_data`)
);

-- Explain predictions (see which transformed features matter)
SELECT * FROM ML.EXPLAIN_PREDICT(
  MODEL `project.dataset.model`,
  (SELECT * FROM `project.dataset.new_data`),
  STRUCT(3 AS top_k_features)
);

-- Global feature importance
SELECT * FROM ML.GLOBAL_EXPLAIN(
  MODEL `project.dataset.model`
);
```

### 9.4 Debugging Transformations

```sql
-- View transformed features (before training)
CREATE OR REPLACE TEMP TABLE transformed_preview AS
SELECT
  -- Apply same TRANSFORM logic
  ML.STANDARD_SCALER(age) OVER() AS age_scaled,
  ML.MIN_MAX_SCALER(income) OVER() AS income_norm,
  ML.BUCKETIZE(age, [25, 35, 50, 65]) AS age_group,
  *
FROM `project.dataset.data`
LIMIT 100;

SELECT * FROM transformed_preview;

-- Check for NULL values after transformation
SELECT
  COUNTIF(age_scaled IS NULL) AS nulls_age,
  COUNTIF(income_norm IS NULL) AS nulls_income,
  COUNTIF(age_group IS NULL) AS nulls_age_group,
  COUNT(*) AS total_rows
FROM transformed_preview;

-- Check feature distributions
SELECT
  age_group,
  COUNT(*) AS count,
  AVG(age_scaled) AS avg_age_scaled,
  AVG(income_norm) AS avg_income_norm
FROM transformed_preview
GROUP BY age_group
ORDER BY age_group;
```

---

## Quick Reference: Transformation Functions

| Function | Purpose | Input | Output | Use Case |
|----------|---------|-------|--------|----------|
| **ML.STANDARD_SCALER** | Z-score normalization | Numerical | Scaled (mean=0, std=1) | Linear models, distance-based |
| **ML.MIN_MAX_SCALER** | Min-max scaling | Numerical | Normalized [0, 1] | Neural networks, bounded features |
| **ML.MAX_ABS_SCALER** | Scale by max abs | Numerical | Scaled [-1, 1] | Sparse data, centered data |
| **ML.BUCKETIZE** | Discretize continuous | Numerical | Categorical (0, 1, 2, ...) | Non-linear patterns, reduce outliers |
| **ML.QUANTILE_BUCKETIZE** | Equal-frequency bins | Numerical | Categorical (0, 1, 2, ...) | Skewed distributions |
| **ML.FEATURE_CROSS** | Interaction features | Categorical STRUCTs | Crossed categorical | Capture interactions |
| **ML.POLYNOMIAL_EXPAND** | Polynomial features | Numerical STRUCT | Polynomial terms | Non-linear relationships |
| **ML.NGRAMS** | N-gram extraction | ARRAY<STRING> | ARRAY<STRING> | Text classification |
| **ML.TF_IDF** | TF-IDF weighting | ARRAY<STRING> | ARRAY<STRUCT> | Text importance |

---

## Exam Tips: BQML Transformations

### Most Tested Concepts

1. **TRANSFORM clause is automatic** during prediction
2. **ML.STANDARD_SCALER vs ML.MIN_MAX_SCALER** - when to use each
3. **ML.FEATURE_CROSS** for categorical interactions
4. **ML.BUCKETIZE** for discretizing continuous features
5. **ML.NGRAMS** for text features
6. **Handling missing values** with IFNULL/COALESCE
7. **Feature engineering** patterns (ratios, differences, flags)
8. **Cyclical encoding** for periodic features (month, day)

### Common Exam Scenarios

**Scenario 1**: "Features on different scales (age vs income)"
- ✅ Use ML.STANDARD_SCALER or ML.MIN_MAX_SCALER

**Scenario 2**: "Text classification problem"
- ✅ Use ML.NGRAMS with SPLIT(LOWER(text), ' ')

**Scenario 3**: "Interaction between city and product matters"
- ✅ Use ML.FEATURE_CROSS(STRUCT(city, product))

**Scenario 4**: "Non-linear relationship between age and outcome"
- ✅ Use ML.BUCKETIZE(age, [25, 35, 50, 65])

**Scenario 5**: "Missing values in numerical features"
- ✅ Use IFNULL(feature, default_value) and add missing indicator

**Scenario 6**: "Day of week matters (Monday similar to Tuesday, Sunday similar to Saturday)"
- ✅ Use cyclical encoding: SIN/COS transformation

---

## Summary

### Key Takeaways

1. ✅ **TRANSFORM clause** - transformations saved with model, applied automatically
2. ✅ **Scale numerical features** - ML.STANDARD_SCALER, ML.MIN_MAX_SCALER
3. ✅ **Bucketize for non-linearity** - ML.BUCKETIZE, ML.QUANTILE_BUCKETIZE
4. ✅ **Feature crosses for interactions** - ML.FEATURE_CROSS
5. ✅ **N-grams for text** - ML.NGRAMS with text preprocessing
6. ✅ **Handle missing values** - IFNULL, COALESCE, missing indicators
7. ✅ **Cyclical encoding** - SIN/COS for periodic features
8. ✅ **Feature engineering** - ratios, differences, aggregations
9. ✅ **Exclude ID columns** - * EXCEPT(id, date)
10. ✅ **Debug transformations** - preview before training

### Common Patterns

| Task | Transformation Pattern |
|------|----------------------|
| **Scale features** | `ML.STANDARD_SCALER(feature) OVER()` |
| **Discretize** | `ML.BUCKETIZE(feature, [10, 20, 30])` |
| **Interaction** | `ML.FEATURE_CROSS(STRUCT(f1, f2))` |
| **Text** | `ML.NGRAMS(SPLIT(LOWER(text), ' '), [1,2])` |
| **Missing** | `IFNULL(feature, 0) AS feature` |
| **Cyclical** | `SIN(2*π*month/12), COS(2*π*month/12)` |
| **Ratio** | `f1 / NULLIF(f2, 0) AS ratio` |
| **Time** | `DATE_DIFF(CURRENT_DATE(), date, DAY)` |

This comprehensive guide covers all BQML transformation capabilities for the GCP Data Engineer certification! 🎯