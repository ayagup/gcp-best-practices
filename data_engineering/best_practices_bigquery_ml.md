# BigQuery ML Best Practices

*Last Updated: December 26, 2025*

## Overview

BigQuery ML (BQML) enables users to create and execute machine learning models directly in BigQuery using standard SQL queries. It democratizes machine learning by eliminating the need to move data and allows data analysts to build models using familiar SQL syntax.

---

## 1. Model Selection

### Supported Model Types

**Best Practices:**
- Choose appropriate model type for your use case
- Understand model capabilities and limitations
- Start with simpler models before complex ones
- Consider training time and cost

```sql
-- Model Types Reference
/*
LINEAR_REG: Linear regression for continuous values
LOGISTIC_REG: Binary or multi-class classification
KMEANS: Unsupervised clustering
MATRIX_FACTORIZATION: Recommendation systems
DNN_CLASSIFIER: Deep neural network classification
DNN_REGRESSOR: Deep neural network regression
BOOSTED_TREE_CLASSIFIER: XGBoost classification
BOOSTED_TREE_REGRESSOR: XGBoost regression
AUTOML_CLASSIFIER: AutoML classification
AUTOML_REGRESSOR: AutoML regression
ARIMA_PLUS: Time series forecasting
TENSORFLOW: Import TensorFlow models
ONNX: Import ONNX models
*/

-- Example: Choosing the right model
-- For binary classification (churn prediction)
CREATE OR REPLACE MODEL `my_dataset.customer_churn_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['churned'],
  auto_class_weights=TRUE  -- Handle class imbalance
) AS
SELECT
  age,
  tenure_months,
  monthly_spend,
  support_tickets,
  product_category,
  churned
FROM
  `my_dataset.customer_data`
WHERE
  partition_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY);
```

### Model Selection Decision Tree

**Use Case Mapping:**
```sql
-- Regression (predicting continuous values)
-- Example: Predict customer lifetime value
CREATE OR REPLACE MODEL `my_dataset.clv_model`
OPTIONS(
  model_type='LINEAR_REG',  -- Or DNN_REGRESSOR, BOOSTED_TREE_REGRESSOR
  input_label_cols=['lifetime_value']
) AS
SELECT
  age,
  income,
  tenure_months,
  purchase_frequency,
  average_order_value,
  lifetime_value
FROM
  `my_dataset.customer_metrics`;

-- Classification (predicting categories)
-- Example: Customer segmentation
CREATE OR REPLACE MODEL `my_dataset.segment_model`
OPTIONS(
  model_type='LOGISTIC_REG',  -- Or DNN_CLASSIFIER, BOOSTED_TREE_CLASSIFIER
  input_label_cols=['segment']
) AS
SELECT
  * EXCEPT(customer_id, segment),
  segment
FROM
  `my_dataset.customer_features`;

-- Clustering (unsupervised grouping)
-- Example: Customer clustering
CREATE OR REPLACE MODEL `my_dataset.customer_clusters`
OPTIONS(
  model_type='KMEANS',
  num_clusters=5,
  distance_type='EUCLIDEAN',
  standardize_features=TRUE
) AS
SELECT
  * EXCEPT(customer_id)
FROM
  `my_dataset.customer_features`;

-- Recommendation (collaborative filtering)
-- Example: Product recommendations
CREATE OR REPLACE MODEL `my_dataset.product_recommendations`
OPTIONS(
  model_type='MATRIX_FACTORIZATION',
  user_col='customer_id',
  item_col='product_id',
  rating_col='rating',
  num_factors=10
) AS
SELECT
  customer_id,
  product_id,
  rating
FROM
  `my_dataset.product_ratings`;

-- Time Series Forecasting
-- Example: Sales forecasting
CREATE OR REPLACE MODEL `my_dataset.sales_forecast`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='date',
  time_series_data_col='sales',
  time_series_id_col='store_id',
  holiday_region='US'
) AS
SELECT
  date,
  store_id,
  sales
FROM
  `my_dataset.daily_sales`;
```

---

## 2. Data Preparation

### Feature Engineering

**Best Practices:**
- Transform data before model creation
- Handle missing values appropriately
- Create meaningful features
- Use TRANSFORM clause for preprocessing

```sql
-- Feature engineering with TRANSFORM
CREATE OR REPLACE MODEL `my_dataset.churn_model_v2`
TRANSFORM(
  -- Numeric transformations
  age,
  tenure_months,
  CAST(monthly_spend AS FLOAT64) AS monthly_spend,
  support_tickets,
  
  -- Derived features
  monthly_spend / NULLIF(tenure_months, 0) AS spend_per_month,
  support_tickets / NULLIF(tenure_months, 0) AS tickets_per_month,
  
  -- Categorical encoding (automatic one-hot encoding)
  product_category,
  region,
  
  -- Date features
  EXTRACT(MONTH FROM signup_date) AS signup_month,
  EXTRACT(DAYOFWEEK FROM signup_date) AS signup_day_of_week,
  
  -- Binning
  CASE
    WHEN age < 25 THEN 'young'
    WHEN age BETWEEN 25 AND 40 THEN 'adult'
    WHEN age BETWEEN 41 AND 60 THEN 'middle_age'
    ELSE 'senior'
  END AS age_group,
  
  -- Text features (for DNN models)
  ML.NGRAMS(SPLIT(customer_notes, ' '), [1, 2]) AS notes_tokens,
  
  -- Target variable
  churned
)
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['churned'],
  max_iterations=50,
  early_stop=TRUE,
  min_rel_progress=0.01
) AS
SELECT
  age,
  tenure_months,
  monthly_spend,
  support_tickets,
  product_category,
  region,
  signup_date,
  customer_notes,
  churned
FROM
  `my_dataset.customer_data`
WHERE
  partition_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY);
```

### Handling Missing Data

**Strategies for Null Values:**
```sql
-- Strategy 1: Filter out rows with missing values
CREATE OR REPLACE MODEL `my_dataset.model_no_nulls`
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['target']
) AS
SELECT
  feature1,
  feature2,
  feature3,
  target
FROM
  `my_dataset.training_data`
WHERE
  feature1 IS NOT NULL
  AND feature2 IS NOT NULL
  AND feature3 IS NOT NULL
  AND target IS NOT NULL;

-- Strategy 2: Impute missing values
CREATE OR REPLACE MODEL `my_dataset.model_imputed`
TRANSFORM(
  -- Replace nulls with mean (for numeric)
  IFNULL(age, (SELECT AVG(age) FROM `my_dataset.training_data`)) AS age,
  
  -- Replace nulls with median
  IFNULL(income, (SELECT APPROX_QUANTILES(income, 2)[OFFSET(1)] 
                  FROM `my_dataset.training_data`)) AS income,
  
  -- Replace nulls with mode (for categorical)
  IFNULL(category, 'unknown') AS category,
  
  -- Forward fill for time series
  COALESCE(
    value,
    LAG(value) OVER (PARTITION BY group_id ORDER BY date)
  ) AS value,
  
  target
)
OPTIONS(
  model_type='LINEAR_REG',
  input_label_cols=['target']
) AS
SELECT * FROM `my_dataset.training_data`;
```

### Data Splitting

**Train/Test Split:**
```sql
-- Create training and evaluation sets
-- Method 1: Using MOD for consistent split
CREATE OR REPLACE MODEL `my_dataset.model_with_split`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label'],
  data_split_method='SEQ',
  data_split_eval_fraction=0.2,
  data_split_col='split_col'
) AS
SELECT
  features.*,
  label,
  MOD(ABS(FARM_FINGERPRINT(CAST(id AS STRING))), 10) AS split_col
FROM
  `my_dataset.training_data` AS features;

-- Method 2: Using random split
CREATE OR REPLACE MODEL `my_dataset.model_random_split`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label'],
  data_split_method='RANDOM',
  data_split_eval_fraction=0.2
) AS
SELECT
  * EXCEPT(id)
FROM
  `my_dataset.training_data`;

-- Method 3: Time-based split for time series
CREATE OR REPLACE MODEL `my_dataset.model_time_split`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='date',
  time_series_data_col='value',
  data_split_method='CUSTOM',
  data_split_col='is_training'
) AS
SELECT
  date,
  value,
  CASE
    WHEN date < '2024-01-01' THEN TRUE  -- Training
    ELSE FALSE  -- Evaluation
  END AS is_training
FROM
  `my_dataset.time_series_data`;
```

---

## 3. Model Training

### Training Configuration

**Best Practices:**
- Set appropriate hyperparameters
- Use early stopping to prevent overfitting
- Monitor training progress
- Balance training time vs. accuracy

```sql
-- Comprehensive training configuration
CREATE OR REPLACE MODEL `my_dataset.optimized_model`
TRANSFORM(
  * EXCEPT(customer_id, timestamp)
)
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['churned'],
  
  -- Data split
  data_split_method='RANDOM',
  data_split_eval_fraction=0.2,
  
  -- Class balance
  auto_class_weights=TRUE,
  
  -- Training parameters
  max_iterations=100,
  early_stop=TRUE,
  min_rel_progress=0.01,
  learn_rate=0.1,
  subsample=0.8,
  
  -- Tree parameters
  max_tree_depth=10,
  tree_method='HIST',
  min_tree_child_weight=1,
  
  -- Regularization
  l1_reg=0.0,
  l2_reg=0.1,
  
  -- Feature selection
  enable_global_explain=TRUE,
  
  -- Model metadata
  model_registry='vertex_ai',
  vertex_ai_model_version_aliases=['champion', 'production']
) AS
SELECT
  *
FROM
  `my_dataset.customer_features`
WHERE
  partition_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY);
```

### Deep Neural Networks

**DNN Configuration:**
```sql
-- DNN Classifier with custom architecture
CREATE OR REPLACE MODEL `my_dataset.dnn_classifier`
TRANSFORM(
  -- Numeric features
  ML.STANDARD_SCALER(age) OVER() AS age_scaled,
  ML.STANDARD_SCALER(income) OVER() AS income_scaled,
  ML.STANDARD_SCALER(tenure_months) OVER() AS tenure_scaled,
  
  -- Categorical features (embedding)
  product_category,
  region,
  
  -- Target
  churned
)
OPTIONS(
  model_type='DNN_CLASSIFIER',
  input_label_cols=['churned'],
  
  -- Network architecture
  hidden_units=[128, 64, 32],  -- Three hidden layers
  activation_fn='RELU',
  dropout=0.2,
  
  -- Training parameters
  batch_size=32,
  max_iterations=100,
  early_stop=TRUE,
  min_rel_progress=0.01,
  learn_rate=0.001,
  learn_rate_strategy='LINE_SEARCH',
  optimizer='ADAM',
  
  -- Regularization
  l1_reg=0.0,
  l2_reg=0.01,
  
  -- Class balance
  auto_class_weights=TRUE
) AS
SELECT
  age,
  income,
  tenure_months,
  product_category,
  region,
  churned
FROM
  `my_dataset.customer_data`;
```

### AutoML Models

**Automated Model Selection:**
```sql
-- AutoML Classifier - automatically selects best model
CREATE OR REPLACE MODEL `my_dataset.automl_classifier`
OPTIONS(
  model_type='AUTOML_CLASSIFIER',
  input_label_cols=['label'],
  budget_hours=1.0,  -- Training time budget
  optimization_objective='MAXIMIZE_AU_ROC',
  
  -- Feature selection
  enable_global_explain=TRUE,
  
  -- Early stopping
  early_stop=TRUE
) AS
SELECT
  * EXCEPT(id)
FROM
  `my_dataset.training_data`;

-- AutoML Regressor
CREATE OR REPLACE MODEL `my_dataset.automl_regressor`
OPTIONS(
  model_type='AUTOML_REGRESSOR',
  input_label_cols=['price'],
  budget_hours=2.0,
  optimization_objective='MINIMIZE_RMSE',
  enable_global_explain=TRUE
) AS
SELECT
  * EXCEPT(id, timestamp)
FROM
  `my_dataset.housing_data`;
```

---

## 4. Model Evaluation

### Evaluation Metrics

**Best Practices:**
- Use appropriate metrics for model type
- Evaluate on holdout test set
- Compare against baseline
- Analyze confusion matrix

```sql
-- Evaluate model performance
SELECT
  *
FROM
  ML.EVALUATE(MODEL `my_dataset.churn_model`);

-- Classification metrics output:
-- precision, recall, accuracy, f1_score, log_loss, roc_auc

-- Detailed evaluation with threshold
SELECT
  *
FROM
  ML.EVALUATE(MODEL `my_dataset.churn_model`,
    (SELECT * FROM `my_dataset.test_data`),
    STRUCT(0.6 AS threshold));  -- Custom threshold

-- Regression metrics
SELECT
  *
FROM
  ML.EVALUATE(MODEL `my_dataset.price_model`);
-- mean_absolute_error, mean_squared_error, 
-- mean_squared_log_error, median_absolute_error, r2_score

-- Multi-class classification evaluation
SELECT
  *
FROM
  ML.EVALUATE(MODEL `my_dataset.multiclass_model`);

-- Confusion matrix for classification
SELECT
  *
FROM
  ML.CONFUSION_MATRIX(MODEL `my_dataset.churn_model`,
    (SELECT * FROM `my_dataset.test_data`));

-- ROC curve
SELECT
  *
FROM
  ML.ROC_CURVE(MODEL `my_dataset.churn_model`,
    (SELECT * FROM `my_dataset.test_data`));

-- Feature importance
SELECT
  *
FROM
  ML.FEATURE_IMPORTANCE(MODEL `my_dataset.churn_model`);

-- Global explanations
SELECT
  *
FROM
  ML.GLOBAL_EXPLAIN(MODEL `my_dataset.churn_model`);
```

### Advanced Evaluation

**Cross-Validation and Custom Metrics:**
```sql
-- Evaluate with custom data
SELECT
  *
FROM
  ML.EVALUATE(MODEL `my_dataset.churn_model`,
    (
      SELECT
        *
      FROM
        `my_dataset.customer_data`
      WHERE
        partition_date = '2025-01-01'
    ));

-- Compare multiple models
WITH model_a_eval AS (
  SELECT
    'model_a' AS model_name,
    *
  FROM
    ML.EVALUATE(MODEL `my_dataset.model_a`)
),
model_b_eval AS (
  SELECT
    'model_b' AS model_name,
    *
  FROM
    ML.EVALUATE(MODEL `my_dataset.model_b`)
)
SELECT
  model_name,
  precision,
  recall,
  f1_score,
  roc_auc
FROM
  model_a_eval
UNION ALL
SELECT
  model_name,
  precision,
  recall,
  f1_score,
  roc_auc
FROM
  model_b_eval
ORDER BY
  roc_auc DESC;

-- Time series evaluation
SELECT
  *
FROM
  ML.EVALUATE(MODEL `my_dataset.sales_forecast`,
    STRUCT(30 AS horizon,  -- Forecast horizon
           0.9 AS confidence_level));
```

---

## 5. Making Predictions

### Batch Predictions

**Best Practices:**
- Use batch prediction for large datasets
- Store predictions in tables for analysis
- Include prediction probabilities
- Use appropriate threshold

```sql
-- Basic prediction
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs
FROM
  ML.PREDICT(MODEL `my_dataset.churn_model`,
    (SELECT * FROM `my_dataset.customers_to_score`));

-- Prediction with custom threshold
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(1)].prob AS churn_probability,
  CASE
    WHEN predicted_churned_probs[OFFSET(1)].prob > 0.7 THEN 'high_risk'
    WHEN predicted_churned_probs[OFFSET(1)].prob > 0.4 THEN 'medium_risk'
    ELSE 'low_risk'
  END AS risk_category
FROM
  ML.PREDICT(MODEL `my_dataset.churn_model`,
    (SELECT * FROM `my_dataset.customers_to_score`))
ORDER BY
  churn_probability DESC;

-- Save predictions to table
CREATE OR REPLACE TABLE `my_dataset.churn_predictions` AS
SELECT
  customer_id,
  prediction_date,
  predicted_churned,
  predicted_churned_probs[OFFSET(1)].prob AS churn_probability
FROM
  ML.PREDICT(MODEL `my_dataset.churn_model`,
    (
      SELECT
        *,
        CURRENT_DATE() AS prediction_date
      FROM
        `my_dataset.customers_to_score`
    ));

-- Regression predictions
SELECT
  property_id,
  predicted_price,
  predicted_price AS estimated_value
FROM
  ML.PREDICT(MODEL `my_dataset.price_model`,
    (SELECT * FROM `my_dataset.properties_to_value`));
```

### Recommendation Predictions

**Collaborative Filtering:**
```sql
-- Get product recommendations for customers
SELECT
  customer_id,
  ARRAY_AGG(
    STRUCT(product_id, predicted_rating)
    ORDER BY predicted_rating DESC
    LIMIT 10
  ) AS top_recommendations
FROM
  ML.PREDICT(MODEL `my_dataset.product_recommendations`,
    (
      SELECT
        customer_id,
        product_id
      FROM
        `my_dataset.customers`
      CROSS JOIN
        `my_dataset.products`
    ))
WHERE
  predicted_rating > 4.0  -- Only high-rated predictions
GROUP BY
  customer_id;

-- Recommend similar items
SELECT
  *
FROM
  ML.RECOMMEND(MODEL `my_dataset.product_recommendations`,
    STRUCT(1234 AS customer_id, 5 AS num_recommendations));
```

### Time Series Forecasting

**ARIMA Plus Forecasts:**
```sql
-- Forecast future values
SELECT
  *
FROM
  ML.FORECAST(MODEL `my_dataset.sales_forecast`,
    STRUCT(30 AS horizon,  -- 30 days ahead
           0.95 AS confidence_level));

-- Forecast with explanations
SELECT
  forecast_timestamp,
  forecast_value,
  standard_error,
  confidence_level,
  prediction_interval_lower_bound,
  prediction_interval_upper_bound,
  confidence_interval_lower_bound,
  confidence_interval_upper_bound
FROM
  ML.FORECAST(MODEL `my_dataset.sales_forecast`,
    STRUCT(90 AS horizon,
           0.95 AS confidence_level));

-- Detect anomalies in time series
SELECT
  *
FROM
  ML.DETECT_ANOMALIES(MODEL `my_dataset.sales_forecast`,
    STRUCT(0.95 AS anomaly_prob_threshold),
    (SELECT * FROM `my_dataset.recent_sales`));
```

### Clustering Predictions

**Assign Clusters:**
```sql
-- Assign customers to clusters
SELECT
  customer_id,
  CENTROID_ID AS cluster_id,
  NEAREST_CENTROIDS_DISTANCE[OFFSET(0)].CENTROID_ID AS nearest_cluster,
  NEAREST_CENTROIDS_DISTANCE[OFFSET(0)].DISTANCE AS distance_to_cluster
FROM
  ML.PREDICT(MODEL `my_dataset.customer_clusters`,
    (SELECT * EXCEPT(customer_id) FROM `my_dataset.customers`));

-- Analyze cluster characteristics
WITH cluster_assignments AS (
  SELECT
    customer_id,
    CENTROID_ID AS cluster_id
  FROM
    ML.PREDICT(MODEL `my_dataset.customer_clusters`,
      (SELECT * FROM `my_dataset.customer_features`))
)
SELECT
  ca.cluster_id,
  COUNT(*) AS cluster_size,
  AVG(c.age) AS avg_age,
  AVG(c.lifetime_value) AS avg_ltv,
  AVG(c.tenure_months) AS avg_tenure
FROM
  cluster_assignments ca
JOIN
  `my_dataset.customers` c
ON
  ca.customer_id = c.customer_id
GROUP BY
  ca.cluster_id
ORDER BY
  cluster_size DESC;
```

---

## 6. Model Explainability

### Feature Attribution

**Best Practices:**
- Use explainability for model trust
- Analyze feature importance
- Understand prediction drivers
- Identify bias

```sql
-- Global feature importance
SELECT
  *
FROM
  ML.FEATURE_IMPORTANCE(MODEL `my_dataset.churn_model`)
ORDER BY
  importance DESC;

-- Explain specific predictions
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(1)].prob AS churn_probability,
  attribution.*
FROM
  ML.EXPLAIN_PREDICT(MODEL `my_dataset.churn_model`,
    (
      SELECT *
      FROM `my_dataset.customers_to_score`
      WHERE customer_id IN ('cust-123', 'cust-456', 'cust-789')
    ),
    STRUCT(3 AS top_k_features));

-- Aggregate feature attributions
WITH explanations AS (
  SELECT
    customer_id,
    predicted_churned,
    attribution.feature_name,
    attribution.attribution AS feature_attribution
  FROM
    ML.EXPLAIN_PREDICT(MODEL `my_dataset.churn_model`,
      (SELECT * FROM `my_dataset.customers_to_score`),
      STRUCT(5 AS top_k_features)),
    UNNEST(top_feature_attributions) AS attribution
)
SELECT
  feature_name,
  COUNT(*) AS prediction_count,
  AVG(feature_attribution) AS avg_attribution,
  STDDEV(feature_attribution) AS stddev_attribution
FROM
  explanations
WHERE
  predicted_churned = 1  -- Focus on churn predictions
GROUP BY
  feature_name
ORDER BY
  ABS(avg_attribution) DESC;
```

### Model Diagnostics

**Understanding Model Behavior:**
```sql
-- Get training statistics
SELECT
  *
FROM
  ML.TRAINING_INFO(MODEL `my_dataset.churn_model`)
ORDER BY
  iteration DESC;

-- Analyze iteration metrics
SELECT
  iteration,
  loss,
  eval_loss,
  duration_ms,
  learning_rate
FROM
  ML.TRAINING_INFO(MODEL `my_dataset.churn_model`)
ORDER BY
  iteration;

-- Check for overfitting
WITH training_metrics AS (
  SELECT
    iteration,
    loss AS training_loss,
    eval_loss AS validation_loss,
    ABS(loss - eval_loss) AS loss_diff
  FROM
    ML.TRAINING_INFO(MODEL `my_dataset.churn_model`)
)
SELECT
  iteration,
  training_loss,
  validation_loss,
  loss_diff,
  CASE
    WHEN loss_diff > 0.1 THEN 'Potential overfitting'
    ELSE 'Normal'
  END AS status
FROM
  training_metrics
ORDER BY
  iteration DESC
LIMIT 10;

-- Model weights (for linear models)
SELECT
  *
FROM
  ML.WEIGHTS(MODEL `my_dataset.linear_model`)
ORDER BY
  ABS(weight) DESC;

-- Centroids (for clustering)
SELECT
  *
FROM
  ML.CENTROIDS(MODEL `my_dataset.customer_clusters`);
```

---

## 7. Model Management

### Model Versioning

**Best Practices:**
- Version models systematically
- Document model changes
- Compare model versions
- Maintain model registry

```sql
-- Create model with version info
CREATE OR REPLACE MODEL `my_dataset.churn_model_v2`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['churned'],
  labels=[
    ('version', '2.0'),
    ('created_by', 'data-team'),
    ('purpose', 'production'),
    ('last_updated', '2025-01-15')
  ]
) AS
SELECT * FROM `my_dataset.training_data`;

-- List all models
SELECT
  model_catalog,
  model_schema,
  model_name,
  model_type,
  creation_time,
  labels
FROM
  `my_dataset.INFORMATION_SCHEMA.MODELS`
ORDER BY
  creation_time DESC;

-- Compare model performance
WITH model_v1 AS (
  SELECT 'v1' AS version, *
  FROM ML.EVALUATE(MODEL `my_dataset.churn_model_v1`)
),
model_v2 AS (
  SELECT 'v2' AS version, *
  FROM ML.EVALUATE(MODEL `my_dataset.churn_model_v2`)
)
SELECT
  version,
  precision,
  recall,
  f1_score,
  roc_auc,
  log_loss
FROM model_v1
UNION ALL
SELECT
  version,
  precision,
  recall,
  f1_score,
  roc_auc,
  log_loss
FROM model_v2
ORDER BY roc_auc DESC;

-- Get model metadata
SELECT
  *
FROM
  `my_dataset.INFORMATION_SCHEMA.MODELS`
WHERE
  model_name = 'churn_model';

-- Model lineage
SELECT
  creation_time,
  model_type,
  training_runs[SAFE_OFFSET(0)].training_options AS training_options
FROM
  `my_dataset.INFORMATION_SCHEMA.MODELS`
WHERE
  model_name LIKE 'churn_model%'
ORDER BY
  creation_time DESC;
```

### Model Export and Import

**Model Portability:**
```sql
-- Export model to Cloud Storage
EXPORT MODEL `my_dataset.churn_model`
OPTIONS(
  uri='gs://my-bucket/models/churn_model/*'
);

-- Import TensorFlow model
CREATE OR REPLACE MODEL `my_dataset.imported_tf_model`
OPTIONS(
  model_type='TENSORFLOW',
  model_path='gs://my-bucket/tf-models/saved_model/*'
);

-- Import ONNX model
CREATE OR REPLACE MODEL `my_dataset.imported_onnx_model`
OPTIONS(
  model_type='ONNX',
  model_path='gs://my-bucket/onnx-models/model.onnx'
);

-- Register model in Vertex AI
CREATE OR REPLACE MODEL `my_dataset.vertex_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label'],
  model_registry='vertex_ai',
  vertex_ai_model_id='projects/my-project/locations/us-central1/models/12345'
) AS
SELECT * FROM `my_dataset.training_data`;
```

### Model Monitoring

**Track Model Performance:**
```sql
-- Create monitoring table
CREATE OR REPLACE TABLE `my_dataset.model_predictions_log` AS
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(1)].prob AS churn_probability,
  CURRENT_TIMESTAMP() AS prediction_timestamp
FROM
  ML.PREDICT(MODEL `my_dataset.churn_model`,
    (SELECT * FROM `my_dataset.customers_to_score`));

-- Monitor prediction distribution
SELECT
  DATE(prediction_timestamp) AS prediction_date,
  COUNT(*) AS total_predictions,
  COUNTIF(predicted_churned = 1) AS predicted_churns,
  AVG(churn_probability) AS avg_churn_probability,
  STDDEV(churn_probability) AS stddev_churn_probability
FROM
  `my_dataset.model_predictions_log`
GROUP BY
  prediction_date
ORDER BY
  prediction_date DESC;

-- Detect prediction drift
WITH daily_stats AS (
  SELECT
    DATE(prediction_timestamp) AS prediction_date,
    AVG(churn_probability) AS avg_probability,
    STDDEV(churn_probability) AS stddev_probability
  FROM
    `my_dataset.model_predictions_log`
  GROUP BY
    prediction_date
),
baseline AS (
  SELECT
    AVG(avg_probability) AS baseline_avg,
    AVG(stddev_probability) AS baseline_stddev
  FROM
    daily_stats
  WHERE
    prediction_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
                       AND DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
)
SELECT
  ds.prediction_date,
  ds.avg_probability,
  b.baseline_avg,
  ABS(ds.avg_probability - b.baseline_avg) AS drift,
  CASE
    WHEN ABS(ds.avg_probability - b.baseline_avg) > 0.1 THEN 'Alert'
    ELSE 'Normal'
  END AS status
FROM
  daily_stats ds
CROSS JOIN
  baseline b
WHERE
  ds.prediction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
ORDER BY
  ds.prediction_date DESC;
```

---

## 8. Hyperparameter Tuning

### Manual Tuning

**Best Practices:**
- Test different hyperparameters systematically
- Use cross-validation approach
- Document tuning experiments
- Compare results objectively

```sql
-- Create multiple models with different hyperparameters
CREATE OR REPLACE MODEL `my_dataset.model_hp_1`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['label'],
  max_iterations=50,
  learn_rate=0.1,
  subsample=0.8,
  max_tree_depth=6
) AS
SELECT * FROM `my_dataset.training_data`;

CREATE OR REPLACE MODEL `my_dataset.model_hp_2`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['label'],
  max_iterations=100,
  learn_rate=0.05,
  subsample=0.7,
  max_tree_depth=8
) AS
SELECT * FROM `my_dataset.training_data`;

CREATE OR REPLACE MODEL `my_dataset.model_hp_3`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['label'],
  max_iterations=150,
  learn_rate=0.01,
  subsample=0.9,
  max_tree_depth=10
) AS
SELECT * FROM `my_dataset.training_data`;

-- Compare hyperparameter tuning results
WITH hp1_eval AS (
  SELECT 'hp_1' AS experiment, 50 AS max_iter, 0.1 AS lr, *
  FROM ML.EVALUATE(MODEL `my_dataset.model_hp_1`)
),
hp2_eval AS (
  SELECT 'hp_2' AS experiment, 100 AS max_iter, 0.05 AS lr, *
  FROM ML.EVALUATE(MODEL `my_dataset.model_hp_2`)
),
hp3_eval AS (
  SELECT 'hp_3' AS experiment, 150 AS max_iter, 0.01 AS lr, *
  FROM ML.EVALUATE(MODEL `my_dataset.model_hp_3`)
)
SELECT
  experiment,
  max_iter,
  lr AS learning_rate,
  roc_auc,
  precision,
  recall,
  f1_score
FROM hp1_eval
UNION ALL
SELECT experiment, max_iter, lr, roc_auc, precision, recall, f1_score FROM hp2_eval
UNION ALL
SELECT experiment, max_iter, lr, roc_auc, precision, recall, f1_score FROM hp3_eval
ORDER BY roc_auc DESC;
```

### Grid Search Pattern

**Systematic Hyperparameter Search:**
```sql
-- Create script for grid search
DECLARE hyperparameters ARRAY<STRUCT<learn_rate FLOAT64, max_depth INT64>>;
SET hyperparameters = [
  (0.01, 5),
  (0.01, 10),
  (0.05, 5),
  (0.05, 10),
  (0.1, 5),
  (0.1, 10)
];

-- Create table to store results
CREATE OR REPLACE TABLE `my_dataset.hyperparameter_results` (
  model_name STRING,
  learn_rate FLOAT64,
  max_depth INT64,
  roc_auc FLOAT64,
  precision FLOAT64,
  recall FLOAT64,
  f1_score FLOAT64,
  training_time_seconds INT64
);

-- Note: In practice, you would loop through hyperparameters
-- and create/evaluate models programmatically
```

---

## 9. Integration with Vertex AI

### Model Registration

**Best Practices:**
- Register models in Vertex AI for MLOps
- Use version aliases
- Enable model monitoring
- Implement CI/CD

```sql
-- Create model and register in Vertex AI
CREATE OR REPLACE MODEL `my_dataset.production_model`
OPTIONS(
  model_type='BOOSTED_TREE_CLASSIFIER',
  input_label_cols=['label'],
  
  -- Vertex AI registration
  model_registry='vertex_ai',
  vertex_ai_model_version_aliases=['champion', 'production', 'v1'],
  
  -- Enable explainability
  enable_global_explain=TRUE,
  
  -- Model metadata
  labels=[
    ('env', 'production'),
    ('team', 'ml-team'),
    ('use_case', 'churn-prediction')
  ]
) AS
SELECT * FROM `my_dataset.training_data`;

-- Export for Vertex AI deployment
EXPORT MODEL `my_dataset.production_model`
OPTIONS(
  uri='gs://my-bucket/vertex-models/churn-model/*'
);
```

---

## 10. Cost Optimization

### Best Practices

**Reduce Training Costs:**
```sql
-- Use data sampling for large datasets
CREATE OR REPLACE MODEL `my_dataset.sampled_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT
  *
FROM
  `my_dataset.large_training_data`
WHERE
  RAND() < 0.1;  -- 10% sample

-- Use clustered tables for faster queries
CREATE OR REPLACE TABLE `my_dataset.training_data_clustered`
CLUSTER BY customer_segment
AS SELECT * FROM `my_dataset.training_data`;

-- Partition training data by date
CREATE OR REPLACE TABLE `my_dataset.training_data_partitioned`
PARTITION BY DATE(event_date)
CLUSTER BY customer_segment
AS SELECT * FROM `my_dataset.training_data`;

-- Use partitioned tables in training
CREATE OR REPLACE MODEL `my_dataset.efficient_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT
  * EXCEPT(event_date)
FROM
  `my_dataset.training_data_partitioned`
WHERE
  event_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY);

-- Monitor slot usage
SELECT
  user_email,
  job_id,
  total_slot_ms,
  total_bytes_processed,
  creation_time
FROM
  `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  statement_type = 'CREATE_MODEL'
  AND creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
ORDER BY
  total_slot_ms DESC;
```

---

## 11. Common Anti-Patterns

### ❌ Anti-Pattern 1: Not Using TRANSFORM
**Problem:** Features need preprocessing at prediction time
**Solution:** Use TRANSFORM clause for consistent preprocessing

### ❌ Anti-Pattern 2: Ignoring Data Leakage
**Problem:** Including future information in training
**Solution:** Carefully check feature engineering for temporal leaks

### ❌ Anti-Pattern 3: No Model Evaluation
**Problem:** Deploying models without testing
**Solution:** Always evaluate on holdout test set

### ❌ Anti-Pattern 4: Over-Complex Models
**Problem:** Using DNN when linear regression suffices
**Solution:** Start simple, increase complexity if needed

### ❌ Anti-Pattern 5: Not Monitoring Predictions
**Problem:** Model degradation goes unnoticed
**Solution:** Log predictions and monitor distributions

---

## 12. Quick Reference Checklist

### Model Creation
- [ ] Choose appropriate model type
- [ ] Use TRANSFORM for feature engineering
- [ ] Handle missing values
- [ ] Split data appropriately
- [ ] Set reasonable hyperparameters

### Training
- [ ] Monitor training progress
- [ ] Enable early stopping
- [ ] Handle class imbalance
- [ ] Document training configuration
- [ ] Version models properly

### Evaluation
- [ ] Evaluate on test set
- [ ] Check multiple metrics
- [ ] Analyze feature importance
- [ ] Compare against baseline
- [ ] Check for bias

### Deployment
- [ ] Test predictions
- [ ] Monitor performance
- [ ] Log predictions
- [ ] Set up alerts
- [ ] Document model usage

### Maintenance
- [ ] Schedule retraining
- [ ] Monitor data drift
- [ ] Update features as needed
- [ ] Archive old versions
- [ ] Document changes

---

*Best Practices for Google Cloud Data Engineer Certification*
