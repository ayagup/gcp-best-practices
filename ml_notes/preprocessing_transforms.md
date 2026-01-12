# Data Preprocessing Transforms for Model Training

## Overview

Data preprocessing transforms are essential steps to prepare raw data for machine learning models. Proper preprocessing improves model performance, training speed, and generalization.

## Categories of Preprocessing Transforms

### 1. **Numerical Feature Transforms**

#### 1.1 Scaling/Normalization
- **Min-Max Scaling (Normalization)**
  - Scales features to [0, 1] range
  - Formula: `(x - min) / (max - min)`
  - Use: When features have different scales, neural networks
  ```python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)
  ```

- **Standardization (Z-score normalization)**
  - Centers data around mean=0, std=1
  - Formula: `(x - mean) / std`
  - Use: When features follow Gaussian distribution, SVM, logistic regression
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_standardized = scaler.fit_transform(X)
  ```

- **Robust Scaling**
  - Uses median and IQR instead of mean/std
  - Formula: `(x - median) / IQR`
  - Use: When data has outliers
  ```python
  from sklearn.preprocessing import RobustScaler
  scaler = RobustScaler()
  X_robust = scaler.fit_transform(X)
  ```

- **MaxAbs Scaling**
  - Scales by maximum absolute value to [-1, 1]
  - Use: Sparse data, preserves zero entries
  ```python
  from sklearn.preprocessing import MaxAbsScaler
  scaler = MaxAbsScaler()
  X_scaled = scaler.fit_transform(X)
  ```

#### 1.2 Distribution Transforms
- **Log Transform**
  - Handles skewed distributions
  - Formula: `log(x + 1)` or `log(x)`
  - Use: Right-skewed data (income, prices)
  ```python
  import numpy as np
  X_log = np.log1p(X)  # log(1 + x)
  ```

- **Square Root Transform**
  - Reduces right skew
  - Formula: `sqrt(x)`
  - Use: Count data, Poisson-distributed features

- **Box-Cox Transform**
  - Automatic power transformation
  - Finds optimal lambda parameter
  - Use: Making data more Gaussian
  ```python
  from sklearn.preprocessing import PowerTransformer
  pt = PowerTransformer(method='box-cox')
  X_transformed = pt.fit_transform(X)
  ```

- **Yeo-Johnson Transform**
  - Like Box-Cox but handles negative values
  - Use: Mixed positive/negative data
  ```python
  from sklearn.preprocessing import PowerTransformer
  pt = PowerTransformer(method='yeo-johnson')
  X_transformed = pt.fit_transform(X)
  ```

#### 1.3 Binning/Discretization
- **Equal-Width Binning**
  - Divides range into equal intervals
  - Use: Converting continuous to categorical
  ```python
  from sklearn.preprocessing import KBinsDiscretizer
  kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
  X_binned = kbd.fit_transform(X)
  ```

- **Equal-Frequency Binning (Quantile)**
  - Each bin has same number of samples
  - Use: Handling skewed distributions
  ```python
  kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
  X_binned = kbd.fit_transform(X)
  ```

- **Custom Binning**
  - Domain-specific bin edges
  - Use: Age groups, income brackets
  ```python
  import pandas as pd
  bins = [0, 18, 35, 60, 100]
  labels = ['child', 'young_adult', 'adult', 'senior']
  df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
  ```

### 2. **Categorical Feature Transforms**

#### 2.1 Encoding Techniques
- **Label Encoding**
  - Converts categories to integers (0, 1, 2, ...)
  - Use: Ordinal features, tree-based models
  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  X_encoded = le.fit_transform(X)
  ```

- **One-Hot Encoding**
  - Creates binary column for each category
  - Use: Nominal features, linear models, neural networks
  ```python
  from sklearn.preprocessing import OneHotEncoder
  ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
  X_encoded = ohe.fit_transform(X)
  
  # Pandas
  pd.get_dummies(df, columns=['category'])
  ```

- **Ordinal Encoding**
  - Maps categories to ordered integers
  - Use: When categories have natural order
  ```python
  from sklearn.preprocessing import OrdinalEncoder
  oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])
  X_encoded = oe.fit_transform(X)
  ```

- **Target Encoding (Mean Encoding)**
  - Replaces category with target mean
  - Use: High-cardinality features
  - Warning: Can cause overfitting, use cross-validation
  ```python
  # Manual implementation
  target_mean = df.groupby('category')['target'].mean()
  df['category_encoded'] = df['category'].map(target_mean)
  ```

- **Frequency Encoding**
  - Replaces category with its frequency
  - Use: High-cardinality features
  ```python
  freq = df['category'].value_counts()
  df['category_freq'] = df['category'].map(freq)
  ```

- **Binary Encoding**
  - Converts to binary representation
  - Use: High-cardinality features (fewer dimensions than one-hot)
  ```python
  import category_encoders as ce
  encoder = ce.BinaryEncoder(cols=['category'])
  X_encoded = encoder.fit_transform(X)
  ```

- **Hash Encoding**
  - Uses hash function to map categories
  - Use: Very high-cardinality, streaming data
  ```python
  from sklearn.feature_extraction import FeatureHasher
  hasher = FeatureHasher(n_features=10, input_type='string')
  X_hashed = hasher.transform(X)
  ```

#### 2.2 Handling High Cardinality
- **Top-N Categories + "Other"**
  - Keep most frequent categories, group rest
  ```python
  top_categories = df['category'].value_counts().head(10).index
  df['category_grouped'] = df['category'].apply(
      lambda x: x if x in top_categories else 'Other'
  )
  ```

- **Embedding Layers**
  - Learn dense representations (neural networks)
  - Use: Deep learning with categorical features
  ```python
  from tensorflow.keras.layers import Embedding
  embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
  ```

### 3. **Text Feature Transforms**

#### 3.1 Text Cleaning
- **Lowercasing**
  ```python
  text = text.lower()
  ```

- **Remove Punctuation**
  ```python
  import string
  text = text.translate(str.maketrans('', '', string.punctuation))
  ```

- **Remove Stop Words**
  ```python
  from nltk.corpus import stopwords
  stop_words = set(stopwords.words('english'))
  words = [w for w in words if w not in stop_words]
  ```

- **Stemming/Lemmatization**
  ```python
  from nltk.stem import PorterStemmer, WordNetLemmatizer
  stemmer = PorterStemmer()
  lemmatizer = WordNetLemmatizer()
  stemmed = [stemmer.stem(word) for word in words]
  lemmatized = [lemmatizer.lemmatize(word) for word in words]
  ```

#### 3.2 Text Vectorization
- **Bag of Words (CountVectorizer)**
  - Word frequency counts
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  vectorizer = CountVectorizer(max_features=1000)
  X_bow = vectorizer.fit_transform(texts)
  ```

- **TF-IDF (Term Frequency-Inverse Document Frequency)**
  - Weighted word importance
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  vectorizer = TfidfVectorizer(max_features=1000)
  X_tfidf = vectorizer.fit_transform(texts)
  ```

- **N-grams**
  - Capture word sequences
  ```python
  vectorizer = CountVectorizer(ngram_range=(1, 3))
  X_ngrams = vectorizer.fit_transform(texts)
  ```

- **Word Embeddings**
  - Word2Vec, GloVe, FastText
  - Dense vector representations
  ```python
  from gensim.models import Word2Vec
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
  ```

- **Contextual Embeddings**
  - BERT, GPT, etc.
  - Context-aware representations

### 4. **Date/Time Feature Transforms**

#### 4.1 Temporal Decomposition
- **Extract Components**
  ```python
  df['year'] = df['date'].dt.year
  df['month'] = df['date'].dt.month
  df['day'] = df['date'].dt.day
  df['dayofweek'] = df['date'].dt.dayofweek
  df['hour'] = df['date'].dt.hour
  df['quarter'] = df['date'].dt.quarter
  df['week'] = df['date'].dt.isocalendar().week
  ```

- **Cyclical Encoding**
  - Captures cyclical nature (months, hours)
  ```python
  df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
  df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
  df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
  df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
  ```

- **Time Since Event**
  ```python
  df['days_since_signup'] = (df['current_date'] - df['signup_date']).dt.days
  ```

- **Is Weekend/Holiday**
  ```python
  df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
  ```

### 5. **Missing Value Handling**

#### 5.1 Imputation Strategies
- **Mean/Median/Mode Imputation**
  ```python
  from sklearn.impute import SimpleImputer
  imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
  X_imputed = imputer.fit_transform(X)
  ```

- **Constant Value Imputation**
  ```python
  imputer = SimpleImputer(strategy='constant', fill_value=0)
  X_imputed = imputer.fit_transform(X)
  ```

- **Forward Fill / Backward Fill**
  ```python
  df.fillna(method='ffill')  # Forward fill
  df.fillna(method='bfill')  # Backward fill
  ```

- **K-Nearest Neighbors Imputation**
  ```python
  from sklearn.impute import KNNImputer
  imputer = KNNImputer(n_neighbors=5)
  X_imputed = imputer.fit_transform(X)
  ```

- **Iterative Imputation (MICE)**
  ```python
  from sklearn.impute import IterativeImputer
  imputer = IterativeImputer(random_state=0)
  X_imputed = imputer.fit_transform(X)
  ```

- **Domain-Specific Imputation**
  - Use business logic or external data

#### 5.2 Missing Indicator
- **Add Missing Flag**
  ```python
  from sklearn.impute import MissingIndicator
  indicator = MissingIndicator()
  X_missing_mask = indicator.fit_transform(X)
  ```

### 6. **Outlier Handling**

#### 6.1 Detection Methods
- **IQR Method**
  ```python
  Q1 = df['feature'].quantile(0.25)
  Q3 = df['feature'].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_clean = df[(df['feature'] >= lower_bound) & (df['feature'] <= upper_bound)]
  ```

- **Z-Score Method**
  ```python
  from scipy import stats
  z_scores = np.abs(stats.zscore(df['feature']))
  df_clean = df[z_scores < 3]
  ```

- **Percentile Method**
  ```python
  lower = df['feature'].quantile(0.01)
  upper = df['feature'].quantile(0.99)
  df_clean = df[(df['feature'] >= lower) & (df['feature'] <= upper)]
  ```

#### 6.2 Handling Strategies
- **Remove Outliers**
- **Cap/Winsorize**
  ```python
  from scipy.stats.mstats import winsorize
  df['feature_winsorized'] = winsorize(df['feature'], limits=[0.05, 0.05])
  ```
- **Transform (log, sqrt)**
- **Treat Separately (create indicator)**

### 7. **Feature Engineering Transforms**

#### 7.1 Polynomial Features
- **Create Interactions and Powers**
  ```python
  from sklearn.preprocessing import PolynomialFeatures
  poly = PolynomialFeatures(degree=2, include_bias=False)
  X_poly = poly.fit_transform(X)
  ```

#### 7.2 Mathematical Transformations
- **Ratios**
  ```python
  df['debt_to_income'] = df['debt'] / df['income']
  ```

- **Differences**
  ```python
  df['price_change'] = df['current_price'] - df['previous_price']
  ```

- **Aggregations**
  ```python
  df['total_purchases'] = df[['online_purchases', 'store_purchases']].sum(axis=1)
  ```

#### 7.3 Domain-Specific Features
- **BMI from height/weight**
  ```python
  df['bmi'] = df['weight'] / (df['height'] ** 2)
  ```

- **Customer Lifetime Value**
- **Recency, Frequency, Monetary (RFM)**

### 8. **Image Transforms**

#### 8.1 Preprocessing
- **Resizing**
  ```python
  from tensorflow.keras.preprocessing.image import img_to_array, load_img
  img = load_img('image.jpg', target_size=(224, 224))
  ```

- **Normalization**
  ```python
  img_array = img_array / 255.0  # Scale to [0, 1]
  # Or use mean/std normalization
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  ```

- **Center Cropping**
- **Color Space Conversion** (RGB to Grayscale)

#### 8.2 Augmentation (Training Only)
- **Horizontal/Vertical Flip**
- **Rotation**
- **Zoom**
- **Brightness/Contrast Adjustment**
- **Random Crop**
  ```python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  datagen = ImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      zoom_range=0.2
  )
  ```

### 9. **Dimensionality Reduction**

- **Principal Component Analysis (PCA)**
  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=0.95)  # Keep 95% variance
  X_pca = pca.fit_transform(X)
  ```

- **Linear Discriminant Analysis (LDA)**
  ```python
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  lda = LinearDiscriminantAnalysis(n_components=2)
  X_lda = lda.fit_transform(X, y)
  ```

- **t-SNE**
  ```python
  from sklearn.manifold import TSNE
  tsne = TSNE(n_components=2)
  X_tsne = tsne.fit_transform(X)
  ```

- **UMAP**
  ```python
  import umap
  reducer = umap.UMAP(n_components=2)
  X_umap = reducer.fit_transform(X)
  ```

- **Feature Selection**
  - Variance Threshold
  - SelectKBest
  - Recursive Feature Elimination (RFE)

### 10. **Data Balancing (For Classification)**

- **Oversampling**
  - Random Oversampling
  - SMOTE (Synthetic Minority Over-sampling)
  ```python
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(random_state=42)
  X_resampled, y_resampled = smote.fit_resample(X, y)
  ```

- **Undersampling**
  - Random Undersampling
  - NearMiss
  ```python
  from imblearn.under_sampling import RandomUnderSampler
  rus = RandomUnderSampler(random_state=42)
  X_resampled, y_resampled = rus.fit_resample(X, y)
  ```

- **Combined Methods**
  - SMOTEENN
  - SMOTETomek

- **Class Weights**
  ```python
  from sklearn.utils.class_weight import compute_class_weight
  class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
  ```

### 11. **Time Series Specific Transforms**

- **Lag Features**
  ```python
  df['lag_1'] = df['value'].shift(1)
  df['lag_7'] = df['value'].shift(7)
  ```

- **Rolling Statistics**
  ```python
  df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
  df['rolling_std_7'] = df['value'].rolling(window=7).std()
  ```

- **Differencing**
  ```python
  df['diff_1'] = df['value'].diff(1)
  df['diff_seasonal'] = df['value'].diff(12)  # For monthly data
  ```

- **Exponential Moving Average**
  ```python
  df['ema'] = df['value'].ewm(span=10, adjust=False).mean()
  ```

- **Fourier Features**
  ```python
  from scipy import fft
  fft_values = fft.fft(df['value'].values)
  ```

### 12. **GCP-Specific Transforms (TensorFlow Transform in TFX)**

```python
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    outputs = {}
    
    # Numerical scaling
    outputs['scaled_feature'] = tft.scale_to_0_1(inputs['numerical_feature'])
    outputs['normalized_feature'] = tft.scale_to_z_score(inputs['numerical_feature'])
    
    # Categorical encoding
    outputs['encoded_category'] = tft.compute_and_apply_vocabulary(
        inputs['category'],
        top_k=100,
        num_oov_buckets=1
    )
    
    # Bucketization
    outputs['binned_feature'] = tft.bucketize(
        inputs['numerical_feature'],
        num_buckets=10
    )
    
    # Handle missing values
    outputs['imputed_feature'] = tft.sparse_tensor_to_dense_with_shape(
        inputs['sparse_feature'],
        default_value=0.0
    )
    
    return outputs
```

## Transform Pipeline Best Practices

### 1. **Order Matters**
Typical pipeline order:
1. Handle missing values
2. Remove/handle outliers
3. Encode categorical variables
4. Create new features
5. Scale/normalize numerical features
6. Dimensionality reduction (if needed)

### 2. **Train/Test Split Before Transforms**
- Always split data BEFORE applying transforms
- Fit transforms only on training data
- Apply fitted transforms to validation/test data

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit on training data only
scaler = StandardScaler()
scaler.fit(X_train)

# Transform both
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use fitted scaler
```

### 3. **Use Pipelines for Reproducibility**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numerical_features = ['age', 'income']
categorical_features = ['gender', 'city']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

model_pipeline.fit(X_train, y_train)
```

### 4. **Save Transformers for Production**
```python
import joblib

# Save fitted transformers
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Load in production
scaler = joblib.load('scaler.pkl')
X_new_scaled = scaler.transform(X_new)
```

## Transform Selection Guide

| Data Type | Recommended Transforms |
|-----------|------------------------|
| **Numerical (Continuous)** | StandardScaler, MinMaxScaler, Log Transform |
| **Numerical (Counts)** | Log Transform, Square Root, Poisson Scaling |
| **Numerical (Skewed)** | Log, Box-Cox, Yeo-Johnson |
| **Categorical (Low Cardinality)** | One-Hot Encoding |
| **Categorical (High Cardinality)** | Target Encoding, Hash Encoding, Embeddings |
| **Categorical (Ordinal)** | Ordinal Encoding, Label Encoding |
| **Text** | TF-IDF, Word Embeddings (Word2Vec, BERT) |
| **Dates** | Cyclical Encoding, Extract Components |
| **Images** | Normalization, Resizing, Augmentation |
| **Time Series** | Lag Features, Rolling Stats, Differencing |

## Model-Specific Requirements

### Neural Networks
- **Required**: Normalization/Standardization (all features in similar range)
- **Optional**: Dimensionality reduction for high-dimensional data

### Tree-Based Models (Random Forest, XGBoost)
- **Not Required**: Scaling/normalization
- **Required**: Encode categorical variables (Label or One-Hot)
- **Beneficial**: Handle missing values explicitly

### Linear Models (Logistic Regression, SVM)
- **Required**: Feature scaling (StandardScaler)
- **Required**: One-Hot encoding for categorical
- **Beneficial**: Polynomial features for non-linear relationships

### K-Nearest Neighbors
- **Critical**: Feature scaling (distances are scale-dependent)
- **Required**: Handle missing values
- **Beneficial**: Dimensionality reduction

## GCP Tools for Preprocessing

| Tool | Use Case |
|------|----------|
| **TensorFlow Transform (TFT)** | Preprocessing in TFX pipelines |
| **Dataflow** | Large-scale data preprocessing |
| **BigQuery ML** | SQL-based transforms for tabular data |
| **Dataprep** | Visual data preparation |
| **Vertex AI Feature Store** | Feature serving and versioning |
| **tf.data API** | Online preprocessing during training |

## Random Forest vs Deep Neural Networks (DNN): Preprocessing Comparison

### Overview

Random Forest and DNN models have **fundamentally different preprocessing requirements** due to their architectural differences. Understanding these differences is crucial for efficient model development and optimal model performance.

---

### 1. Feature Scaling/Normalization

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Required?** | ❌ **NO** | ✅ **YES - Critical** |
| **Reason** | Tree splits are invariant to monotonic transformations | Gradient descent requires similar feature scales |
| **Impact if skipped** | None - works perfectly | Poor convergence, unstable training, vanishing/exploding gradients |
| **Recommended Method** | Not needed | StandardScaler or MinMaxScaler |

**Example:**
```python
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dataset with different scales
X_train = np.array([
    [25, 50000, 3],      # age, income, children
    [35, 80000, 2],
    [45, 120000, 1]
])

# Random Forest - No scaling needed
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)  # Works great with raw features

# DNN - Scaling REQUIRED
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

dnn_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
dnn_model.fit(X_train_scaled, y_train)  # Needs scaled features
```

**Why DNN needs scaling:**
- Income (50k-120k) dominates gradients vs age (25-45)
- Network weights initialize around 0, expects normalized inputs
- Different scales cause different learning rates per feature

---

### 2. Categorical Encoding

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Method** | Label Encoding or One-Hot | One-Hot Encoding (preferred) or Embeddings |
| **Label Encoding OK?** | ✅ YES | ❌ NO (creates false ordinal relationships) |
| **One-Hot Encoding** | ✅ Works, but increases memory | ✅ Preferred for low cardinality |
| **High Cardinality** | Label/Target Encoding | Embedding Layers |
| **Handles Native** | Can handle strings (sklearn) | Must be numeric |

**Example:**
```python
# Dataset
categories = ['red', 'blue', 'green', 'red', 'blue']

# Random Forest - Label encoding is fine
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_encoded = le.fit_transform(categories)  # [2, 0, 1, 2, 0]
rf_model.fit(X_encoded.reshape(-1, 1), y)  # Works!

# DNN - One-hot encoding or embeddings needed
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
X_onehot = ohe.fit_transform(np.array(categories).reshape(-1, 1))
# [[0, 0, 1],   # red
#  [1, 0, 0],   # blue
#  [0, 1, 0],   # green
#  [0, 0, 1],   # red
#  [1, 0, 0]]   # blue

dnn_model.fit(X_onehot, y)

# Or use embedding layer for high cardinality
input_layer = keras.layers.Input(shape=(1,))
embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=8)(input_layer)
```

**Why DNN needs one-hot/embeddings:**
- Label encoding: `blue=0, green=1, red=2` implies `red > green > blue`
- DNN learns: `green = (blue + red) / 2` ❌
- Random Forest splits: "Is color == blue?" ✅

---

### 3. Missing Value Handling

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Native Handling** | ✅ Partial (sklearn uses surrogates) | ❌ NO - must impute |
| **Required?** | Recommended but not always critical | ✅ Mandatory |
| **Best Practice** | Impute + add missing indicator | Impute (mean/median/mode or model-based) |
| **NaN Impact** | Can work around | Training fails |

**Example:**
```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Dataset with missing values
X = pd.DataFrame({
    'age': [25, np.nan, 35, 45],
    'income': [50000, 60000, np.nan, 80000]
})

# Random Forest - Can sometimes handle NaN
rf_model = RandomForestClassifier()
# Some implementations handle NaN, but best to impute

# DNN - MUST impute
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

dnn_model.fit(X_imputed, y)  # No NaN allowed

# Best practice for both: Impute + add indicator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

imputer_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True))
])
X_processed = imputer_pipeline.fit_transform(X)
```

---

### 4. Outlier Handling

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Sensitive?** | ❌ Robust to outliers | ✅ Very sensitive |
| **Reason** | Tree splits partition data | Outliers distort weight updates |
| **Action Needed** | Often none | Remove, cap, or transform |
| **Impact** | Minimal | Significant performance degradation |

**Example:**
```python
# Dataset with outlier
X_train = np.array([
    [25, 50000],
    [30, 55000],
    [35, 60000],
    [40, 5000000]  # Outlier: 5M income
])

# Random Forest - Handles gracefully
rf_model.fit(X_train, y_train)  # Still works well

# DNN - Outlier causes problems
# Option 1: Remove outliers
from scipy import stats
z_scores = np.abs(stats.zscore(X_train))
X_clean = X_train[np.all(z_scores < 3, axis=1)]

# Option 2: Cap outliers (Winsorization)
from scipy.stats.mstats import winsorize
X_train[:, 1] = winsorize(X_train[:, 1], limits=[0.05, 0.05])

# Option 3: Log transform
X_train[:, 1] = np.log1p(X_train[:, 1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
dnn_model.fit(X_scaled, y_train)
```

---

### 5. Feature Engineering

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Need explicit interactions?** | ❌ NO - discovers automatically | ✅ YES - helps significantly |
| **Polynomial features** | Usually unnecessary | Can improve performance |
| **Domain features** | Beneficial | Very beneficial |
| **Feature complexity** | Handles non-linearity natively | Benefits from pre-engineered features |

**Example:**
```python
from sklearn.preprocessing import PolynomialFeatures

X = np.array([
    [2, 3],
    [3, 4],
    [4, 5]
])

# Random Forest - No need for interaction terms
rf_model.fit(X, y)  # Automatically finds: if X1 > 2 AND X2 < 4 then...

# DNN - Benefits from explicit interactions
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Original: [2, 3]
# Becomes: [2, 3, 4, 6, 9]  # [x1, x2, x1^2, x1*x2, x2^2]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
dnn_model.fit(X_scaled, y)  # Better performance with interactions
```

---

### 6. Data Volume Requirements

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Min samples** | 100s-1000s | 10,000s-1,000,000s |
| **Overfitting risk** | Lower (with proper params) | Higher (needs more data) |
| **Data efficiency** | ✅ Works well with small data | ❌ Needs large datasets |
| **Augmentation** | Not typically used | Critical for images/small datasets |

---

### 7. Feature Distribution

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Needs Gaussian?** | ❌ NO | ✅ Helps significantly |
| **Skewed distributions** | Handles naturally | Should transform (log, Box-Cox) |
| **Multi-modal** | No problem | Can cause issues |

**Example:**
```python
# Skewed income data
income = np.array([30000, 35000, 40000, 45000, 500000])  # Right-skewed

# Random Forest - works with raw data
rf_model.fit(income.reshape(-1, 1), y)

# DNN - transform to more Gaussian
income_log = np.log1p(income)  # Log transform
# Or Box-Cox
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')
income_transformed = pt.fit_transform(income.reshape(-1, 1))

scaler = StandardScaler()
income_scaled = scaler.fit_transform(income_transformed)
dnn_model.fit(income_scaled, y)
```

---

### 8. Dimensionality

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **High dimensions** | Handles well (1000s of features) | Can struggle, needs more data |
| **Feature selection** | Built-in importance | Recommended for very high dims |
| **Curse of dimensionality** | Less affected | More affected |
| **PCA/reduction** | Usually not needed | Can help with 1000+ features |

---

### 9. Imbalanced Classes

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Native handling** | `class_weight='balanced'` | Custom loss weights or sampling |
| **SMOTE/oversampling** | Optional | Recommended |
| **Undersampling** | Risky (loses data) | Very risky (needs data) |

**Example:**
```python
from imblearn.over_sampling import SMOTE

# Imbalanced: 95% class 0, 5% class 1

# Random Forest - class weights
rf_model = RandomForestClassifier(class_weight='balanced')
rf_model.fit(X_train, y_train)  # Often sufficient

# DNN - SMOTE + class weights
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Or use class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

dnn_model.fit(X_resampled, y_resampled, 
              class_weight=class_weight_dict)
```

---

### 10. Text Data

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Input format** | TF-IDF, Bag of Words | Word embeddings, sequences |
| **Sequence modeling** | ❌ NO | ✅ YES (RNN, LSTM, Transformer) |
| **Pre-trained models** | Not applicable | BERT, GPT (transfer learning) |

**Example:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = ['I love this', 'I hate this', 'This is great']

# Random Forest - TF-IDF
vectorizer = TfidfVectorizer(max_features=100)
X_tfidf = vectorizer.fit_transform(texts)
rf_model.fit(X_tfidf, y)

# DNN - Sequences + Embeddings
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X_padded = pad_sequences(sequences, maxlen=10)

model = keras.Sequential([
    keras.layers.Embedding(1000, 32, input_length=10),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])
model.fit(X_padded, y)
```

---

### 11. Computational Cost

| Aspect | Random Forest | Deep Neural Networks |
|--------|---------------|---------------------|
| **Preprocessing time** | Minimal | Significant |
| **Training time** | Fast-Medium | Slow (GPU needed) |
| **Inference time** | Fast | Fast (optimized) - Slow (unoptimized) |
| **Iteration speed** | Fast experimentation | Slower experimentation |

---

## Complete Preprocessing Pipelines

### Random Forest Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Minimal preprocessing needed
numerical_features = ['age', 'income', 'score']
categorical_features = ['gender', 'city', 'product']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('label', LabelEncoder())  # or OrdinalEncoder
        ]), categorical_features)
    ])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)
```

### DNN Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from tensorflow import keras

# Comprehensive preprocessing needed
numerical_features = ['age', 'income', 'score']
categorical_features = ['gender', 'city', 'product']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier_removal', PowerTransformer(method='yeo-johnson')),  # Handle skew
    ('scaler', StandardScaler())  # CRITICAL
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))  # Must be one-hot
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit preprocessor
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Build DNN
dnn_model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_processed.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

dnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Handle class imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

dnn_model.fit(
    X_train_processed, 
    y_train,
    validation_data=(X_test_processed, y_test),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[keras.callbacks.EarlyStopping(patience=5)]
)
```

---

## Decision Matrix: Which Model to Choose?

| Scenario | Choose Random Forest | Choose DNN |
|----------|---------------------|------------|
| **Small dataset** (<10k samples) | ✅ | ❌ |
| **Large dataset** (>100k samples) | ✅ | ✅ |
| **Tabular data** | ✅ Excellent | ⚠️ Good, but RF often better |
| **Images/Audio/Video** | ❌ | ✅ |
| **Text (short)** | ✅ (with TF-IDF) | ✅ (with embeddings) |
| **Text (long/sequential)** | ❌ | ✅ |
| **Mixed data types** | ✅ | ⚠️ (needs careful preprocessing) |
| **Need interpretability** | ✅ Feature importance | ❌ Black box |
| **Limited preprocessing time** | ✅ | ❌ |
| **Limited compute** | ✅ | ❌ (needs GPU) |
| **Fast iteration needed** | ✅ | ❌ |
| **State-of-the-art performance** (structured) | ✅ | ⚠️ Similar |
| **State-of-the-art performance** (unstructured) | ❌ | ✅ |

---

## Summary Table

| Preprocessing Step | Random Forest | DNN | Why Different? |
|-------------------|---------------|-----|----------------|
| **Scaling** | Not needed | Required | Gradient descent needs similar scales |
| **Categorical encoding** | Label encoding OK | One-hot/embeddings | Prevents false ordinal relationships |
| **Missing values** | Partially handled | Must impute | Neural nets can't process NaN |
| **Outliers** | Robust | Very sensitive | Outliers distort gradient updates |
| **Feature engineering** | Optional | Beneficial | Trees find interactions automatically |
| **Distribution transform** | Not needed | Helpful | Gaussian-like data trains better |
| **Dimensionality reduction** | Rarely needed | Sometimes helpful | High dims need more training data |
| **Data volume** | Works with less | Needs much more | DNNs have many parameters to learn |
| **Imbalanced classes** | Class weights | Sampling + weights | DNNs need balanced training |

---

## Key Takeaways

### Random Forest Advantages
- **Minimal preprocessing** required
- **Fast to experiment** with
- **Robust to outliers** and skewed distributions
- **Works with small datasets**
- **Built-in feature interactions**
- **Interpretable** (feature importance)

### DNN Advantages
- **Superior for unstructured data** (images, text, audio)
- **Better at complex patterns** with enough data
- **Transfer learning** available
- **Sequence modeling** capabilities

### Preprocessing Philosophy
- **Random Forest**: "Clean the data, let the model do the work"
- **DNN**: "Carefully prepare the data, then train the model"

### For GCP Data Engineer Exam
- Understand that **tabular data** often works better with Random Forest/XGBoost
- **Images/text/audio** require DNNs
- Vertex AI supports both via AutoML or custom training
- TFX pipelines (TensorFlow Transform) are designed for DNN preprocessing
- Consider preprocessing computational cost in production

## General Key Takeaways

1. **Preprocessing is data-dependent** - understand your data first
2. **Avoid data leakage** - fit on training data only
3. **Use pipelines** - ensures consistency and reproducibility
4. **Document transforms** - critical for model maintenance
5. **Test on holdout data** - validate transform effectiveness
6. **Monitor in production** - ensure transforms work on new data
7. **Version control** - track transform configurations
8. **Consider computational cost** - some transforms are expensive at scale
