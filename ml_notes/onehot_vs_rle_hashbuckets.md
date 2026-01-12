# One-Hot Encoding vs Hash Buckets vs Run-Length Encoding (RLE)

## 1. One-Hot Encoding

**Purpose**: Convert categorical variables into binary vectors

**How it works**:
- Creates a new binary column for each unique category
- Value is 1 if category is present, 0 otherwise

**Example**:
```
Color: ['Red', 'Blue', 'Green']

Color_Red  Color_Blue  Color_Green
    1          0           0        → Red
    0          1           0        → Blue
    0          0           1        → Green
```

**Pros**:
- Simple and interpretable
- Works well with tree-based models
- No information loss

**Cons**:
- **High dimensionality**: Creates many columns for high-cardinality features
- **Sparse matrices**: Wastes memory
- **Doesn't handle unseen categories** at prediction time

**Use When**:
- Low cardinality (< 50 unique values)
- Need interpretability
- Tree-based models (Random Forest, XGBoost)

````python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# One-Hot Encoding
data = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(data[['color']])
print(pd.DataFrame(encoded, columns=encoder.get_feature_names_out()))
````

---

## 2. Hash Buckets (Feature Hashing)

**Purpose**: Map categorical values to fixed number of buckets using hash function

**How it works**:
- Apply hash function to category
- Use modulo operation to map to fixed number of buckets
- Multiple categories can map to same bucket (collision)

**Example**:
```
Hash function with 4 buckets:
'California' → hash % 4 = 2 → Bucket_2
'Texas'      → hash % 4 = 1 → Bucket_1  
'New York'   → hash % 4 = 2 → Bucket_2  (collision!)
'Florida'    → hash % 4 = 0 → Bucket_0
```

**Pros**:
- **Fixed dimensionality**: Controls feature space size
- **Handles unseen categories**: New values automatically hashed
- **Memory efficient**: No need to store vocabulary
- **Fast**: O(1) encoding time

**Cons**:
- **Hash collisions**: Different categories map to same bucket
- **Not interpretable**: Can't decode back to original
- **Information loss**: Due to collisions

**Use When**:
- Very high cardinality (millions of categories)
- Memory constraints
- Online learning (streaming data)
- Text data (words, n-grams)

````python
from sklearn.feature_extraction import FeatureHasher

# Hash Buckets
data = [{'color': 'red'}, {'color': 'blue'}, {'color': 'green'}]
hasher = FeatureHasher(n_features=8, input_type='dict')
hashed = hasher.transform(data)
print(hashed.toarray())

# Google Cloud BigQuery ML Example
"""
CREATE MODEL `project.dataset.model`
OPTIONS(
  model_type='logistic_reg',
  input_label_cols=['label']
) AS
SELECT
  label,
  FARM_FINGERPRINT(category_column) AS hashed_feature
FROM `project.dataset.table`
"""
````

---

## 3. Run-Length Encoding (RLE)

**Purpose**: Compress sequential data by storing value and run length

**How it works**:
- Store pairs of (value, count) instead of repeating values
- Compresses consecutive repeated values

**Example**:
```
Original: [A, A, A, B, B, C, C, C, C, C, A, A]
RLE:      [(A, 3), (B, 2), (C, 5), (A, 2)]

Original: 1 1 1 1 0 0 0 1 1 1 1 1
RLE:      (1, 4) (0, 3) (1, 5)
```

**Pros**:
- **Excellent compression** for sequential repeated data
- **Lossless**: No information lost
- **Fast decoding**: O(n) reconstruction

**Cons**:
- **Only for sequential data**: Requires order
- **Poor for random data**: Can increase size
- **Not for ML features**: Mainly for compression/storage

**Use When**:
- Image compression (binary images, masks)
- Video encoding
- Time series with long constant periods
- Data storage/transmission

````python
import numpy as np

# Run-Length Encoding
def run_length_encode(data):
    """Encode sequential data using RLE"""
    if len(data) == 0:
        return []
    
    encoded = []
    prev_val = data[0]
    count = 1
    
    for val in data[1:]:
        if val == prev_val:
            count += 1
        else:
            encoded.append((prev_val, count))
            prev_val = val
            count = 1
    
    encoded.append((prev_val, count))
    return encoded

def run_length_decode(encoded):
    """Decode RLE back to original"""
    decoded = []
    for val, count in encoded:
        decoded.extend([val] * count)
    return decoded

# Example
data = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
encoded = run_length_encode(data)
print(f"Original: {data}")
print(f"Encoded: {encoded}")
print(f"Decoded: {run_length_decode(encoded)}")

# Image compression example
binary_image = np.array([
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1]
])
for row in binary_image:
    print(f"Row: {row} → RLE: {run_length_encode(row)}")
````

---

## Comparison Table

| Feature | One-Hot | Hash Buckets | RLE |
|---------|---------|--------------|-----|
| **Purpose** | Categorical encoding | High-cardinality encoding | Data compression |
| **Output Dimensions** | # unique values | Fixed (user-defined) | Variable (# runs) |
| **Information Loss** | None | Yes (collisions) | None |
| **Memory Usage** | High | Low | Very Low |
| **Unseen Categories** | ❌ Fails | ✅ Handles | N/A |
| **Interpretability** | ✅ High | ❌ Low | ✅ High |
| **Use Case** | ML features | ML features | Storage/Compression |
| **Data Type** | Categorical | Categorical | Sequential |

---

## Google Cloud BigQuery ML Context

````sql
-- One-Hot Encoding (Implicit)
CREATE MODEL `project.dataset.onehot_model`
OPTIONS(model_type='logistic_reg') AS
SELECT
  label,
  category_col  -- BigQuery ML auto one-hot encodes string columns
FROM `project.dataset.training_data`;

-- Hash Buckets (Explicit)
CREATE MODEL `project.dataset.hash_model`
OPTIONS(model_type='dnn_classifier') AS
SELECT
  label,
  ML.FEATURE_CROSS(STRUCT(
    FARM_FINGERPRINT(CAST(user_id AS STRING)) AS user_id_hash,
    FARM_FINGERPRINT(product_id) AS product_id_hash
  )) AS crossed_feature
FROM `project.dataset.training_data`;

-- Bucketized Features (Similar to hashing but explicit ranges)
CREATE MODEL `project.dataset.bucket_model`
OPTIONS(model_type='linear_reg') AS
SELECT
  label,
  ML.BUCKETIZE(age, [0, 18, 35, 50, 65, 100]) AS age_bucket
FROM `project.dataset.training_data`;
````

---

## When to Use Each

**One-Hot Encoding**:
- ✅ Low-cardinality features (cities, countries, departments)
- ✅ Tree-based models
- ✅ Need interpretability

**Hash Buckets**:
- ✅ High-cardinality features (user IDs, product IDs, zip codes)
- ✅ Memory constraints
- ✅ Online learning systems
- ✅ Recommendation systems

**Run-Length Encoding**:
- ✅ Binary images/masks
- ✅ Video frames
- ✅ Time series with long constant periods
- ✅ Data transmission/storage (not for ML training)