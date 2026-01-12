# Collaborative Filtering

**Collaborative filtering** is a recommendation technique that predicts user preferences based on the collective behavior and opinions of many users.

## Core Concept
*"Users who agreed in the past will agree in the future"* - recommendations are based on user similarity or item similarity patterns.

## Types of Collaborative Filtering

### 1. **User-Based Collaborative Filtering**
- Finds users similar to the target user
- Recommends items that similar users liked

````python
# Example: User-based approach
# User A likes items [1, 2, 3]
# User B likes items [1, 2, 4]
# Since A and B are similar, recommend item 4 to User A
````

### 2. **Item-Based Collaborative Filtering**
- Finds items similar to what the user already likes
- More stable than user-based (items change less than user preferences)

````python
# Example: Item-based approach
# User likes Item X
# Items Y and Z are similar to X
# Recommend Y and Z to the user
````

### 3. **Matrix Factorization (Modern Approach)**
- Decomposes user-item interaction matrix into latent factors
- Used by Netflix, Spotify, etc.

````python
import tensorflow as tf

# Simple matrix factorization model
class MatrixFactorization(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=50):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)
    
    def call(self, inputs):
        user_id, item_id = inputs
        user_vec = self.user_embedding(user_id)
        item_vec = self.item_embedding(item_id)
        # Dot product for rating prediction
        return tf.reduce_sum(user_vec * item_vec, axis=1)
````

## Advantages
- No domain knowledge needed
- Discovers hidden patterns
- Improves over time with more data

## Challenges
- **Cold start problem**: New users/items have no history. Thus may not be suitable when fresh content is being generated and needs to be recommended.
- **Sparsity**: Most users rate few items
- **Scalability**: Large user/item matrices

## Google Cloud Implementation
Use **Vertex AI Matching Engine** or **BigQuery ML** for collaborative filtering at scale.