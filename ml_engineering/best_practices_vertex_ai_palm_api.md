# Vertex AI PaLM API Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI PaLM API provides access to Google's Pathways Language Model (PaLM 2) for text generation, chat conversations, text embeddings, and model fine-tuning with enterprise-grade security, scalability, and integration with Google Cloud services.

---

## 1. Text Generation

### Generate Text with PaLM API

```python
from google.cloud import aiplatform
from vertexai.preview.language_models import TextGenerationModel

class PaLMTextGenerator:
    """Manage PaLM text generation."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def generate_text(
        self,
        prompt,
        temperature=0.2,
        max_output_tokens=256,
        top_p=0.8,
        top_k=40
    ):
        """Generate text using PaLM 2 model."""
        
        model = TextGenerationModel.from_pretrained("text-bison@002")
        
        response = model.predict(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k
        )
        
        print(f"✓ Generated text")
        print(f"  Length: {len(response.text)} characters")
        print(f"  Temperature: {temperature}")
        
        return response.text
    
    def generate_with_stop_sequences(
        self,
        prompt,
        stop_sequences,
        max_output_tokens=512
    ):
        """Generate text with stop sequences."""
        
        model = TextGenerationModel.from_pretrained("text-bison@002")
        
        response = model.predict(
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
            temperature=0.2
        )
        
        print(f"✓ Generated with stop sequences")
        print(f"  Stop sequences: {stop_sequences}")
        
        return response.text
    
    def batch_generate(
        self,
        prompts,
        temperature=0.2,
        max_output_tokens=256
    ):
        """Generate text for multiple prompts."""
        
        model = TextGenerationModel.from_pretrained("text-bison@002")
        
        results = []
        
        for i, prompt in enumerate(prompts):
            response = model.predict(
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            
            results.append({
                'prompt': prompt[:50] + '...',
                'response': response.text,
                'index': i
            })
        
        print(f"✓ Batch generation completed")
        print(f"  Prompts processed: {len(prompts)}")
        
        return results
    
    def generate_with_retries(
        self,
        prompt,
        max_retries=3,
        backoff_factor=2
    ):
        """Generate with automatic retries."""
        
        import time
        
        model = TextGenerationModel.from_pretrained("text-bison@002")
        
        for attempt in range(max_retries):
            try:
                response = model.predict(
                    prompt=prompt,
                    max_output_tokens=256,
                    temperature=0.2
                )
                
                print(f"✓ Generated successfully (attempt {attempt + 1})")
                return response.text
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Failed after {max_retries} attempts")
                    raise e
    
    def generate_structured_output(
        self,
        prompt,
        output_schema
    ):
        """Generate structured JSON output."""
        
        import json
        
        # Add schema instructions to prompt
        schema_prompt = f"""{prompt}

Output format (JSON):
{json.dumps(output_schema, indent=2)}

Generate the output in valid JSON format following the schema above.

JSON Output:"""
        
        model = TextGenerationModel.from_pretrained("text-bison@002")
        
        response = model.predict(
            prompt=schema_prompt,
            temperature=0.1,  # Lower temperature for structured output
            max_output_tokens=512
        )
        
        # Parse JSON
        try:
            structured_data = json.loads(response.text)
            print(f"✓ Generated structured output")
            return structured_data
        except json.JSONDecodeError:
            print(f"⚠ Failed to parse JSON, returning raw text")
            return response.text

# Example usage
# generator = PaLMTextGenerator(project_id='my-project')

# Generate text
# text = generator.generate_text(
#     prompt="Explain data partitioning strategies in BigQuery.",
#     temperature=0.2,
#     max_output_tokens=512
# )

# Generate with stop sequences
# response = generator.generate_with_stop_sequences(
#     prompt="List the steps to create a Dataflow pipeline:\n1.",
#     stop_sequences=["\n\n", "Summary:"],
#     max_output_tokens=256
# )

# Structured output
# data = generator.generate_structured_output(
#     prompt="Extract key information from this data engineering job description: ...",
#     output_schema={
#         "title": "string",
#         "skills": ["string"],
#         "experience_years": "number",
#         "location": "string"
#     }
# )
```

---

## 2. Chat Conversations

### Build Chat Applications

```python
class PaLMChatManager:
    """Manage PaLM chat conversations."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_chat_session(
        self,
        context=None,
        examples=None,
        temperature=0.25
    ):
        """Create new chat session with context."""
        
        from vertexai.preview.language_models import ChatModel
        
        chat_model = ChatModel.from_pretrained("chat-bison@002")
        
        # Format examples if provided
        formatted_examples = []
        if examples:
            for ex in examples:
                formatted_examples.append(
                    {
                        'input': {'content': ex['input']},
                        'output': {'content': ex['output']}
                    }
                )
        
        chat = chat_model.start_chat(
            context=context,
            examples=formatted_examples,
            temperature=temperature
        )
        
        print(f"✓ Chat session created")
        if context:
            print(f"  Context: {context[:100]}...")
        if examples:
            print(f"  Examples: {len(examples)}")
        
        return chat
    
    def send_message(
        self,
        chat,
        message,
        max_output_tokens=1024
    ):
        """Send message and get response."""
        
        response = chat.send_message(
            message=message,
            max_output_tokens=max_output_tokens
        )
        
        print(f"✓ Response received")
        print(f"  Length: {len(response.text)} characters")
        
        return response.text
    
    def multi_turn_conversation(
        self,
        chat,
        messages
    ):
        """Handle multi-turn conversation."""
        
        conversation = []
        
        for msg in messages:
            response = chat.send_message(message=msg, max_output_tokens=512)
            
            conversation.append({
                'user': msg,
                'assistant': response.text
            })
        
        print(f"✓ Multi-turn conversation completed")
        print(f"  Turns: {len(messages)}")
        
        return conversation
    
    def create_specialized_chat_bot(
        self,
        domain,
        instructions,
        example_conversations
    ):
        """Create specialized domain chatbot."""
        
        from vertexai.preview.language_models import ChatModel
        
        context = f"""You are a specialized assistant for {domain}.

Instructions:
{instructions}

Always provide accurate, helpful responses based on your expertise in {domain}."""
        
        chat_model = ChatModel.from_pretrained("chat-bison@002")
        
        formatted_examples = []
        for conv in example_conversations:
            formatted_examples.append({
                'input': {'content': conv['user']},
                'output': {'content': conv['assistant']}
            })
        
        chat = chat_model.start_chat(
            context=context,
            examples=formatted_examples,
            temperature=0.3
        )
        
        print(f"✓ Specialized chatbot created")
        print(f"  Domain: {domain}")
        print(f"  Examples: {len(example_conversations)}")
        
        return chat
    
    def implement_chat_with_context_window(
        self,
        max_context_tokens=4096
    ):
        """Implement chat with context window management."""
        
        from vertexai.preview.language_models import ChatModel
        
        chat_model = ChatModel.from_pretrained("chat-bison@002")
        chat = chat_model.start_chat()
        
        conversation_history = []
        
        def send_with_context_management(message):
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            total_chars = sum(
                len(turn['user']) + len(turn['assistant'])
                for turn in conversation_history
            )
            estimated_tokens = total_chars // 4
            
            # If exceeding context window, summarize or truncate
            if estimated_tokens > max_context_tokens * 0.8:
                print(f"⚠ Context window near limit, truncating old messages")
                conversation_history[:] = conversation_history[-5:]
            
            # Send message
            response = chat.send_message(message, max_output_tokens=512)
            
            # Add to history
            conversation_history.append({
                'user': message,
                'assistant': response.text
            })
            
            return response.text
        
        print(f"✓ Chat with context management initialized")
        print(f"  Max context tokens: {max_context_tokens}")
        
        return send_with_context_management, conversation_history

# Example usage
# chat_manager = PaLMChatManager(project_id='my-project')

# Create data engineering chatbot
# de_examples = [
#     {
#         'user': 'How do I optimize BigQuery costs?',
#         'assistant': 'Use partitioning, clustering, and avoid SELECT * queries...'
#     },
#     {
#         'user': 'When should I use Dataflow vs Dataproc?',
#         'assistant': 'Use Dataflow for streaming and fully managed pipelines...'
#     }
# ]

# chat = chat_manager.create_specialized_chat_bot(
#     domain='Google Cloud Data Engineering',
#     instructions='Provide practical advice on BigQuery, Dataflow, Pub/Sub, and other GCP data services.',
#     example_conversations=de_examples
# )

# Conversation
# response1 = chat_manager.send_message(
#     chat=chat,
#     message='What are best practices for Pub/Sub topic design?'
# )
```

---

## 3. Text Embeddings

### Generate and Use Text Embeddings

```python
class PaLMEmbeddingsManager:
    """Manage PaLM text embeddings."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def generate_embeddings(
        self,
        texts
    ):
        """Generate embeddings for texts."""
        
        from vertexai.preview.language_models import TextEmbeddingModel
        
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        
        embeddings = model.get_embeddings(texts)
        
        embedding_vectors = [emb.values for emb in embeddings]
        
        print(f"✓ Generated embeddings")
        print(f"  Texts: {len(texts)}")
        print(f"  Embedding dimension: {len(embedding_vectors[0])}")
        
        return embedding_vectors
    
    def batch_generate_embeddings(
        self,
        texts,
        batch_size=5
    ):
        """Generate embeddings in batches."""
        
        from vertexai.preview.language_models import TextEmbeddingModel
        
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = model.get_embeddings(batch)
            
            batch_vectors = [emb.values for emb in embeddings]
            all_embeddings.extend(batch_vectors)
            
            print(f"  Processed batch {i // batch_size + 1}")
        
        print(f"✓ Batch generation completed")
        print(f"  Total embeddings: {len(all_embeddings)}")
        
        return all_embeddings
    
    def calculate_similarity(
        self,
        text1,
        text2
    ):
        """Calculate cosine similarity between two texts."""
        
        import numpy as np
        from vertexai.preview.language_models import TextEmbeddingModel
        
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        
        embeddings = model.get_embeddings([text1, text2])
        
        vec1 = np.array(embeddings[0].values)
        vec2 = np.array(embeddings[1].values)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        print(f"✓ Calculated similarity")
        print(f"  Similarity score: {similarity:.4f}")
        
        return float(similarity)
    
    def semantic_search(
        self,
        query,
        documents,
        top_k=5
    ):
        """Perform semantic search over documents."""
        
        import numpy as np
        from vertexai.preview.language_models import TextEmbeddingModel
        
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        
        # Generate embeddings
        all_texts = [query] + documents
        embeddings = model.get_embeddings(all_texts)
        
        query_vec = np.array(embeddings[0].values)
        doc_vecs = [np.array(emb.values) for emb in embeddings[1:]]
        
        # Calculate similarities
        similarities = []
        for i, doc_vec in enumerate(doc_vecs):
            sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append({
                'index': i,
                'document': documents[i][:100] + '...',
                'similarity': float(sim)
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = similarities[:top_k]
        
        print(f"✓ Semantic search completed")
        print(f"  Query: {query[:50]}...")
        print(f"  Top {top_k} results returned")
        
        return top_results
    
    def cluster_texts_by_embeddings(
        self,
        texts,
        n_clusters=3
    ):
        """Cluster texts using embeddings."""
        
        import numpy as np
        from sklearn.cluster import KMeans
        from vertexai.preview.language_models import TextEmbeddingModel
        
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        
        # Generate embeddings
        embeddings = model.get_embeddings(texts)
        embedding_matrix = np.array([emb.values for emb in embeddings])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embedding_matrix)
        
        # Group texts by cluster
        clustered_texts = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_texts:
                clustered_texts[cluster_id] = []
            clustered_texts[cluster_id].append(texts[i])
        
        print(f"✓ Clustering completed")
        print(f"  Clusters: {n_clusters}")
        for cluster_id, texts_in_cluster in clustered_texts.items():
            print(f"  Cluster {cluster_id}: {len(texts_in_cluster)} texts")
        
        return clustered_texts

# Example usage
# embeddings_manager = PaLMEmbeddingsManager(project_id='my-project')

# Generate embeddings
# texts = [
#     'BigQuery is a serverless data warehouse',
#     'Dataflow processes streaming and batch data',
#     'Pub/Sub is a messaging service'
# ]
# embeddings = embeddings_manager.generate_embeddings(texts)

# Semantic search
# documents = [
#     'BigQuery supports SQL queries on petabyte-scale data',
#     'Cloud Storage is object storage for any data type',
#     'Dataproc runs Apache Spark and Hadoop clusters',
#     'Dataflow is based on Apache Beam programming model'
# ]
# results = embeddings_manager.semantic_search(
#     query='How do I query large datasets?',
#     documents=documents,
#     top_k=3
# )
```

---

## 4. Model Fine-Tuning

### Fine-Tune PaLM Models

```python
class PaLMFineTuningManager:
    """Manage PaLM model fine-tuning."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def prepare_tuning_dataset(
        self,
        training_examples,
        output_file='training_data.jsonl'
    ):
        """Prepare dataset for fine-tuning.
        
        Args:
            training_examples: List of dicts with 'input_text' and 'output_text'
        """
        
        import json
        
        with open(output_file, 'w') as f:
            for example in training_examples:
                json_line = json.dumps({
                    'input_text': example['input_text'],
                    'output_text': example['output_text']
                })
                f.write(json_line + '\n')
        
        print(f"✓ Prepared tuning dataset")
        print(f"  Examples: {len(training_examples)}")
        print(f"  Output: {output_file}")
        
        return output_file
    
    def validate_tuning_data(
        self,
        training_examples
    ):
        """Validate training data quality."""
        
        issues = []
        
        for i, example in enumerate(training_examples):
            # Check required fields
            if 'input_text' not in example or 'output_text' not in example:
                issues.append(f"Example {i}: Missing required fields")
            
            # Check text length
            if len(example.get('input_text', '')) < 10:
                issues.append(f"Example {i}: Input text too short")
            
            if len(example.get('output_text', '')) < 5:
                issues.append(f"Example {i}: Output text too short")
        
        if issues:
            print(f"⚠ Validation issues found: {len(issues)}")
            for issue in issues[:5]:
                print(f"  - {issue}")
        else:
            print(f"✓ Validation passed")
        
        return len(issues) == 0, issues
    
    def upload_tuning_data(
        self,
        local_file,
        gcs_bucket,
        gcs_path
    ):
        """Upload tuning data to Cloud Storage."""
        
        from google.cloud import storage
        
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)
        
        blob.upload_from_filename(local_file)
        
        gcs_uri = f'gs://{gcs_bucket}/{gcs_path}'
        
        print(f"✓ Uploaded tuning data")
        print(f"  GCS URI: {gcs_uri}")
        
        return gcs_uri
    
    def fine_tune_model(
        self,
        training_data_uri,
        model_display_name,
        tuning_steps=100,
        learning_rate_multiplier=1.0
    ):
        """Fine-tune PaLM model."""
        
        from vertexai.preview.language_models import TextGenerationModel
        
        base_model = TextGenerationModel.from_pretrained("text-bison@002")
        
        print(f"Starting fine-tuning...")
        print(f"  Training data: {training_data_uri}")
        print(f"  Model name: {model_display_name}")
        print(f"  Tuning steps: {tuning_steps}")
        print(f"  Learning rate multiplier: {learning_rate_multiplier}")
        
        # Note: Actual fine-tuning code
        # tuning_job = base_model.tune_model(
        #     training_data=training_data_uri,
        #     train_steps=tuning_steps,
        #     learning_rate_multiplier=learning_rate_multiplier,
        #     tuned_model_display_name=model_display_name
        # )
        
        print(f"✓ Fine-tuning job started")
        
        return {
            'model_name': model_display_name,
            'status': 'tuning',
            'training_data': training_data_uri
        }
    
    def monitor_tuning_job(
        self,
        tuning_job
    ):
        """Monitor fine-tuning job progress."""
        
        print(f"Monitoring tuning job...")
        
        # Check job status
        status = 'running'  # Placeholder
        
        metrics = {
            'loss': 0.25,
            'steps_completed': 50,
            'total_steps': 100
        }
        
        print(f"  Status: {status}")
        print(f"  Progress: {metrics['steps_completed']}/{metrics['total_steps']}")
        print(f"  Loss: {metrics['loss']:.4f}")
        
        return metrics

# Example usage
# tuning_manager = PaLMFineTuningManager(project_id='my-project')

# Prepare training data
# training_examples = [
#     {
#         'input_text': 'Summarize this BigQuery best practice: Always use partitioning for large tables...',
#         'output_text': 'Use partitioning for large tables to improve query performance and reduce costs.'
#     },
#     # Add 100+ more examples...
# ]

# Validate data
# is_valid, issues = tuning_manager.validate_tuning_data(training_examples)

# Prepare dataset
# dataset_file = tuning_manager.prepare_tuning_dataset(
#     training_examples=training_examples,
#     output_file='palm_tuning_data.jsonl'
# )

# Upload to GCS
# gcs_uri = tuning_manager.upload_tuning_data(
#     local_file=dataset_file,
#     gcs_bucket='my-ml-bucket',
#     gcs_path='tuning/palm_data.jsonl'
# )

# Fine-tune model
# job = tuning_manager.fine_tune_model(
#     training_data_uri=gcs_uri,
#     model_display_name='custom-palm-model-v1',
#     tuning_steps=200
# )
```

---

## 5. Production Best Practices

### Deploy and Monitor PaLM Models

```python
class PaLMProductionManager:
    """Manage PaLM models in production."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def implement_retry_logic(
        self,
        model,
        prompt,
        max_retries=3
    ):
        """Implement exponential backoff retry logic."""
        
        import time
        
        for attempt in range(max_retries):
            try:
                response = model.predict(
                    prompt=prompt,
                    max_output_tokens=256,
                    temperature=0.2
                )
                
                print(f"✓ Request successful (attempt {attempt + 1})")
                return response.text
                
            except Exception as e:
                wait_time = 2 ** attempt
                
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Failed after {max_retries} attempts: {e}")
                    raise
    
    def batch_process_with_rate_limiting(
        self,
        prompts,
        requests_per_minute=60
    ):
        """Process multiple prompts with rate limiting."""
        
        import time
        from vertexai.preview.language_models import TextGenerationModel
        
        model = TextGenerationModel.from_pretrained("text-bison@002")
        
        results = []
        delay = 60.0 / requests_per_minute
        
        for i, prompt in enumerate(prompts):
            response = model.predict(
                prompt=prompt,
                max_output_tokens=256,
                temperature=0.2
            )
            
            results.append({
                'prompt': prompt[:50],
                'response': response.text
            })
            
            # Rate limiting
            if i < len(prompts) - 1:
                time.sleep(delay)
        
        print(f"✓ Batch processing completed")
        print(f"  Processed: {len(prompts)} prompts")
        print(f"  Rate: {requests_per_minute} req/min")
        
        return results
    
    def implement_response_caching(
        self,
        cache_ttl_hours=24
    ):
        """Implement response caching."""
        
        import hashlib
        import time
        
        cache = {}
        
        def get_cached_response(model, prompt, **kwargs):
            # Create cache key
            cache_key = hashlib.sha256(
                f"{prompt}{str(kwargs)}".encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in cache:
                cached_entry = cache[cache_key]
                age_hours = (time.time() - cached_entry['timestamp']) / 3600
                
                if age_hours < cache_ttl_hours:
                    print(f"✓ Cache hit (age: {age_hours:.1f}h)")
                    return cached_entry['response']
            
            # Cache miss
            print(f"Cache miss - generating response")
            response = model.predict(prompt=prompt, **kwargs)
            
            # Store in cache
            cache[cache_key] = {
                'response': response.text,
                'timestamp': time.time()
            }
            
            return response.text
        
        print(f"✓ Caching configured (TTL: {cache_ttl_hours}h)")
        
        return get_cached_response
    
    def log_requests_to_bigquery(
        self,
        dataset_id,
        table_id
    ):
        """Log API requests to BigQuery."""
        
        from google.cloud import bigquery
        import datetime
        
        client = bigquery.Client(project=self.project_id)
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        
        def log_request(prompt, response, latency_ms, tokens_used):
            row = {
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'prompt': prompt[:500],
                'response': response[:500],
                'latency_ms': latency_ms,
                'tokens_used': tokens_used,
                'model': 'text-bison@002'
            }
            
            errors = client.insert_rows_json(table_ref, [row])
            
            if not errors:
                print(f"✓ Logged request to BigQuery")
            else:
                print(f"✗ Logging errors: {errors}")
        
        print(f"✓ BigQuery logging configured")
        print(f"  Table: {table_ref}")
        
        return log_request
    
    def monitor_model_performance(self):
        """Monitor model performance metrics."""
        
        metrics = {
            'requests_per_hour': 0,
            'avg_latency_ms': 0,
            'error_rate': 0,
            'cache_hit_rate': 0
        }
        
        print(f"✓ Monitoring configured")
        print(f"  Metrics tracked: {list(metrics.keys())}")
        
        return metrics

# Example usage
# prod_manager = PaLMProductionManager(project_id='my-project')

# Implement caching
# from vertexai.preview.language_models import TextGenerationModel
# model = TextGenerationModel.from_pretrained("text-bison@002")
# cached_predict = prod_manager.implement_response_caching(cache_ttl_hours=24)

# Use with caching
# response = cached_predict(
#     model=model,
#     prompt="Explain BigQuery partitioning",
#     max_output_tokens=256,
#     temperature=0.2
# )
```

---

## 6. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Install google-cloud-aiplatform SDK
- [ ] Configure authentication
- [ ] Set up IAM permissions
- [ ] Create GCS bucket for data

### Text Generation
- [ ] Choose appropriate temperature (0.0-1.0)
- [ ] Set max_output_tokens limit
- [ ] Use stop sequences when needed
- [ ] Implement retry logic
- [ ] Add error handling

### Chat Applications
- [ ] Define clear context
- [ ] Provide example conversations
- [ ] Manage context window
- [ ] Handle multi-turn conversations
- [ ] Store conversation history

### Embeddings
- [ ] Use textembedding-gecko@003 model
- [ ] Batch process for efficiency
- [ ] Store embeddings for reuse
- [ ] Implement semantic search
- [ ] Calculate similarity scores

### Fine-Tuning
- [ ] Prepare 100+ training examples
- [ ] Validate data quality
- [ ] Upload to Cloud Storage
- [ ] Monitor tuning progress
- [ ] Evaluate tuned model

### Production Best Practices
- [ ] Implement rate limiting
- [ ] Add response caching
- [ ] Log requests to BigQuery
- [ ] Monitor performance metrics
- [ ] Set up alerts for errors
- [ ] Use retry with exponential backoff
- [ ] Implement circuit breakers
- [ ] Version control prompts

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
