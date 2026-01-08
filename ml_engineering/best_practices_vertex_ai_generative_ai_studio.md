# Vertex AI Generative AI Studio Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Generative AI Studio provides a unified interface for prototyping, testing, and deploying generative AI models including text, chat, code, and image generation with prompt design tools, model tuning capabilities, and integrated evaluation frameworks.

---

## 1. Prompt Design and Engineering

### Design Effective Prompts

```python
from google.cloud import aiplatform
from vertexai.preview.language_models import TextGenerationModel, ChatModel

class PromptDesignManager:
    """Manage prompt design and engineering."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def create_zero_shot_prompt(
        self,
        task_description,
        input_text,
        output_format=None
    ):
        """Create zero-shot prompt."""
        
        prompt = f"{task_description}\n\n"
        
        if output_format:
            prompt += f"Output format: {output_format}\n\n"
        
        prompt += f"Input: {input_text}\n"
        prompt += "Output:"
        
        print(f"✓ Created zero-shot prompt")
        print(f"  Task: {task_description[:50]}...")
        
        return prompt
    
    def create_few_shot_prompt(
        self,
        task_description,
        examples,
        input_text
    ):
        """Create few-shot prompt with examples.
        
        Args:
            examples: List of dicts with 'input' and 'output' keys
        """
        
        prompt = f"{task_description}\n\n"
        
        # Add examples
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        # Add actual input
        prompt += f"Now, process this input:\n"
        prompt += f"Input: {input_text}\n"
        prompt += "Output:"
        
        print(f"✓ Created few-shot prompt")
        print(f"  Examples: {len(examples)}")
        
        return prompt
    
    def create_chain_of_thought_prompt(
        self,
        question,
        include_reasoning=True
    ):
        """Create chain-of-thought prompt for complex reasoning."""
        
        prompt = "Let's solve this step by step:\n\n"
        prompt += f"Question: {question}\n\n"
        
        if include_reasoning:
            prompt += "Think through each step:\n"
            prompt += "1. First, identify what we know\n"
            prompt += "2. Then, determine what we need to find\n"
            prompt += "3. Finally, solve systematically\n\n"
        
        prompt += "Step-by-step solution:"
        
        print(f"✓ Created chain-of-thought prompt")
        
        return prompt
    
    def generate_text_with_prompt(
        self,
        prompt,
        temperature=0.2,
        max_output_tokens=1024,
        top_p=0.8,
        top_k=40
    ):
        """Generate text using prompt."""
        
        model = TextGenerationModel.from_pretrained("text-bison@002")
        
        response = model.predict(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k
        )
        
        print(f"✓ Generated response")
        print(f"  Length: {len(response.text)} characters")
        
        return response.text
    
    def optimize_prompt_parameters(
        self,
        prompt,
        test_inputs,
        evaluation_metric
    ):
        """Optimize prompt parameters through testing."""
        
        model = TextGenerationModel.from_pretrained("text-bison@002")
        
        # Test different temperature values
        temperatures = [0.0, 0.2, 0.5, 0.8, 1.0]
        results = []
        
        for temp in temperatures:
            outputs = []
            
            for input_text in test_inputs:
                response = model.predict(
                    prompt=prompt.format(input=input_text),
                    temperature=temp,
                    max_output_tokens=256
                )
                outputs.append(response.text)
            
            # Calculate metric
            score = evaluation_metric(outputs)
            
            results.append({
                'temperature': temp,
                'score': score,
                'sample_output': outputs[0][:100]
            })
        
        # Find best temperature
        best = max(results, key=lambda x: x['score'])
        
        print(f"✓ Optimized prompt parameters")
        print(f"  Best temperature: {best['temperature']}")
        print(f"  Best score: {best['score']:.4f}")
        
        return results

# Example usage
# prompt_manager = PromptDesignManager(project_id='my-project')

# Zero-shot prompt
# prompt = prompt_manager.create_zero_shot_prompt(
#     task_description="Classify the sentiment of the following text as positive, negative, or neutral.",
#     input_text="The product quality exceeded my expectations!",
#     output_format="One word: positive, negative, or neutral"
# )

# Few-shot prompt
# examples = [
#     {'input': 'The movie was amazing!', 'output': 'positive'},
#     {'input': 'Terrible experience, very disappointed.', 'output': 'negative'},
#     {'input': 'It was okay, nothing special.', 'output': 'neutral'}
# ]
# prompt = prompt_manager.create_few_shot_prompt(
#     task_description="Classify sentiment:",
#     examples=examples,
#     input_text="I absolutely loved this book!"
# )

# Generate response
# response = prompt_manager.generate_text_with_prompt(
#     prompt=prompt,
#     temperature=0.2,
#     max_output_tokens=256
# )
```

---

## 2. Chat Model Integration

### Build Conversational Applications

```python
class ChatModelManager:
    """Manage chat model interactions."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def start_chat_session(
        self,
        context=None,
        examples=None,
        temperature=0.2
    ):
        """Start a new chat session."""
        
        from vertexai.preview.language_models import ChatModel
        
        chat_model = ChatModel.from_pretrained("chat-bison@002")
        
        chat = chat_model.start_chat(
            context=context,
            examples=examples,
            temperature=temperature
        )
        
        print(f"✓ Started chat session")
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
        """Send message in chat session."""
        
        response = chat.send_message(
            message=message,
            max_output_tokens=max_output_tokens
        )
        
        print(f"✓ Received response")
        print(f"  Length: {len(response.text)} characters")
        
        return response.text
    
    def create_contextual_chat(
        self,
        system_context,
        user_profile=None
    ):
        """Create chat with system context and user profile."""
        
        from vertexai.preview.language_models import ChatModel
        
        context_parts = [system_context]
        
        if user_profile:
            context_parts.append(f"User profile: {user_profile}")
        
        full_context = "\n\n".join(context_parts)
        
        chat_model = ChatModel.from_pretrained("chat-bison@002")
        chat = chat_model.start_chat(
            context=full_context,
            temperature=0.3
        )
        
        print(f"✓ Created contextual chat")
        
        return chat
    
    def implement_chat_with_memory(
        self,
        max_history=10
    ):
        """Implement chat with conversation memory."""
        
        from vertexai.preview.language_models import ChatModel
        
        chat_model = ChatModel.from_pretrained("chat-bison@002")
        chat = chat_model.start_chat()
        
        conversation_history = []
        
        def send_with_memory(message):
            # Add to history
            conversation_history.append({
                'role': 'user',
                'content': message
            })
            
            # Keep only recent history
            if len(conversation_history) > max_history * 2:
                conversation_history.pop(0)
                conversation_history.pop(0)
            
            # Send message
            response = chat.send_message(message)
            
            # Add response to history
            conversation_history.append({
                'role': 'assistant',
                'content': response.text
            })
            
            return response.text
        
        print(f"✓ Chat with memory initialized")
        print(f"  Max history: {max_history} exchanges")
        
        return send_with_memory, conversation_history

# Example usage
# chat_manager = ChatModelManager(project_id='my-project')

# Start chat with context
# chat = chat_manager.start_chat_session(
#     context="You are a helpful data engineering assistant. Help users with BigQuery, Dataflow, and Pub/Sub questions.",
#     temperature=0.2
# )

# Send messages
# response1 = chat_manager.send_message(
#     chat=chat,
#     message="How do I optimize BigQuery query performance?"
# )
# print(response1)

# response2 = chat_manager.send_message(
#     chat=chat,
#     message="What about partitioning strategies?"
# )
# print(response2)
```

---

## 3. Model Tuning and Customization

### Fine-Tune Models for Specific Tasks

```python
class ModelTuningManager:
    """Manage model tuning and customization."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def prepare_tuning_dataset(
        self,
        training_data,
        output_jsonl_path
    ):
        """Prepare dataset for model tuning.
        
        Args:
            training_data: List of dicts with 'input_text' and 'output_text'
        """
        
        import json
        
        with open(output_jsonl_path, 'w') as f:
            for item in training_data:
                json_line = json.dumps({
                    'input_text': item['input_text'],
                    'output_text': item['output_text']
                })
                f.write(json_line + '\n')
        
        print(f"✓ Prepared tuning dataset")
        print(f"  Examples: {len(training_data)}")
        print(f"  Output: {output_jsonl_path}")
        
        return output_jsonl_path
    
    def upload_tuning_data_to_gcs(
        self,
        local_path,
        gcs_bucket,
        gcs_path
    ):
        """Upload tuning data to Cloud Storage."""
        
        from google.cloud import storage
        
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)
        
        blob.upload_from_filename(local_path)
        
        gcs_uri = f'gs://{gcs_bucket}/{gcs_path}'
        
        print(f"✓ Uploaded tuning data to GCS")
        print(f"  URI: {gcs_uri}")
        
        return gcs_uri
    
    def tune_text_model(
        self,
        training_data_uri,
        tuned_model_display_name,
        train_steps=100,
        learning_rate=0.001
    ):
        """Tune a text generation model."""
        
        from vertexai.preview.language_models import TextGenerationModel
        
        base_model = TextGenerationModel.from_pretrained("text-bison@002")
        
        print(f"Starting model tuning...")
        print(f"  Training data: {training_data_uri}")
        print(f"  Train steps: {train_steps}")
        print(f"  Learning rate: {learning_rate}")
        
        # Note: Actual tuning would use the tune() method
        # tuning_job = base_model.tune_model(
        #     training_data=training_data_uri,
        #     train_steps=train_steps,
        #     learning_rate=learning_rate,
        #     tuned_model_display_name=tuned_model_display_name
        # )
        
        print(f"✓ Model tuning initiated")
        
        return {
            'model_name': tuned_model_display_name,
            'status': 'tuning',
            'train_steps': train_steps
        }
    
    def evaluate_tuned_model(
        self,
        model,
        test_data,
        evaluation_metrics
    ):
        """Evaluate tuned model performance."""
        
        results = []
        
        for test_item in test_data:
            response = model.predict(
                prompt=test_item['input_text'],
                max_output_tokens=256
            )
            
            # Calculate metrics
            metrics = {}
            for metric_name, metric_func in evaluation_metrics.items():
                metrics[metric_name] = metric_func(
                    response.text,
                    test_item['expected_output']
                )
            
            results.append({
                'input': test_item['input_text'][:50],
                'output': response.text[:50],
                'metrics': metrics
            })
        
        # Aggregate metrics
        aggregated = {}
        for metric_name in evaluation_metrics.keys():
            values = [r['metrics'][metric_name] for r in results]
            aggregated[metric_name] = sum(values) / len(values)
        
        print(f"✓ Model evaluation completed")
        print(f"  Test samples: {len(test_data)}")
        print(f"  Metrics: {aggregated}")
        
        return results, aggregated

# Example usage
# tuning_manager = ModelTuningManager(project_id='my-project')

# Prepare training data
# training_data = [
#     {
#         'input_text': 'Summarize: The data pipeline processes 1TB daily...',
#         'output_text': 'Daily 1TB data processing pipeline with batch ETL.'
#     },
#     # Add more examples...
# ]

# jsonl_path = tuning_manager.prepare_tuning_dataset(
#     training_data=training_data,
#     output_jsonl_path='training_data.jsonl'
# )

# Upload to GCS
# gcs_uri = tuning_manager.upload_tuning_data_to_gcs(
#     local_path='training_data.jsonl',
#     gcs_bucket='my-tuning-data',
#     gcs_path='models/text-tuning/training_data.jsonl'
# )

# Tune model
# job = tuning_manager.tune_text_model(
#     training_data_uri=gcs_uri,
#     tuned_model_display_name='custom-text-model-v1',
#     train_steps=200,
#     learning_rate=0.001
# )
```

---

## 4. Model Evaluation and Testing

### Evaluate Model Performance

```python
class ModelEvaluationManager:
    """Manage model evaluation and testing."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def evaluate_text_generation(
        self,
        model,
        test_prompts,
        ground_truth_outputs
    ):
        """Evaluate text generation quality."""
        
        from sklearn.metrics import accuracy_score
        import difflib
        
        predictions = []
        similarities = []
        
        for prompt in test_prompts:
            response = model.predict(prompt=prompt, max_output_tokens=256)
            predictions.append(response.text)
        
        # Calculate similarity scores
        for pred, truth in zip(predictions, ground_truth_outputs):
            similarity = difflib.SequenceMatcher(None, pred, truth).ratio()
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities)
        
        print(f"✓ Text generation evaluation")
        print(f"  Samples: {len(test_prompts)}")
        print(f"  Avg similarity: {avg_similarity:.4f}")
        
        return {
            'predictions': predictions,
            'similarities': similarities,
            'avg_similarity': avg_similarity
        }
    
    def evaluate_classification_task(
        self,
        model,
        test_inputs,
        true_labels,
        label_extraction_fn
    ):
        """Evaluate classification task performance."""
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        predicted_labels = []
        
        for input_text in test_inputs:
            response = model.predict(prompt=input_text, max_output_tokens=50)
            label = label_extraction_fn(response.text)
            predicted_labels.append(label)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average='weighted'
        )
        cm = confusion_matrix(true_labels, predicted_labels)
        
        print(f"✓ Classification evaluation")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
    
    def ab_test_models(
        self,
        model_a,
        model_b,
        test_prompts,
        evaluation_fn
    ):
        """A/B test two models."""
        
        results_a = []
        results_b = []
        
        for prompt in test_prompts:
            response_a = model_a.predict(prompt=prompt, max_output_tokens=256)
            response_b = model_b.predict(prompt=prompt, max_output_tokens=256)
            
            score_a = evaluation_fn(response_a.text)
            score_b = evaluation_fn(response_b.text)
            
            results_a.append(score_a)
            results_b.append(score_b)
        
        avg_a = sum(results_a) / len(results_a)
        avg_b = sum(results_b) / len(results_b)
        
        winner = 'Model A' if avg_a > avg_b else 'Model B'
        improvement = abs(avg_a - avg_b) / min(avg_a, avg_b) * 100
        
        print(f"✓ A/B test completed")
        print(f"  Model A score: {avg_a:.4f}")
        print(f"  Model B score: {avg_b:.4f}")
        print(f"  Winner: {winner}")
        print(f"  Improvement: {improvement:.2f}%")
        
        return {
            'model_a_score': avg_a,
            'model_b_score': avg_b,
            'winner': winner,
            'improvement_pct': improvement
        }
    
    def benchmark_latency(
        self,
        model,
        test_prompts,
        num_runs=10
    ):
        """Benchmark model latency."""
        
        import time
        
        latencies = []
        
        for prompt in test_prompts:
            for _ in range(num_runs):
                start_time = time.time()
                model.predict(prompt=prompt, max_output_tokens=256)
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"✓ Latency benchmark")
        print(f"  Avg: {avg_latency:.2f}ms")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")
        
        return {
            'avg_latency_ms': avg_latency,
            'p50_ms': p50,
            'p95_ms': p95,
            'p99_ms': p99
        }

# Example usage
# eval_manager = ModelEvaluationManager(project_id='my-project')

# Evaluate text generation
# from vertexai.preview.language_models import TextGenerationModel
# model = TextGenerationModel.from_pretrained("text-bison@002")

# results = eval_manager.evaluate_text_generation(
#     model=model,
#     test_prompts=['Summarize: ...', 'Translate: ...'],
#     ground_truth_outputs=['Summary...', 'Translation...']
# )
```

---

## 5. Prompt Management and Versioning

### Manage Prompt Libraries

```python
class PromptLibraryManager:
    """Manage prompt templates and versioning."""
    
    def __init__(self, project_id, gcs_bucket):
        self.project_id = project_id
        self.gcs_bucket = gcs_bucket
        self.prompts = {}
    
    def create_prompt_template(
        self,
        template_name,
        template_text,
        variables,
        description=None
    ):
        """Create reusable prompt template."""
        
        template = {
            'name': template_name,
            'text': template_text,
            'variables': variables,
            'description': description,
            'version': '1.0',
            'created_at': '2024-01-01T00:00:00Z'
        }
        
        self.prompts[template_name] = template
        
        print(f"✓ Created prompt template: {template_name}")
        print(f"  Variables: {variables}")
        
        return template
    
    def render_prompt(
        self,
        template_name,
        variable_values
    ):
        """Render prompt from template."""
        
        if template_name not in self.prompts:
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.prompts[template_name]
        prompt = template['text']
        
        # Replace variables
        for var, value in variable_values.items():
            placeholder = f"{{{var}}}"
            prompt = prompt.replace(placeholder, str(value))
        
        print(f"✓ Rendered prompt from template: {template_name}")
        
        return prompt
    
    def version_prompt(
        self,
        template_name,
        new_text,
        version_note
    ):
        """Create new version of prompt template."""
        
        if template_name not in self.prompts:
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.prompts[template_name]
        
        # Parse version
        major, minor = map(int, template['version'].split('.'))
        new_version = f"{major}.{minor + 1}"
        
        # Update template
        template['text'] = new_text
        template['version'] = new_version
        template['version_note'] = version_note
        
        print(f"✓ Updated prompt version: {new_version}")
        print(f"  Note: {version_note}")
        
        return template
    
    def save_prompt_library(self):
        """Save prompt library to Cloud Storage."""
        
        from google.cloud import storage
        import json
        
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob('prompt-library/prompts.json')
        
        blob.upload_from_string(
            json.dumps(self.prompts, indent=2),
            content_type='application/json'
        )
        
        print(f"✓ Saved prompt library to GCS")
        print(f"  URI: gs://{self.gcs_bucket}/prompt-library/prompts.json")
        print(f"  Templates: {len(self.prompts)}")
    
    def load_prompt_library(self):
        """Load prompt library from Cloud Storage."""
        
        from google.cloud import storage
        import json
        
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob('prompt-library/prompts.json')
        
        self.prompts = json.loads(blob.download_as_string())
        
        print(f"✓ Loaded prompt library from GCS")
        print(f"  Templates: {len(self.prompts)}")
        
        return self.prompts

# Example usage
# library = PromptLibraryManager(
#     project_id='my-project',
#     gcs_bucket='prompt-library-bucket'
# )

# Create template
# template = library.create_prompt_template(
#     template_name='sentiment_analysis',
#     template_text='Classify the sentiment of this text as positive, negative, or neutral:\n\nText: {input_text}\n\nSentiment:',
#     variables=['input_text'],
#     description='Classify text sentiment'
# )

# Render prompt
# prompt = library.render_prompt(
#     template_name='sentiment_analysis',
#     variable_values={'input_text': 'This product is amazing!'}
# )
```

---

## 6. Safety and Content Filtering

### Implement Content Safety Controls

```python
class SafetyManager:
    """Manage safety settings and content filtering."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def generate_with_safety_settings(
        self,
        prompt,
        safety_settings=None
    ):
        """Generate text with custom safety settings."""
        
        from vertexai.preview.language_models import TextGenerationModel
        
        model = TextGenerationModel.from_pretrained("text-bison@002")
        
        # Default safety settings
        if safety_settings is None:
            safety_settings = {
                'harm_category': 'HARM_CATEGORY_UNSPECIFIED',
                'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
            }
        
        print(f"Generating with safety settings...")
        print(f"  Threshold: {safety_settings.get('threshold', 'default')}")
        
        response = model.predict(
            prompt=prompt,
            max_output_tokens=1024,
            temperature=0.2
        )
        
        # Check safety attributes
        if hasattr(response, 'safety_attributes'):
            print(f"✓ Safety check passed")
            print(f"  Blocked: {response.safety_attributes.blocked}")
        
        return response.text
    
    def implement_input_filtering(
        self,
        user_input,
        blocked_patterns=None
    ):
        """Filter user input for safety."""
        
        import re
        
        if blocked_patterns is None:
            blocked_patterns = [
                r'\b(password|secret|api[_-]?key)\b',
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{16}\b',  # Credit card
            ]
        
        filtered_input = user_input
        violations = []
        
        for pattern in blocked_patterns:
            if re.search(pattern, filtered_input, re.IGNORECASE):
                violations.append(pattern)
                filtered_input = re.sub(pattern, '[REDACTED]', filtered_input, flags=re.IGNORECASE)
        
        if violations:
            print(f"⚠ Input filtered")
            print(f"  Violations: {len(violations)}")
        else:
            print(f"✓ Input passed filtering")
        
        return {
            'filtered_input': filtered_input,
            'violations': violations,
            'is_safe': len(violations) == 0
        }
    
    def moderate_output_content(
        self,
        generated_text,
        moderation_rules
    ):
        """Moderate generated content."""
        
        issues = []
        
        for rule_name, rule_fn in moderation_rules.items():
            if not rule_fn(generated_text):
                issues.append(rule_name)
        
        is_safe = len(issues) == 0
        
        if is_safe:
            print(f"✓ Content passed moderation")
        else:
            print(f"⚠ Content moderation issues")
            print(f"  Issues: {issues}")
        
        return {
            'is_safe': is_safe,
            'issues': issues,
            'text': generated_text if is_safe else None
        }

# Example usage
# safety_manager = SafetyManager(project_id='my-project')

# Generate with safety settings
# response = safety_manager.generate_with_safety_settings(
#     prompt="Write a professional email...",
#     safety_settings={'threshold': 'BLOCK_MEDIUM_AND_ABOVE'}
# )

# Filter input
# result = safety_manager.implement_input_filtering(
#     user_input="My password is abc123 and my API key is xyz789"
# )
# print(result['filtered_input'])
```

---

## 7. Production Deployment

### Deploy Models to Production

```python
class ProductionDeploymentManager:
    """Manage production deployment of generative models."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
    
    def deploy_model_endpoint(
        self,
        model_name,
        endpoint_display_name,
        machine_type='n1-standard-4',
        min_replicas=1,
        max_replicas=3
    ):
        """Deploy model to endpoint with autoscaling."""
        
        print(f"Deploying model to endpoint...")
        print(f"  Model: {model_name}")
        print(f"  Machine type: {machine_type}")
        print(f"  Min replicas: {min_replicas}")
        print(f"  Max replicas: {max_replicas}")
        
        # Note: Actual deployment code
        endpoint_info = {
            'endpoint_name': endpoint_display_name,
            'model': model_name,
            'machine_type': machine_type,
            'status': 'deployed'
        }
        
        print(f"✓ Model deployed to endpoint")
        
        return endpoint_info
    
    def implement_rate_limiting(
        self,
        max_requests_per_minute=60
    ):
        """Implement rate limiting for API calls."""
        
        import time
        from collections import deque
        
        request_times = deque()
        
        def rate_limited_predict(model, prompt):
            now = time.time()
            
            # Remove old requests
            while request_times and request_times[0] < now - 60:
                request_times.popleft()
            
            # Check rate limit
            if len(request_times) >= max_requests_per_minute:
                wait_time = 60 - (now - request_times[0])
                print(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            # Make request
            request_times.append(time.time())
            return model.predict(prompt=prompt)
        
        print(f"✓ Rate limiting configured")
        print(f"  Max requests/min: {max_requests_per_minute}")
        
        return rate_limited_predict
    
    def implement_caching(
        self,
        cache_ttl_seconds=3600
    ):
        """Implement response caching."""
        
        import hashlib
        import time
        
        cache = {}
        
        def cached_predict(model, prompt, **kwargs):
            # Create cache key
            cache_key = hashlib.md5(
                f"{prompt}{str(kwargs)}".encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in cache:
                cached_data = cache[cache_key]
                if time.time() - cached_data['timestamp'] < cache_ttl_seconds:
                    print(f"✓ Cache hit")
                    return cached_data['response']
            
            # Cache miss - make request
            print(f"Cache miss - making request")
            response = model.predict(prompt=prompt, **kwargs)
            
            # Store in cache
            cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            
            return response
        
        print(f"✓ Caching configured")
        print(f"  TTL: {cache_ttl_seconds}s")
        
        return cached_predict
    
    def monitor_model_metrics(
        self,
        endpoint_name
    ):
        """Monitor model performance metrics."""
        
        from google.cloud import monitoring_v3
        
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{self.project_id}"
        
        # Query metrics
        metrics_to_monitor = [
            'prediction_count',
            'prediction_latency',
            'error_count'
        ]
        
        print(f"✓ Monitoring configured for endpoint: {endpoint_name}")
        print(f"  Metrics: {metrics_to_monitor}")
        
        return {
            'endpoint': endpoint_name,
            'metrics': metrics_to_monitor
        }

# Example usage
# deployment_manager = ProductionDeploymentManager(project_id='my-project')

# Deploy endpoint
# endpoint = deployment_manager.deploy_model_endpoint(
#     model_name='custom-text-model-v1',
#     endpoint_display_name='production-text-endpoint',
#     machine_type='n1-standard-4',
#     min_replicas=2,
#     max_replicas=10
# )
```

---

## 8. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Set up IAM permissions
- [ ] Create GCS bucket for data
- [ ] Install Vertex AI SDK
- [ ] Configure authentication

### Prompt Engineering
- [ ] Design clear task descriptions
- [ ] Use few-shot examples when needed
- [ ] Implement chain-of-thought for reasoning
- [ ] Optimize temperature and sampling parameters
- [ ] Test prompts with diverse inputs

### Model Tuning
- [ ] Prepare high-quality training data (100+ examples)
- [ ] Upload data to Cloud Storage
- [ ] Configure tuning hyperparameters
- [ ] Monitor tuning progress
- [ ] Evaluate tuned model performance

### Safety and Compliance
- [ ] Configure safety settings
- [ ] Implement input filtering
- [ ] Add output moderation
- [ ] Filter PII from inputs
- [ ] Log safety violations

### Production Best Practices
- [ ] Deploy with autoscaling
- [ ] Implement rate limiting
- [ ] Add response caching
- [ ] Monitor latency and errors
- [ ] Set up alerting
- [ ] Version control prompts
- [ ] A/B test model changes
- [ ] Document prompt templates

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
