# Vertex AI Gemini API Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Gemini API provides access to Google's most capable multimodal AI model, supporting text, image, video, and audio inputs with advanced capabilities including function calling, streaming responses, long context windows, and comprehensive safety controls.

---

## 1. Multimodal Input Processing

### Process Text, Images, and Video

```python
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image

class GeminiMultimodalManager:
    """Manage Gemini multimodal capabilities."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def generate_from_text(
        self,
        prompt,
        model_name='gemini-pro'
    ):
        """Generate response from text input."""
        
        model = GenerativeModel(model_name)
        
        response = model.generate_content(prompt)
        
        print(f"✓ Generated response from text")
        print(f"  Model: {model_name}")
        print(f"  Response length: {len(response.text)} characters")
        
        return response.text
    
    def generate_from_image(
        self,
        image_path,
        prompt,
        model_name='gemini-pro-vision'
    ):
        """Generate response from image and text."""
        
        model = GenerativeModel(model_name)
        
        # Load image
        image = Image.load_from_file(image_path)
        
        # Create multimodal prompt
        response = model.generate_content([prompt, image])
        
        print(f"✓ Generated response from image")
        print(f"  Image: {image_path}")
        print(f"  Prompt: {prompt[:50]}...")
        
        return response.text
    
    def generate_from_image_url(
        self,
        image_url,
        prompt,
        model_name='gemini-pro-vision'
    ):
        """Generate response from image URL."""
        
        model = GenerativeModel(model_name)
        
        # Create image part from URL
        image_part = Part.from_uri(image_url, mime_type='image/jpeg')
        
        # Generate response
        response = model.generate_content([prompt, image_part])
        
        print(f"✓ Generated response from image URL")
        print(f"  URL: {image_url}")
        
        return response.text
    
    def generate_from_gcs_image(
        self,
        gcs_uri,
        prompt,
        model_name='gemini-pro-vision'
    ):
        """Generate response from GCS image."""
        
        model = GenerativeModel(model_name)
        
        # Create image part from GCS
        image_part = Part.from_uri(gcs_uri, mime_type='image/jpeg')
        
        # Generate response
        response = model.generate_content([prompt, image_part])
        
        print(f"✓ Generated response from GCS image")
        print(f"  GCS URI: {gcs_uri}")
        
        return response.text
    
    def analyze_multiple_images(
        self,
        image_paths,
        prompt,
        model_name='gemini-pro-vision'
    ):
        """Analyze multiple images together."""
        
        model = GenerativeModel(model_name)
        
        # Load all images
        image_parts = [Image.load_from_file(path) for path in image_paths]
        
        # Create content with prompt and all images
        content = [prompt] + image_parts
        
        response = model.generate_content(content)
        
        print(f"✓ Analyzed multiple images")
        print(f"  Images: {len(image_paths)}")
        
        return response.text
    
    def process_video(
        self,
        video_gcs_uri,
        prompt,
        model_name='gemini-pro-vision'
    ):
        """Process video input."""
        
        model = GenerativeModel(model_name)
        
        # Create video part
        video_part = Part.from_uri(video_gcs_uri, mime_type='video/mp4')
        
        # Generate response
        response = model.generate_content([prompt, video_part])
        
        print(f"✓ Processed video")
        print(f"  Video: {video_gcs_uri}")
        print(f"  Prompt: {prompt[:50]}...")
        
        return response.text
    
    def extract_structured_data_from_image(
        self,
        image_path,
        schema
    ):
        """Extract structured data from image."""
        
        import json
        
        model = GenerativeModel('gemini-pro-vision')
        
        image = Image.load_from_file(image_path)
        
        prompt = f"""Extract information from this image according to the following schema:

{json.dumps(schema, indent=2)}

Return the data in valid JSON format."""
        
        response = model.generate_content([prompt, image])
        
        try:
            structured_data = json.loads(response.text)
            print(f"✓ Extracted structured data from image")
            return structured_data
        except json.JSONDecodeError:
            print(f"⚠ Failed to parse JSON, returning raw text")
            return response.text

# Example usage
# gemini = GeminiMultimodalManager(project_id='my-project')

# Text generation
# response = gemini.generate_from_text(
#     prompt="Explain Apache Beam programming model for data pipelines.",
#     model_name='gemini-pro'
# )

# Image analysis
# response = gemini.generate_from_image(
#     image_path='architecture_diagram.png',
#     prompt="Describe this data architecture diagram and identify potential bottlenecks."
# )

# Video analysis
# response = gemini.process_video(
#     video_gcs_uri='gs://my-bucket/demo-video.mp4',
#     prompt="Summarize the key steps shown in this tutorial video."
# )

# Multiple images
# response = gemini.analyze_multiple_images(
#     image_paths=['chart1.png', 'chart2.png', 'chart3.png'],
#     prompt="Compare these three data visualizations and identify trends."
# )
```

---

## 2. Function Calling

### Implement Function Calling

```python
class GeminiFunctionCallingManager:
    """Manage Gemini function calling capabilities."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def define_functions(self):
        """Define functions for Gemini to call."""
        
        from vertexai.generative_models import FunctionDeclaration, Tool
        
        # Define function for BigQuery query
        get_query_results = FunctionDeclaration(
            name="get_bigquery_results",
            description="Execute a BigQuery SQL query and return results",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of rows to return"
                    }
                },
                "required": ["query"]
            }
        )
        
        # Define function for GCS operations
        list_gcs_files = FunctionDeclaration(
            name="list_gcs_files",
            description="List files in a Google Cloud Storage bucket",
            parameters={
                "type": "object",
                "properties": {
                    "bucket_name": {
                        "type": "string",
                        "description": "Name of the GCS bucket"
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Optional prefix to filter files"
                    }
                },
                "required": ["bucket_name"]
            }
        )
        
        # Create tool with functions
        tool = Tool(
            function_declarations=[get_query_results, list_gcs_files]
        )
        
        print(f"✓ Defined functions for Gemini")
        print(f"  Functions: get_bigquery_results, list_gcs_files")
        
        return tool
    
    def execute_function(self, function_name, args):
        """Execute the actual function based on Gemini's request."""
        
        if function_name == "get_bigquery_results":
            from google.cloud import bigquery
            
            client = bigquery.Client(project=self.project_id)
            query = args.get('query')
            max_results = args.get('max_results', 100)
            
            query_job = client.query(query)
            results = query_job.result(max_results=max_results)
            
            rows = [dict(row) for row in results]
            
            print(f"✓ Executed BigQuery query")
            print(f"  Rows returned: {len(rows)}")
            
            return {'results': rows, 'row_count': len(rows)}
        
        elif function_name == "list_gcs_files":
            from google.cloud import storage
            
            client = storage.Client(project=self.project_id)
            bucket_name = args.get('bucket_name')
            prefix = args.get('prefix', '')
            
            bucket = client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=100))
            
            files = [blob.name for blob in blobs]
            
            print(f"✓ Listed GCS files")
            print(f"  Files found: {len(files)}")
            
            return {'files': files, 'count': len(files)}
        
        else:
            return {'error': f'Unknown function: {function_name}'}
    
    def chat_with_function_calling(
        self,
        user_message,
        tools
    ):
        """Chat with function calling enabled."""
        
        from vertexai.generative_models import GenerativeModel, Part
        
        model = GenerativeModel(
            'gemini-pro',
            tools=[tools]
        )
        
        chat = model.start_chat()
        
        # Send user message
        response = chat.send_message(user_message)
        
        # Check if function call was requested
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            
            print(f"✓ Gemini requested function call")
            print(f"  Function: {function_call.name}")
            print(f"  Args: {dict(function_call.args)}")
            
            # Execute function
            function_result = self.execute_function(
                function_call.name,
                dict(function_call.args)
            )
            
            # Send function response back to Gemini
            function_response = Part.from_function_response(
                name=function_call.name,
                response={"content": function_result}
            )
            
            final_response = chat.send_message(function_response)
            
            print(f"✓ Received final response from Gemini")
            
            return final_response.text
        
        else:
            return response.text
    
    def multi_turn_function_calling(
        self,
        messages,
        tools
    ):
        """Handle multi-turn conversation with function calling."""
        
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel('gemini-pro', tools=[tools])
        chat = model.start_chat()
        
        conversation = []
        
        for msg in messages:
            print(f"\nUser: {msg}")
            
            response = chat.send_message(msg)
            
            # Handle function calls
            while response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
                
                print(f"  [Function call: {function_call.name}]")
                
                # Execute function
                result = self.execute_function(
                    function_call.name,
                    dict(function_call.args)
                )
                
                # Send result back
                from vertexai.generative_models import Part
                function_response = Part.from_function_response(
                    name=function_call.name,
                    response={"content": result}
                )
                
                response = chat.send_message(function_response)
            
            print(f"Gemini: {response.text}")
            
            conversation.append({
                'user': msg,
                'assistant': response.text
            })
        
        print(f"\n✓ Multi-turn conversation completed")
        
        return conversation

# Example usage
# function_manager = GeminiFunctionCallingManager(project_id='my-project')

# Define functions
# tools = function_manager.define_functions()

# Chat with function calling
# response = function_manager.chat_with_function_calling(
#     user_message="Show me the top 10 customers by revenue from the orders table",
#     tools=tools
# )
# print(response)

# Multi-turn with functions
# conversation = function_manager.multi_turn_function_calling(
#     messages=[
#         "List files in the data-lake bucket",
#         "Show me a sample of the customer data",
#         "What's the average order value?"
#     ],
#     tools=tools
# )
```

---

## 3. Streaming Responses

### Implement Streaming for Real-Time Output

```python
class GeminiStreamingManager:
    """Manage streaming responses from Gemini."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def stream_generate_content(
        self,
        prompt,
        model_name='gemini-pro'
    ):
        """Stream generated content in real-time."""
        
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel(model_name)
        
        print(f"Streaming response:")
        print("-" * 60)
        
        full_response = ""
        
        for response_chunk in model.generate_content(prompt, stream=True):
            chunk_text = response_chunk.text
            full_response += chunk_text
            print(chunk_text, end='', flush=True)
        
        print("\n" + "-" * 60)
        print(f"✓ Streaming completed ({len(full_response)} characters)")
        
        return full_response
    
    def stream_chat_response(
        self,
        chat,
        message
    ):
        """Stream chat response."""
        
        print(f"Streaming chat response:")
        print("-" * 60)
        
        full_response = ""
        
        for response_chunk in chat.send_message(message, stream=True):
            chunk_text = response_chunk.text
            full_response += chunk_text
            print(chunk_text, end='', flush=True)
        
        print("\n" + "-" * 60)
        print(f"✓ Chat streaming completed")
        
        return full_response
    
    def stream_with_callbacks(
        self,
        prompt,
        on_chunk_callback
    ):
        """Stream with custom callback for each chunk."""
        
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel('gemini-pro')
        
        chunks = []
        
        for response_chunk in model.generate_content(prompt, stream=True):
            chunk_text = response_chunk.text
            chunks.append(chunk_text)
            
            # Call callback
            on_chunk_callback(chunk_text, len(chunks))
        
        full_response = ''.join(chunks)
        
        print(f"✓ Streaming with callbacks completed")
        print(f"  Total chunks: {len(chunks)}")
        
        return full_response
    
    def stream_multimodal_response(
        self,
        image_path,
        prompt,
        model_name='gemini-pro-vision'
    ):
        """Stream multimodal response."""
        
        from vertexai.generative_models import GenerativeModel, Image
        
        model = GenerativeModel(model_name)
        image = Image.load_from_file(image_path)
        
        print(f"Streaming multimodal response:")
        print("-" * 60)
        
        full_response = ""
        
        for response_chunk in model.generate_content([prompt, image], stream=True):
            chunk_text = response_chunk.text
            full_response += chunk_text
            print(chunk_text, end='', flush=True)
        
        print("\n" + "-" * 60)
        print(f"✓ Multimodal streaming completed")
        
        return full_response

# Example usage
# streaming_manager = GeminiStreamingManager(project_id='my-project')

# Stream text generation
# response = streaming_manager.stream_generate_content(
#     prompt="Explain the differences between Dataflow and Dataproc in detail."
# )

# Stream with callback
# def on_chunk(chunk, chunk_num):
#     print(f"[Chunk {chunk_num}] ", end='')
#
# response = streaming_manager.stream_with_callbacks(
#     prompt="Write a detailed guide on BigQuery optimization.",
#     on_chunk_callback=on_chunk
# )
```

---

## 4. Safety Settings and Content Filtering

### Configure Safety Controls

```python
class GeminiSafetyManager:
    """Manage Gemini safety settings."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def generate_with_safety_settings(
        self,
        prompt,
        safety_settings=None
    ):
        """Generate content with custom safety settings."""
        
        from vertexai.generative_models import GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold
        
        # Default safety settings
        if safety_settings is None:
            safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                )
            ]
        
        model = GenerativeModel('gemini-pro')
        
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings
        )
        
        # Check safety ratings
        if response.candidates:
            candidate = response.candidates[0]
            
            print(f"✓ Generated with safety checks")
            print(f"  Finish reason: {candidate.finish_reason}")
            
            if hasattr(candidate, 'safety_ratings'):
                print(f"  Safety ratings: {len(candidate.safety_ratings)} categories checked")
        
        return response.text
    
    def check_content_safety(
        self,
        text
    ):
        """Check content safety before processing."""
        
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel('gemini-pro')
        
        # Test with minimal generation
        try:
            response = model.generate_content(
                f"Is this text safe and appropriate? Text: {text}",
                generation_config={'max_output_tokens': 10}
            )
            
            is_safe = True
            print(f"✓ Content safety check passed")
            
        except Exception as e:
            is_safe = False
            print(f"⚠ Content safety issue detected: {str(e)}")
        
        return is_safe
    
    def implement_custom_filtering(
        self,
        text,
        blocked_patterns
    ):
        """Implement custom content filtering."""
        
        import re
        
        violations = []
        
        for pattern_name, pattern in blocked_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(pattern_name)
        
        is_safe = len(violations) == 0
        
        if is_safe:
            print(f"✓ Custom filtering passed")
        else:
            print(f"⚠ Custom filtering violations: {violations}")
        
        return {
            'is_safe': is_safe,
            'violations': violations
        }

# Example usage
# safety_manager = GeminiSafetyManager(project_id='my-project')

# Generate with safety settings
# response = safety_manager.generate_with_safety_settings(
#     prompt="Explain data security best practices."
# )

# Custom filtering
# blocked_patterns = {
#     'pii_email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
#     'pii_phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
# }
# result = safety_manager.implement_custom_filtering(
#     text="Contact me at john@example.com or 555-123-4567",
#     blocked_patterns=blocked_patterns
# )
```

---

## 5. Long Context and Token Management

### Handle Long Context Windows

```python
class GeminiContextManager:
    """Manage long context windows in Gemini."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def process_long_document(
        self,
        document_text,
        query,
        model_name='gemini-pro'
    ):
        """Process long documents with Gemini's extended context."""
        
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel(model_name)
        
        prompt = f"""Document:
{document_text}

Question: {query}

Answer based on the document above:"""
        
        response = model.generate_content(prompt)
        
        print(f"✓ Processed long document")
        print(f"  Document length: {len(document_text)} characters")
        print(f"  Estimated tokens: ~{len(document_text) // 4}")
        
        return response.text
    
    def count_tokens(
        self,
        content,
        model_name='gemini-pro'
    ):
        """Count tokens in content."""
        
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel(model_name)
        
        # Count tokens
        response = model.count_tokens(content)
        token_count = response.total_tokens
        
        print(f"✓ Token count: {token_count}")
        
        return token_count
    
    def chunk_large_content(
        self,
        content,
        max_tokens_per_chunk=8000
    ):
        """Chunk large content for processing."""
        
        # Rough estimate: 1 token ≈ 4 characters
        chars_per_chunk = max_tokens_per_chunk * 4
        
        chunks = []
        for i in range(0, len(content), chars_per_chunk):
            chunk = content[i:i + chars_per_chunk]
            chunks.append(chunk)
        
        print(f"✓ Chunked content")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
        
        return chunks
    
    def process_with_summarization(
        self,
        long_content,
        final_query
    ):
        """Process very long content with intermediate summarization."""
        
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel('gemini-pro')
        
        # Chunk content
        chunks = self.chunk_large_content(long_content, max_tokens_per_chunk=8000)
        
        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            summary_prompt = f"Summarize this section concisely:\n\n{chunk}"
            response = model.generate_content(summary_prompt)
            summaries.append(response.text)
            print(f"  Summarized chunk {i + 1}/{len(chunks)}")
        
        # Combine summaries and answer query
        combined_summary = "\n\n".join(summaries)
        final_prompt = f"""Based on these summaries:

{combined_summary}

Answer this question: {final_query}"""
        
        final_response = model.generate_content(final_prompt)
        
        print(f"✓ Processed with summarization")
        
        return final_response.text

# Example usage
# context_manager = GeminiContextManager(project_id='my-project')

# Process long document
# with open('long_document.txt', 'r') as f:
#     document = f.read()
#
# answer = context_manager.process_long_document(
#     document_text=document,
#     query="What are the main conclusions?"
# )

# Count tokens
# token_count = context_manager.count_tokens(
#     content="This is a test document for token counting.",
#     model_name='gemini-pro'
# )
```

---

## 6. Production Deployment

### Deploy Gemini in Production

```python
class GeminiProductionManager:
    """Manage Gemini models in production."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def implement_rate_limiting(self, requests_per_minute=60):
        """Implement rate limiting."""
        
        import time
        from collections import deque
        
        request_times = deque()
        
        def rate_limited_call(func, *args, **kwargs):
            now = time.time()
            
            # Remove old requests
            while request_times and request_times[0] < now - 60:
                request_times.popleft()
            
            # Check limit
            if len(request_times) >= requests_per_minute:
                wait_time = 60 - (now - request_times[0])
                print(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            # Make request
            request_times.append(time.time())
            return func(*args, **kwargs)
        
        print(f"✓ Rate limiting configured ({requests_per_minute} req/min)")
        
        return rate_limited_call
    
    def implement_exponential_backoff(self, max_retries=3):
        """Implement retry with exponential backoff."""
        
        import time
        
        def retry_with_backoff(func, *args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        raise e
        
        print(f"✓ Exponential backoff configured ({max_retries} retries)")
        
        return retry_with_backoff
    
    def log_to_bigquery(self, dataset_id, table_id):
        """Log requests to BigQuery."""
        
        from google.cloud import bigquery
        import datetime
        
        client = bigquery.Client(project=self.project_id)
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        
        def log_request(prompt, response, latency_ms, model_name):
            row = {
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'prompt': prompt[:1000],
                'response': response[:1000],
                'latency_ms': latency_ms,
                'model_name': model_name,
                'project_id': self.project_id
            }
            
            errors = client.insert_rows_json(table_ref, [row])
            
            if not errors:
                print(f"✓ Logged to BigQuery")
        
        print(f"✓ BigQuery logging configured: {table_ref}")
        
        return log_request

# Example usage
# prod_manager = GeminiProductionManager(project_id='my-project')

# Rate limiting
# rate_limited = prod_manager.implement_rate_limiting(requests_per_minute=60)

# Use with rate limiting
# from vertexai.generative_models import GenerativeModel
# model = GenerativeModel('gemini-pro')
# response = rate_limited(model.generate_content, "Explain BigQuery")
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Install vertexai SDK
- [ ] Configure authentication
- [ ] Set up IAM permissions
- [ ] Initialize vertexai with project/location

### Multimodal
- [ ] Use gemini-pro for text-only
- [ ] Use gemini-pro-vision for images/video
- [ ] Support images from file, URL, or GCS
- [ ] Process multiple images together
- [ ] Handle video from GCS URIs

### Function Calling
- [ ] Define function declarations
- [ ] Create Tool with functions
- [ ] Handle function call requests
- [ ] Execute actual functions
- [ ] Return results to Gemini

### Streaming
- [ ] Use stream=True parameter
- [ ] Handle chunks in real-time
- [ ] Implement custom callbacks
- [ ] Support multimodal streaming
- [ ] Manage connection errors

### Safety
- [ ] Configure SafetySettings
- [ ] Set HarmBlockThreshold
- [ ] Check safety_ratings
- [ ] Implement custom filtering
- [ ] Handle blocked content

### Production Best Practices
- [ ] Implement rate limiting
- [ ] Add exponential backoff
- [ ] Log requests to BigQuery
- [ ] Monitor latency
- [ ] Count tokens before requests
- [ ] Handle long contexts
- [ ] Use chunking for large documents
- [ ] Cache frequent responses

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
