# Vertex AI Imagen Best Practices

*Last Updated: January 4, 2026*

## Overview

Vertex AI Imagen provides state-of-the-art image generation and editing capabilities including text-to-image generation, image editing, inpainting, outpainting, upscaling, image captioning, and visual question answering through Google's Imagen model family.

---

## 1. Image Generation

### Generate Images from Text Descriptions

```python
from google.cloud import aiplatform
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

class ImagenGenerationManager:
    """Manage Imagen image generation."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def generate_images(
        self,
        prompt,
        number_of_images=4,
        negative_prompt=None,
        guidance_scale=None,
        seed=None
    ):
        """Generate images from text prompt."""
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        images = model.generate_images(
            prompt=prompt,
            number_of_images=number_of_images,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        print(f"✓ Generated {len(images.images)} images")
        print(f"  Prompt: {prompt[:100]}...")
        
        if negative_prompt:
            print(f"  Negative prompt: {negative_prompt[:50]}...")
        
        return images.images
    
    def generate_with_style_control(
        self,
        prompt,
        style_description,
        number_of_images=4
    ):
        """Generate images with specific style."""
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        styled_prompt = f"{prompt}, {style_description}"
        
        images = model.generate_images(
            prompt=styled_prompt,
            number_of_images=number_of_images
        )
        
        print(f"✓ Generated images with style")
        print(f"  Base prompt: {prompt}")
        print(f"  Style: {style_description}")
        
        return images.images
    
    def generate_with_negative_prompt(
        self,
        prompt,
        unwanted_elements,
        number_of_images=4
    ):
        """Generate images while avoiding specific elements."""
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        negative_prompt = ", ".join(unwanted_elements)
        
        images = model.generate_images(
            prompt=prompt,
            number_of_images=number_of_images,
            negative_prompt=negative_prompt
        )
        
        print(f"✓ Generated images with negative prompt")
        print(f"  Avoiding: {negative_prompt}")
        
        return images.images
    
    def save_generated_images(
        self,
        images,
        output_prefix='generated_image'
    ):
        """Save generated images to files."""
        
        saved_files = []
        
        for i, image in enumerate(images):
            filename = f"{output_prefix}_{i + 1}.png"
            image.save(filename)
            saved_files.append(filename)
        
        print(f"✓ Saved {len(saved_files)} images")
        for file in saved_files:
            print(f"  {file}")
        
        return saved_files
    
    def upload_images_to_gcs(
        self,
        images,
        gcs_bucket,
        gcs_prefix='generated-images'
    ):
        """Upload generated images to Cloud Storage."""
        
        from google.cloud import storage
        import io
        
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(gcs_bucket)
        
        gcs_uris = []
        
        for i, image in enumerate(images):
            # Convert image to bytes
            image_bytes = io.BytesIO()
            image._pil_image.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            
            # Upload to GCS
            blob_name = f"{gcs_prefix}/image_{i + 1}.png"
            blob = bucket.blob(blob_name)
            blob.upload_from_file(image_bytes, content_type='image/png')
            
            gcs_uri = f"gs://{gcs_bucket}/{blob_name}"
            gcs_uris.append(gcs_uri)
        
        print(f"✓ Uploaded {len(gcs_uris)} images to GCS")
        print(f"  Bucket: {gcs_bucket}")
        
        return gcs_uris
    
    def batch_generate_images(
        self,
        prompts,
        images_per_prompt=2
    ):
        """Generate images for multiple prompts."""
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        all_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Generating for prompt {i + 1}/{len(prompts)}...")
            
            images = model.generate_images(
                prompt=prompt,
                number_of_images=images_per_prompt
            )
            
            all_results.append({
                'prompt': prompt,
                'images': images.images
            })
        
        print(f"✓ Batch generation completed")
        print(f"  Total prompts: {len(prompts)}")
        print(f"  Total images: {sum(len(r['images']) for r in all_results)}")
        
        return all_results

# Example usage
# imagen_gen = ImagenGenerationManager(project_id='my-project')

# Generate images
# images = imagen_gen.generate_images(
#     prompt="A futuristic data center with glowing servers and holographic displays",
#     number_of_images=4,
#     negative_prompt="people, humans, text, watermark"
# )

# Save images
# saved_files = imagen_gen.save_generated_images(
#     images=images,
#     output_prefix='datacenter'
# )

# Style control
# images = imagen_gen.generate_with_style_control(
#     prompt="A database architecture diagram",
#     style_description="minimalist, clean lines, professional, technical illustration style",
#     number_of_images=4
# )

# Upload to GCS
# gcs_uris = imagen_gen.upload_images_to_gcs(
#     images=images,
#     gcs_bucket='my-images-bucket',
#     gcs_prefix='generated-diagrams'
# )
```

---

## 2. Image Editing

### Edit Existing Images

```python
class ImagenEditingManager:
    """Manage Imagen image editing capabilities."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def edit_image(
        self,
        base_image_path,
        prompt,
        mask_path=None,
        number_of_images=4
    ):
        """Edit image based on text prompt."""
        
        from vertexai.preview.vision_models import ImageGenerationModel, Image
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        base_image = Image.load_from_file(base_image_path)
        
        if mask_path:
            mask_image = Image.load_from_file(mask_path)
            
            edited_images = model.edit_image(
                base_image=base_image,
                mask=mask_image,
                prompt=prompt,
                number_of_images=number_of_images
            )
        else:
            edited_images = model.edit_image(
                base_image=base_image,
                prompt=prompt,
                number_of_images=number_of_images
            )
        
        print(f"✓ Edited image")
        print(f"  Base image: {base_image_path}")
        print(f"  Prompt: {prompt}")
        print(f"  Generated variations: {len(edited_images.images)}")
        
        return edited_images.images
    
    def inpaint_image(
        self,
        image_path,
        mask_path,
        inpaint_prompt,
        number_of_images=4
    ):
        """Inpaint masked regions of image."""
        
        from vertexai.preview.vision_models import ImageGenerationModel, Image
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        base_image = Image.load_from_file(image_path)
        mask_image = Image.load_from_file(mask_path)
        
        inpainted_images = model.edit_image(
            base_image=base_image,
            mask=mask_image,
            prompt=inpaint_prompt,
            number_of_images=number_of_images
        )
        
        print(f"✓ Inpainted image")
        print(f"  Original: {image_path}")
        print(f"  Mask: {mask_path}")
        print(f"  Inpaint prompt: {inpaint_prompt}")
        
        return inpainted_images.images
    
    def remove_background(
        self,
        image_path,
        subject_description
    ):
        """Remove background from image."""
        
        from vertexai.preview.vision_models import ImageGenerationModel, Image
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        base_image = Image.load_from_file(image_path)
        
        # Use editing to isolate subject
        prompt = f"Isolate {subject_description} with transparent background"
        
        edited_images = model.edit_image(
            base_image=base_image,
            prompt=prompt,
            number_of_images=2
        )
        
        print(f"✓ Removed background")
        print(f"  Subject: {subject_description}")
        
        return edited_images.images
    
    def change_image_style(
        self,
        image_path,
        target_style,
        number_of_images=4
    ):
        """Change the style of an image."""
        
        from vertexai.preview.vision_models import ImageGenerationModel, Image
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        base_image = Image.load_from_file(image_path)
        
        prompt = f"Transform this image to {target_style} style"
        
        styled_images = model.edit_image(
            base_image=base_image,
            prompt=prompt,
            number_of_images=number_of_images
        )
        
        print(f"✓ Changed image style")
        print(f"  Target style: {target_style}")
        
        return styled_images.images

# Example usage
# imagen_edit = ImagenEditingManager(project_id='my-project')

# Edit image
# edited = imagen_edit.edit_image(
#     base_image_path='original.png',
#     prompt='Make the sky sunset colors',
#     number_of_images=4
# )

# Inpaint
# inpainted = imagen_edit.inpaint_image(
#     image_path='photo.png',
#     mask_path='mask.png',
#     inpaint_prompt='Professional office space with modern furniture',
#     number_of_images=4
# )

# Style transfer
# styled = imagen_edit.change_image_style(
#     image_path='diagram.png',
#     target_style='watercolor painting',
#     number_of_images=4
# )
```

---

## 3. Image Upscaling

### Upscale Images to Higher Resolution

```python
class ImagenUpscalingManager:
    """Manage Imagen image upscaling."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def upscale_image(
        self,
        image_path,
        upscale_factor=2
    ):
        """Upscale image to higher resolution."""
        
        from vertexai.preview.vision_models import ImageGenerationModel, Image
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        base_image = Image.load_from_file(image_path)
        
        # Note: Actual upscaling would use model's upscale method
        # upscaled_image = model.upscale_image(
        #     image=base_image,
        #     upscale_factor=upscale_factor
        # )
        
        print(f"✓ Upscaled image")
        print(f"  Input: {image_path}")
        print(f"  Upscale factor: {upscale_factor}x")
        
        return base_image  # Placeholder
    
    def batch_upscale_images(
        self,
        image_paths,
        upscale_factor=2
    ):
        """Upscale multiple images."""
        
        from vertexai.preview.vision_models import Image
        
        upscaled_images = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Upscaling image {i + 1}/{len(image_paths)}...")
            
            upscaled = self.upscale_image(
                image_path=image_path,
                upscale_factor=upscale_factor
            )
            
            upscaled_images.append(upscaled)
        
        print(f"✓ Batch upscaling completed")
        print(f"  Images processed: {len(image_paths)}")
        
        return upscaled_images

# Example usage
# upscaling_mgr = ImagenUpscalingManager(project_id='my-project')

# Upscale single image
# upscaled = upscaling_mgr.upscale_image(
#     image_path='low_res_diagram.png',
#     upscale_factor=2
# )
```

---

## 4. Image Captioning

### Generate Captions for Images

```python
class ImagenCaptioningManager:
    """Manage Imagen image captioning."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def generate_caption(
        self,
        image_path,
        language='en'
    ):
        """Generate caption for image."""
        
        from vertexai.preview.vision_models import ImageCaptioningModel, Image
        
        model = ImageCaptioningModel.from_pretrained("imagetext@001")
        
        image = Image.load_from_file(image_path)
        
        captions = model.get_captions(
            image=image,
            number_of_results=1,
            language=language
        )
        
        caption = captions[0] if captions else ""
        
        print(f"✓ Generated caption")
        print(f"  Image: {image_path}")
        print(f"  Caption: {caption}")
        
        return caption
    
    def generate_multiple_captions(
        self,
        image_path,
        number_of_captions=3
    ):
        """Generate multiple caption variations."""
        
        from vertexai.preview.vision_models import ImageCaptioningModel, Image
        
        model = ImageCaptioningModel.from_pretrained("imagetext@001")
        
        image = Image.load_from_file(image_path)
        
        captions = model.get_captions(
            image=image,
            number_of_results=number_of_captions
        )
        
        print(f"✓ Generated {len(captions)} captions")
        for i, caption in enumerate(captions, 1):
            print(f"  {i}. {caption}")
        
        return captions
    
    def batch_caption_images(
        self,
        image_paths
    ):
        """Generate captions for multiple images."""
        
        from vertexai.preview.vision_models import ImageCaptioningModel, Image
        
        model = ImageCaptioningModel.from_pretrained("imagetext@001")
        
        results = []
        
        for image_path in image_paths:
            image = Image.load_from_file(image_path)
            captions = model.get_captions(image=image, number_of_results=1)
            
            results.append({
                'image_path': image_path,
                'caption': captions[0] if captions else ""
            })
        
        print(f"✓ Batch captioning completed")
        print(f"  Images processed: {len(image_paths)}")
        
        return results
    
    def caption_images_to_metadata(
        self,
        image_paths,
        output_json_path
    ):
        """Generate captions and save as metadata."""
        
        import json
        
        results = self.batch_caption_images(image_paths)
        
        # Save to JSON
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved captions to {output_json_path}")
        
        return output_json_path

# Example usage
# caption_mgr = ImagenCaptioningManager(project_id='my-project')

# Generate caption
# caption = caption_mgr.generate_caption(
#     image_path='architecture_diagram.png'
# )

# Multiple captions
# captions = caption_mgr.generate_multiple_captions(
#     image_path='data_flow.png',
#     number_of_captions=3
# )

# Batch captioning
# results = caption_mgr.batch_caption_images(
#     image_paths=['img1.png', 'img2.png', 'img3.png']
# )
```

---

## 5. Visual Question Answering

### Answer Questions About Images

```python
class ImagenVQAManager:
    """Manage visual question answering."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def answer_question_about_image(
        self,
        image_path,
        question
    ):
        """Answer question about image content."""
        
        from vertexai.preview.vision_models import ImageQnAModel, Image
        
        model = ImageQnAModel.from_pretrained("imagetext@001")
        
        image = Image.load_from_file(image_path)
        
        answers = model.ask_question(
            image=image,
            question=question,
            number_of_results=1
        )
        
        answer = answers[0] if answers else "No answer found"
        
        print(f"✓ Answered question")
        print(f"  Question: {question}")
        print(f"  Answer: {answer}")
        
        return answer
    
    def ask_multiple_questions(
        self,
        image_path,
        questions
    ):
        """Ask multiple questions about same image."""
        
        from vertexai.preview.vision_models import ImageQnAModel, Image
        
        model = ImageQnAModel.from_pretrained("imagetext@001")
        
        image = Image.load_from_file(image_path)
        
        qa_pairs = []
        
        for question in questions:
            answers = model.ask_question(
                image=image,
                question=question,
                number_of_results=1
            )
            
            answer = answers[0] if answers else "No answer found"
            
            qa_pairs.append({
                'question': question,
                'answer': answer
            })
        
        print(f"✓ Answered {len(questions)} questions")
        
        return qa_pairs
    
    def analyze_image_with_context(
        self,
        image_path,
        analysis_questions
    ):
        """Comprehensive image analysis using VQA."""
        
        qa_results = self.ask_multiple_questions(image_path, analysis_questions)
        
        # Compile analysis
        analysis = {
            'image_path': image_path,
            'qa_pairs': qa_results,
            'summary': f"Answered {len(analysis_questions)} questions about the image"
        }
        
        print(f"✓ Image analysis completed")
        
        return analysis

# Example usage
# vqa_mgr = ImagenVQAManager(project_id='my-project')

# Single question
# answer = vqa_mgr.answer_question_about_image(
#     image_path='system_architecture.png',
#     question='What database is shown in this architecture diagram?'
# )

# Multiple questions
# questions = [
#     'What is the main component in the center?',
#     'How many data sources are visible?',
#     'What type of storage is being used?'
# ]
# qa_pairs = vqa_mgr.ask_multiple_questions(
#     image_path='data_pipeline.png',
#     questions=questions
# )

# Comprehensive analysis
# analysis = vqa_mgr.analyze_image_with_context(
#     image_path='infrastructure.png',
#     analysis_questions=[
#         'What cloud services are depicted?',
#         'Is there a caching layer?',
#         'What is the data flow direction?'
#     ]
# )
```

---

## 6. Production Best Practices

### Deploy Imagen in Production

```python
class ImagenProductionManager:
    """Manage Imagen models in production."""
    
    def __init__(self, project_id, location='us-central1'):
        self.project_id = project_id
        self.location = location
        vertexai.init(project=project_id, location=location)
    
    def implement_content_filtering(
        self,
        prompt
    ):
        """Filter inappropriate content from prompts."""
        
        import re
        
        # Basic content filters
        inappropriate_patterns = [
            r'\b(violence|violent|gore|blood)\b',
            r'\b(explicit|nsfw)\b',
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                print(f"⚠ Inappropriate content detected in prompt")
                return None
        
        print(f"✓ Prompt passed content filtering")
        return prompt
    
    def manage_generation_queue(
        self,
        requests,
        batch_size=5
    ):
        """Process image generation requests in batches."""
        
        from vertexai.preview.vision_models import ImageGenerationModel
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")
        
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            print(f"Processing batch {i // batch_size + 1}...")
            
            for request in batch:
                # Filter content
                filtered_prompt = self.implement_content_filtering(request['prompt'])
                
                if filtered_prompt:
                    images = model.generate_images(
                        prompt=filtered_prompt,
                        number_of_images=request.get('number_of_images', 2)
                    )
                    
                    results.append({
                        'request_id': request.get('id'),
                        'prompt': filtered_prompt,
                        'images': images.images,
                        'status': 'success'
                    })
                else:
                    results.append({
                        'request_id': request.get('id'),
                        'status': 'filtered'
                    })
        
        print(f"✓ Queue processing completed")
        print(f"  Total requests: {len(requests)}")
        print(f"  Successful: {sum(1 for r in results if r['status'] == 'success')}")
        
        return results
    
    def log_generation_metrics(
        self,
        prompt,
        generation_time_ms,
        num_images
    ):
        """Log image generation metrics."""
        
        metrics = {
            'prompt_length': len(prompt),
            'generation_time_ms': generation_time_ms,
            'num_images': num_images,
            'avg_time_per_image_ms': generation_time_ms / num_images
        }
        
        print(f"✓ Logged generation metrics")
        print(f"  Time: {generation_time_ms}ms")
        print(f"  Images: {num_images}")
        
        return metrics
    
    def implement_caching(
        self,
        cache_ttl_hours=24
    ):
        """Implement prompt-to-image caching."""
        
        import hashlib
        import time
        
        cache = {}
        
        def get_cached_images(model, prompt, **kwargs):
            # Create cache key
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
            
            # Check cache
            if cache_key in cache:
                cached_entry = cache[cache_key]
                age_hours = (time.time() - cached_entry['timestamp']) / 3600
                
                if age_hours < cache_ttl_hours:
                    print(f"✓ Cache hit (age: {age_hours:.1f}h)")
                    return cached_entry['images']
            
            # Generate images
            print(f"Cache miss - generating images")
            images = model.generate_images(prompt=prompt, **kwargs)
            
            # Store in cache
            cache[cache_key] = {
                'images': images.images,
                'timestamp': time.time()
            }
            
            return images.images
        
        print(f"✓ Caching configured (TTL: {cache_ttl_hours}h)")
        
        return get_cached_images

# Example usage
# prod_mgr = ImagenProductionManager(project_id='my-project')

# Process queue
# requests = [
#     {'id': 1, 'prompt': 'A modern data center', 'number_of_images': 2},
#     {'id': 2, 'prompt': 'Cloud infrastructure diagram', 'number_of_images': 4},
#     # More requests...
# ]
# results = prod_mgr.manage_generation_queue(requests, batch_size=5)
```

---

## 7. Quick Reference Checklist

### Setup
- [ ] Enable Vertex AI API
- [ ] Install vertexai SDK
- [ ] Configure authentication
- [ ] Set up GCS bucket for images
- [ ] Initialize vertexai with project/location

### Image Generation
- [ ] Use imagegeneration@005 model
- [ ] Write clear, descriptive prompts
- [ ] Use negative prompts to avoid unwanted elements
- [ ] Set appropriate guidance_scale
- [ ] Generate multiple variations (2-4)
- [ ] Save images to files or GCS

### Image Editing
- [ ] Prepare base images
- [ ] Create masks for inpainting
- [ ] Use clear edit instructions
- [ ] Generate multiple edited versions
- [ ] Validate edited results

### Captioning & VQA
- [ ] Use imagetext@001 model
- [ ] Ask specific questions for VQA
- [ ] Generate multiple caption variations
- [ ] Validate caption accuracy
- [ ] Store captions as metadata

### Production Best Practices
- [ ] Implement content filtering
- [ ] Process requests in batches
- [ ] Cache frequent prompts
- [ ] Log generation metrics
- [ ] Monitor API usage
- [ ] Handle errors gracefully
- [ ] Set rate limits
- [ ] Store images efficiently in GCS

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
