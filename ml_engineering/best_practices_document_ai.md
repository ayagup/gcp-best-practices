# Document AI Best Practices

*Last Updated: January 4, 2026*

## Overview

Document AI provides specialized processors for extracting structured data from documents including invoices, receipts, forms, contracts, and custom document types using machine learning.

---

## 1. Specialized Processors

### Invoice Parser

```python
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

class DocumentAIProcessor:
    """Wrapper for Document AI operations."""
    
    def __init__(self, project_id, location, processor_id):
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id
        
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
        
        self.processor_name = self.client.processor_path(
            project_id, location, processor_id
        )
    
    def process_document(self, file_path, mime_type='application/pdf'):
        """Process document with specified processor."""
        
        with open(file_path, 'rb') as file:
            file_content = file.read()
        
        raw_document = documentai.RawDocument(
            content=file_content,
            mime_type=mime_type
        )
        
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document
        )
        
        print(f"Processing document: {file_path}")
        
        result = self.client.process_document(request=request)
        
        return result.document

def parse_invoice(processor, file_path):
    """Parse invoice and extract key fields."""
    
    document = processor.process_document(file_path)
    
    # Extract entities
    invoice_data = {}
    
    for entity in document.entities:
        entity_type = entity.type_
        entity_value = entity.mention_text
        confidence = entity.confidence
        
        invoice_data[entity_type] = {
            'value': entity_value,
            'confidence': confidence
        }
        
        # Extract nested properties
        if entity.properties:
            invoice_data[entity_type]['properties'] = {}
            for prop in entity.properties:
                invoice_data[entity_type]['properties'][prop.type_] = {
                    'value': prop.mention_text,
                    'confidence': prop.confidence
                }
    
    # Display key fields
    print("\n=== Invoice Details ===\n")
    
    key_fields = [
        'invoice_id', 'invoice_date', 'due_date',
        'supplier_name', 'supplier_address',
        'total_amount', 'net_amount', 'total_tax_amount',
        'currency'
    ]
    
    for field in key_fields:
        if field in invoice_data:
            value = invoice_data[field]['value']
            conf = invoice_data[field]['confidence']
            print(f"{field}: {value} ({conf:.2%})")
    
    # Extract line items
    print("\n=== Line Items ===\n")
    
    for entity in document.entities:
        if entity.type_ == 'line_item':
            line_item = {}
            for prop in entity.properties:
                line_item[prop.type_] = prop.mention_text
            
            if line_item:
                print(f"Item: {line_item.get('line_item/description', 'N/A')}")
                print(f"  Quantity: {line_item.get('line_item/quantity', 'N/A')}")
                print(f"  Unit Price: {line_item.get('line_item/unit_price', 'N/A')}")
                print(f"  Amount: {line_item.get('line_item/amount', 'N/A')}\n")
    
    return invoice_data

# Example usage
processor = DocumentAIProcessor(
    project_id='my-project',
    location='us',
    processor_id='invoice-processor-id'
)

# invoice_data = parse_invoice(processor, 'invoice.pdf')
```

### Form Parser

```python
def parse_form(processor, file_path):
    """Parse form and extract form fields."""
    
    document = processor.process_document(file_path)
    
    form_fields = {}
    
    # Extract form fields from pages
    for page in document.pages:
        for form_field in page.form_fields:
            # Field name
            field_name = ''
            if form_field.field_name.text_anchor.text_segments:
                field_name = get_text(document.text, form_field.field_name.text_anchor)
            
            # Field value
            field_value = ''
            if form_field.field_value.text_anchor.text_segments:
                field_value = get_text(document.text, form_field.field_value.text_anchor)
            
            # Confidence
            confidence = form_field.field_name.confidence
            
            form_fields[field_name] = {
                'value': field_value,
                'confidence': confidence
            }
    
    print("\n=== Form Fields ===\n")
    
    for field_name, field_data in form_fields.items():
        if field_name:  # Skip empty field names
            print(f"{field_name}: {field_data['value']} ({field_data['confidence']:.2%})")
    
    return form_fields

def get_text(document_text, text_anchor):
    """Extract text from text anchor."""
    
    text_segments = text_anchor.text_segments
    text = ''
    
    for segment in text_segments:
        start_index = segment.start_index if segment.start_index else 0
        end_index = segment.end_index
        text += document_text[start_index:end_index]
    
    return text.strip()
```

---

## 2. OCR and Text Extraction

```python
def extract_text_from_document(processor, file_path):
    """Extract all text from document with OCR."""
    
    document = processor.process_document(file_path)
    
    # Full text
    print("=== Full Text ===\n")
    print(document.text)
    
    # Extract by pages
    print("\n=== Text by Page ===\n")
    
    for page_num, page in enumerate(document.pages, 1):
        print(f"Page {page_num}:")
        
        # Get page dimensions
        print(f"  Size: {page.dimension.width} x {page.dimension.height}")
        print(f"  Unit: {page.dimension.unit}")
        
        # Extract paragraphs
        if page.paragraphs:
            print(f"  Paragraphs: {len(page.paragraphs)}")
            
            for i, paragraph in enumerate(page.paragraphs[:3], 1):  # First 3
                para_text = get_text(document.text, paragraph.layout.text_anchor)
                print(f"    {i}. {para_text[:100]}...")
        
        # Extract lines
        if page.lines:
            print(f"  Lines: {len(page.lines)}")
        
        # Extract tokens (words)
        if page.tokens:
            print(f"  Tokens: {len(page.tokens)}")
        
        print()
    
    return document

def extract_tables(processor, file_path):
    """Extract tables from document."""
    
    document = processor.process_document(file_path)
    
    print("=== Tables ===\n")
    
    for page_num, page in enumerate(document.pages, 1):
        if not page.tables:
            continue
        
        print(f"Page {page_num}: {len(page.tables)} table(s)\n")
        
        for table_num, table in enumerate(page.tables, 1):
            print(f"Table {table_num}:")
            print(f"  Rows: {len(table.body_rows)}")
            print(f"  Columns: {len(table.header_rows[0].cells) if table.header_rows else 0}")
            
            # Extract header
            if table.header_rows:
                header_row = table.header_rows[0]
                headers = []
                
                for cell in header_row.cells:
                    cell_text = get_text(document.text, cell.layout.text_anchor)
                    headers.append(cell_text)
                
                print(f"  Headers: {', '.join(headers)}")
            
            # Extract first few data rows
            for row_num, row in enumerate(table.body_rows[:3], 1):
                row_data = []
                
                for cell in row.cells:
                    cell_text = get_text(document.text, cell.layout.text_anchor)
                    row_data.append(cell_text)
                
                print(f"  Row {row_num}: {' | '.join(row_data)}")
            
            print()
    
    return document
```

---

## 3. Custom Document Extractors

```python
from google.cloud import documentai_v1beta3 as documentai_beta

class CustomExtractorTrainer:
    """Train custom document extractors."""
    
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        self.client = documentai_beta.DocumentProcessorServiceClient(client_options=opts)
        
        self.parent = f"projects/{project_id}/locations/{location}"
    
    def create_processor(self, display_name, processor_type='CUSTOM_EXTRACTION_PROCESSOR'):
        """Create custom processor."""
        
        processor = documentai_beta.Processor(
            display_name=display_name,
            type_=processor_type
        )
        
        request = documentai_beta.CreateProcessorRequest(
            parent=self.parent,
            processor=processor
        )
        
        response = self.client.create_processor(request=request)
        
        print(f"Created processor: {response.name}")
        print(f"  Display Name: {response.display_name}")
        print(f"  Type: {response.type_}")
        
        return response
    
    def prepare_training_data(self, documents_with_labels):
        """Prepare training data for custom extractor."""
        
        training_data = []
        
        for doc in documents_with_labels:
            sample = {
                'document_uri': doc['gcs_uri'],
                'labels': []
            }
            
            # Add entity labels
            for entity in doc.get('entities', []):
                label = {
                    'entity_type': entity['type'],
                    'mention_text': entity['text'],
                    'page_ref': entity.get('page', 0),
                }
                
                if 'bounding_box' in entity:
                    label['bounding_box'] = entity['bounding_box']
                
                sample['labels'].append(label)
            
            training_data.append(sample)
        
        print(f"Prepared {len(training_data)} training samples")
        
        return training_data

# Example: Define custom document schema
custom_schema = {
    'entity_types': [
        {
            'name': 'customer_name',
            'type': 'text',
            'required': True,
        },
        {
            'name': 'policy_number',
            'type': 'text',
            'required': True,
        },
        {
            'name': 'effective_date',
            'type': 'date',
            'required': True,
        },
        {
            'name': 'premium_amount',
            'type': 'money',
            'required': True,
        },
    ]
}
```

---

## 4. Batch Processing

```python
from google.cloud import storage

def batch_process_documents(
    project_id,
    location,
    processor_id,
    gcs_input_uri,
    gcs_output_uri,
    mime_type='application/pdf'
):
    """Process multiple documents in batch."""
    
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    processor_name = client.processor_path(project_id, location, processor_id)
    
    # Configure input
    gcs_documents = documentai.GcsDocuments(
        documents=[
            documentai.GcsDocument(
                gcs_uri=gcs_input_uri,
                mime_type=mime_type
            )
        ]
    )
    
    input_config = documentai.BatchDocumentsInputConfig(
        gcs_documents=gcs_documents
    )
    
    # Configure output
    output_config = documentai.DocumentOutputConfig(
        gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=gcs_output_uri
        )
    )
    
    request = documentai.BatchProcessRequest(
        name=processor_name,
        input_documents=input_config,
        document_output_config=output_config,
    )
    
    print(f"Starting batch processing...")
    print(f"  Input: {gcs_input_uri}")
    print(f"  Output: {gcs_output_uri}")
    
    operation = client.batch_process_documents(request)
    
    print("Waiting for operation to complete...")
    operation.result(timeout=600)
    
    print("Batch processing complete!")
    
    # List output files
    storage_client = storage.Client()
    
    match = re.match(r'gs://([^/]+)/(.+)', gcs_output_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)
    
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    print(f"\nOutput files ({len(blobs)}):")
    for blob in blobs:
        print(f"  gs://{bucket_name}/{blob.name}")
    
    return blobs

# Example usage
import re

# batch_process_documents(
#     project_id='my-project',
#     location='us',
#     processor_id='invoice-processor-id',
#     gcs_input_uri='gs://my-bucket/invoices/*.pdf',
#     gcs_output_uri='gs://my-bucket/output/'
# )
```

---

## 5. Document Quality Assessment

```python
def assess_document_quality(processor, file_path):
    """Assess document quality for processing."""
    
    document = processor.process_document(file_path)
    
    quality_report = {
        'pages': len(document.pages),
        'has_text': bool(document.text),
        'text_length': len(document.text),
        'avg_confidence': 0,
        'low_confidence_pages': [],
        'issues': [],
    }
    
    total_confidence = 0
    confidence_count = 0
    
    for page_num, page in enumerate(document.pages, 1):
        # Check for detected text
        if not page.tokens:
            quality_report['issues'].append(f"Page {page_num}: No text detected")
        
        # Calculate average confidence
        page_confidences = []
        
        for token in page.tokens:
            if hasattr(token.layout, 'confidence'):
                page_confidences.append(token.layout.confidence)
                total_confidence += token.layout.confidence
                confidence_count += 1
        
        if page_confidences:
            avg_page_conf = sum(page_confidences) / len(page_confidences)
            
            if avg_page_conf < 0.7:
                quality_report['low_confidence_pages'].append({
                    'page': page_num,
                    'confidence': avg_page_conf
                })
    
    if confidence_count > 0:
        quality_report['avg_confidence'] = total_confidence / confidence_count
    
    # Generate report
    print("\n=== Document Quality Report ===\n")
    print(f"Pages: {quality_report['pages']}")
    print(f"Text detected: {quality_report['has_text']}")
    print(f"Text length: {quality_report['text_length']} characters")
    print(f"Average confidence: {quality_report['avg_confidence']:.2%}")
    
    if quality_report['low_confidence_pages']:
        print(f"\nLow confidence pages ({len(quality_report['low_confidence_pages'])}):")
        for page_info in quality_report['low_confidence_pages']:
            print(f"  Page {page_info['page']}: {page_info['confidence']:.2%}")
    
    if quality_report['issues']:
        print(f"\nIssues ({len(quality_report['issues'])}):")
        for issue in quality_report['issues']:
            print(f"  - {issue}")
    
    # Recommendations
    print("\nRecommendations:")
    
    if quality_report['avg_confidence'] < 0.7:
        print("  - Document quality is low. Consider:")
        print("    * Rescanning at higher resolution")
        print("    * Improving image quality")
        print("    * Using original digital document if available")
    elif quality_report['avg_confidence'] < 0.85:
        print("  - Document quality is moderate. Consider:")
        print("    * Reviewing extracted data carefully")
        print("    * Manual verification of key fields")
    else:
        print("  ✓ Document quality is good")
    
    return quality_report
```

---

## 6. Document Classification

```python
def classify_document(processor, file_path):
    """Classify document type."""
    
    document = processor.process_document(file_path)
    
    # Extract document type from entities
    doc_type = None
    doc_type_confidence = 0
    
    for entity in document.entities:
        if entity.type_ == 'document_type':
            doc_type = entity.mention_text
            doc_type_confidence = entity.confidence
            break
    
    # Heuristic classification if not found
    if not doc_type:
        text_lower = document.text.lower()
        
        if 'invoice' in text_lower and 'total' in text_lower:
            doc_type = 'invoice'
            doc_type_confidence = 0.8
        elif 'receipt' in text_lower or 'paid' in text_lower:
            doc_type = 'receipt'
            doc_type_confidence = 0.7
        elif 'contract' in text_lower or 'agreement' in text_lower:
            doc_type = 'contract'
            doc_type_confidence = 0.75
        else:
            doc_type = 'unknown'
            doc_type_confidence = 0.5
    
    print(f"Document Type: {doc_type} ({doc_type_confidence:.2%})")
    
    return doc_type, doc_type_confidence

def split_and_classify_documents(processor, multi_doc_file):
    """Split multi-page document and classify each section."""
    
    document = processor.process_document(multi_doc_file)
    
    # Detect document boundaries
    doc_boundaries = detect_document_boundaries(document)
    
    print(f"Detected {len(doc_boundaries)} documents in file\n")
    
    classified_docs = []
    
    for i, (start_page, end_page) in enumerate(doc_boundaries, 1):
        # Extract section
        section_text = extract_page_range(document, start_page, end_page)
        
        # Classify
        doc_type, confidence = classify_section(section_text)
        
        classified_docs.append({
            'section': i,
            'pages': f"{start_page}-{end_page}",
            'type': doc_type,
            'confidence': confidence,
        })
        
        print(f"Document {i}: Pages {start_page}-{end_page}")
        print(f"  Type: {doc_type} ({confidence:.2%})\n")
    
    return classified_docs

def detect_document_boundaries(document):
    """Detect boundaries between multiple documents."""
    
    # Simple heuristic: large gaps or specific keywords
    boundaries = [(1, 1)]
    
    for i, page in enumerate(document.pages[1:], 2):
        # Check for document start indicators
        page_text = get_page_text(document, page)
        
        if any(keyword in page_text.lower() for keyword in
               ['invoice', 'contract', 'agreement', 'receipt']):
            # Start of new document
            if i - boundaries[-1][1] > 1:
                boundaries[-1] = (boundaries[-1][0], i - 1)
                boundaries.append((i, i))
            else:
                boundaries[-1] = (boundaries[-1][0], i)
        else:
            boundaries[-1] = (boundaries[-1][0], i)
    
    return boundaries

def get_page_text(document, page):
    """Extract text from specific page."""
    # Simplified implementation
    return document.text
```

---

## 7. Error Handling and Validation

```python
class DocumentProcessor:
    """Production-ready document processor."""
    
    def __init__(self, processor):
        self.processor = processor
        self.validation_errors = []
    
    def process_with_validation(self, file_path, required_fields=None):
        """Process document with validation."""
        
        if required_fields is None:
            required_fields = []
        
        try:
            document = self.processor.process_document(file_path)
        except Exception as e:
            print(f"✗ Processing error: {e}")
            return None
        
        # Extract data
        extracted_data = self.extract_entities(document)
        
        # Validate
        is_valid = self.validate_extraction(extracted_data, required_fields)
        
        if is_valid:
            print("✓ Document processed and validated successfully")
        else:
            print("⚠️  Document processed with validation errors:")
            for error in self.validation_errors:
                print(f"  - {error}")
        
        return {
            'data': extracted_data,
            'valid': is_valid,
            'errors': self.validation_errors,
        }
    
    def extract_entities(self, document):
        """Extract entities from document."""
        
        entities = {}
        
        for entity in document.entities:
            entities[entity.type_] = {
                'value': entity.mention_text,
                'confidence': entity.confidence,
            }
        
        return entities
    
    def validate_extraction(self, data, required_fields):
        """Validate extracted data."""
        
        self.validation_errors = []
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                self.validation_errors.append(f"Missing required field: {field}")
            elif data[field]['confidence'] < 0.7:
                self.validation_errors.append(
                    f"Low confidence for {field}: {data[field]['confidence']:.2%}"
                )
        
        # Validate data types
        if 'invoice_date' in data:
            if not self.validate_date(data['invoice_date']['value']):
                self.validation_errors.append("Invalid invoice date format")
        
        if 'total_amount' in data:
            if not self.validate_amount(data['total_amount']['value']):
                self.validation_errors.append("Invalid total amount format")
        
        return len(self.validation_errors) == 0
    
    def validate_date(self, date_str):
        """Validate date format."""
        from datetime import datetime
        
        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
        
        for fmt in date_formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        
        return False
    
    def validate_amount(self, amount_str):
        """Validate amount format."""
        import re
        
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[^\d.,]', '', amount_str)
        
        try:
            float(cleaned.replace(',', ''))
            return True
        except ValueError:
            return False
```

---

## 8. Quick Reference Checklist

### Setup
- [ ] Enable Document AI API
- [ ] Choose appropriate processor type
- [ ] Configure authentication
- [ ] Set up Cloud Storage for batch processing
- [ ] Test with sample documents

### Processing
- [ ] Assess document quality first
- [ ] Use appropriate processor for document type
- [ ] Handle low-confidence extractions
- [ ] Validate extracted data
- [ ] Implement error handling

### Production
- [ ] Use batch processing for large volumes
- [ ] Monitor processing accuracy
- [ ] Set up validation rules
- [ ] Handle multi-page documents
- [ ] Implement retry logic

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
