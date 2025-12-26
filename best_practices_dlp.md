# Data Loss Prevention (DLP) Best Practices

*Last Updated: December 25, 2025*

## Overview

Cloud Data Loss Prevention (DLP) is a fully managed service designed to discover, classify, and protect sensitive data. It provides powerful content inspection, de-identification, and risk analysis capabilities for structured and unstructured data across Google Cloud and beyond.

---

## 1. InfoType Detection

### Built-in InfoTypes

**Best Practices:**
- Use built-in InfoTypes for common sensitive data
- Test detection accuracy
- Configure likelihood thresholds
- Understand regional variations

```python
from google.cloud import dlp_v2

def inspect_content_basic(project_id, content_string):
    """Inspect content for sensitive data using built-in InfoTypes."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Configure detection
    inspect_config = {
        "info_types": [
            {"name": "EMAIL_ADDRESS"},
            {"name": "PHONE_NUMBER"},
            {"name": "CREDIT_CARD_NUMBER"},
            {"name": "US_SOCIAL_SECURITY_NUMBER"},
            {"name": "PERSON_NAME"},
            {"name": "DATE_OF_BIRTH"},
            {"name": "STREET_ADDRESS"},
            {"name": "PASSPORT"},
            {"name": "IBAN_CODE"},
            {"name": "IP_ADDRESS"},
            {"name": "MAC_ADDRESS"},
            {"name": "MEDICAL_RECORD_NUMBER"},
            {"name": "DRIVER_LICENSE_NUMBER"},
        ],
        "min_likelihood": dlp_v2.Likelihood.LIKELY,
        "include_quote": True,
        "limits": {
            "max_findings_per_item": 0,  # No limit
            "max_findings_per_request": 0,
        }
    }
    
    item = {"value": content_string}
    
    response = client.inspect_content(
        request={
            "parent": parent,
            "inspect_config": inspect_config,
            "item": item,
        }
    )
    
    # Process findings
    if response.result.findings:
        print(f"Found {len(response.result.findings)} sensitive data instances:")
        for finding in response.result.findings:
            print(f"\n  InfoType: {finding.info_type.name}")
            print(f"  Likelihood: {finding.likelihood.name}")
            print(f"  Quote: {finding.quote}")
            print(f"  Location: {finding.location.byte_range.start}-{finding.location.byte_range.end}")
    else:
        print("No sensitive data found")
    
    return response

# Example usage
content = """
Customer Record:
Name: John Doe
Email: john.doe@example.com
Phone: (555) 123-4567
SSN: 123-45-6789
Credit Card: 4532-1234-5678-9010
Address: 123 Main St, Anytown, CA 90210
DOB: 01/15/1980
"""

inspect_content_basic("my-project", content)
```

### Custom InfoTypes

**Create Domain-Specific InfoTypes:**
```python
def create_custom_infotype_inspect(project_id, content_string):
    """Inspect content with custom InfoTypes."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Custom InfoType: Employee ID
    custom_info_types = [
        {
            "info_type": {"name": "EMPLOYEE_ID"},
            "regex": {"pattern": r"EMP-\d{6}"},
            "likelihood": dlp_v2.Likelihood.LIKELY,
        },
        {
            "info_type": {"name": "CUSTOMER_ID"},
            "regex": {"pattern": r"CUST-[A-Z]{3}-\d{8}"},
            "likelihood": dlp_v2.Likelihood.LIKELY,
        },
        {
            "info_type": {"name": "ORDER_NUMBER"},
            "regex": {"pattern": r"ORD-\d{10}"},
            "likelihood": dlp_v2.Likelihood.LIKELY,
        },
        {
            "info_type": {"name": "INTERNAL_PROJECT_CODE"},
            "dictionary": {
                "word_list": {
                    "words": [
                        "PROJECT_ALPHA",
                        "PROJECT_BETA",
                        "PROJECT_GAMMA",
                        "CONFIDENTIAL_INITIATIVE",
                    ]
                }
            },
            "likelihood": dlp_v2.Likelihood.VERY_LIKELY,
        },
    ]
    
    inspect_config = {
        "custom_info_types": custom_info_types,
        "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
        "include_quote": True,
    }
    
    item = {"value": content_string}
    
    response = client.inspect_content(
        request={
            "parent": parent,
            "inspect_config": inspect_config,
            "item": item,
        }
    )
    
    if response.result.findings:
        print(f"Found {len(response.result.findings)} custom InfoType matches:")
        for finding in response.result.findings:
            print(f"  {finding.info_type.name}: {finding.quote}")
    
    return response

# Example
custom_content = """
Employee Report:
Employee ID: EMP-123456
Customer ID: CUST-ABC-12345678
Order Number: ORD-9876543210
Project: PROJECT_ALPHA
"""

create_custom_infotype_inspect("my-project", custom_content)
```

### Stored InfoTypes

**Create Reusable InfoTypes:**
```python
def create_stored_infotype(project_id):
    """Create a stored InfoType for large dictionaries."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Large dictionary stored in BigQuery
    stored_info_type_config = {
        "display_name": "VIP Customer List",
        "description": "List of VIP customer identifiers",
        "large_custom_dictionary": {
            "big_query_field": {
                "table": {
                    "project_id": project_id,
                    "dataset_id": "customer_data",
                    "table_id": "vip_customers",
                },
                "field": {
                    "name": "customer_id"
                },
            }
        },
    }
    
    response = client.create_stored_info_type(
        request={
            "parent": parent,
            "config": stored_info_type_config,
            "stored_info_type_id": "vip-customer-list",
        }
    )
    
    print(f"Created stored InfoType: {response.name}")
    return response

def use_stored_infotype(project_id, content_string, stored_info_type_name):
    """Use stored InfoType in inspection."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    inspect_config = {
        "custom_info_types": [
            {
                "info_type": {"name": "VIP_CUSTOMER"},
                "stored_type": {"name": stored_info_type_name},
            }
        ],
        "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
    }
    
    item = {"value": content_string}
    
    response = client.inspect_content(
        request={
            "parent": parent,
            "inspect_config": inspect_config,
            "item": item,
        }
    )
    
    return response
```

---

## 2. Inspection Templates

### Create Reusable Templates

**Best Practices:**
- Create templates for common inspection scenarios
- Version control template changes
- Document template purposes
- Share templates across organization

```python
def create_inspection_template(project_id):
    """Create inspection template for PII detection."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    inspect_config = {
        "info_types": [
            {"name": "EMAIL_ADDRESS"},
            {"name": "PHONE_NUMBER"},
            {"name": "PERSON_NAME"},
            {"name": "CREDIT_CARD_NUMBER"},
            {"name": "US_SOCIAL_SECURITY_NUMBER"},
            {"name": "DATE_OF_BIRTH"},
            {"name": "STREET_ADDRESS"},
        ],
        "min_likelihood": dlp_v2.Likelihood.LIKELY,
        "limits": {
            "max_findings_per_item": 100,
            "max_findings_per_request": 1000,
        },
        "include_quote": True,
        "rule_set": [
            {
                "info_types": [{"name": "EMAIL_ADDRESS"}],
                "rules": [
                    {
                        "exclusion_rule": {
                            "regex": {"pattern": r".*@example\.com$"},
                            "matching_type": dlp_v2.MatchingType.MATCHING_TYPE_FULL_MATCH,
                        }
                    }
                ],
            }
        ],
    }
    
    inspect_template = {
        "display_name": "PII Detection Template",
        "description": "Standard template for detecting personally identifiable information",
        "inspect_config": inspect_config,
    }
    
    response = client.create_inspect_template(
        request={
            "parent": parent,
            "inspect_template": inspect_template,
            "template_id": "pii-detection-template",
        }
    )
    
    print(f"Created inspection template: {response.name}")
    return response

def use_inspection_template(project_id, template_name, content_string):
    """Use inspection template to scan content."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    item = {"value": content_string}
    
    response = client.inspect_content(
        request={
            "parent": parent,
            "inspect_template_name": template_name,
            "item": item,
        }
    )
    
    return response
```

### Healthcare-Specific Template

**HIPAA Compliance:**
```python
def create_healthcare_template(project_id):
    """Create inspection template for healthcare data (HIPAA)."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    inspect_config = {
        "info_types": [
            # Patient identifiers
            {"name": "PERSON_NAME"},
            {"name": "DATE_OF_BIRTH"},
            {"name": "PHONE_NUMBER"},
            {"name": "EMAIL_ADDRESS"},
            {"name": "STREET_ADDRESS"},
            {"name": "US_SOCIAL_SECURITY_NUMBER"},
            {"name": "PASSPORT"},
            {"name": "DRIVER_LICENSE_NUMBER"},
            
            # Medical identifiers
            {"name": "MEDICAL_RECORD_NUMBER"},
            {"name": "US_HEALTHCARE_NPI"},
            {"name": "FDA_CODE"},
            {"name": "ICD9_CODE"},
            {"name": "ICD10_CODE"},
            
            # Financial
            {"name": "CREDIT_CARD_NUMBER"},
            {"name": "IBAN_CODE"},
        ],
        "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
        "include_quote": False,  # Don't quote PHI in logs
    }
    
    inspect_template = {
        "display_name": "HIPAA PHI Detection",
        "description": "Template for detecting Protected Health Information",
        "inspect_config": inspect_config,
    }
    
    response = client.create_inspect_template(
        request={
            "parent": parent,
            "inspect_template": inspect_template,
            "template_id": "hipaa-phi-detection",
        }
    )
    
    return response
```

---

## 3. De-identification Techniques

### Masking

**Best Practices:**
- Preserve data format
- Use consistent masking characters
- Consider partial masking for usability
- Test de-identified data utility

```python
def deidentify_with_mask(project_id, content_string, info_types):
    """De-identify content using masking."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Configure masking
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                {
                    "info_types": info_types,
                    "primitive_transformation": {
                        "character_mask_config": {
                            "masking_character": "*",
                            "number_to_mask": 0,  # Mask all characters
                        }
                    },
                }
            ]
        }
    }
    
    inspect_config = {
        "info_types": info_types,
    }
    
    item = {"value": content_string}
    
    response = client.deidentify_content(
        request={
            "parent": parent,
            "deidentify_config": deidentify_config,
            "inspect_config": inspect_config,
            "item": item,
        }
    )
    
    print(f"Original: {content_string}")
    print(f"De-identified: {response.item.value}")
    
    return response

# Example: Mask email and phone
deidentify_with_mask(
    project_id="my-project",
    content_string="Contact: john.doe@example.com, (555) 123-4567",
    info_types=[{"name": "EMAIL_ADDRESS"}, {"name": "PHONE_NUMBER"}]
)
```

### Partial Masking

**Preserve Usability:**
```python
def deidentify_with_partial_mask(project_id, content_string):
    """Partially mask sensitive data to preserve usability."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Partial mask: show last 4 digits of credit card
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                {
                    "info_types": [{"name": "CREDIT_CARD_NUMBER"}],
                    "primitive_transformation": {
                        "character_mask_config": {
                            "masking_character": "*",
                            "number_to_mask": 12,  # Mask first 12 digits
                            "reverse_order": False,  # Start from beginning
                        }
                    },
                },
                {
                    "info_types": [{"name": "EMAIL_ADDRESS"}],
                    "primitive_transformation": {
                        "character_mask_config": {
                            "masking_character": "*",
                            "number_to_mask": 5,  # Mask 5 characters
                            "characters_to_ignore": [
                                {"characters_to_skip": "@."}
                            ],
                        }
                    },
                },
            ]
        }
    }
    
    inspect_config = {
        "info_types": [
            {"name": "CREDIT_CARD_NUMBER"},
            {"name": "EMAIL_ADDRESS"},
        ],
    }
    
    item = {"value": content_string}
    
    response = client.deidentify_content(
        request={
            "parent": parent,
            "deidentify_config": deidentify_config,
            "inspect_config": inspect_config,
            "item": item,
        }
    )
    
    return response
```

### Replacement

**Replace with Surrogate Values:**
```python
def deidentify_with_replacement(project_id, content_string):
    """Replace sensitive data with placeholder text."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                {
                    "info_types": [{"name": "EMAIL_ADDRESS"}],
                    "primitive_transformation": {
                        "replace_config": {
                            "new_value": {"string_value": "[EMAIL_REDACTED]"}
                        }
                    },
                },
                {
                    "info_types": [{"name": "PHONE_NUMBER"}],
                    "primitive_transformation": {
                        "replace_config": {
                            "new_value": {"string_value": "[PHONE_REDACTED]"}
                        }
                    },
                },
                {
                    "info_types": [{"name": "PERSON_NAME"}],
                    "primitive_transformation": {
                        "replace_config": {
                            "new_value": {"string_value": "[NAME_REDACTED]"}
                        }
                    },
                },
            ]
        }
    }
    
    inspect_config = {
        "info_types": [
            {"name": "EMAIL_ADDRESS"},
            {"name": "PHONE_NUMBER"},
            {"name": "PERSON_NAME"},
        ],
    }
    
    item = {"value": content_string}
    
    response = client.deidentify_content(
        request={
            "parent": parent,
            "deidentify_config": deidentify_config,
            "inspect_config": inspect_config,
            "item": item,
        }
    )
    
    return response
```

### Tokenization with Crypto-Based Hash

**Consistent Pseudonymization:**
```python
def deidentify_with_crypto_hash(project_id, content_string, crypto_key_name):
    """De-identify using crypto-based tokenization."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Crypto-based tokenization
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                {
                    "info_types": [
                        {"name": "EMAIL_ADDRESS"},
                        {"name": "PHONE_NUMBER"},
                    ],
                    "primitive_transformation": {
                        "crypto_hash_config": {
                            "crypto_key": {
                                "kms_wrapped": {
                                    "wrapped_key": crypto_key_name,
                                    "crypto_key_name": f"projects/{project_id}/locations/global/keyRings/dlp-keyring/cryptoKeys/dlp-key",
                                }
                            }
                        }
                    },
                }
            ]
        }
    }
    
    inspect_config = {
        "info_types": [
            {"name": "EMAIL_ADDRESS"},
            {"name": "PHONE_NUMBER"},
        ],
    }
    
    item = {"value": content_string}
    
    response = client.deidentify_content(
        request={
            "parent": parent,
            "deidentify_config": deidentify_config,
            "inspect_config": inspect_config,
            "item": item,
        }
    )
    
    return response
```

### Format-Preserving Encryption (FPE)

**Maintain Data Format:**
```python
def deidentify_with_fpe(project_id, content_string, wrapped_key, alphabet):
    """De-identify using Format-Preserving Encryption."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # FPE preserves format (e.g., SSN remains XXX-XX-XXXX format)
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                {
                    "info_types": [
                        {"name": "US_SOCIAL_SECURITY_NUMBER"},
                        {"name": "CREDIT_CARD_NUMBER"},
                    ],
                    "primitive_transformation": {
                        "crypto_replace_ffx_fpe_config": {
                            "crypto_key": {
                                "kms_wrapped": {
                                    "wrapped_key": wrapped_key,
                                    "crypto_key_name": f"projects/{project_id}/locations/global/keyRings/dlp-keyring/cryptoKeys/dlp-key",
                                }
                            },
                            "alphabet": alphabet,  # "NUMERIC" for SSN/credit card
                            "surrogate_info_type": {"name": "SSN_TOKEN"},
                        }
                    },
                }
            ]
        }
    }
    
    inspect_config = {
        "info_types": [
            {"name": "US_SOCIAL_SECURITY_NUMBER"},
            {"name": "CREDIT_CARD_NUMBER"},
        ],
    }
    
    item = {"value": content_string}
    
    response = client.deidentify_content(
        request={
            "parent": parent,
            "deidentify_config": deidentify_config,
            "inspect_config": inspect_config,
            "item": item,
        }
    )
    
    return response

# Example: FPE for SSN (preserves XXX-XX-XXXX format)
# Original: 123-45-6789
# De-identified: 987-65-4321 (different number, same format)
```

### Date Shifting

**Preserve Temporal Relationships:**
```python
def deidentify_with_date_shift(project_id, content_string, date_fields):
    """Shift dates while preserving intervals."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    deidentify_config = {
        "record_transformations": {
            "field_transformations": [
                {
                    "fields": date_fields,
                    "primitive_transformation": {
                        "date_shift_config": {
                            "upper_bound_days": 30,
                            "lower_bound_days": -30,
                            "context": {"name": "patient_id"},  # Same shift for same patient
                        }
                    },
                }
            ]
        }
    }
    
    # For structured data (CSV, JSON, etc.)
    item = {"table": content_string}  # Use table format
    
    response = client.deidentify_content(
        request={
            "parent": parent,
            "deidentify_config": deidentify_config,
            "item": item,
        }
    )
    
    return response
```

---

## 4. De-identification Templates

### Create De-identification Templates

**Best Practices:**
- Create templates for different use cases
- Document transformation logic
- Test templates thoroughly
- Version control changes

```python
def create_deidentify_template(project_id):
    """Create comprehensive de-identification template."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                # Mask email addresses
                {
                    "info_types": [{"name": "EMAIL_ADDRESS"}],
                    "primitive_transformation": {
                        "character_mask_config": {
                            "masking_character": "*",
                            "number_to_mask": 5,
                            "characters_to_ignore": [
                                {"characters_to_skip": "@."}
                            ],
                        }
                    },
                },
                # Replace names
                {
                    "info_types": [{"name": "PERSON_NAME"}],
                    "primitive_transformation": {
                        "replace_config": {
                            "new_value": {"string_value": "[NAME_REDACTED]"}
                        }
                    },
                },
                # Partial mask credit cards (show last 4)
                {
                    "info_types": [{"name": "CREDIT_CARD_NUMBER"}],
                    "primitive_transformation": {
                        "character_mask_config": {
                            "masking_character": "*",
                            "number_to_mask": 12,
                        }
                    },
                },
                # Redact SSN completely
                {
                    "info_types": [{"name": "US_SOCIAL_SECURITY_NUMBER"}],
                    "primitive_transformation": {
                        "replace_config": {
                            "new_value": {"string_value": "[SSN_REDACTED]"}
                        }
                    },
                },
            ]
        }
    }
    
    deidentify_template = {
        "display_name": "Standard PII De-identification",
        "description": "Standard template for de-identifying PII in customer data",
        "deidentify_config": deidentify_config,
    }
    
    response = client.create_deidentify_template(
        request={
            "parent": parent,
            "deidentify_template": deidentify_template,
            "template_id": "standard-pii-deidentify",
        }
    )
    
    print(f"Created de-identification template: {response.name}")
    return response
```

---

## 5. Scanning BigQuery Tables

### Inspect BigQuery Tables

**Best Practices:**
- Schedule regular scans
- Use sampling for large tables
- Set appropriate limits
- Monitor scan costs

```python
def inspect_bigquery_table(
    project_id,
    dataset_id,
    table_id,
    template_name=None,
    max_findings=1000
):
    """Inspect BigQuery table for sensitive data."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # BigQuery table reference
    big_query_options = {
        "table_reference": {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "table_id": table_id,
        },
        "rows_limit": 10000,  # Sample size
        "sample_method": dlp_v2.BigQueryOptions.SampleMethod.RANDOM_START,
        "identifying_fields": [{"name": "user_id"}],  # Context field
    }
    
    storage_config = {
        "big_query_options": big_query_options,
    }
    
    # Use template or define inspect config
    if template_name:
        inspect_job_config = {
            "inspect_template_name": template_name,
            "storage_config": storage_config,
        }
    else:
        inspect_config = {
            "info_types": [
                {"name": "EMAIL_ADDRESS"},
                {"name": "PHONE_NUMBER"},
                {"name": "CREDIT_CARD_NUMBER"},
                {"name": "US_SOCIAL_SECURITY_NUMBER"},
            ],
            "min_likelihood": dlp_v2.Likelihood.LIKELY,
            "limits": {
                "max_findings_per_request": max_findings,
            },
            "include_quote": False,  # Don't store sensitive quotes
        }
        
        inspect_job_config = {
            "inspect_config": inspect_config,
            "storage_config": storage_config,
        }
    
    # Configure actions (save to BigQuery)
    actions = [
        {
            "save_findings": {
                "output_config": {
                    "table": {
                        "project_id": project_id,
                        "dataset_id": "dlp_findings",
                        "table_id": f"findings_{table_id}",
                    }
                }
            }
        }
    ]
    
    inspect_job_config["actions"] = actions
    
    # Create DLP job
    response = client.create_dlp_job(
        request={
            "parent": parent,
            "inspect_job": inspect_job_config,
        }
    )
    
    print(f"Inspection job started: {response.name}")
    return response

def check_dlp_job(project_id, job_name):
    """Check status of DLP job."""
    
    client = dlp_v2.DlpServiceClient()
    
    job = client.get_dlp_job(request={"name": job_name})
    
    print(f"Job state: {job.state.name}")
    
    if job.state == dlp_v2.DlpJob.JobState.DONE:
        print(f"Job completed: {job.inspect_details.result.info_type_stats}")
        
        for stat in job.inspect_details.result.info_type_stats:
            print(f"  {stat.info_type.name}: {stat.count} findings")
    
    return job
```

### De-identify BigQuery Tables

**Create De-identified Copy:**
```python
def deidentify_bigquery_table(
    project_id,
    source_dataset,
    source_table,
    dest_dataset,
    dest_table,
    deidentify_template_name
):
    """De-identify BigQuery table and save to new table."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Source table
    big_query_source = {
        "table_reference": {
            "project_id": project_id,
            "dataset_id": source_dataset,
            "table_id": source_table,
        }
    }
    
    # Destination table
    big_query_dest = {
        "project_id": project_id,
        "dataset_id": dest_dataset,
        "table_id": dest_table,
    }
    
    # Transformation config
    transformation_details_storage_config = {
        "table": big_query_dest
    }
    
    # Create transformation job
    risk_job = {
        "privacy_metric": {
            "categorical_stats_config": {
                "field": {"name": "user_id"}
            }
        },
        "source_table": big_query_source,
        "actions": [
            {
                "save_findings": {
                    "output_config": {
                        "table": transformation_details_storage_config
                    }
                }
            }
        ],
    }
    
    # Use de-identification template
    job_config = {
        "risk_job": risk_job,
    }
    
    response = client.create_dlp_job(
        request={
            "parent": parent,
            "risk_job": risk_job,
        }
    )
    
    print(f"De-identification job started: {response.name}")
    return response
```

---

## 6. Scanning Cloud Storage

### Inspect Cloud Storage Bucket

**Best Practices:**
- Use file type filters
- Limit file sizes
- Set sampling rates
- Handle large files appropriately

```python
def inspect_gcs_files(
    project_id,
    bucket_name,
    file_patterns,
    template_name=None
):
    """Inspect files in Cloud Storage bucket."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Cloud Storage configuration
    cloud_storage_options = {
        "file_set": {
            "url": f"gs://{bucket_name}/*"
        },
        "bytes_limit_per_file": 1073741824,  # 1 GB
        "file_types": [
            dlp_v2.FileType.TEXT_FILE,
            dlp_v2.FileType.CSV,
            dlp_v2.FileType.TSV,
        ],
        "files_limit_percent": 90,  # Sample 90% of files
        "sample_method": dlp_v2.CloudStorageOptions.SampleMethod.RANDOM_START,
    }
    
    storage_config = {
        "cloud_storage_options": cloud_storage_options,
    }
    
    # Inspection config
    if template_name:
        inspect_job_config = {
            "inspect_template_name": template_name,
            "storage_config": storage_config,
        }
    else:
        inspect_config = {
            "info_types": [
                {"name": "EMAIL_ADDRESS"},
                {"name": "PHONE_NUMBER"},
                {"name": "CREDIT_CARD_NUMBER"},
                {"name": "PERSON_NAME"},
            ],
            "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
            "limits": {
                "max_findings_per_request": 10000,
            },
        }
        
        inspect_job_config = {
            "inspect_config": inspect_config,
            "storage_config": storage_config,
        }
    
    # Save findings to BigQuery
    actions = [
        {
            "save_findings": {
                "output_config": {
                    "table": {
                        "project_id": project_id,
                        "dataset_id": "dlp_findings",
                        "table_id": f"gcs_findings_{bucket_name}",
                    }
                }
            }
        },
        {
            "pub_sub": {
                "topic": f"projects/{project_id}/topics/dlp-findings"
            }
        }
    ]
    
    inspect_job_config["actions"] = actions
    
    response = client.create_dlp_job(
        request={
            "parent": parent,
            "inspect_job": inspect_job_config,
        }
    )
    
    print(f"GCS inspection job started: {response.name}")
    return response
```

### De-identify Cloud Storage Files

**Batch Processing:**
```bash
# Use DLP API to de-identify files in Cloud Storage
gcloud dlp jobs create deidentify \
    --source-path="gs://my-bucket/sensitive-data/*" \
    --output-path="gs://my-bucket/deidentified-data/" \
    --template="projects/my-project/locations/global/deidentifyTemplates/standard-pii-deidentify" \
    --project=my-project
```

---

## 7. Risk Analysis

### K-Anonymity

**Measure Re-identification Risk:**
```python
def calculate_k_anonymity(project_id, dataset_id, table_id, quasi_ids):
    """Calculate k-anonymity for a table."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Table reference
    source_table = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "table_id": table_id,
    }
    
    # Quasi-identifier fields
    quasi_id_fields = [{"field": {"name": qi}} for qi in quasi_ids]
    
    # K-anonymity config
    privacy_metric = {
        "k_anonymity_config": {
            "quasi_ids": quasi_id_fields,
        }
    }
    
    # Risk analysis job
    risk_job = {
        "privacy_metric": privacy_metric,
        "source_table": source_table,
        "actions": [
            {
                "save_findings": {
                    "output_config": {
                        "table": {
                            "project_id": project_id,
                            "dataset_id": "dlp_risk_analysis",
                            "table_id": f"k_anonymity_{table_id}",
                        }
                    }
                }
            }
        ],
    }
    
    response = client.create_dlp_job(
        request={
            "parent": parent,
            "risk_job": risk_job,
        }
    )
    
    print(f"K-anonymity job started: {response.name}")
    return response

# Example: Analyze customer table
calculate_k_anonymity(
    project_id="my-project",
    dataset_id="customers",
    table_id="profiles",
    quasi_ids=["age", "gender", "zip_code", "country"]
)
```

### L-Diversity

**Measure Sensitive Attribute Diversity:**
```python
def calculate_l_diversity(project_id, dataset_id, table_id, quasi_ids, sensitive_attribute):
    """Calculate l-diversity for a table."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    source_table = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "table_id": table_id,
    }
    
    quasi_id_fields = [{"field": {"name": qi}} for qi in quasi_ids]
    
    privacy_metric = {
        "l_diversity_config": {
            "quasi_ids": quasi_id_fields,
            "sensitive_attribute": {"name": sensitive_attribute},
        }
    }
    
    risk_job = {
        "privacy_metric": privacy_metric,
        "source_table": source_table,
        "actions": [
            {
                "save_findings": {
                    "output_config": {
                        "table": {
                            "project_id": project_id,
                            "dataset_id": "dlp_risk_analysis",
                            "table_id": f"l_diversity_{table_id}",
                        }
                    }
                }
            }
        ],
    }
    
    response = client.create_dlp_job(
        request={
            "parent": parent,
            "risk_job": risk_job,
        }
    )
    
    return response

# Example: Check diversity of medical conditions
calculate_l_diversity(
    project_id="my-project",
    dataset_id="healthcare",
    table_id="patient_records",
    quasi_ids=["age", "zip_code", "gender"],
    sensitive_attribute="medical_condition"
)
```

---

## 8. Monitoring and Compliance

### Monitor DLP Jobs

**Best Practices:**
- Track job success rates
- Monitor finding trends
- Alert on policy violations
- Review costs regularly

```python
def list_dlp_jobs(project_id, job_type=None, days=7):
    """List recent DLP jobs."""
    
    from google.cloud import dlp_v2
    from datetime import datetime, timedelta
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Filter by time
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    filter_str = f'end_time >= "{start_time.isoformat()}"'
    
    if job_type:
        filter_str += f' AND type="{job_type}"'
    
    request = {
        "parent": parent,
        "filter": filter_str,
        "page_size": 100,
    }
    
    jobs = client.list_dlp_jobs(request=request)
    
    print(f"DLP Jobs (last {days} days):")
    for job in jobs:
        print(f"\n  Job: {job.name}")
        print(f"  State: {job.state.name}")
        print(f"  Type: {job.type_.name}")
        print(f"  Start: {job.start_time}")
        print(f"  End: {job.end_time}")
        
        if job.inspect_details:
            print(f"  Findings: {job.inspect_details.result.info_type_stats}")
    
    return jobs
```

### Create Compliance Reports

**Generate Findings Summary:**
```sql
-- Query DLP findings in BigQuery
SELECT
    info_type.name AS sensitive_data_type,
    EXTRACT(DATE FROM finding_timestamp) AS date,
    COUNT(*) AS finding_count,
    SUM(CAST(JSON_EXTRACT_SCALAR(likelihood, '$.enum_value') = 'VERY_LIKELY' AS INT64)) AS high_confidence_findings,
    APPROX_TOP_COUNT(location.content_locations[OFFSET(0)].record_location.table_location.row_index, 10) AS top_affected_rows
FROM
    `my-project.dlp_findings.findings_*`
GROUP BY
    info_type.name,
    date
ORDER BY
    date DESC,
    finding_count DESC;

-- Trend analysis
SELECT
    info_type.name,
    EXTRACT(WEEK FROM finding_timestamp) AS week,
    COUNT(*) AS weekly_findings,
    LAG(COUNT(*)) OVER (PARTITION BY info_type.name ORDER BY EXTRACT(WEEK FROM finding_timestamp)) AS previous_week,
    COUNT(*) - LAG(COUNT(*)) OVER (PARTITION BY info_type.name ORDER BY EXTRACT(WEEK FROM finding_timestamp)) AS change
FROM
    `my-project.dlp_findings.findings_*`
GROUP BY
    info_type.name,
    week
ORDER BY
    week DESC,
    weekly_findings DESC;
```

---

## 9. Cost Optimization

### Best Practices

**Reduce DLP Costs:**
- Use sampling for large datasets
- Apply filters to reduce scan scope
- Use stored InfoTypes for large dictionaries
- Schedule jobs during off-peak hours
- Cache inspection results

```python
def cost_efficient_inspection(project_id, dataset_id, table_id):
    """Inspect table with cost optimization."""
    
    client = dlp_v2.DlpServiceClient()
    parent = f"projects/{project_id}"
    
    # Optimize with sampling and limits
    big_query_options = {
        "table_reference": {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "table_id": table_id,
        },
        "rows_limit": 50000,  # Limit sample size
        "sample_method": dlp_v2.BigQueryOptions.SampleMethod.TOP,  # Faster than random
        "excluded_fields": [  # Skip non-sensitive columns
            {"name": "id"},
            {"name": "created_at"},
            {"name": "updated_at"},
        ],
    }
    
    storage_config = {
        "big_query_options": big_query_options,
    }
    
    # Use targeted InfoTypes
    inspect_config = {
        "info_types": [
            {"name": "EMAIL_ADDRESS"},
            {"name": "PHONE_NUMBER"},
        ],
        "min_likelihood": dlp_v2.Likelihood.LIKELY,  # Higher threshold
        "limits": {
            "max_findings_per_request": 1000,  # Limit findings
        },
        "exclude_info_types": False,
        "include_quote": False,  # Reduce data transfer
    }
    
    inspect_job_config = {
        "inspect_config": inspect_config,
        "storage_config": storage_config,
    }
    
    response = client.create_dlp_job(
        request={
            "parent": parent,
            "inspect_job": inspect_job_config,
        }
    )
    
    return response
```

---

## 10. Common Anti-Patterns

### ❌ Anti-Pattern 1: Over-Masking
**Problem:** Masking all data makes it unusable
**Solution:** Use partial masking or tokenization

### ❌ Anti-Pattern 2: No Testing
**Problem:** De-identified data breaks analytics
**Solution:** Test transformations thoroughly

### ❌ Anti-Pattern 3: Ignoring Performance
**Problem:** Scanning entire tables without sampling
**Solution:** Use appropriate sampling rates

### ❌ Anti-Pattern 4: Weak Key Management
**Problem:** Poor cryptographic key protection
**Solution:** Use Cloud KMS for encryption keys

### ❌ Anti-Pattern 5: No Re-identification Controls
**Problem:** Multiple de-identified datasets can be joined
**Solution:** Use consistent tokenization with proper governance

---

## 11. Quick Reference Checklist

### Setup
- [ ] Enable DLP API
- [ ] Create KMS keys for encryption
- [ ] Set up output datasets for findings
- [ ] Configure Pub/Sub notifications
- [ ] Define IAM roles and permissions

### InfoType Configuration
- [ ] Review built-in InfoTypes
- [ ] Create custom InfoTypes for domain data
- [ ] Create stored InfoTypes for large dictionaries
- [ ] Test detection accuracy
- [ ] Document InfoType usage

### Templates
- [ ] Create inspection templates
- [ ] Create de-identification templates
- [ ] Version control templates
- [ ] Test templates thoroughly
- [ ] Share templates across organization

### Scanning
- [ ] Schedule regular scans
- [ ] Use appropriate sampling
- [ ] Set finding limits
- [ ] Monitor job status
- [ ] Review findings regularly

### De-identification
- [ ] Choose appropriate techniques
- [ ] Test data utility
- [ ] Validate transformations
- [ ] Document methods
- [ ] Implement re-identification controls

### Compliance
- [ ] Define data classification policies
- [ ] Create compliance reports
- [ ] Audit DLP operations
- [ ] Monitor policy violations
- [ ] Review access logs

---

*Best Practices for Google Cloud Data Engineer Certification*
