# GCP Data Engineering Services

## Data Storage Services

### 1. **Cloud Storage**
- Object storage for unstructured data
- Different storage classes (Standard, Nearline, Coldline, Archive)
- Supports data lake architectures
- Integration with other GCP services

### 2. **Cloud SQL**
- Fully managed relational database service
- Supports MySQL, PostgreSQL, and SQL Server
- Automatic backups and replication
- High availability configurations

### 3. **Cloud Spanner**
- Globally distributed relational database
- Horizontally scalable with strong consistency
- ACID transactions at global scale
- SQL interface

### 4. **Cloud Bigtable**
- NoSQL wide-column database
- Low-latency, high-throughput workloads
- Ideal for time-series and IoT data
- HBase API compatible

### 5. **Firestore**
- NoSQL document database
- Real-time synchronization
- Mobile and web application backend
- Serverless architecture

### 6. **Cloud Memorystore**
- Fully managed in-memory data store
- Supports Redis and Memcached
- Sub-millisecond latency
- Session management and caching

## Data Processing & Analytics

### 7. **BigQuery**
- Serverless, highly scalable data warehouse
- SQL-based analytics
- Built-in machine learning (BigQuery ML)
- Real-time analytics and streaming inserts
- Federated queries across multiple sources

### 8. **Dataflow**
- Fully managed stream and batch data processing
- Based on Apache Beam
- Auto-scaling and serverless
- Unified programming model for batch and streaming

### 9. **Dataproc**
- Fully managed Apache Spark and Apache Hadoop service
- Fast cluster provisioning (90 seconds or less)
- Support for Spark, Hadoop, Hive, Pig, etc.
- Integration with other GCP services

### 10. **Cloud Data Fusion**
- Fully managed, cloud-native data integration service
- Visual interface for building ETL/ELT pipelines
- Pre-built transformations and connectors
- Based on open-source CDAP

### 11. **Dataprep by Trifacta**
- Intelligent data preparation service
- Visual interface for data cleaning and transformation
- Automatic schema detection
- Integrated with Dataflow for execution

## Data Ingestion & Integration

### 12. **Pub/Sub**
- Real-time messaging service
- Asynchronous messaging between applications
- Event-driven architectures
- Stream analytics integration

### 13. **Cloud Composer**
- Fully managed Apache Airflow service
- Workflow orchestration and scheduling
- Complex data pipeline management
- Python-based DAG definitions

### 14. **Datastream**
- Serverless change data capture (CDC) service
- Real-time data replication
- Supports Oracle, MySQL, PostgreSQL, and AlloyDB
- Low-latency data movement

### 15. **Transfer Service**
- Data transfer from online sources to Cloud Storage
- Scheduled and one-time transfers
- Support for S3, Azure Blob Storage, and HTTP/HTTPS sources

### 16. **Transfer Appliance**
- Physical device for offline data transfer
- Petabyte-scale data migration
- Secure data transfer to GCP

### 17. **BigQuery Data Transfer Service**
- Automated data movement into BigQuery
- Scheduled imports from SaaS applications
- Support for Google Ads, YouTube, Cloud Storage, etc.

## Data Governance & Catalog

### 18. **Dataplex**
- Intelligent data fabric for unified data management
- Data lake management across organizations
- Automated data discovery and classification
- Centralized security and governance

### 19. **Data Catalog**
- Fully managed metadata management service
- Automatic data discovery
- Search and tag data assets
- Integration with BigQuery, Pub/Sub, and Cloud Storage

### 20. **Data Loss Prevention (DLP)**
- Sensitive data discovery and classification
- PII detection and redaction
- De-identification of sensitive information
- Compliance support (GDPR, HIPAA, etc.)

## Machine Learning & AI Integration

### 21. **Vertex AI**
- Unified ML platform
- AutoML and custom training
- Model deployment and monitoring
- Feature Store for ML features

### 22. **BigQuery ML**
- Machine learning within BigQuery
- SQL-based model creation
- Support for various ML models
- Integration with Vertex AI

### 23. **AI Platform (Legacy)**
- Training and deployment of ML models
- Being replaced by Vertex AI
- Jupyter notebook integration

## Analytics & Visualization

### 24. **Looker**
- Enterprise business intelligence platform
- Data exploration and visualization
- Embedded analytics
- LookML modeling layer

### 25. **Looker Studio (formerly Data Studio)**
- Free data visualization and reporting tool
- Interactive dashboards
- Multiple data source connectors
- Collaboration features

## Streaming Analytics

### 26. **Dataflow (Streaming)**
- Real-time stream processing
- Windowing and late data handling
- Exactly-once processing semantics

## Data Quality & Lineage

### 27. **Dataplex Data Quality**
- Automated data quality checks
- Rule-based validation
- Data profiling
- Integration with data pipelines

### 28. **Data Lineage**
- Track data flow across systems
- Available in Data Catalog and Dataplex
- Understand data dependencies
- Impact analysis

## Additional Services

### 29. **Cloud Functions**
- Serverless compute for event-driven data processing
- Triggered by Pub/Sub, Cloud Storage events
- Lightweight data transformations

### 30. **Cloud Run**
- Containerized serverless applications
- Custom data processing logic
- Auto-scaling capabilities

### 31. **Dataform**
- SQL-based data transformation and modeling
- Version control for data pipelines
- Dependency management
- Integration with BigQuery

### 32. **Analytics Hub**
- Data exchange platform
- Secure data sharing across organizations
- BigQuery-based listings
- Controlled data access

### 33. **Cloud Logging**
- Log management and analysis
- Export logs to BigQuery for analysis
- Real-time log monitoring

### 34. **Cloud Monitoring**
- Infrastructure and application monitoring
- Custom metrics and dashboards
- Integration with data pipelines

## Specialty Services

### 35. **Genomics (Cloud Life Sciences)**
- Genomic data processing
- Bioinformatics pipelines
- Large-scale genomic analysis

### 36. **Healthcare API**
- FHIR, HL7v2, DICOM support
- Healthcare data ingestion and storage
- De-identification services

---

## Service Categories Summary

| Category | Services |
|----------|----------|
| **Storage** | Cloud Storage, Cloud SQL, Cloud Spanner, Cloud Bigtable, Firestore, Memorystore |
| **Processing** | BigQuery, Dataflow, Dataproc, Data Fusion, Dataprep |
| **Ingestion** | Pub/Sub, Datastream, Transfer Service, Transfer Appliance, BigQuery Data Transfer |
| **Orchestration** | Cloud Composer, Cloud Scheduler |
| **Governance** | Dataplex, Data Catalog, DLP |
| **ML/AI** | Vertex AI, BigQuery ML |
| **Visualization** | Looker, Looker Studio |
| **Transformation** | Dataform, Dataprep, Data Fusion |

---

## Common Data Engineering Patterns

### 1. **Batch Processing Pipeline**
Cloud Storage → Dataproc/Dataflow → BigQuery → Looker Studio

### 2. **Streaming Pipeline**
Pub/Sub → Dataflow → BigQuery → Real-time Dashboard

### 3. **Data Lake Architecture**
Cloud Storage (Raw) → Dataproc/Dataflow (Processing) → Cloud Storage (Curated) → BigQuery (Analytics)

### 4. **ETL/ELT Pipeline**
Source Systems → Data Fusion/Dataflow → Cloud Storage/BigQuery → Analytics

### 5. **Real-time Analytics**
IoT Devices → Pub/Sub → Dataflow → Bigtable/BigQuery → Monitoring Dashboard

---

*Last Updated: December 25, 2025*
