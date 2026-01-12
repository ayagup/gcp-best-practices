1. model evaluation has to be done on recall and percision parameters and not human evaluation.


2. Vertex AI Pipelines do not have a native API. Rather you should use kfp/tfx to define your pipeline, compile to json.
Then submit that JSON to vertex AI SDK to schedule it on Vertex AI Platform


3. 