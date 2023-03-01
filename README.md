# Hugging Face LLM inference example 

This is a simple example of how to use the [text-generation-inference](https://github.com/huggingface/text-generation-inference) library to deploy LLMs, like BLOOM, GPT-NeoX, FLAN-T5-XXL to Amaozn SageMaker.

## Requirements

- SageMaker compute quota for p4 instances

## Getting Started

Check out the [sagemaker-notebook](sagemaker-notebook.ipynb) for a step-by-step guide on how to deploy BLOOM to Amazon SageMaker.


## Get infrastructure/disk information

```python
import json
import os
import psutil


def model_fn(model_dir):
    os.system("df -h")


def predict_fn(data, model_and_tokenizer):
    hdd = psutil.disk_usage("/")
    return {
        "total_in_gb": hdd.total / (2**30),
        "used_in_gb": hdd.used / (2**30),
        "free_in_gb": hdd.free / (2**30),
    }

```

upload 

```python
s3_location=f"s3://{sess.default_bucket()}/custom_inference/disk/model.tar.gz"

%cd test
!tar zcvf model.tar.gz *
!aws s3 cp model.tar.gz $s3_location
%cd ..
````

deploy

```python
from sagemaker.huggingface.model import HuggingFaceModel


# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data=s3_location,       # path to your model and script
   role=role,                    # iam role with permissions to create an Endpoint
   transformers_version="4.17",  # transformers version used
   pytorch_version="1.10",        # pytorch version used
   py_version='py38',            # python version used
)

# deploy the endpoint endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.p4d.24xlarge"
    )
```