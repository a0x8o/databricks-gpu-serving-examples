# Deploying Large Language Models on Databricks Model Serving
Welcome to this GitHub repository. Here, we provide example scripts to deploy different Huggingface models on Databricks Model Serving. These examples can also guide you in deploying other models following similar steps. The models included in this repository are:

| Model | Hugging Face Model Repo | Deployment Script |
|-------|------------------------|-------------------|
| Instruction-Tuned LLM with dolly-v2-3b/7b | [link to model](https://huggingface.co/databricks/dolly-v2-7b) | [link to script](examples/dolly-v2-7b(pyfunc).py) |
| Instruction-Tuned LLM with mpt-7b-instruct | [link to model](https://huggingface.co/mosaicml/mpt-7b-instruct) | [link to script](examples/mpt-7b-instruct(pyfunc).py) |
| Instruction-Tuned LLM with falcon-7b-instruct | [link to model](https://huggingface.co/tiiuae/falcon-7b-instruct) | [link to script](examples/falcon-7b-instruct(pyfunc).py) |
| Sentiment Analysis with bert-base-uncased-imdb | [link to model](https://huggingface.co/textattack/bert-base-uncased-imdb) | [link to script](examples/bert-sentiment(pyfunc).py) |
| Text-to-Image Generation with stable-diffusion-2-1 | [link to model](https://huggingface.co/stabilityai/stable-diffusion-2-1) | [link to script](examples/stable-diffusion-2-1(pyfunc).py)|
| Speech-to-Text with whisper-large-v2 | [link to model](https://huggingface.co/openai/whisper-large-v2) | [link to script](examples/whisper-large-v2(pyfunc).py)|
| Code Completion with replit-code-v1-3b | [link to model](https://huggingface.co/replit/replit-code-v1-3b) | [link to script](examples/replit-code-v1-3b(pyfunc).py) |
| Deploying 10B+ LLM: Example using dolly-v2-12b| [link to model](https://huggingface.co/databricks/dolly-v2-12b) | [link to script](examples/dolly-v2-12b(pyfunc).py) |


## Requirements
Before you start, please ensure you meet the following requirements:

- Ensure that you have Nvidia A10/A100 GPUs to run the script.

- Ensure that you have MLflow 2.3+ (MLR 13.1 beta) installed.

- Deployment requires GPU model serving. For more information on GPU model serving, contact the Databricks team.

## How to Use
Clone this repository and navigate to the desired script file. Follow the instructions within the script to deploy the model, ensuring you meet the requirements listed above.

## Contribution
Feel free to contribute to this project by forking this repo and creating pull requests. If you encounter any issues or have any questions, create an issue on this repo, and we'll try our best to respond in a timely manner.

## License
This project is licensed under the terms of the MIT license. For the usage license of the individual models, please check the respective links provided above.
