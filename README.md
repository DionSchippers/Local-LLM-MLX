# Local LLM using MLX
This repository is a demonstration and testing project for every thing to do with LLM's on Apple Silicon Macs using the MLX framework.
It includes examples for Finetuning, RAG systems and LLM's with a more Tool based approach.
The code is intended for educational purposes and experimentation.

## How to navigate the repository
You can simply run main.py to navigate to the different systems. Currently, the systems included are:
```
Fine-tuned-Doctor - A fine-tuned LLM model for medical diagnosis using a small symptoms-diagnosis dataset.
Eredivisie RAG - A Retrieval Augmented Generation system using the small Eredivisie Dataset.
Nederland Expert RAG - A Retrieval Augmented Generation system using the PDF of the Wikipedia Article about the Netherlands.
Conversator - A conversational agent that can remember previous questions and answers.
```
When running main, it will greet and ask which system you want to explore.
You can simply return one of the options, and the system will take you there and explain more to you what it can and can't do.

## Setup Instructions
### create and activate virtual environment
```
python -m venv .venv
source .venv/bin/activate
```
### install required packages
```
pip install -r requirements.txt
```
Current packeges used are:
- mlx-lm
- pandas
- huggingface_hub + cli
- faiss-cpu
- sentence-transformers
- inquirer 

### Hugging Face CLI setup
The Hugging Face platform is used for downloading pre-trained LLM models. To access these models, you need to set up your Hugging Face credentials.
This can be done through the Hugging Face CLI.
Firstly you need to create an account on huggingface.co and get an access token: https://huggingface.co/settings/tokens.
Then run:
```
> hf auth login
```
This will prompt you to enter your access token. Once entered, your credentials will be stored locally, allowing you to download models seamlessly.

## Development
The main.py file is the entry point to the different systems. You can explore and modify the code in the respective folders for each system.
Feel free to experiment with different models, datasets, and configurations to enhance your understanding of LLMs on Apple Silicon Macs using the MLX framework.
If you want to install packages, be sure to add them to requirements.txt for future reference.
```
pip install <package-name>
pip freeze > requirements.txt
```