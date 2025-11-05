# Local LLM using MLX

# create and activate virtual environment
> python -m venv .venv
> source .venv/bin/activate

# ML package optimized for Apple Silicon
> pip install -U mlx-lm

# install other required libraries
> pip install --upgrade pip
> pip install pandas
> pip install huggingface_hub
> pip install "huggingface_hub[cli]"
> pip install faiss-cpu
> pip install sentence-transformers

# create access token to read/write data from hugging-face through the cli
# this token required when login to huggingface cli
https://huggingface.co/settings/tokens

# login into huggingface-cli with access token to download LLM models
> hf auth login