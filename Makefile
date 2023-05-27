PYTHON=3.9
BASENAME=serving-codegen-triton
CONTAINER_NAME=ghcr.io/curt-park/serving-codegen-triton:latest

env:
	conda create -n $(BASENAME)  python=$(PYTHON) -y

setup:
	pip install -r requirements.txt

model:
	wget https://huggingface.co/moyix/codegen-350M-mono-gptj/resolve/main/pytorch_model.bin

client:
	GRADIO_SERVER_NAME=0.0.0.0 python src/client/app.py

# Dev
setup-dev:
	pip install -r requirements-dev.txt

format:
	black .
	isort .

lint:
	PYTHONPATH=src pytest src --flake8 --pylint --mypy
