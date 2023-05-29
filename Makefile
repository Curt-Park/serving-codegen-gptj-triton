PYTHON=3.9
BASENAME=serving-codegen-triton
TRITON_CONTAINER_NAME=registry.gitlab.com/curt-park/tritonserver-ft
TRITON_VERSION=22.12

# Prerequisites
env:
	conda create -n $(BASENAME)  python=$(PYTHON) -y

setup:
	pip install -r requirements.txt


# Client
client:
	GRADIO_SERVER_NAME=0.0.0.0 python src/client/app.py


# Triton
model:
	git clone https://huggingface.co/curt-park/codegen-350M-mono-gptj

triton:
	docker run --gpus "device=0" --shm-size=4G --rm \
		-p 8000:8000 -p 8001:8001 -p 8002:8002  -v $(PWD)/codegen-350M-mono-gptj/model_repository:/models \
		$(TRITON_CONTAINER_NAME):$(TRITON_VERSION) tritonserver --model-repository=/models


# Dev
setup-dev:
	pip install -r requirements-dev.txt
	pre-commit install

format:
	black .
	isort .

lint:
	PYTHONPATH=src pytest src --flake8 --pylint --mypy

load-test:
	PYTHONPATH=src locust -f $(PWD)/test/load_test/locustfile.py APIUser
