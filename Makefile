PYTHON=3.9
BASENAME=serving-codegen-triton
TRITON_CONTAINER_NAME=registry.gitlab.com/curt-park/tritonserver-ft
TRITON_VERSION=22.12

env:
	conda create -n $(BASENAME)  python=$(PYTHON) -y

setup:
	pip install -r requirements.txt

client:
	GRADIO_SERVER_NAME=0.0.0.0 python src/client/app.py


# Triton
model:
	git clone https://huggingface.co/curt-park/codegen-350M-mono-gptj

triton:
	docker run --gpus "device=0" --shm-size=4G --rm \
		-p 8000:8000 -p 8001:8001 -p 8002:8002  -v $(PWD)/codegen-350M-mono-gptj/model_repository:/models \
		$(TRITON_CONTAINER_NAME):$(TRITON_VERSION) tritonserver --model-repository=/models

perf-help:
	docker run -it --ipc=host --net=host nvcr.io/nvidia/tritonserver:$(TRITON_VERSION)-py3-sdk \
		/workspace/install/bin/perf_analyzer --help\

perf:
	docker run -it --ipc=host --net=host nvcr.io/nvidia/tritonserver:$(TRITON_VERSION)-py3-sdk \
		/workspace/install/bin/perf_analyzer \
			-m ensemble --percentile=95 \
			-i grpc -u 0.0.0.0:8001 \
			--concurrency-range 1:1 \
			--input-data  sample_request_ensemble.json
			# --shape INPUT_0:32 --shape INPUT_1:1 --shape INPUT_2:1 --shape INPUT_3:1

curl:
	curl -X POST -H 'Content-Type: application/json' -d @sample_request_ensemble.json http://localhost:8000/v2/models/ensemble/infer


# Dev
setup-dev:
	pip install -r requirements-dev.txt

format:
	black .
	isort .

lint:
	PYTHONPATH=src pytest src --flake8 --pylint --mypy
