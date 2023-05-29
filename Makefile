PYTHON=3.9
BASENAME=serving-codegen-triton
CONTAINER_NAME=ghcr.io/curt-park/serving-codegen-gptj-triton
FT_BACKEND_VERSION=release/v1.4_tag
TRITON_VERSION=22.12

# Cluster
cluster:
	curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="v1.27.2+k3s1" K3S_KUBECONFIG_MODE="644" INSTALL_K3S_EXEC="server --disable=traefik" sh -s - --docker
	mkdir -p ~/.kube
	cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
	kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
	kubectl create -f https://raw.githubusercontent.com/NVIDIA/dcgm-exporter/master/dcgm-exporter.yaml
	helm repo add traefik https://helm.traefik.io/traefik
	helm repo add grafana https://grafana.github.io/helm-charts
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update

.PHONY: charts
charts:
	helm install traefik charts/traefik
	helm install loki charts/loki
	helm install promtail charts/promtail
	helm install prometheus charts/prometheus
	helm install triton charts/triton
	helm install client charts/client

remove-charts:
	helm uninstall client || true
	helm uninstall triton || true
	helm uninstall prometheus || true
	helm uninstall promtail || true
	helm uninstall loki || true
	helm uninstall traefik || true

remove-all-containers:
	docker rm -f $(shell docker ps -a -q)

finalize:
	sh /usr/local/bin/k3s-killall.sh
	sh /usr/local/bin/k3s-uninstall.sh

# Prerequisites for local execution
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


# Docker
docker-build:
	docker build -t $(CONTAINER_NAME):latest .

docker-pull:
	docker pull $(CONTAINER_NAME):latest

docker-push:
	docker push $(CONTAINER_NAME):latest

docker-run:
	docker run -it --rm -p 7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0 \
		-e TRITON_SERVER_URL=$(TRITON_SERVER_URL) $(CONTAINER_NAME):latest

docker-triton-build:
	git clone https://github.com/triton-inference-server/fastertransformer_backend.git
	cd fastertransformer_backend && \
		git fetch origin $(FT_BACKEND_VERSION)  && \
		git checkout $(FT_BACKEND_VERSION) && \
		cp docker/Dockerfile . && \
		docker build -t $(CONTAINER_NAME):server-$(TRITON_VERSION) .
	rm -rf fastertransformer_backend

docker-triton-push:
	docker push $(CONTAINER_NAME):server-$(TRITON_VERSION)

docker-triton-run:
	docker run --gpus "device=0" --shm-size=4G --rm \
		-p 8000:8000 -p 8001:8001 -p 8002:8002  \
		$(CONTAINER_NAME):server-$(TRITON_VERSION) \
		tritonserver --model-repository=/models


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
