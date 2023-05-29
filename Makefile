PYTHON=3.9
BASENAME=serving-codegen-triton
CONTAINER_NAME=registry.gitlab.com/curt-park/serving-codegen-triton:latest
TRITON_CONTAINER_NAME=registry.gitlab.com/curt-park/tritonserver-ft
TRITON_VERSION=22.12

# Cluster
cluster:
	curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="v1.27.2+k3s1" K3S_KUBECONFIG_MODE="644" INSTALL_K3S_EXEC="server --disable=traefik" sh -s - --docker
	mkdir -p ~/.kube
	cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
	kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
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
	helm install dcgm-exporter charts/dcgm-exporter

remove-charts:
	helm uninstall prometheus || true
	helm uninstall promtail || true
	helm uninstall loki || true
	helm uninstall traefik || true

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

triton:
	docker run --gpus "device=0" --shm-size=4G --rm \
		-p 8000:8000 -p 8001:8001 -p 8002:8002  -v $(PWD)/codegen-350M-mono-gptj/model_repository:/models \
		$(TRITON_CONTAINER_NAME):$(TRITON_VERSION) tritonserver --model-repository=/models


# Docker
docker-build:
	docker build -t $(CONTAINER_NAME) .

docker-pull:
	docker pull $(CONTAINER_NAME)

docker-push:
	docker push $(CONTAINER_NAME)

docker-run:
	docker run -it --rm -p 7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0 -e TRITON_SERVER_URL=$(TRITON_SERVER_URL) $(CONTAINER_NAME)


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
