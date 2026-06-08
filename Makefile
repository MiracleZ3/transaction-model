.PHONY: install data tokenize train extract detect all all-pretrained clean test \
        install-dev install-gpu install-nemo \
        docker-build-base docker-build-cpu docker-build-gpu-train docker-build-gpu-infer \
        docker-build docker-push docker-test docker-clean docker-run-help \
        compose-data compose-detect compose-test compose-cpu-pipeline \
        compose-tokenize compose-train compose-extract compose-all compose-down \
        help

PYTHON ?= python
PIPELINE ?= $(PYTHON) scripts/run_pipeline.py

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu]" --extra-index-url https://pypi.nvidia.com

install-nemo:
	pip install -e ".[nemo]"

# 单步执行（带隐式依赖，触发仅当上游缺失）
data:
	$(PYTHON) scripts/step_01_dataset_baseline.py

tokenize: data/TabFormer/temporal_split/train.parquet
	$(PYTHON) scripts/step_02_tokenize_corpus.py

train: data/decoder_corpus/train_corpus.txt
	$(PYTHON) scripts/step_03_train_model.py

extract: models/decoder-foundation-model/config.json
	$(PYTHON) scripts/step_04_extract_embeddings.py

detect: data/embeddings/train_embeddings.npy
	$(PYTHON) scripts/step_05_fraud_detection.py

# 上游产物规则
data/TabFormer/temporal_split/train.parquet:
	$(PYTHON) scripts/step_01_dataset_baseline.py --skip-baseline

data/decoder_corpus/train_corpus.txt:
	$(PYTHON) scripts/step_02_tokenize_corpus.py

models/decoder-foundation-model/config.json:
	@echo "ERROR: 预训练模型缺失，请放置到 models/decoder-foundation-model/"; exit 1

data/embeddings/train_embeddings.npy:
	$(PYTHON) scripts/step_04_extract_embeddings.py

# 全流程
all:
	$(PIPELINE) --steps all

# 全流程（跳过训练，使用预训练模型）
all-pretrained:
	$(PIPELINE) --steps data,tokenize,extract,detect

clean:
	rm -rf data/TabFormer/temporal_split/
	rm -rf data/decoder_corpus/
	rm -rf data/embeddings/
	rm -rf data/outputs/
	rm -rf models/decoder-demo/

test:
	pytest tests/ -v

# ═══════════════════════════════════════════════════════════════════
# Docker: 镜像构建 / 推送 / 清理
# ═══════════════════════════════════════════════════════════════════
DOCKER_REG   ?= local
DOCKER_TAG   ?= latest
DOCKER_FLAGS ?= --progress=plain

# 1. 基础镜像（cpu / gpu 镜像 FROM 它，必须先构建）
docker-build-base:
	docker build $(DOCKER_FLAGS) -f docker/Dockerfile.base \
	    -t $(DOCKER_REG)/tm-base:$(DOCKER_TAG) .

# 2. CPU 镜像（继承 base）：~1.2 GB，覆盖 Step 1/5 + test + 可视化
docker-build-cpu: docker-build-base
	docker build $(DOCKER_FLAGS) -f docker/Dockerfile.cpu \
	    --build-arg REGISTRY=$(DOCKER_REG) --build-arg TAG=$(DOCKER_TAG) \
	    -t $(DOCKER_REG)/tm-cpu:$(DOCKER_TAG) .

# 3. GPU 训练镜像（CUDA + cuDF/cuML/cuPy + NeMo）：~10 GB，覆盖 Step 2/3
docker-build-gpu-train:
	docker build $(DOCKER_FLAGS) -f docker/Dockerfile.gpu.train \
	    -t $(DOCKER_REG)/tm-gpu-train:$(DOCKER_TAG) .

# 4. GPU 推理镜像（CUDA + cuDF + transformers）：~8 GB，覆盖 Step 4
docker-build-gpu-infer:
	docker build $(DOCKER_FLAGS) -f docker/Dockerfile.gpu.infer \
	    -t $(DOCKER_REG)/tm-gpu-infer:$(DOCKER_TAG) .

# 5. 一键构建全部（CPU + 两套 GPU；要求宿主可拉 nvidia/cuda）
docker-build: docker-build-cpu docker-build-gpu-train docker-build-gpu-infer
	@echo "All images built:"
	@docker images --format "  {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep "^$(DOCKER_REG)/tm-"

# 6. 仅构建 CPU（开发机 / 无 nvidia 驱动时）
docker-build-cpu-only: docker-build-cpu
	@echo "CPU image built (skip GPU)."

# 7. 测试容器
docker-test:
	docker run --rm $(DOCKER_REG)/tm-cpu:$(DOCKER_TAG) test

docker-run-help:
	docker run --rm $(DOCKER_REG)/tm-cpu:$(DOCKER_TAG) --help

# 8. 批量推送（仅当 DOCKER_REG ≠ local）
docker-push:
	@if [ "$(DOCKER_REG)" = "local" ]; then \
	    echo "Set DOCKER_REG to a remote registry, e.g.:"; \
	    echo "  make docker-push DOCKER_REG=ghcr.io/miraclez3 DOCKER_TAG=$(DOCKER_TAG)"; \
	    exit 1; \
	fi
	docker push $(DOCKER_REG)/tm-base:$(DOCKER_TAG)
	docker push $(DOCKER_REG)/tm-cpu:$(DOCKER_TAG)
	docker push $(DOCKER_REG)/tm-gpu-train:$(DOCKER_TAG)
	docker push $(DOCKER_REG)/tm-gpu-infer:$(DOCKER_TAG)

# 9. 清理本项目的所有镜像
docker-clean:
	-docker rmi $(DOCKER_REG)/tm-cpu:$(DOCKER_TAG) \
	    $(DOCKER_REG)/tm-gpu-train:$(DOCKER_TAG) \
	    $(DOCKER_REG)/tm-gpu-infer:$(DOCKER_TAG) \
	    $(DOCKER_REG)/tm-base:$(DOCKER_TAG) 2>/dev/null
	-docker image prune -f --filter label="org.opencontainers.image.title=transaction-model*"

# ═══════════════════════════════════════════════════════════════════
# Docker Compose: 按 profile 启动单个 service（--abort-on-container-exit
# 保证 service 退出后 docker compose 自己也退出）
# ═══════════════════════════════════════════════════════════════════
COMPOSE := docker compose

compose-data:
	$(COMPOSE) --profile data up --abort-on-container-exit

compose-detect:
	$(COMPOSE) --profile detect up --abort-on-container-exit

compose-test:
	$(COMPOSE) --profile test up --abort-on-container-exit

compose-cpu-pipeline:
	$(COMPOSE) --profile cpu-pipeline up --abort-on-container-exit

compose-tokenize:
	$(COMPOSE) --profile tokenize up --abort-on-container-exit

compose-train:
	$(COMPOSE) --profile train up --abort-on-container-exit

compose-extract:
	$(COMPOSE) --profile extract up --abort-on-container-exit

compose-all:
	$(COMPOSE) --profile all up --abort-on-container-exit

compose-down:
	$(COMPOSE) down --remove-orphans

# ═══════════════════════════════════════════════════════════════════
# Help
# ═══════════════════════════════════════════════════════════════════
help:
	@echo "Transaction Model Makefile targets:"
	@echo ""
	@echo "  Native (host Python):"
	@echo "    install              pip install -e ."
	@echo "    install-dev / install-gpu / install-nemo"
	@echo "    data / tokenize / train / extract / detect"
	@echo "    all / all-pretrained"
	@echo "    test                 pytest"
	@echo "    clean                remove generated artifacts"
	@echo ""
	@echo "  Docker images:"
	@echo "    docker-build-base"
	@echo "    docker-build-cpu        (also builds base)"
	@echo "    docker-build-gpu-train"
	@echo "    docker-build-gpu-infer"
	@echo "    docker-build            (all 4)"
	@echo "    docker-build-cpu-only   (only CPU, no NVIDIA needed)"
	@echo "    docker-test / docker-run-help"
	@echo "    docker-push             (set DOCKER_REG=remote)"
	@echo "    docker-clean"
	@echo ""
	@echo "  Docker Compose (one service per profile):"
	@echo "    compose-data / compose-detect / compose-test"
	@echo "    compose-cpu-pipeline"
	@echo "    compose-tokenize / compose-train / compose-extract"
	@echo "    compose-all"
	@echo "    compose-down"
	@echo ""
	@echo "  Override example:"
	@echo "    DOCKER_TAG=\$(git rev-parse --short HEAD) make docker-build-cpu"
	@echo "    TRAIN_NUM_GPUS=8 make compose-train"
	@echo "    DOCKER_REG=ghcr.io/miraclez3 DOCKER_TAG=v1.0 make docker-push"
