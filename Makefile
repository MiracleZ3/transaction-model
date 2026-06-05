.PHONY: install data tokenize train extract detect all all-pretrained clean test install-dev

PYTHON ?= python
PIPELINE ?= $(PYTHON) scripts/run_pipeline.py

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu]"

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
