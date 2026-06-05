.PHONY: install data tokenize train extract detect all all-pretrained clean test

PYTHON ?= python
PIPELINE ?= $(PYTHON) scripts/run_pipeline.py

install:
	pip install -e .

# 单步执行
data:
	$(PYTHON) scripts/step_01_dataset_baseline.py

tokenize:
	$(PYTHON) scripts/step_02_tokenize_corpus.py

train:
	$(PYTHON) scripts/step_03_train_model.py

extract:
	$(PYTHON) scripts/step_04_extract_embeddings.py

detect:
	$(PYTHON) scripts/step_05_fraud_detection.py

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
