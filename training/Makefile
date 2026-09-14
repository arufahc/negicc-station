# Makefile for Appearance Model Training Pipeline (DINOv3)

PYTHON ?= python3
CORPUS_DIR ?= /home/alpha/Projects/FilmCorpus/training
CACHE_DIR ?= cache
OUTPUT_DIR ?= ../models/dinov3-small_portra400
METADATA ?= ../models/dinov3-small_portra400/metadata.json
PROFILE ?= portra400
EPOCHS ?= 60
BATCH_SIZE ?= 64
MAX_SAMPLES ?= 400
DEVICE ?= cpu

.PHONY: all help train_portra400 test clean download_backbone

help:
	@echo "Appearance Model Training Pipeline (DINOv3-Small)"
	@echo "Usage:"
	@echo "  make all                  - Download backbone, extract features, and train default model"
	@echo "  make train_portra400      - Train Portra 400 appearance model using metadata.json"
	@echo "  make test                 - Quick test run (2 epochs, 10 samples)"
	@echo "  make download_backbone    - Download DINOv3-small backbone weights locally"
	@echo "  make clean                - Clean training cache and temporary files"

all: download_backbone train_portra400

download_backbone:
	@echo "Ensuring DINOv3 backbone is cached locally..."
	@$(PYTHON) -c "import os; from transformers import AutoModel; out='../models/dinov3-small'; os.makedirs(out, exist_ok=True); AutoModel.from_pretrained('Tooony133/dinov3-vits16-pretrain-lvd1689m').save_pretrained(out) if not os.path.isfile(os.path.join(out, 'model.safetensors')) else print('DINOv3 backbone already present.'); print('DINOv3 backbone ready at ../models/dinov3-small.')"

train_portra400: download_backbone
	$(PYTHON) train_appearance_model.py \
		--metadata $(METADATA) \
		--corpus-dir $(CORPUS_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--cache-dir $(CACHE_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--max-samples $(MAX_SAMPLES) \
		--device $(DEVICE)

test: download_backbone
	$(PYTHON) train_appearance_model.py \
		--metadata $(METADATA) \
		--corpus-dir $(CORPUS_DIR) \
		--output-dir test_output \
		--cache-dir $(CACHE_DIR) \
		--epochs 2 \
		--batch-size 16 \
		--max-samples 10 \
		--device $(DEVICE)

clean:
	rm -rf cache test_output __pycache__ *.pyc
