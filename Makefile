SHELL = /bin/bash
# This makefile runs the pipeline that:
# 1. Processes the raw dataset.
# 2. Builds the model.
# 3. ??

BASE_DIR := .
DATA_DIR := $(BASE_DIR)/data
SRC_DIR  := $(BASE_DIR)/src
FILENAME_BASE := dataset

PYTHONPATH := $(SRC_DIR):$(PYTHONPATH)

ALL: $(DATA_DIR)/$(FILENAME_BASE)_norm.txt

$(DATA_DIR)/$(FILENAME_BASE).csv: $(DATA_DIR)/$(FILENAME_BASE).json
	python $(SRC_DIR)/dataset_preprocessor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(FILENAME_BASE).json \
		--output_filename=$(FILENAME_BASE).csv

$(DATA_DIR)/$(FILENAME_BASE)_norm.csv: $(DATA_DIR)/$(FILENAME_BASE).csv
	python $(SRC_DIR)/dataset_normalizer.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(FILENAME_BASE).csv \
		--output_filename=$(FILENAME_BASE)_norm.csv

$(DATA_DIR)/$(FILENAME_BASE)_norm.txt: $(DATA_DIR)/$(FILENAME_BASE)_norm.csv
	python $(SRC_DIR)/token_extractor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(FILENAME_BASE)_norm.csv \
		--output_filename=$(FILENAME_BASE)_norm.txt
