SHELL = /bin/bash
# This makefile runs the pipeline that:
# 1. Processes the raw dataset.
# 2. Builds the model.
# 3. ??

BASE_DIR := .
DATA_DIR := $(BASE_DIR)/data
SRC_DIR  := $(BASE_DIR)/src
NAME_BASE := dataset
PYTHONPATH := $(SRC_DIR):$(PYTHONPATH)

ALL: datasets codesets

datasets: $(DATA_DIR)/$(NAME_BASE)_norm.txt

codesets: $(DATA_DIR)/$(NAME_BASE)_code.csv $(DATA_DIR)/$(NAME_BASE)_autocode.csv

$(DATA_DIR)/$(NAME_BASE).csv: $(DATA_DIR)/$(NAME_BASE).json
	python $(SRC_DIR)/dataset_preprocessor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE).json \
		--output_filename=$(NAME_BASE).csv

$(DATA_DIR)/$(NAME_BASE)_norm.csv: $(DATA_DIR)/$(NAME_BASE).csv
	python $(SRC_DIR)/dataset_normalizer.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE).csv \
		--output_filename=$(NAME_BASE)_norm.csv

$(DATA_DIR)/$(NAME_BASE)_norm.txt: $(DATA_DIR)/$(NAME_BASE)_norm.csv
	python $(SRC_DIR)/token_extractor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE)_norm.csv \
		--output_filename=$(NAME_BASE)_norm.txt

$(DATA_DIR)/$(NAME_BASE)_code.csv: $(DATA_DIR)/$(NAME_BASE)_norm.csv
	python $(SRC_DIR)/coding_processor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE)_norm.csv \
		--output_filename=$(NAME_BASE)_code.csv \
		--size=50 \
		--company_names=['adani']

$(DATA_DIR)/$(NAME_BASE)_autocode.csv: $(DATA_DIR)/$(NAME_BASE)_norm.csv
	python $(SRC_DIR)/autocoding_processor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE)_norm.csv \
		--output_filename=$(NAME_BASE)_autocode.csv \
		--company_tweets=False
