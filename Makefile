SHELL = /bin/bash
# This makefile runs the pipeline that:
# 1. Processes the raw dataset
# 2. Builds the word vectorizations
# 3. Trains/tests the model

BASE_DIR := .
DATA_DIR := $(BASE_DIR)/data
SRC_DIR  := $(BASE_DIR)/src
NAME_BASE := dataset

FASTTEXT := /code/fastText/fasttext
PYTHON := PYTHONPATH=$(BASE_DIR) python

ALL: datasets codesets wordvecs models

datasets: $(DATA_DIR)/$(NAME_BASE)_norm.txt

codesets: $(DATA_DIR)/$(NAME_BASE)_code.csv $(DATA_DIR)/$(NAME_BASE)_autocode.csv

wordvecs: $(DATA_DIR)/$(NAME_BASE)_wordvec_all100.vec

models: $(DATA_DIR)/model.pkl

$(DATA_DIR)/$(NAME_BASE).csv: $(DATA_DIR)/$(NAME_BASE).json
	$(PYTHON) $(SRC_DIR)/dataset_preprocessor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE).json \
		--output_filename=$(NAME_BASE).csv

$(DATA_DIR)/$(NAME_BASE)_norm.csv: $(DATA_DIR)/$(NAME_BASE).csv
	$(PYTHON) $(SRC_DIR)/dataset_normalizer.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE).csv \
		--output_filename=$(NAME_BASE)_norm.csv

$(DATA_DIR)/$(NAME_BASE)_norm.txt: $(DATA_DIR)/$(NAME_BASE)_norm.csv
	$(PYTHON) $(SRC_DIR)/token_extractor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE)_norm.csv \
		--output_filename=$(NAME_BASE)_norm.txt

$(DATA_DIR)/$(NAME_BASE)_code.csv: $(DATA_DIR)/$(NAME_BASE)_norm.csv
	$(PYTHON) $(SRC_DIR)/coding_processor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE)_norm.csv \
		--output_filename=$(NAME_BASE)_code.csv \
		--size=50 \
		--company_names=['adani']

$(DATA_DIR)/$(NAME_BASE)_autocode.csv: $(DATA_DIR)/$(NAME_BASE)_norm.csv
	$(PYTHON) $(SRC_DIR)/autocoding_processor.py \
		--dataset_path=$(DATA_DIR) \
		--input_filename=$(NAME_BASE)_norm.csv \
		--output_filename=$(NAME_BASE)_autocode.csv \
		--company_tweets=False

$(DATA_DIR)/$(NAME_BASE)_wordvec_all100.vec: $(DATA_DIR)/$(NAME_BASE)_norm.txt
	$(FASTTEXT) skipgram -input $(DATA_DIR)/$(NAME_BASE)_norm.txt -output $(DATA_DIR)/$(NAME_BASE)_wordvec_all100 -dim 100

$(DATA_DIR)/model.pkl: $(DATA_DIR)/$(NAME_BASE)_autocode.csv $(DATA_DIR)/$(NAME_BASE)_wordvec_all100.vec
	$(PYTHON) $(SRC_DIR)/model_build.py \
		--dataset_path=$(DATA_DIR) \
		--trainset_filename=$(NAME_BASE)_autocode.csv \
		--word_vectors_filename=$(NAME_BASE)_wordvec_all100.vec \
		--model_filename=model.pkl

test: $(DATA_DIR)/model.pkl $(DATA_DIR)/coding/gold_20180514_majority.csv
	$(PYTHON) $(SRC_DIR)/model_test.py \
		--dataset_path=$(DATA_DIR) \
		--testset_filename=coding/gold_20180514_majority.csv \
		--model_filename=model.pkl

clean:
	rm -f $(DATA_DIR)/$(NAME_BASE).csv
	rm -f $(DATA_DIR)/$(NAME_BASE)_norm.csv
	rm -f $(DATA_DIR)/$(NAME_BASE)_norm.txt
	rm -f $(DATA_DIR)/$(NAME_BASE)_code.csv
	rm -f $(DATA_DIR)/$(NAME_BASE)_autocode.csv
	rm -f $(DATA_DIR)/$(NAME_BASE)_wordvec_all100.vec
	rm -f $(DATA_DIR)/$(NAME_BASE)_wordvec_all100.bin
	rm -f $(DATA_DIR)/model.pkl
	rm -r $(BASE_DIR)/__main__.log
