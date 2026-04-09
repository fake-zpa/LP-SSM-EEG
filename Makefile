.PHONY: all setup check-env download-chbmit download-sleepedf download-eegmmidb \
        verify-downloads preprocess-chbmit preprocess-sleepedf \
        smoke-test train-baseline train-lp-ssm train-all \
        eval-all ablation analysis clean

PROJECT_ROOT := $(shell pwd)
PYTHON := python
SRC := src/cli/main.py

all: setup

setup:
	bash scripts/setup_env.sh

check-env:
	$(PYTHON) $(SRC) system-info

download-chbmit:
	$(PYTHON) $(SRC) download --dataset chbmit

download-sleepedf:
	$(PYTHON) $(SRC) download --dataset sleepedf

download-eegmmidb:
	$(PYTHON) $(SRC) download --dataset eegmmidb

download-smoke:
	$(PYTHON) $(SRC) download --dataset chbmit --subjects chb01 --max-files 5

verify-downloads:
	$(PYTHON) $(SRC) verify --dataset chbmit
	$(PYTHON) $(SRC) verify --dataset sleepedf

preprocess-chbmit:
	$(PYTHON) $(SRC) preprocess --dataset chbmit --config configs/data/chbmit.yaml

preprocess-sleepedf:
	$(PYTHON) $(SRC) preprocess --dataset sleepedf --config configs/data/sleepedf.yaml

smoke-test:
	bash scripts/run_smoke_test.sh

train-baseline:
	$(PYTHON) $(SRC) train --model eegnet --config configs/train/train_default.yaml

train-mamba:
	$(PYTHON) $(SRC) train --model mamba_baseline --config configs/train/amp_single_4090.yaml

train-lp-ssm:
	$(PYTHON) $(SRC) train --model lp_ssm_eeg --config configs/train/amp_single_4090.yaml

train-all:
	bash scripts/run_main_experiments.sh

eval-all:
	$(PYTHON) $(SRC) eval --all

ablation:
	bash scripts/run_ablation.sh

analysis:
	bash scripts/run_analysis.sh

clean-processed:
	rm -rf data/processed/*
	rm -rf data/manifests/*
	rm -rf data/interim/*

clean-logs:
	rm -rf logs/train/* logs/eval/* logs/preprocess/*

help:
	@echo "LP-SSM-EEG Project Makefile"
	@echo "----------------------------"
	@echo "make setup           -- install dependencies"
	@echo "make check-env       -- print system/GPU info"
	@echo "make download-smoke  -- download CHB-MIT chb01 (5 files, smoke test)"
	@echo "make download-chbmit -- download full CHB-MIT"
	@echo "make smoke-test      -- run full smoke test pipeline"
	@echo "make train-lp-ssm    -- train LP-SSM-EEG"
	@echo "make train-all       -- run all main experiments"
	@echo "make ablation        -- run ablation experiments"
	@echo "make analysis        -- run analysis & generate figures"
