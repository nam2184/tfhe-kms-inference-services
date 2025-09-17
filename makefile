SHELL := /bin/bash

CONDA_ENV = honours
CONDA_ACTIVATE = source ~/miniconda3/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

.PHONY: inference kms

inference:
	$(CONDA_ACTIVATE) && nohup python inference-server.py > inference.log 2>&1 &

kms:
	$(CONDA_ACTIVATE) && nohup python kms.py > kms.log 2>&1 &

