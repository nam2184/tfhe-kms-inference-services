SHELL := /bin/bash

CONDA_ENV = honours
CONDA_ACTIVATE = source ~/miniconda3/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

.PHONY: inference kms stop-inference stop-kms stop-all

inference:
	$(CONDA_ACTIVATE) && nohup python inference-server.py > inference.log 2>&1 &

kms:
	$(CONDA_ACTIVATE) && nohup python kms.py > kms.log 2>&1 &

stop-inference:
	@pkill -f "python inference-server.py" || echo "inference-server not running"

stop-kms:
	@pkill -f "python kms.py" || echo "kms not running"

stop-all: stop-inference stop-kms

