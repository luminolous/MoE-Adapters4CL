#!bin/bash

# Scenario 2: tasks are domains (6 tasks), evaluated cumulatively.
# Usage:
#   bash run_domainnet_domains.sh

set -e

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main.py \
  --config-path configs/domain \
  --config-name domainnet_domains-MoE-Adapters.yaml \
  dataset_root="../datasets/" \
  class_order="class_orders/domainnet.yaml"
