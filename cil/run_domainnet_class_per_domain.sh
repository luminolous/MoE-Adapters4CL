#!bin/bash

# Usage:
#   bash run_domainnet_class_per_domain.sh <domain> [init] [inc]
# Example:
#   bash run_domainnet_class_per_domain.sh clipart 50 50

set -e

DOMAIN=${1:-clipart}
INIT=${2:-50}
INC=${3:-50}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main.py \
  --config-path configs/class \
  --config-name domainnet_50-50-MoE-Adapters.yaml \
  dataset_root="../datasets/" \
  class_order="class_orders/domainnet.yaml" \
  domainnet_domain=${DOMAIN} \
  initial_increment=${INIT} \
  increment=${INC}
