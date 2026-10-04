#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
conda create -n llm_inference python=3.11 pip -y
conda run -n llm_inference pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
conda run -n llm_inference pip install 'torch==2.8.0+cu126' -r requirements.txt
