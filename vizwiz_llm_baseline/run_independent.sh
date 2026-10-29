#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
RESULTS=${RESULTS:-results/independent}
BATCH_SIZE=${BATCH_SIZE:-144}
IFS=',' read -ra gpu_ids <<< "${GPUS:-0,1,2}"
mkdir -p "$RESULTS"

worker() {
  local gpu=$1 shard=$2 model folder
  local models=(Qwen3.5-0.8B Qwen3.5-2B)
  if (( shard % 2 )); then models=(Qwen3.5-2B Qwen3.5-0.8B); fi
  for model in "${models[@]}"; do
    folder=qwen35_08b
    if [[ "$model" == Qwen3.5-2B ]]; then folder=qwen35_2b; fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u infer_independent.py \
      --model "$model" --output "$RESULTS/$folder/shard$shard.jsonl" \
      --batch-size "$BATCH_SIZE" --shard "$shard" --num-shards "${#gpu_ids[@]}" \
      > "$RESULTS/${folder}_shard$shard.log" 2>&1
  done
}

pids=()
for shard in "${!gpu_ids[@]}"; do
  worker "${gpu_ids[$shard]}" "$shard" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
