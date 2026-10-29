# VizWiz Quality Baseline

Zero-shot image quality classification with Qwen3.5-0.8B and Qwen3.5-2B using independent Yes/No scoring.

Enter the baseline directory.
```bash
cd vizwiz_baseline
```

Create the conda environment.
```bash
bash setup_env.sh
```

Activate the environment.
```bash
conda activate llm_inference
```

Choose where to store the dataset and model weights (default: `./data`).
```bash
export VIZWIZ_ROOT=/path/to/storage
```

Download the test set and prepare labels.
```bash
python prepare_data.py --download
```

Download both models.
```bash
python download_models.py
```

Run both models on GPUs 0, 1, and 2 with 144 image-question sequences per batch.
```bash
GPUS=0,1,2 BATCH_SIZE=144 bash run_independent.sh
```

Compute the evaluation metrics: per-label precision at recall 1.0/0.9/0.8/0.7, and the mean rank of true issues under Platt-calibrated scores. Calibration is fit on `--calibration-results` / `--calibration-truth` (validation by default) and applied unchanged to the evaluation split; outputs go to `results/independent/evaluation_revised/`.
```bash
python evaluate_revised.py --calibration-results results/independent_val --calibration-truth "$VIZWIZ_ROOT/datasets/vizwiz_quality/val_labels.json"
```
