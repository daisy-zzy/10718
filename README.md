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

Generate metrics, plots, and `results/independent/evaluation/report.html`.
```bash
python evaluate.py --bootstrap 1000
```

Run the evaluation checks.
```bash
python -m unittest test_evaluation.py
```
