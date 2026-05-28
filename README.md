# MIL-based Voice Classification for Parkinson's Disease Detection

This project applies Multiple Instance Learning (MIL) to classify voice recordings as either Parkinson's Disease (PD) or Healthy Control (HC). Each recording is treated as a *bag* of short overlapping spectrogram segments (instances), and the model learns to classify the bag without per-segment labels.

## Model architecture

1. **SpectrogramResNet** — ResNet-18 adapted for single-channel spectrograms; extracts a feature vector from each segment.
2. **DistributionPoolingFilter** — aggregates all per-segment features into a bag-level distribution histogram.
3. **MLPClassifier** — classifies the histogram as PD (1) or HC (0).

Supports three spectrogram types: `mel` (log-mel), `linear` (log-power FFT), `mfcc`.

## Dataset structure

```
data/
├── HC/          # Healthy Control recordings
│   ├── subject01.wav
│   └── ...
└── PD/          # Parkinson's Disease recordings
    ├── subject02.wav
    └── ...
```

The dataset is automatically split into train / val / test (65 / 15 / 20 %) stratified at the recording level.

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `tune.py` | Optuna hyperparameter search |
| `visualize_results.py` | Evaluate a checkpoint — confusion matrix + ROC curve |
| `save_example_spectrograms.py` | Save example preprocessed spectrograms as PNG |
| `preprocessing.py` | Audio pipeline: load → segment → spectrogram → tensor |
| `create_dataloader.py` | `VoiceDataset` and `build_voice_datasets` |
| `model.py` | `MILModel` (ResNet + Distribution Pooling + MLP) |
| `Learner.py` | PyTorch Lightning wrapper with metrics |
| `distribution_pooling_filter.py` | Distribution Pooling Filter implementation |
| `requirements.txt` | Python dependencies |

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --data_dir /path/to/dataset
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | *required* | Folder with `PD/` and `HC/` subfolders |
| `--num_epochs` | `300` | Training epochs |
| `--batch_size` | `4` | Bags per batch |
| `--learning_rate` | `1e-4` | Adam learning rate |
| `--weight_decay` | `1e-4` | L2 regularisation |
| `--sample_rate` | `22050` | Audio sample rate (Hz) |
| `--spec_type` | `mel` | Spectrogram type: `mel`, `linear`, `mfcc` |
| `--segment_duration` | `0.25` | Segment length (seconds) |
| `--hop_duration` | `0.125` | Hop between segments (seconds) |
| `--freq_min` / `--freq_max` | `None` | Frequency band to keep (Hz) |
| `--mlp_hidden_sizes` | `192 96` | MLP hidden layer sizes |
| `--mlp_dropout` | `0.5` | MLP dropout rate |
| `--model_save_dir` | `saved_models/` | Where to save checkpoints |
| `--metrics_save_dir` | `metrics/` | TensorBoard log directory |
| `--device` | `cuda` | `cuda` or `cpu` |

## Hyperparameter search

Uses Optuna to tune learning rate, weight decay, batch size, sample rate, and MLP architecture. Results are saved as CSV and PNG plots, and each trial is logged to TensorBoard.

```bash
python tune.py --data_dir /path/to/dataset --n_trials 50 --epochs_per_trial 50
```

Resume an interrupted study by reusing the same `--study_name`:

```bash
python tune.py --data_dir /path/to/dataset --study_name mil_tuning --n_trials 100
```

After the search finishes, a ready-to-use `train.py` command is printed with the best parameters.

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_trials` | `50` | Number of Optuna trials |
| `--epochs_per_trial` | `50` | Max epochs per trial |
| `--study_name` | `mil_tuning` | Study name (reuse to resume) |
| `--results_dir` | `tuning_results/` | Directory for DB, CSV, and PNG plots |
| `--log_dir` | `tuning_results/tb_logs/` | TensorBoard logs |

## Evaluation

```bash
python visualize_results.py --data_dir /path/to/dataset --model_dir saved_models/model.ckpt
```

Saves a figure with a confusion matrix and ROC curve. Pass the same architecture arguments (`--mlp_hidden_sizes`, `--mlp_dropout`, etc.) that were used during training.

## TensorBoard

```bash
tensorboard --logdir metrics/
```

For hyperparameter search logs:

```bash
tensorboard --logdir tuning_results/tb_logs/
```

## Visualise spectrograms

```bash
python save_example_spectrograms.py --data_dir /path/to/dataset
```

Saves a side-by-side MEL | LINEAR | MFCC comparison for a few recordings from each class to `data/example_spectrograms/`.

## Reference

Distribution Pooling Filter:
```
@article{DBLP:journals/corr/abs-1802-04712,
  author    = {Maximilian Ilse and Jakub M. Tomczak and Max Welling},
  title     = {Attention-based Deep Multiple Instance Learning},
  journal   = {CoRR},
  volume    = {abs/1802.04712},
  year      = {2018},
  url       = {http://arxiv.org/abs/1802.04712},
}
```
