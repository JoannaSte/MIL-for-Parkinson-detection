"""Optuna hyperparameter search for MIL voice classification.

Tuned parameters:
  Training:  learning_rate, weight_decay, batch_size
  Audio:     sample_rate
  MLP arch:  dropout_rate, n_layers, hidden_size_0..n

Usage:
  python tune.py --data_dir /path/to/dataset --n_trials 50 --epochs_per_trial 50

Resume an interrupted study (reuse same --study_name and --results_dir):
  python tune.py --data_dir /path/to/dataset --study_name mil_tuning --n_trials 100

Requires: pip install optuna
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightning as pl
import optuna
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from create_dataloader import build_voice_datasets, mil_collate_fn
from Learner import Learner
from model import MILModel
from preprocessing import (
    SAMPLE_RATE, SEGMENT_DURATION, HOP_DURATION,
    N_MELS, N_FFT, HOP_LENGTH, IMG_SIZE,
    SPEC_TYPE, N_MFCC,
)

METRIC = "val_f1"


class _PruningCallback(pl.Callback):
    """Reports val_f1 to Optuna each epoch and prunes unpromising trials."""

    def __init__(self, trial: optuna.Trial, monitor: str) -> None:
        self._trial = trial
        self._monitor = monitor

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        value = trainer.callback_metrics.get(self._monitor)
        if value is None:
            return
        self._trial.report(float(value), step=trainer.current_epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


def _build_objective(args, summary_writer: SummaryWriter):
    def objective(trial: optuna.Trial) -> float:
        # --- Training hyperparameters ---
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])

        # --- Audio ---
        sample_rate = trial.suggest_categorical("sample_rate", [16000, 22050])

        # --- MLP architecture ---
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.7)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_sizes = tuple(
            trial.suggest_categorical(f"hidden_size_{i}", [32, 64, 96, 128, 192, 256])
            for i in range(n_layers)
        )

        audio_kwargs = dict(
            sample_rate=sample_rate,
            segment_duration=args.segment_duration,
            hop_duration=args.hop_duration,
            spec_type=args.spec_type,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mfcc=args.n_mfcc,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            img_size=args.img_size,
        )

        train_ds, val_ds, _ = build_voice_datasets(
            args.data_dir, shuffle_bag=False, **audio_kwargs
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            num_workers=args.num_workers, shuffle=True,
            collate_fn=mil_collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            num_workers=args.num_workers, shuffle=False,
            collate_fn=mil_collate_fn,
        )

        model = MILModel(
            num_features=args.num_features,
            num_bins=args.num_bins,
            sigma=args.sigma,
            mlp_hidden_sizes=hidden_sizes,
            mlp_dropout=dropout_rate,
        )
        learner = Learner(model=model, learning_rate=lr, weight_decay=wd)

        # Each trial gets its own TensorBoard run so you can compare curves
        trial_logger = TensorBoardLogger(
            save_dir=str(args.log_dir),
            name=args.study_name,
            version=f"trial_{trial.number:03d}",
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs_per_trial,
            accelerator=args.device,
            logger=trial_logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[_PruningCallback(trial, METRIC)],
        )

        try:
            trainer.fit(learner, train_loader, val_loader)
        except optuna.exceptions.TrialPruned:
            raise

        final_val_f1 = trainer.callback_metrics.get(METRIC, torch.tensor(0.0)).item()

        # Log cross-trial summary: one scalar per trial number for easy comparison
        summary_writer.add_scalar("optuna/val_f1", final_val_f1, trial.number)
        summary_writer.add_scalar("optuna/learning_rate", lr, trial.number)
        summary_writer.add_scalar("optuna/weight_decay", wd, trial.number)
        summary_writer.add_scalar("optuna/batch_size", float(batch_size), trial.number)
        summary_writer.add_scalar("optuna/sample_rate", float(sample_rate), trial.number)
        summary_writer.add_scalar("optuna/dropout_rate", dropout_rate, trial.number)
        summary_writer.add_scalar("optuna/n_layers", float(n_layers), trial.number)
        summary_writer.flush()

        return final_val_f1

    return objective


def _save_optuna_plots(study: optuna.Study, results_dir: Path) -> None:
    """Save Optuna visualization plots as PNG files using the matplotlib backend."""
    import importlib
    try:
        _mpl = importlib.import_module("optuna.visualization.matplotlib")
    except ImportError:
        print("\noptuna matplotlib visualization not available, skipping plots.")
        return

    plot_optimization_history = _mpl.plot_optimization_history
    plot_param_importances = _mpl.plot_param_importances
    plot_parallel_coordinate = _mpl.plot_parallel_coordinate
    plot_slice = _mpl.plot_slice

    print("\nSaving Optuna plots...")

    for filename, fn in [
        ("optimization_history.png", plot_optimization_history),
        ("parallel_coordinate.png", plot_parallel_coordinate),
    ]:
        try:
            fn(study)
            fig = plt.gcf()
            path = results_dir / filename
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  {path}")
        except Exception as e:
            print(f"  {filename}: skipped ({e})")

    # param_importances requires scikit-learn
    try:
        plot_param_importances(study)
        fig = plt.gcf()
        path = results_dir / "param_importances.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")
    except Exception as e:
        print(f"  param_importances.png: skipped ({e})")

    # slice: one subplot per parameter showing how each param affects val_f1
    try:
        plot_slice(study)
        fig = plt.gcf()
        path = results_dir / "slice.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")
    except Exception as e:
        print(f"  slice.png: skipped ({e})")


def _print_best(study: optuna.Study, args) -> None:
    best = study.best_trial
    p = best.params
    n_layers = p.get("n_layers", 1)
    hidden = [p[f"hidden_size_{i}"] for i in range(n_layers) if f"hidden_size_{i}" in p]

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Best trial #{best.number}   {METRIC} = {best.value:.4f}")
    print(sep)
    print(f"  {'learning_rate':<18} {p['learning_rate']:.3e}")
    print(f"  {'weight_decay':<18} {p['weight_decay']:.3e}")
    print(f"  {'batch_size':<18} {p['batch_size']}")
    print(f"  {'sample_rate':<18} {p['sample_rate']}")
    print(f"  {'dropout_rate':<18} {p['dropout_rate']:.3f}")
    print(f"  {'n_layers':<18} {n_layers}")
    print(f"  {'hidden_sizes':<18} {hidden}")
    print(sep)

    hidden_str = " ".join(str(h) for h in hidden)
    print("\nReady-to-use train.py command:")
    print(
        f"  python train.py"
        f" --data_dir {args.data_dir}"
        f" --learning_rate {p['learning_rate']:.3e}"
        f" --weight_decay {p['weight_decay']:.3e}"
        f" --batch_size {p['batch_size']}"
        f" --sample_rate {p['sample_rate']}"
        f" --mlp_hidden_sizes {hidden_str}"
        f" --mlp_dropout {p['dropout_rate']:.3f}"
    )
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for MIL voice classification"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Root folder with PD/ and HC/ subfolders")

    # Optuna settings
    parser.add_argument("--n_trials", default=50, type=int,
                        help="Total number of Optuna trials to run")
    parser.add_argument("--epochs_per_trial", default=50, type=int,
                        help="Max training epochs per trial")
    parser.add_argument("--study_name", default="mil_tuning",
                        help="Optuna study name (reuse to resume)")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL; default: sqlite:///<results_dir>/<study_name>.db")
    parser.add_argument("--results_dir", default=None,
                        help="Directory for the SQLite DB, CSV results and PNG plots")
    parser.add_argument("--log_dir", default=None,
                        help="TensorBoard log directory (default: <results_dir>/tb_logs)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=0, type=int)

    # Fixed audio params (not tuned)
    parser.add_argument("--segment_duration", default=SEGMENT_DURATION, type=float)
    parser.add_argument("--hop_duration", default=HOP_DURATION, type=float)
    parser.add_argument("--spec_type", default=SPEC_TYPE, choices=["mel", "linear", "mfcc"])
    parser.add_argument("--n_mels", default=N_MELS, type=int)
    parser.add_argument("--n_fft", default=N_FFT, type=int)
    parser.add_argument("--hop_length", default=HOP_LENGTH, type=int)
    parser.add_argument("--n_mfcc", default=N_MFCC, type=int)
    parser.add_argument("--freq_min", default=None, type=float)
    parser.add_argument("--freq_max", default=None, type=float)
    parser.add_argument("--img_size", default=IMG_SIZE, type=int)

    # Fixed model params (not tuned)
    parser.add_argument("--num_features", default=32, type=int)
    parser.add_argument("--num_bins", default=11, type=int)
    parser.add_argument("--sigma", default=0.05, type=float)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        args.device = "cpu"

    results_dir = Path(args.results_dir) if args.results_dir else Path.cwd() / "tuning_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    args.log_dir = Path(args.log_dir) if args.log_dir else results_dir / "tb_logs"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage or f"sqlite:///{results_dir / (args.study_name + '.db')}"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )

    # Shared writer logs one point per trial (cross-trial progress)
    summary_writer = SummaryWriter(log_dir=str(args.log_dir / "optuna_summary"))
    try:
        study.optimize(_build_objective(args, summary_writer), n_trials=args.n_trials)
    finally:
        summary_writer.close()

    # Save CSV with all trials
    try:
        csv_path = results_dir / f"{args.study_name}_results.csv"
        study.trials_dataframe().to_csv(csv_path, index=False)
        print(f"\nAll trial results saved to {csv_path}")
    except Exception:
        pass

    # Save PNG plots
    _save_optuna_plots(study, results_dir)

    # Print best params + ready-to-run command
    _print_best(study, args)

    print(f"\nOpen TensorBoard with:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
