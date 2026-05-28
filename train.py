import argparse
import os
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from create_dataloader import build_voice_datasets, mil_collate_fn
from Learner import Learner
from model import MILModel
from preprocessing import (
    SAMPLE_RATE, SEGMENT_DURATION, HOP_DURATION,
    N_MELS, N_FFT, HOP_LENGTH, IMG_SIZE,
    SPEC_TYPE, N_MFCC,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train MIL model for PD vs HC voice classification"
    )

    # --- Required ---
    parser.add_argument("--data_dir", required=True,
                        help="Root folder containing PD/ and HC/ subfolders with WAV files",
                        dest="data_dir")

    # --- Output paths ---
    parser.add_argument("--model_save_dir", default=None, dest="model_save_dir")
    parser.add_argument("--metrics_save_dir", default=None, dest="metrics_save_dir")
    parser.add_argument("--experiment_name", default="logs", dest="experiment_name",
                        help="Subfolder name inside metrics_save_dir for this run")

    # --- Model hyperparameters ---
    parser.add_argument("--num_features", default=32, type=int, dest="num_features",
                        help="Number of features extracted per spectrogram instance")
    parser.add_argument("--num_bins", default=11, type=int, dest="num_bins",
                        help="Number of bins in the Distribution Pooling filter")
    parser.add_argument("--sigma", default=0.05, type=float, dest="sigma",
                        help="Sigma (bandwidth) in the Distribution Pooling filter")
    parser.add_argument("--mlp_hidden_sizes", default=[192, 96], nargs="+", type=int,
                        dest="mlp_hidden_sizes",
                        help="Hidden layer sizes for MLPClassifier, e.g. --mlp_hidden_sizes 192 96")
    parser.add_argument("--mlp_dropout", default=0.5, type=float, dest="mlp_dropout",
                        help="Dropout rate for MLPClassifier")

    # --- Training hyperparameters ---
    parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
    parser.add_argument("--num_workers", default=0, type=int, dest="num_workers")
    parser.add_argument("--learning_rate", default=1e-4, type=float, dest="learning_rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument("--num_epochs", default=300, type=int, dest="num_epochs")
    parser.add_argument("--save_top_k_models", default=1, type=int, dest="save_top_k_models")
    parser.add_argument("--device", default="cuda", dest="device")

    # --- Audio preprocessing parameters ---
    parser.add_argument("--sample_rate", default=SAMPLE_RATE, type=int, dest="sample_rate")
    parser.add_argument("--segment_duration", default=SEGMENT_DURATION, type=float, dest="segment_duration",
                        help="Duration (seconds) of each spectrogram segment")
    parser.add_argument("--hop_duration", default=HOP_DURATION, type=float, dest="hop_duration",
                        help="Hop size (seconds) between consecutive segments")
    parser.add_argument("--n_mels", default=N_MELS, type=int, dest="n_mels")
    parser.add_argument("--n_fft", default=N_FFT, type=int, dest="n_fft")
    parser.add_argument("--hop_length", default=HOP_LENGTH, type=int, dest="hop_length")
    parser.add_argument("--img_size", default=IMG_SIZE, type=int, dest="img_size",
                        help="Height and width (pixels) of the resized spectrogram")
    parser.add_argument("--spec_type", default=SPEC_TYPE, choices=["mel", "linear", "mfcc"],
                        dest="spec_type", help="Spectrogram type: mel, linear (FFT), or mfcc")
    parser.add_argument("--n_mfcc", default=N_MFCC, type=int, dest="n_mfcc",
                        help="Number of MFCC coefficients (only used when --spec_type mfcc)")
    parser.add_argument("--freq_min", default=None, type=float, dest="freq_min",
                        help="Lower frequency bound in Hz to keep (mel/linear only)")
    parser.add_argument("--freq_max", default=None, type=float, dest="freq_max",
                        help="Upper frequency bound in Hz to keep (mel/linear only)")
    parser.add_argument("--time_start", default=None, type=float, dest="time_start",
                        help="Start of time crop in seconds (within each segment)")
    parser.add_argument("--time_end", default=None, type=float, dest="time_end",
                        help="End of time crop in seconds (within each segment)")
    parser.add_argument("--shuffle_bag", default=False, type=lambda x: x.lower() == "true",
                        dest="shuffle_bag",
                        help="If True, shuffle instance order within each bag during training")

    args = parser.parse_args()

    # Resolve output directories
    cwd = Path.cwd()
    if not args.model_save_dir or not os.path.isdir(args.model_save_dir):
        args.model_save_dir = str(cwd / "saved_models")
    if not args.metrics_save_dir or not os.path.isdir(args.metrics_save_dir):
        args.metrics_save_dir = str(cwd / "metrics")

    # Fall back to CPU if CUDA is not available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        args.device = "cpu"
        args.num_workers = 0

    audio_kwargs = dict(
        sample_rate=args.sample_rate,
        segment_duration=args.segment_duration,
        hop_duration=args.hop_duration,
        spec_type=args.spec_type,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mfcc=args.n_mfcc,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        time_start=args.time_start,
        time_end=args.time_end,
        img_size=args.img_size,
    )

    print("Building datasets...")
    train_ds, val_ds, _ = build_voice_datasets(args.data_dir, shuffle_bag=args.shuffle_bag, **audio_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True,
        collate_fn=mil_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        collate_fn=mil_collate_fn,
    )

    print("Creating model...")
    model = MILModel(
        num_features=args.num_features,
        num_bins=args.num_bins,
        sigma=args.sigma,
        mlp_hidden_sizes=tuple(args.mlp_hidden_sizes),
        mlp_dropout=args.mlp_dropout,
    )
    learner = Learner(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logger = TensorBoardLogger(save_dir=args.metrics_save_dir, name=args.experiment_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_save_dir,
        filename="model-{epoch:02d}-{validation_loss:.3f}",
        monitor="validation_loss",
        save_top_k=args.save_top_k_models,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="validation_loss",
        patience=1000,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=args.num_epochs,
        default_root_dir=args.model_save_dir,
        log_every_n_steps=1,
        accelerator=args.device,
    )

    print("Start training")
    trainer.fit(model=learner, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training complete")


if __name__ == "__main__":
    main()
