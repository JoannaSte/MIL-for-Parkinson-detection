import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
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
        description="Evaluate MIL model for PD vs HC voice classification"
    )

    # --- Required ---
    parser.add_argument("--model_dir", required=True,
                        help="Path to the saved model checkpoint (.ckpt)", dest="model_dir")
    parser.add_argument("--data_dir", required=True,
                        help="Root folder containing PD/ and HC/ subfolders", dest="data_dir")

    # --- Output ---
    parser.add_argument("--save_path", default=None, dest="save_path")
    parser.add_argument("--file_name", default="results.png", dest="file_name")

    # --- Model hyperparameters (must match training) ---
    parser.add_argument("--num_features", default=32, type=int, dest="num_features")
    parser.add_argument("--num_bins", default=11, type=int, dest="num_bins")
    parser.add_argument("--sigma", default=0.05, type=float, dest="sigma")
    parser.add_argument("--mlp_hidden_sizes", default=[192, 96], nargs="+", type=int,
                        dest="mlp_hidden_sizes",
                        help="Hidden layer sizes, e.g. --mlp_hidden_sizes 192 96")
    parser.add_argument("--mlp_dropout", default=0.5, type=float, dest="mlp_dropout")
    parser.add_argument("--num_workers", default=0, type=int, dest="num_workers")
    parser.add_argument("--device", default="cuda", dest="device")

    # --- Audio preprocessing (must match training) ---
    parser.add_argument("--sample_rate", default=SAMPLE_RATE, type=int, dest="sample_rate")
    parser.add_argument("--segment_duration", default=SEGMENT_DURATION, type=float, dest="segment_duration")
    parser.add_argument("--hop_duration", default=HOP_DURATION, type=float, dest="hop_duration")
    parser.add_argument("--n_mels", default=N_MELS, type=int, dest="n_mels")
    parser.add_argument("--n_fft", default=N_FFT, type=int, dest="n_fft")
    parser.add_argument("--hop_length", default=HOP_LENGTH, type=int, dest="hop_length")
    parser.add_argument("--img_size", default=IMG_SIZE, type=int, dest="img_size")
    parser.add_argument("--spec_type", default=SPEC_TYPE, choices=["mel", "linear", "mfcc"],
                        dest="spec_type")
    parser.add_argument("--n_mfcc", default=N_MFCC, type=int, dest="n_mfcc")
    parser.add_argument("--freq_min", default=None, type=float, dest="freq_min")
    parser.add_argument("--freq_max", default=None, type=float, dest="freq_max")
    parser.add_argument("--time_start", default=None, type=float, dest="time_start")
    parser.add_argument("--time_end", default=None, type=float, dest="time_end")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        args.num_workers = 0

    save_dir = Path(args.save_path) if args.save_path else Path.cwd()
    save_file = save_dir / args.file_name

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

    print("Building test dataset...")
    _, _, test_ds = build_voice_datasets(args.data_dir, **audio_kwargs)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, shuffle=False,
                             collate_fn=mil_collate_fn)

    print("Loading model from checkpoint...")
    model = MILModel(
        num_features=args.num_features,
        num_bins=args.num_bins,
        sigma=args.sigma,
        mlp_hidden_sizes=tuple(args.mlp_hidden_sizes),
        mlp_dropout=args.mlp_dropout,
    )
    learner = Learner.load_from_checkpoint(args.model_dir, model=model, map_location=args.device)
    learner.eval()
    learner.to(args.device)

    print("Running predictions...")
    true_labels = []
    pred_labels = []
    pred_probs = []   # probability of PD (class 1)

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(args.device)
            logits = learner(data)                          # [1, 2]
            probs = torch.softmax(logits, dim=1)[:, 1]     # P(PD)
            pred = logits.argmax(dim=1)

            true_labels.append(label.cpu().item())
            pred_labels.append(pred.cpu().item())
            pred_probs.append(probs.cpu().item())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=["HC", "PD"]))

    # Compute ROC / AUC
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Confusion matrix
    im = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["HC", "PD"])
    axes[0].set_yticklabels(["HC", "PD"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]),
                         ha="center", va="center", fontsize=14,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_ylabel("True label")
    axes[0].set_xlabel("Predicted label")

    # 2. ROC curve
    axes[1].plot(fpr, tpr, color="purple", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve (PD vs HC)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_file, dpi=150)
    print(f"Plot saved to: {save_file}")


if __name__ == "__main__":
    main()
