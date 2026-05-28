"""Save example preprocessed spectrograms as PNG images.

For each WAV file shows mel / linear / mfcc side by side on one figure.

Usage:
  python save_example_spectrograms.py --data_dir /path/to/dataset
  python save_example_spectrograms.py --data_dir /path/to/dataset --n_per_class 3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import (
    SAMPLE_RATE, SEGMENT_DURATION, HOP_DURATION,
    N_MELS, N_FFT, HOP_LENGTH, IMG_SIZE, N_MFCC,
    load_audio, segment_audio, compute_spectrogram, spectrogram_to_tensor,
)

SPEC_TYPES = ["mel", "linear", "mfcc"]


def save_comparison(wav_path: str, label: str, args, output_dir: Path) -> None:
    waveform = load_audio(wav_path, args.sample_rate)
    segments = segment_audio(waveform, args.sample_rate, args.segment_duration, args.hop_duration)
    seg = segments[0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, spec_type in zip(axes, SPEC_TYPES):
        spec = compute_spectrogram(
            seg,
            sample_rate=args.sample_rate,
            spec_type=spec_type,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mfcc=args.n_mfcc,
        )
        img = spectrogram_to_tensor(spec, args.img_size)  # [1, img_size, img_size]

        ax.imshow(img.squeeze().numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(spec_type.upper(), fontsize=13)
        ax.axis("off")

    stem = Path(wav_path).stem
    fig.suptitle(f"[{label}]  {stem}", fontsize=11)
    fig.tight_layout()

    out_path = output_dir / f"{label}_{stem}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_per_class", default=2, type=int)
    parser.add_argument("--sample_rate", default=SAMPLE_RATE, type=int)
    parser.add_argument("--segment_duration", default=SEGMENT_DURATION, type=float)
    parser.add_argument("--hop_duration", default=HOP_DURATION, type=float)
    parser.add_argument("--n_mels", default=N_MELS, type=int)
    parser.add_argument("--n_fft", default=N_FFT, type=int)
    parser.add_argument("--hop_length", default=HOP_LENGTH, type=int)
    parser.add_argument("--n_mfcc", default=N_MFCC, type=int)
    parser.add_argument("--img_size", default=IMG_SIZE, type=int)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "example_spectrograms"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    for label in ("HC", "PD"):
        class_dir = data_dir / label
        if not class_dir.exists():
            print(f"Folder not found, skipping: {class_dir}")
            continue
        wav_files = sorted(class_dir.glob("*.wav"))[: args.n_per_class]
        print(f"\n[{label}] {len(wav_files)} file(s)...")
        for wav_path in wav_files:
            try:
                save_comparison(str(wav_path), label, args, output_dir)
            except Exception as e:
                print(f"  ERROR {wav_path.name}: {e}")

    print(f"\nDone. Images in: {output_dir}")


if __name__ == "__main__":
    main()
