from pathlib import Path

import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

from preprocessing import (
    SAMPLE_RATE, SEGMENT_DURATION, HOP_DURATION,
    N_MELS, N_FFT, HOP_LENGTH, IMG_SIZE,
    SPEC_TYPE, N_MFCC,
    wav_to_spectrogram_bag,
)


class VoiceDataset(data_utils.Dataset):
    """MIL dataset for voice recordings.

    Each bag = one WAV file (one patient recording).
    Each instance = mel spectrogram of one audio segment.
    Label: 0 = HC (Healthy Control), 1 = PD (Parkinson's Disease).
    """

    def __init__(self, file_paths, labels,
                 sample_rate=SAMPLE_RATE,
                 segment_duration=SEGMENT_DURATION,
                 hop_duration=HOP_DURATION,
                 spec_type=SPEC_TYPE,
                 n_mels=N_MELS,
                 n_fft=N_FFT,
                 hop_length=HOP_LENGTH,
                 n_mfcc=N_MFCC,
                 freq_min=None,
                 freq_max=None,
                 time_start=None,
                 time_end=None,
                 img_size=IMG_SIZE,
                 shuffle_bag=False):
        self.file_paths = file_paths
        self.labels = labels
        self.shuffle_bag = shuffle_bag
        self.audio_kwargs = dict(
            sample_rate=sample_rate,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            spec_type=spec_type,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mfcc=n_mfcc,
            freq_min=freq_min,
            freq_max=freq_max,
            time_start=time_start,
            time_end=time_end,
            img_size=img_size,
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav_path = self.file_paths[idx]
        label = self.labels[idx]

        # bag shape: [num_segments, 1, img_size, img_size]
        bag = wav_to_spectrogram_bag(wav_path, **self.audio_kwargs)

        if self.shuffle_bag:
            idx = torch.randperm(bag.shape[0])
            bag = bag[idx]

        return bag, torch.tensor(label, dtype=torch.long)


def mil_collate_fn(batch):
    """Pad bags to the same number of segments so they can be stacked."""
    bags, labels = zip(*batch)
    max_n = max(b.shape[0] for b in bags)
    padded = []
    for b in bags:
        n, c, h, w = b.shape
        if n < max_n:
            pad = torch.zeros(max_n - n, c, h, w, dtype=b.dtype)
            b = torch.cat([b, pad], dim=0)
        padded.append(b)
    return torch.stack(padded), torch.stack(labels)


def build_voice_datasets(data_dir,
                         test_size=0.2,
                         val_size=0.15,
                         random_state=42,
                         shuffle_bag=False,
                         **audio_kwargs):
    """Scan data_dir/PD and data_dir/HC for WAV files and build train/val/test splits.

    Split is at the recording (patient) level to avoid data leakage.
    Returns three VoiceDataset instances: (train, val, test).
    """
    data_dir = Path(data_dir)

    all_paths = []
    all_labels = []

    for label, class_name in enumerate(["HC", "PD"]):
        class_path = data_dir / class_name
        if not class_path.exists():
            raise FileNotFoundError(f"Expected folder not found: {class_path}")
        wav_files = sorted(class_path.glob("*.wav"))
        for wav_file in wav_files:
            all_paths.append(str(wav_file))
            all_labels.append(label)

    print(f"Found {len(all_paths)} recordings  "
          f"(HC: {all_labels.count(0)}, PD: {all_labels.count(1)})")

    # First split off the test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=random_state,
    )

    # Then split train vs validation
    relative_val = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=relative_val,
        stratify=train_val_labels,
        random_state=random_state,
    )

    print(f"Split → Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    train_ds = VoiceDataset(train_paths, train_labels, shuffle_bag=shuffle_bag, **audio_kwargs)
    val_ds = VoiceDataset(val_paths, val_labels, shuffle_bag=False, **audio_kwargs)
    test_ds = VoiceDataset(test_paths, test_labels, shuffle_bag=False, **audio_kwargs)

    return train_ds, val_ds, test_ds
