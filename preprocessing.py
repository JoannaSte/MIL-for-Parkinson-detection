import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.transforms as transforms


SAMPLE_RATE = 22050
SEGMENT_DURATION = 0.25 
HOP_DURATION = 0.125     # seconds between segment starts (50% overlap)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
IMG_SIZE = 128
SPEC_TYPE = "mel"        # "mel", "linear", "mfcc"
N_MFCC = 40


def load_audio(path, target_sr=SAMPLE_RATE):
    """Load WAV file with soundfile (no FFmpeg needed), resample, convert to mono."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # data: (samples, channels) → tensor [channels, samples]
    waveform = torch.from_numpy(data.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform  # [1, num_samples]


def segment_audio(waveform, sample_rate=SAMPLE_RATE,
                  segment_duration=SEGMENT_DURATION, hop_duration=HOP_DURATION):
    """Split waveform into overlapping fixed-length segments.

    If the recording is shorter than one segment, it is zero-padded.
    Returns list of tensors each shaped [1, segment_samples].
    """
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)
    total_samples = waveform.shape[1]

    segments = []
    start = 0
    while start + segment_samples <= total_samples:
        segments.append(waveform[:, start:start + segment_samples])
        start += hop_samples

    if not segments:
        pad_size = segment_samples - total_samples
        waveform = F.pad(waveform, (0, pad_size))
        segments.append(waveform)

    return segments


def compute_spectrogram(segment, sample_rate=SAMPLE_RATE,
                        spec_type=SPEC_TYPE,
                        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                        n_mfcc=N_MFCC,
                        freq_min=None, freq_max=None,
                        time_start=None, time_end=None):
    """Convert an audio segment to a 2-D spectrogram tensor.

    spec_type:
        "mel"    – log-mel spectrogram; freq axis = mel bins.
        "linear" – log-power spectrogram; freq axis = Hz bins (FFT).
        "mfcc"   – Mel-Frequency Cepstral Coefficients.

    freq_min / freq_max (Hz):
        "mel"    – sets the filter-bank lower / upper bound.
        "linear" – slices the FFT frequency bins after computing the spectrogram.
        "mfcc"   – ignored (cepstral coefficients have no Hz axis).

    time_start / time_end (seconds):
        Crops the time axis of the resulting spectrogram for all spec types.

    Returns tensor [1, freq_bins, time_frames].
    """
    if spec_type == "mel":
        f_min = freq_min if freq_min is not None else 20.0
        f_max = freq_max if freq_max is not None else 8000.0
        spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )(segment)
        spec = T.AmplitudeToDB(top_db=80)(spec)

    elif spec_type == "linear":
        spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)(segment)
        spec = T.AmplitudeToDB(top_db=80)(spec)
        if freq_min is not None or freq_max is not None:
            total_bins = spec.shape[1]   # n_fft // 2 + 1
            nyquist = sample_rate / 2.0
            lo = int((freq_min or 0.0) / nyquist * total_bins)
            hi = int((freq_max or nyquist) / nyquist * total_bins) + 1
            spec = spec[:, lo:min(hi, total_bins), :]

    elif spec_type == "mfcc":
        spec = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels},
        )(segment)

    else:
        raise ValueError(f"Unknown spec_type {spec_type!r}. Choose 'mel', 'linear', or 'mfcc'.")

    # Time crop (applied after computing the spectrogram)
    if time_start is not None or time_end is not None:
        total_frames = spec.shape[2]
        frame_start = int((time_start or 0.0) * sample_rate / hop_length)
        frame_end = (int(time_end * sample_rate / hop_length)
                     if time_end is not None else total_frames)
        frame_end = min(frame_end, total_frames)
        spec = spec[:, :, max(frame_start, 0):frame_end]

    return spec  # [1, freq_bins, time_frames]


def spectrogram_to_tensor(spec, img_size=IMG_SIZE):
    """Normalize spectrogram to [0, 1] and resize to square img_size x img_size."""
    s_min = spec.min()
    s_max = spec.max()
    spec = (spec - s_min) / (s_max - s_min + 1e-8)
    spec = transforms.Resize((img_size, img_size), antialias=True)(spec)
    return spec  # [1, img_size, img_size]


def wav_to_spectrogram_bag(wav_path, sample_rate=SAMPLE_RATE,
                           segment_duration=SEGMENT_DURATION, hop_duration=HOP_DURATION,
                           spec_type=SPEC_TYPE,
                           n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                           n_mfcc=N_MFCC,
                           freq_min=None, freq_max=None,
                           time_start=None, time_end=None,
                           img_size=IMG_SIZE):
    """Full pipeline: WAV file → bag of spectrogram tensors.

    Returns tensor [num_segments, 1, img_size, img_size].
    """
    waveform = load_audio(wav_path, sample_rate)
    segments = segment_audio(waveform, sample_rate, segment_duration, hop_duration)

    bag = []
    for segment in segments:
        spec = compute_spectrogram(
            segment, sample_rate,
            spec_type=spec_type,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            n_mfcc=n_mfcc,
            freq_min=freq_min, freq_max=freq_max,
            time_start=time_start, time_end=time_end,
        )
        spec = spectrogram_to_tensor(spec, img_size)
        bag.append(spec)

    return torch.stack(bag)  # [num_segments, 1, img_size, img_size]