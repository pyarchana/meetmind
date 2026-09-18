"""
Log mel features for the wake word detector.

Shaped around how the browser actually delivers audio. The mic fires every
1024 samples (64ms at 16kHz), so the detector cannot wait for a tidy one
second buffer to arrive whole. It keeps a rolling window and recomputes on
each callback, which is what RollingWindow does.

No librosa or torchaudio here. The mel filterbank is about thirty lines, and
pulling a large dependency into a component whose whole selling point is
running cheaply on every mic frame would be a poor trade.
"""

import numpy as np

SAMPLE_RATE = 16000
MIC_FRAME = 1024          # what audio.js sends per callback, 64ms
WINDOW_SECONDS = 1.0      # how much history the detector looks at
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

FFT_SIZE = 512
HOP_SAMPLES = 160         # 10ms, the usual speech hop
WIN_SAMPLES = 400         # 25ms analysis window
N_MELS = 40
FMIN = 20.0
FMAX = SAMPLE_RATE / 2


def hz_to_mel(hz):
    """HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=float) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(n_mels=N_MELS, fft_size=FFT_SIZE, sample_rate=SAMPLE_RATE,
                   fmin=FMIN, fmax=FMAX):
    """
    Triangular filters, evenly spaced on the mel scale.

    Returns (n_mels, fft_size // 2 + 1). Each row is one filter, peaking at
    its centre frequency and falling to zero at its neighbours' centres.
    """
    n_bins = fft_size // 2 + 1
    bin_hz = np.linspace(0, sample_rate / 2, n_bins)

    edges_mel = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    edges_hz = mel_to_hz(edges_mel)

    bank = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left, centre, right = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]

        rising = (bin_hz - left) / max(centre - left, 1e-9)
        falling = (right - bin_hz) / max(right - centre, 1e-9)
        bank[m] = np.clip(np.minimum(rising, falling), 0.0, None)

    return bank


_BANK = mel_filterbank()
_WINDOW = np.hanning(WIN_SAMPLES).astype(np.float32)


def frame_signal(samples, frame_length=WIN_SAMPLES, hop=HOP_SAMPLES):
    """Slice into overlapping analysis frames. Drops any ragged tail."""
    samples = np.asarray(samples, dtype=np.float32)
    if len(samples) < frame_length:
        return np.zeros((0, frame_length), dtype=np.float32)

    count = 1 + (len(samples) - frame_length) // hop
    indices = np.arange(frame_length)[None, :] + hop * np.arange(count)[:, None]
    return samples[indices]


def log_mel(samples, bank=None):
    """
    Log mel spectrogram, shape (frames, n_mels).

    Floored before the log so silence lands on a finite constant rather than
    negative infinity, which would poison any model trained on it.
    """
    bank = _BANK if bank is None else bank

    frames = frame_signal(samples)
    if len(frames) == 0:
        return np.zeros((0, bank.shape[0]), dtype=np.float32)

    spectrum = np.fft.rfft(frames * _WINDOW, n=FFT_SIZE, axis=1)
    power = np.abs(spectrum, dtype=np.float32) ** 2

    energy = power @ bank.T
    return np.log(np.maximum(energy, 1e-10)).astype(np.float32)


class RollingWindow:
    """
    Keeps the last WINDOW_SAMPLES of audio so features can be recomputed on
    every mic callback.

    Starts zero filled rather than empty, so the very first callback produces
    a full sized feature instead of a short one. A detector that changes input
    shape for the first second of a session is a detector that fails exactly
    when somebody is most likely to speak.
    """

    def __init__(self, window_samples=WINDOW_SAMPLES):
        self.window_samples = window_samples
        self.buffer = np.zeros(window_samples, dtype=np.float32)

    def push(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if len(frame) >= self.window_samples:
            self.buffer = frame[-self.window_samples:].copy()
        else:
            self.buffer = np.concatenate([self.buffer[len(frame):], frame])
        return self.buffer

    def features(self):
        return log_mel(self.buffer)

    def reset(self):
        self.buffer[:] = 0.0
