from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import soundfile as sf
except ImportError as exc:
    raise SystemExit(
        "soundfile is required. Install it with `pip install soundfile`."
    ) from exc


# =============================================================================
# 1.  Band-splitting schemes (paper Sec. V-A)
# =============================================================================

# (start_hz, end_hz, bandwidth_hz) — vocals/other use the V7 scheme (41 bands).
_RAW_SCHEMES: Dict[str, List[Tuple[float, float, float]]] = {
    "vocals": [
        (0,     1000,  100),
        (1000,  4000,  250),
        (4000,  8000,  500),
        (8000,  16000, 1000),
        (16000, 20000, 2000),
    ],
    "bass": [
        (0,     500,   50),
        (500,   1000,  100),
        (1000,  4000,  500),
        (4000,  8000,  1000),
        (8000,  16000, 2000),
    ],
    "drums": [
        (0,     1000,  50),
        (1000,  2000,  100),
        (2000,  4000,  250),
        (4000,  8000,  500),
        (8000,  16000, 1000),
    ],
}
_RAW_SCHEMES["other"] = _RAW_SCHEMES["vocals"]


def get_band_split(target: str, sr: int = 44100, n_fft: int = 2048) -> List[int]:
    """Return per-subband bin counts whose sum is exactly n_fft//2 + 1."""
    if target not in _RAW_SCHEMES:
        raise ValueError(
            f"Unknown target '{target}'. Choose from {list(_RAW_SCHEMES)}.")

    nyquist = sr / 2
    bin_hz = sr / n_fft
    n_bins = n_fft // 2 + 1

    bandwidths_hz: List[float] = []
    cursor = 0.0
    for start, end, bw in _RAW_SCHEMES[target]:
        end = min(end, nyquist)
        if start >= end:
            continue
        f = start
        while f < end - 1e-6:
            step = min(bw, end - f)
            bandwidths_hz.append(step)
            f += step
        cursor = end
    if cursor < nyquist:
        bandwidths_hz.append(nyquist - cursor)

    bin_counts: List[int] = []
    accumulated = 0.0
    for bw in bandwidths_hz:
        accumulated += bw / bin_hz
        take = int(round(accumulated)) - sum(bin_counts)
        if take > 0:
            bin_counts.append(take)

    diff = n_bins - sum(bin_counts)
    if diff != 0:
        bin_counts[-1] += diff
    assert sum(bin_counts) == n_bins
    assert all(b > 0 for b in bin_counts)
    return bin_counts


# =============================================================================
# 2.  Model
# =============================================================================


class BandSplit(nn.Module):
    """Spectrogram → per-band feature tensor (B, N, K, T)."""

    def __init__(self, band_widths: List[int], n_channels: int, feature_dim: int):
        super().__init__()
        self.band_widths = band_widths
        self.n_channels = n_channels
        self.feature_dim = feature_dim
        self.norms = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for w in band_widths:
            in_dim = n_channels * 2 * w
            self.norms.append(nn.GroupNorm(1, in_dim))  # acts as LayerNorm over `in_dim`
            self.fcs.append(nn.Linear(in_dim, feature_dim))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(spec)
        B, C, F_, T = spec.shape
        assert C == self.n_channels
        assert F_ == sum(self.band_widths)

        outs = []
        offset = 0
        for w, norm, fc in zip(self.band_widths, self.norms, self.fcs):
            sub = spec[:, :, offset:offset + w, :]
            offset += w
            sub = torch.view_as_real(sub)                       # (B, C, w, T, 2)
            sub = sub.permute(0, 3, 1, 2, 4).contiguous()       # (B, T, C, w, 2)
            sub = sub.view(B, T, C * w * 2)
            sub_t = norm(sub.transpose(1, 2)).transpose(1, 2)   # (B, T, C*w*2)
            outs.append(fc(sub_t))                              # (B, T, N)

        z = torch.stack(outs, dim=1)                            # (B, K, T, N)
        return z.permute(0, 3, 1, 2).contiguous()               # (B, N, K, T)


class ResidualRNN(nn.Module):
    """LayerNorm → BLSTM → Linear (with residual). Operates on the last axis."""

    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or 2 * feature_dim
        self.norm = nn.LayerNorm(feature_dim)
        self.rnn = nn.LSTM(feature_dim, hidden, num_layers=1,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y, _ = self.rnn(self.norm(x))
        return residual + self.fc(y)


class SequenceBandBlock(nn.Module):
    """One BSRNN block: a sequence-axis RNN followed by a band-axis RNN."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.seq_rnn = ResidualRNN(feature_dim)
        self.band_rnn = ResidualRNN(feature_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, N, K, T = z.shape
        # sequence modelling across T (RNN shared across K bands)
        x = z.permute(0, 2, 3, 1).contiguous().view(B * K, T, N)
        x = self.seq_rnn(x)
        x = x.view(B, K, T, N).permute(0, 3, 1, 2).contiguous()
        # band modelling across K (RNN shared across T frames)
        y = x.permute(0, 3, 2, 1).contiguous().view(B * T, K, N)
        y = self.band_rnn(y)
        return y.view(B, T, K, N).permute(0, 3, 2, 1).contiguous()


class MaskEstimation(nn.Module):
    """Per-subband MLP → complex T-F mask. Uses tanh + GLU output (paper Sec. II-C)."""

    def __init__(self, band_widths: List[int], n_channels: int,
                 feature_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or 4 * feature_dim
        self.band_widths = band_widths
        self.n_channels = n_channels
        self.norms = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for w in band_widths:
            out_dim = n_channels * 2 * w
            self.norms.append(nn.LayerNorm(feature_dim))
            self.mlps.append(nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, out_dim * 2),  # GLU halves the last dim back to out_dim
                nn.GLU(dim=-1),
            ))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        B, N, K, T = q.shape
        q = q.permute(0, 2, 3, 1).contiguous()  # (B, K, T, N)
        masks = []
        for k, (w, norm, mlp) in enumerate(zip(self.band_widths, self.norms, self.mlps)):
            qk = norm(q[:, k])                                  # (B, T, N)
            mk = mlp(qk).view(B, T, self.n_channels, w, 2)
            mk = mk.permute(0, 2, 3, 1, 4).contiguous()         # (B, C, w, T, 2)
            masks.append(torch.view_as_complex(mk))             # (B, C, w, T)
        return torch.cat(masks, dim=2)                          # (B, C, F, T)


class BSRNN(nn.Module):
    """Full band-split RNN model with internal STFT/iSTFT."""

    def __init__(self,
                 band_widths: List[int],
                 n_channels: int = 2,
                 feature_dim: int = 128,
                 num_layers: int = 12,
                 n_fft: int = 2048,
                 hop_length: int = 512):
        super().__init__()
        assert sum(band_widths) == n_fft // 2 + 1
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_channels = n_channels
        self.band_widths = band_widths
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        self.band_split = BandSplit(band_widths, n_channels, feature_dim)
        self.blocks = nn.ModuleList(
            [SequenceBandBlock(feature_dim) for _ in range(num_layers)]
        )
        self.mask_est = MaskEstimation(band_widths, n_channels, feature_dim)

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        B, C, S = wav.shape
        x = wav.reshape(B * C, S)
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window, return_complex=True, center=True)
        return spec.view(B, C, *spec.shape[-2:])

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        B, C, F_, T = spec.shape
        wav = torch.istft(spec.reshape(B * C, F_, T),
                          n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window, length=length, center=True)
        return wav.view(B, C, length)

    def forward(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        length = mixture.shape[-1]
        mix_spec = self.stft(mixture)
        z = self.band_split(mix_spec)
        for block in self.blocks:
            z = block(z)
        mask = self.mask_est(z)
        out_spec = mask * mix_spec
        out_wav = self.istft(out_spec, length=length)
        return {"waveform": out_wav, "spec": out_spec,
                "mask": mask, "mix_spec": mix_spec}


# =============================================================================
# 3.  Loss + metrics
# =============================================================================


class BSRNNLoss(nn.Module):
    """L1(real)+L1(imag)+L1(|spec|) on the spectrogram + L1 on the waveform.

    Adds a magnitude term to the original Eq. 1 because phase-only L1 under-weights
    high-energy bins and stalls SDR; magnitude L1 is the standard upgrade in MSS.
    """

    def __init__(self, time_weight: float = 1.0, spec_weight: float = 1.0,
                 mag_weight: float = 1.0):
        super().__init__()
        self.time_weight = time_weight
        self.spec_weight = spec_weight
        self.mag_weight = mag_weight

    def forward(self, pred_spec, pred_wav, target_spec, target_wav):
        spec_loss = (pred_spec.real - target_spec.real).abs().mean() \
                  + (pred_spec.imag - target_spec.imag).abs().mean()
        mag_loss = (pred_spec.abs() - target_spec.abs()).abs().mean()
        time_loss = (pred_wav - target_wav).abs().mean()
        return (self.spec_weight * spec_loss
                + self.mag_weight * mag_loss
                + self.time_weight * time_loss)


_EPS = 1e-8


def _sdr(target: np.ndarray, estimate: np.ndarray) -> float:
    num = float((target ** 2).sum())
    den = float(((target - estimate) ** 2).sum()) + _EPS
    if num < _EPS:
        return float("nan")
    return 10.0 * math.log10(num / den + _EPS)


def usdr(target: np.ndarray, estimate: np.ndarray) -> float:
    """Utterance-level SDR (MDX 2021 definition)."""
    return _sdr(target, estimate)


def csdr(target: np.ndarray, estimate: np.ndarray,
         sr: int = 44100, chunk_seconds: float = 1.0) -> float:
    """Median SDR over non-overlapping 1-second chunks (SiSEC convention)."""
    chunk = int(round(sr * chunk_seconds))
    n_chunks = target.shape[-1] // chunk
    if n_chunks == 0:
        return _sdr(target, estimate)
    scores = []
    for i in range(n_chunks):
        v = _sdr(target[..., i * chunk:(i + 1) * chunk],
                 estimate[..., i * chunk:(i + 1) * chunk])
        if not math.isnan(v):
            scores.append(v)
    return float(np.median(scores)) if scores else float("nan")


# =============================================================================
# 4.  Source activity detector (paper Sec. IV-A1)
# =============================================================================


def find_salient_segments(audio: np.ndarray, sr: int,
                          segment_seconds: float = 6.0,
                          overlap: float = 0.5,
                          min_active_ratio: float = 0.5) -> List[Tuple[int, int]]:
    """Return [(start, end)] sample-indexed salient segments."""
    if audio.ndim == 1:
        audio = audio[None, :]
    n_samples = audio.shape[-1]
    seg = int(round(segment_seconds * sr))
    hop = max(1, int(round(seg * (1 - overlap))))
    if n_samples < seg:
        return []

    sub = seg // 10
    n_chunks = n_samples // sub
    chunk_db = np.full(n_chunks, -50.0, dtype=np.float64)
    for i in range(n_chunks):
        e = float(((audio[..., i * sub:(i + 1) * sub]).astype(np.float64) ** 2).mean())
        if e > 1e-10:
            chunk_db[i] = 10.0 * math.log10(e + 1e-12)

    above = chunk_db[chunk_db > -30.0]
    if above.size == 0:
        return []
    threshold = float(np.quantile(above, 0.15))

    chunks_per_seg = seg // sub
    segs: List[Tuple[int, int]] = []
    for start in range(0, n_samples - seg + 1, hop):
        first = start // sub
        last = first + chunks_per_seg
        if last > n_chunks:
            break
        active = float((chunk_db[first:last] > threshold).mean())
        if active > min_active_ratio:
            segs.append((start, start + seg))
    return segs


# =============================================================================
# 5.  MoisesDB dataset (built on top of the official `moisesdb` package)
# =============================================================================

# Map MoisesDB stem categories → 4-stem MUSDB-style targets. The keys are the
# stem names exposed by `MoisesDBTrack.stems`.
_MOISES_STEM_MAP = {
    "vocals":            "vocals",
    "bass":              "bass",
    "drums":             "drums",
    "percussion":        "drums",
    "guitar":            "other",
    "piano":             "other",
    "other_keys":        "other",
    "bowed_strings":     "other",
    "plucked_strings":   "other",
    "wind":              "other",
    "other_plucked":     "other",
    "other":             "other",
    "fx":                "other",
}
TARGETS = ("vocals", "bass", "drums", "other")


def _ensure_stereo(wav: np.ndarray) -> np.ndarray:
    """Coerce an array to shape (2, samples), float32."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[None, :]
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav


def _load_audio(path: Path, sr: int) -> np.ndarray:
    """Load a wav/flac file to (2, samples) float32 at the requested SR."""
    wav, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.T  # (channels, samples)
    if file_sr != sr:
        try:
            import resampy
        except ImportError as exc:
            raise SystemExit(
                f"{path} is at {file_sr} Hz; install `resampy` to enable resampling, "
                f"or pre-resample your data to {sr} Hz."
            ) from exc
        wav = resampy.resample(wav, file_sr, sr, axis=-1).astype(np.float32)
    return _ensure_stereo(wav)


@dataclass
class SongStems:
    song_id: str
    stems: Dict[str, np.ndarray]  # target -> (2, samples)


def _open_db(data_path: Path, sr: int):
    """Instantiate a MoisesDB object from the official `moisesdb` package."""
    try:
        from moisesdb.dataset import MoisesDB
    except ImportError as exc:
        raise SystemExit(
            "The `moisesdb` package is required. Install it with\n"
            "    pip install git+https://github.com/moises-ai/moises-db.git\n"
            "and point --data at the directory holding moisesdb_v0.1/."
        ) from exc
    return MoisesDB(data_path=str(data_path), sample_rate=sr)


def _track_to_song(track, sr: int) -> Optional[SongStems]:
    """Convert one `MoisesDBTrack` into a SongStems with the 4-stem targets."""
    accum: Dict[str, Optional[np.ndarray]] = {t: None for t in TARGETS}
    try:
        stems_items = track.stems.items()
    except Exception as e:
        print(f"[warn] skipping track (failed to read stems): {e}", flush=True)
        return None
    for stem_name, audio in stems_items:
        target = _MOISES_STEM_MAP.get(stem_name.lower())
        if target is None:
            continue
        try:
            signal = _ensure_stereo(audio)
        except (FileNotFoundError, OSError) as e:
            print(f"[warn] skipping stem '{stem_name}' (missing file): {e}", flush=True)
            continue
        except Exception as e:
            print(f"[warn] skipping stem '{stem_name}' (error): {e}", flush=True)
            continue
        if accum[target] is None:
            accum[target] = signal
        else:
            ref = accum[target]
            if signal.shape[-1] > ref.shape[-1]:
                ref = np.pad(ref, ((0, 0), (0, signal.shape[-1] - ref.shape[-1])))
            elif signal.shape[-1] < ref.shape[-1]:
                signal = np.pad(signal, ((0, 0), (0, ref.shape[-1] - signal.shape[-1])))
            accum[target] = ref + signal

    present = [v for v in accum.values() if v is not None]
    if not present:
        return None
    max_len = max(v.shape[-1] for v in present)
    for k in TARGETS:
        if accum[k] is None:
            accum[k] = np.zeros((2, max_len), dtype=np.float32)
        elif accum[k].shape[-1] < max_len:
            accum[k] = np.pad(accum[k], ((0, 0), (0, max_len - accum[k].shape[-1])))
    song_id = getattr(track, "id", None) or getattr(track, "name", None) or "song"
    return SongStems(song_id=str(song_id), stems={k: accum[k] for k in TARGETS})


def discover_songs(data_path: Path, sr: int = 44100) -> List[int]:
    """Return the list of track indices in the MoisesDB at `data_path`."""
    db = _open_db(data_path, sr)
    return list(range(len(db)))


def split_songs(indices: Sequence[int], val_frac: float = 0.10,
                test_frac: float = 0.10, seed: int = 0
                ) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic song-level train / val / test split over track indices."""
    idxs = list(indices)
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n = len(idxs)
    n_test = max(1, int(round(n * test_frac))) if n > 1 else 0
    n_val = max(1, int(round(n * val_frac))) if n > 2 else 0
    test = idxs[:n_test]
    val = idxs[n_test:n_test + n_val]
    train = idxs[n_test + n_val:]
    return train, val, test


def load_songs(data_path: Path, indices: Sequence[int], sr: int) -> List[SongStems]:
    """Materialise a list of SongStems for the given track indices."""
    db = _open_db(data_path, sr)
    songs: List[SongStems] = []
    loop = tqdm(indices)
    loop.set_description(f"loading songs from path: {data_path}")
    for i in loop:
        try:
            song = _track_to_song(db[i], sr)
            if song is not None:
                songs.append(song)
        except (FileNotFoundError, OSError) as e:
            print(f"[warn] skipping track {i} (missing file): {e}", flush=True)
        except Exception as e:
            print(f"[warn] skipping track {i} (error): {e}", flush=True)
    return songs


class MoisesDBSegments(Dataset):
    """On-the-fly segment sampler with mix-and-augment data simulation.

    Each `__getitem__` returns a (mixture, target) pair where the mixture is
    built by summing a random chunk from the target track and random chunks
    of the residual sources, drawn from *different* songs to amplify
    diversity (Uhlich et al., 2017).
    """

    def __init__(self, songs: Sequence[SongStems], target: str,
                 sr: int = 44100, segment_seconds: float = 3.0,
                 samples_per_epoch: int = 10000,
                 sad_segment_seconds: float = 6.0,
                 gain_db_range: Tuple[float, float] = (-10.0, 10.0),
                 dropout_prob: float = 0.1):
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}")
        if not songs:
            raise RuntimeError("MoisesDBSegments received an empty song list.")
        self.target = target
        self.sr = sr
        self.segment = int(round(segment_seconds * sr))
        self.samples_per_epoch = samples_per_epoch
        self.gain_db_range = gain_db_range
        self.dropout_prob = dropout_prob
        self._songs: List[SongStems] = list(songs)
        self._salient: List[Dict[str, List[Tuple[int, int]]]] = [
            {t: find_salient_segments(s.stems[t], sr,
                                      segment_seconds=sad_segment_seconds)
             for t in TARGETS}
            for s in self._songs
        ]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_chunk(self, target: str) -> np.ndarray:
        """Random salient chunk of `self.segment` samples for this target."""
        for _ in range(20):
            idx = random.randrange(len(self._songs))
            song = self._songs[idx]
            wav = song.stems[target]
            sals = self._salient[idx][target]
            if not sals:
                # fall back to any window
                if wav.shape[-1] < self.segment:
                    continue
                start = random.randint(0, wav.shape[-1] - self.segment)
            else:
                seg_start, seg_end = random.choice(sals)
                seg_end = min(seg_end, wav.shape[-1])
                if seg_end - self.segment < seg_start:
                    start = seg_start
                else:
                    start = random.randint(seg_start, seg_end - self.segment)
            return wav[:, start:start + self.segment].astype(np.float32)
        # last-resort silence
        return np.zeros((2, self.segment), dtype=np.float32)

    def _apply_aug(self, chunk: np.ndarray) -> np.ndarray:
        if random.random() < self.dropout_prob:
            return np.zeros_like(chunk)
        gain_db = random.uniform(*self.gain_db_range)
        chunk = chunk * (10 ** (gain_db / 20.0))
        if chunk.shape[0] == 2 and random.random() < 0.5:
            chunk = chunk[::-1].copy()  # L/R swap
        if random.random() < 0.5:
            chunk = -chunk  # polarity flip
        return chunk

    def __getitem__(self, _idx: int) -> Dict[str, torch.Tensor]:
        target_chunk = self._apply_aug(self._sample_chunk(self.target))
        residual = np.zeros_like(target_chunk)
        for other in TARGETS:
            if other == self.target:
                continue
            residual = residual + self._apply_aug(self._sample_chunk(other))
        mixture = target_chunk + residual

        # rescale together so the mixture is in [-1, 1]
        peak = float(np.max(np.abs(mixture))) + 1e-8
        if peak > 1.0:
            mixture = mixture / peak
            target_chunk = target_chunk / peak

        return {
            "mixture": torch.from_numpy(mixture).float(),
            "target":  torch.from_numpy(target_chunk).float(),
        }


# =============================================================================
# 6.  Inference (overlap-add)
# =============================================================================


@torch.no_grad()
def separate(model: BSRNN, wav: torch.Tensor, *, sr: int,
             segment_seconds: float = 3.0, hop_seconds: float = 1.5,
             device: Optional[torch.device] = None) -> torch.Tensor:
    """Overlap-add separation for a long (channels, samples) waveform."""
    device = device or next(model.parameters()).device
    model.eval()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(model.n_channels, 1)

    seg = int(round(segment_seconds * sr))
    hop = int(round(hop_seconds * sr))
    pad = seg - hop  # reflective context at both edges
    padded = F.pad(wav.unsqueeze(0), (pad, pad + seg)).squeeze(0)
    total = padded.shape[-1]

    # triangular synthesis window for smooth overlap-add
    window = torch.bartlett_window(seg, periodic=False, device=device)
    out = torch.zeros_like(padded, device=device)
    weight = torch.zeros(padded.shape[-1], device=device)

    start = 0
    while start + seg <= total:
        chunk = padded[:, start:start + seg].to(device).unsqueeze(0)
        est = model(chunk)["waveform"].squeeze(0)               # (C, seg)
        out[:, start:start + seg] += est * window
        weight[start:start + seg] += window
        start += hop

    out = out / weight.clamp_min(1e-8)
    out = out[:, pad:pad + wav.shape[-1]]
    return out.cpu()


# =============================================================================
# 7.  Training
# =============================================================================


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "mixture": torch.stack([b["mixture"] for b in batch], dim=0),
        "target":  torch.stack([b["target"]  for b in batch], dim=0),
    }


def train_one_epoch(model, loader, loss_fn, optimizer, device, *,
                    grad_clip: float = 5.0, log_every: int = 50) -> float:
    model.train()
    running = 0.0
    n = 0
    for step, batch in enumerate(loader):
        mixture = batch["mixture"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        out = model(mixture)
        target_spec = model.stft(target)
        loss = loss_fn(out["spec"], out["waveform"], target_spec, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running += loss.item() * mixture.size(0)
        n += mixture.size(0)
        if step % log_every == 0:
            print(f"  step {step:5d}/{len(loader)}  loss={loss.item():.4f}",
                  flush=True)
    return running / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        mixture = batch["mixture"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        out = model(mixture)
        target_spec = model.stft(target)
        loss = loss_fn(out["spec"], out["waveform"], target_spec, target)
        total += loss.item() * mixture.size(0)
        n += mixture.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_songs(model, songs: Sequence[SongStems], target: str, *,
                   sr: int, device, segment_seconds: float = 3.0,
                   hop_seconds: float = 1.5) -> Dict[str, float]:
    """Run separation on each song and report mean uSDR / median cSDR."""
    usdrs, csdrs = [], []
    for song in songs:
        mixture = sum(song.stems[t] for t in TARGETS)
        mix_t = torch.from_numpy(mixture).float()
        est = separate(model, mix_t, sr=sr, segment_seconds=segment_seconds,
                       hop_seconds=hop_seconds, device=device).numpy()
        ref = song.stems[target]
        L = min(ref.shape[-1], est.shape[-1])
        usdrs.append(usdr(ref[:, :L], est[:, :L]))
        csdrs.append(csdr(ref[:, :L], est[:, :L], sr=sr))
        print(f"  {song.song_id:30s}  uSDR={usdrs[-1]:6.2f}  cSDR={csdrs[-1]:6.2f}",
              flush=True)

    return {
        "mean_uSDR":  float(np.nanmean(usdrs)) if usdrs else float("nan"),
        "median_cSDR": float(np.nanmedian(csdrs)) if csdrs else float("nan"),
        "n_songs": len(usdrs),
    }


def build_model(target: str, *, sr: int, n_fft: int, hop_length: int,
                feature_dim: int, num_layers: int, n_channels: int) -> BSRNN:
    bws = get_band_split(target, sr=sr, n_fft=n_fft)
    return BSRNN(band_widths=bws, n_channels=n_channels, feature_dim=feature_dim,
                 num_layers=num_layers, n_fft=n_fft, hop_length=hop_length)


def save_checkpoint(path: Path, model: BSRNN, optimizer, epoch: int,
                    val_loss: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "val_loss": val_loss,
        "config": {
            "target": args.target,
            "sr": args.sr,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "feature_dim": args.feature_dim,
            "num_layers": args.num_layers,
            "n_channels": args.n_channels,
            "band_widths": model.band_widths,
        },
    }, str(path))


def load_checkpoint(path: Path, device) -> Tuple[BSRNN, dict]:
    state = torch.load(str(path), map_location=device)
    cfg = state["config"]
    model = BSRNN(band_widths=cfg["band_widths"], n_channels=cfg["n_channels"],
                  feature_dim=cfg["feature_dim"], num_layers=cfg["num_layers"],
                  n_fft=cfg["n_fft"], hop_length=cfg["hop_length"]).to(device)
    model.load_state_dict(state["model_state"])
    return model, cfg

def plot_training_history(history_path: Path, output_path: Path) -> None:
    if not history_path.exists():
        print(f"[evaluate] no history file found at {history_path}; skipping history plot.",
              flush=True)
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib  is required but is not installed. {e}", flush=True)
        return

    with history_path.open() as fh:
        history = json.load(fh)

    if not history:
        print(f"[evaluate] history file is empty at {history_path}; skipping history plot.",
              flush=True)
        return

    epochs = np.array([row.get("epoch", idx + 1) for idx, row in enumerate(history)])
    skip_keys = {"epoch", "n_songs"}
    numeric_keys: List[str] = []
    for row in history:
        for key, value in row.items():
            if key in skip_keys or key in numeric_keys:
                continue
            if isinstance(value, (int, float)) or value is None:
                numeric_keys.append(key)

    loss_keys = [key for key in numeric_keys if "loss" in key.lower() or key == "train_loss"]
    score_keys = [key for key in numeric_keys if key not in loss_keys]

    if not loss_keys and not score_keys:
        print(f"[evaluate] no numeric history values found in {history_path}; skipping plot.",
              flush=True)
        return

    fig, ax_loss = plt.subplots(figsize=(10, 5))
    ax_score = ax_loss.twinx() if score_keys else None

    for key in loss_keys:
        values = np.array([row.get(key, np.nan) for row in history], dtype=float)
        ax_loss.plot(epochs, values, marker="o", linewidth=1.8, label=key)

    for key in score_keys:
        values = np.array([row.get(key, np.nan) for row in history], dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        ax_score.plot(
            epochs[finite],
            values[finite],
            marker="s",
            linestyle="--",
            linewidth=1.6,
            label=key,
        )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    if ax_score is not None:
        ax_score.set_ylabel("Validation score (dB)")

    handles, labels = ax_loss.get_legend_handles_labels()
    if ax_score is not None:
        score_handles, score_labels = ax_score.get_legend_handles_labels()
        handles.extend(score_handles)
        labels.extend(score_labels)
    if handles:
        ax_loss.legend(handles, labels, loc="best")

    fig.suptitle("Training History")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] saved history plot to {output_path}", flush=True)

# =============================================================================
# 8.  CLI
# =============================================================================


def cmd_train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[train] target={args.target} device={device}", flush=True)

    indices = discover_songs(Path(args.data), sr=args.sr)
    if args.max_songs:
        indices = indices[:args.max_songs]
    if not indices:
        sys.exit(f"No songs found under {args.data}")
    train_idx, val_idx, test_idx = split_songs(
        indices, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    print(f"[train] split: train={len(train_idx)}  val={len(val_idx)}  "
          f"test={len(test_idx)}", flush=True)

    print("[train] preloading songs via moisesdb...", flush=True)
    train_songs = load_songs(Path(args.data), train_idx, sr=args.sr)
    val_songs = load_songs(Path(args.data), val_idx, sr=args.sr)

    train_set = MoisesDBSegments(
        train_songs, target=args.target, sr=args.sr,
        segment_seconds=args.segment,
        samples_per_epoch=args.samples_per_epoch)
    val_set = MoisesDBSegments(
        val_songs, target=args.target, sr=args.sr,
        segment_seconds=args.segment,
        samples_per_epoch=max(64, args.samples_per_epoch // 20))

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate, pin_memory=True)

    model = build_model(args.target, sr=args.sr, n_fft=args.n_fft,
                        hop_length=args.hop_length, feature_dim=args.feature_dim,
                        num_layers=args.num_layers, n_channels=args.n_channels
                        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] BSRNN params: {n_params/1e6:.2f}M  "
          f"subbands: {len(model.band_widths)}", flush=True)

    loss_fn = BSRNNLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # Cosine annealing over the full epoch budget — flat enough at the start
    # to actually fit, then decays to ~lr/100 instead of crashing in 2 epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf")  # we now select on val uSDR (higher = better)
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n[epoch {epoch}/{args.epochs}]", flush=True)
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                     device, grad_clip=args.grad_clip,
                                     log_every=args.log_every)
        val_loss = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        # Per-epoch SDR on val songs — loss is a poor proxy for SDR.
        val_sdr_metrics = {"mean_uSDR": float("nan"), "median_cSDR": float("nan")}
        if val_songs and (epoch % args.sdr_every == 0 or epoch == args.epochs):
            sdr_subset = val_songs[:args.sdr_max_songs] if args.sdr_max_songs else val_songs
            print(f"[epoch {epoch}] computing SDR on {len(sdr_subset)} val song(s)...",
                  flush=True)
            val_sdr_metrics = evaluate_songs(
                model, sdr_subset, target=args.target, sr=args.sr, device=device,
                segment_seconds=args.segment, hop_seconds=args.segment / 2)

        print(f"[epoch {epoch}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"val_uSDR={val_sdr_metrics['mean_uSDR']:.3f}  "
              f"val_cSDR={val_sdr_metrics['median_cSDR']:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_loss,
                        "val_uSDR": val_sdr_metrics["mean_uSDR"],
                        "val_cSDR": val_sdr_metrics["median_cSDR"]})
        with (out_dir / "history.json").open("w") as fh:
            json.dump(history, fh, indent=2)

        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, val_loss, args)
        # Pick best by uSDR when available; fall back to -val_loss otherwise.
        current = val_sdr_metrics["mean_uSDR"]
        if not math.isfinite(current):
            current = -val_loss
        if current > best_metric:
            best_metric = current
            no_improve = 0
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch,
                            val_loss, args)
            print(f"[epoch {epoch}] new best (metric={best_metric:.4f})", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[train] early stopping after {no_improve} stagnant epochs",
                      flush=True)
                break

    # remember the test-song split (track indices) so we can reuse it later
    with (out_dir / "splits.json").open("w") as fh:
        json.dump({
            "data_path": str(args.data),
            "sr": args.sr,
            "train": list(train_idx),
            "val":   list(val_idx),
            "test":  list(test_idx),
        }, fh, indent=2)
    print(f"[train] done. best val metric: {best_metric:.4f}", flush=True)


def cmd_separate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, cfg = load_checkpoint(Path(args.ckpt), device)
    sr = cfg["sr"]
    print(f"[separate] loaded {args.ckpt} (target={cfg['target']}, sr={sr})",
          flush=True)

    wav = _load_audio(Path(args.input), sr)
    wav_t = torch.from_numpy(wav).float()
    est = separate(model, wav_t, sr=sr,
                   segment_seconds=args.segment, hop_seconds=args.hop,
                   device=device).numpy()
    sf.write(args.output, est.T, sr)
    print(f"[separate] wrote {args.output}", flush=True)


def cmd_evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, cfg = load_checkpoint(Path(args.ckpt), device)
    sr = cfg["sr"]

    if args.songs_json:
        with open(args.songs_json) as fh:
            split = json.load(fh)
        data_path = Path(args.data or split.get("data_path"))
        if not data_path:
            sys.exit("Either --data or 'data_path' in splits.json must be set.")
        test_idx = list(split["test"])
    else:
        if not args.data:
            sys.exit("--data is required when --songs-json is not given.")
        data_path = Path(args.data)
        indices = discover_songs(data_path, sr=sr)
        _, _, test_idx = split_songs(
            indices, val_frac=args.val_frac, test_frac=args.test_frac,
            seed=args.seed)

    if args.max_songs:
        test_idx = test_idx[:args.max_songs]
    print(f"[evaluate] {len(test_idx)} test songs (loading via moisesdb)...",
          flush=True)
    songs = load_songs(data_path, test_idx, sr=sr)
    metrics = evaluate_songs(model, songs, target=cfg["target"], sr=sr,
                             device=device, segment_seconds=args.segment,
                             hop_seconds=args.hop)
    print(f"\n[evaluate] mean uSDR    = {metrics['mean_uSDR']:.3f} dB")
    print(f"[evaluate] median cSDR  = {metrics['median_cSDR']:.3f} dB")
    print(f"[evaluate] songs scored = {metrics['n_songs']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)

    history_path = out_dir / "history.json"
    history_plot = out_dir / "history_plot.png"
    plot_training_history(history_path, history_plot)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- shared model / audio knobs --------------------------------------
    def add_model_args(ap):
        ap.add_argument("--sr", type=int, default=44100)
        ap.add_argument("--n-fft", type=int, default=2048)
        ap.add_argument("--hop-length", type=int, default=512)
        ap.add_argument("--feature-dim", type=int, default=128)
        ap.add_argument("--num-layers", type=int, default=12)
        ap.add_argument("--n-channels", type=int, default=2)

    # train ----------------------------------------------------------------
    tr = sub.add_parser("train", help="train a BSRNN on MoisesDB")
    tr.add_argument("--data", required=True, type=str,
                    help="Path passed to moisesdb.dataset.MoisesDB(data_path=...). "
                         "Typically the directory containing moisesdb_v0.1/.")
    tr.add_argument("--target", choices=TARGETS, required=True)
    tr.add_argument("--out-dir", default="runs/bsrnn")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--segment", type=float, default=3.0)
    tr.add_argument("--samples-per-epoch", type=int, default=800)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=1e-2)
    tr.add_argument("--grad-clip", type=float, default=5.0)
    tr.add_argument("--patience", type=int, default=15)
    tr.add_argument("--sdr-every", type=int, default=2,
                    help="evaluate val SDR every N epochs (1 = every epoch)")
    tr.add_argument("--sdr-max-songs", type=int, default=0,
                    help="cap on val songs used for SDR (0 = all)")
    tr.add_argument("--num-workers", type=int, default=2)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--val-frac", type=float, default=0.10)
    tr.add_argument("--test-frac", type=float, default=0.10)
    tr.add_argument("--max-songs", type=int, default=0,
                    help="optional cap on the number of songs loaded (for quick runs)")
    tr.add_argument("--log-every", type=int, default=50)
    add_model_args(tr)
    tr.set_defaults(func=cmd_train)

    # separate -------------------------------------------------------------
    sp = sub.add_parser("separate", help="run inference on a wav file")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--segment", type=float, default=3.0)
    sp.add_argument("--hop", type=float, default=1.5)
    sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sp.set_defaults(func=cmd_separate)

    # evaluate -------------------------------------------------------------
    ev = sub.add_parser("evaluate", help="evaluate uSDR/cSDR on the test split")
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--data", default=None,
                    help="MoisesDB root (used if --songs-json is not given)")
    ev.add_argument("--songs-json", default=None,
                    help="splits.json saved during training")
    ev.add_argument("--out-dir", default="runs/bsrnn")
    ev.add_argument("--history-json", default=None,
                    help="history.json saved during training")
    ev.add_argument("--history-plot", default=None,
                    help="path to save history during training")
    ev.add_argument("--segment", type=float, default=3.0)
    ev.add_argument("--hop", type=float, default=1.5)
    ev.add_argument("--max-songs", type=int, default=0)
    ev.add_argument("--val-frac", type=float, default=0.10)
    ev.add_argument("--test-frac", type=float, default=0.10)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ev.set_defaults(func=cmd_evaluate)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()