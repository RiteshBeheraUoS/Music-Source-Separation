from __future__ import annotations

# raise RuntimeError("THIS IS THE FILE BEING EXECUTED")
import warnings
warnings.filterwarnings('ignore')

try:
    import os
    from moisesdb.dataset import MoisesDB
    from moisesdb.track import MoisesDBTrack, pad_to_len
    from moisesdb.utils import save_audio
except ImportError as exc:
    raise SystemExit(
        "moisesdb is required. Install it with `pip install git+https://github.com/moises-ai/moises-db.git`."
    ) from exc

try:
    import museval
except ImportError as exc:
    raise SystemExit(
        "museval is required. Install it with `pip install museval`."
    ) from exc


try:
    from tqdm import tqdm
    import argparse, json, math, random, sys
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Dict, List, Optional, Sequence, Tuple
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "failed to import required basic packages. Please install them and try again."
    ) from exc


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
except ImportError as exc:
    raise SystemExit(
        "failed to import Pytorch libraries. Please install them 'pip install torch torchvision torchaudio'."
    ) from exc

try:
    import soundfile as sf
except ImportError as exc:
    raise SystemExit(
        "soundfile is required. Install it with `pip install soundfile`."
    ) from exc


# =============================================================================
# 1.  MoisesDB dataset (built on top of the official `moisesdb` package)
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
    mixture: torch.Tensor
    stems: Dict[str, torch.Tensor]


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
    accum: Dict[str, Optional[torch.Tensor]] = {t: None for t in TARGETS}

    # --- get mixture directly ---
    try:
        mixture = _ensure_stereo(track.audio)
    except Exception as e:
        print(f"[warn] skipping track {track.id} :'{track.name}' (audio failed): {e}", flush=True)
        return None

    # --- build targets from stems ---
    try:
        stems_items = track.stems.items()
    except Exception as e:
        print(f"[warn] skipping track {track.id} :'{track.name}' (failed to read stems): {e}", flush=True)
        return None

    for stem_name, audio in stems_items:
        target = _MOISES_STEM_MAP.get(stem_name.lower())
        if target is None:
            continue

        try:
            signal = _ensure_stereo(audio)
        except Exception:
            continue

        if accum[target] is None:
            accum[target] = signal
        else:
            ref = accum[target]
            target_len = max(ref.shape[-1], signal.shape[-1])
            accum[target] = (
                pad_to_len(ref, target_len - ref.shape[-1]) +
                pad_to_len(signal, target_len - signal.shape[-1])
            )

    # --- align lengths to mixture ---
    max_len = mixture.shape[-1]

    for k in TARGETS:
        if accum[k] is None:
            accum[k] = np.zeros((2, max_len), dtype=np.float32)
        elif accum[k].shape[-1] < max_len:
            accum[k] = pad_to_len(accum[k], max_len - accum[k].shape[-1])

    # --- convert to torch tensors ---
    mixture_tensor = torch.from_numpy(mixture).float()

    stems_tensor: Dict[str, torch.Tensor] = {
        k: torch.from_numpy(v).float() for k, v in accum.items()
    }

    song_id = getattr(track, "name", None) or getattr(track, "id", None) or "song"

    return SongStems(
        song_id=str(song_id),
        mixture=mixture_tensor,
        stems=stems_tensor
    )


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
    def __init__(
        self,
        songs: Sequence[SongStems],
        target: str,
        sr: int = 44100,
        segment_seconds: float = 3.0,
        samples_per_epoch: int = 10000,
        augment: bool = True,
    ):
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}")

        self.songs = list(songs)
        self.target = target
        self.sr = sr
        self.segment = int(segment_seconds * sr)
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment

        self.source_pool = TARGETS

    def __len__(self):
        return self.samples_per_epoch

    # -------------------------
    # segment sampler
    # -------------------------
    def _random_segment(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (2, T)
        if wav.shape[-1] <= self.segment:
            pad = self.segment - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad))

        start = torch.randint(0, wav.shape[-1] - self.segment + 1, (1,)).item()
        return wav[:, start:start + self.segment]

    # -------------------------
    # mix-audio augmentation
    # -------------------------
    def _mix_same_source(self, seg: torch.Tensor, pool: list[torch.Tensor]) -> torch.Tensor:
        # smix = s1 + s2
        if len(pool) == 0:
            return seg

        other = pool[torch.randint(0, len(pool), (1,)).item()]
        return seg + other

    # -------------------------
    # get segment pool for a source
    # -------------------------
    def _sample_pool(self, song: SongStems, source: str, n: int = 2):
        wav = song.stems[source]
        segs = []
        for _ in range(n):
            segs.append(self._random_segment(wav))
        return segs

    # -------------------------
    # main sample
    # -------------------------
    def __getitem__(self, _idx: int):
        song = self.songs[torch.randint(0, len(self.songs), (1,)).item()]

        # store augmented sources
        source_signals = {}

        for src in TARGETS:
            wav = song.stems[src]

            seg = self._random_segment(wav)

            # -------------------------
            # conditional augmentation
            # -------------------------
            if self.augment and src != "bass":
                pool = self._sample_pool(song, src, n=2)
                seg = self._mix_same_source(seg, pool)

            source_signals[src] = seg

        # -------------------------
        # build mixture
        # -------------------------
        mixture = sum(source_signals[src] for src in TARGETS)

        # optional normalization
        peak = mixture.abs().max().clamp(min=1e-8)
        if peak > 1.0:
            mixture = mixture / peak
            source_signals = {k: v / peak for k, v in source_signals.items()}

        return {
            "mixture": mixture.float(),
            "target": source_signals[self.target].float(),
        }


# =============================================================================
# 2.  Model
# =============================================================================

class RCM(nn.Module):
    """
    Residual Convolutional Module (Fig. 2).
    BN -> GELU -> Conv3x3 -> BN -> GELU -> Conv3x3,  with 1x1 shortcut.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )
        self.shortcut = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.shortcut(x)

class ConvBlock(nn.Module):
    """Four stacked RCMs, with an optional channel-changing 1x1 projection first."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.rcms = nn.Sequential(*[RCM(out_ch) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rcms(self.proj(x))

class EncoderBlock(nn.Module):
    """ConvBlock (keeps spatial size, adjusts channels) + 2x2 AvgPool down-sample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)       # saved for skip connection
        return self.down(skip), skip

class DecoderBlock(nn.Module):
    """Bilinear 2x up-sample, skip-cat, then ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class _SeqAttention(nn.Module):
    """
    Shared structure for TSA and FSA (Fig. 3b):
      LN -> MSA -> residual -> LN -> 2-layer MLP(GELU) -> residual.
    Operates on a flattened sequence of shape (batch*, S, C).
    """

    def __init__(self, channels: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*, S, C)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x

class TSABlock(nn.Module):
    """
    Time Sequence Attention: MSA along T for every frequency bin independently.
    Input/output: (B, C, F, T)
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.sa = _SeqAttention(channels, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * F, T, C)   # (B*F, T, C)
        x = self.sa(x)
        return x.view(B, F, T, C).permute(0, 3, 1, 2).contiguous()

class FSABlock(nn.Module):
    """
    Frequency Sequence Attention: MSA along F for every time frame independently.
    Input/output: (B, C, F, T)
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.sa = _SeqAttention(channels, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * T, F, C)   # (B*T, F, C)
        x = self.sa(x)
        return x.view(B, T, F, C).permute(0, 3, 2, 1).contiguous()

def _window_partition(x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
    """(B, H, W, C) -> (B * nWh * nWw, wh*ww, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh * ww, C)


def _window_reverse(windows: torch.Tensor, wh: int, ww: int,
                    H: int, W: int) -> torch.Tensor:
    """(B * nWh * nWw, wh*ww, C) -> (B, H, W, C)"""
    nWh, nWw = H // wh, W // ww
    B = windows.shape[0] // (nWh * nWw)
    x = windows.view(B, nWh, nWw, wh, ww, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def _make_attn_mask(H: int, W: int, wh: int, ww: int,
                    sh: int, sw: int,
                    device: torch.device) -> Optional[torch.Tensor]:
    """
    Build a 2-D additive attention mask for cyclic-shift SW-MSA.
    Returns None for plain W-MSA (no shift).

    The mask shape is (nW, wh*ww, wh*ww).  Each of the nW windows has its
    own (seq, seq) additive bias.  We keep it 3-D here and index the correct
    window slice per-window inside SwinBlock.forward, avoiding any dependency
    on batch-size or num_heads.
    """
    if sh == 0 and sw == 0:
        return None
    img_mask = torch.zeros(1, H, W, 1, device=device)
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1
    mask_windows = _window_partition(img_mask, wh, ww).squeeze(-1)   # (nW, wh*ww)
    mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)      # (nW, wh*ww, wh*ww)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

class SwinBlock(nn.Module):
    """
    Single Swin Transformer block (Fig. 7).
    shift=False -> W-MSA  (Eq. 2-3)
    shift=True  -> SW-MSA (Eq. 4-5)
    Input/output: (B, C, F, T)
    """

    def __init__(self, channels: int, num_heads: int,
                 window_size: Tuple[int, int] = (5, 4),
                 shift: bool = False,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.wh, self.ww = window_size
        self.sh = self.wh // 2 if shift else 0
        self.sw = self.ww // 2 if shift else 0
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, H, W, C = x.shape
        ph = (self.wh - H % self.wh) % self.wh
        pw = (self.ww - W % self.ww) % self.ww
        if ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph))
        return x, ph, pw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        x_hw = x.permute(0, 2, 3, 1)             # (B, H=F, W=T, C)
        residual = x_hw
        x_hw = self.norm1(x_hw)

        x_hw, ph, pw = self._pad_to_window(x_hw)
        Hp, Wp = x_hw.shape[1], x_hw.shape[2]
        nW = (Hp // self.wh) * (Wp // self.ww)   # windows per sample

        # cyclic shift for SW-MSA
        if self.sh or self.sw:
            x_hw = torch.roll(x_hw, shifts=(-self.sh, -self.sw), dims=(1, 2))

        # attn_mask: (nW, seq, seq) – one independent 2-D bias per window slot
        attn_mask = _make_attn_mask(Hp, Wp, self.wh, self.ww,
                                    self.sh, self.sw, x.device)

        # windows: (B*nW, seq, C)
        windows = _window_partition(x_hw, self.wh, self.ww)

        if attn_mask is not None:
            # Process each window slot separately with its own 2-D mask so
            # PyTorch can broadcast it over (batch, heads) automatically.
            # This avoids the (B*nW*heads, seq, seq) shape requirement.
            seq = self.wh * self.ww
            windows = windows.view(B, nW, seq, C)
            attn_outs = []
            for w_idx in range(nW):
                mask_w = attn_mask[w_idx]           # (seq, seq) – 2-D, broadcast-safe
                w_in   = windows[:, w_idx]          # (B, seq, C)
                w_out, _ = self.attn(w_in, w_in, w_in, attn_mask=mask_w)
                attn_outs.append(w_out)
            attn_out = torch.stack(attn_outs, dim=1).view(B * nW, seq, C)
        else:
            attn_out, _ = self.attn(windows, windows, windows)

        x_hw = _window_reverse(attn_out, self.wh, self.ww, Hp, Wp)

        # reverse cyclic shift
        if self.sh or self.sw:
            x_hw = torch.roll(x_hw, shifts=(self.sh, self.sw), dims=(1, 2))

        if ph or pw:
            x_hw = x_hw[:, :F, :T, :].contiguous()

        x_hw = residual + x_hw
        x_hw = x_hw + self.mlp(self.norm2(x_hw))
        return x_hw.permute(0, 3, 1, 2).contiguous()          # (B, C, F, T)

class SwinPair(nn.Module):
    """Two consecutive Swin blocks: W-MSA -> SW-MSA (Fig. 7)."""

    def __init__(self, channels: int, num_heads: int = 4,
                 window_size: Tuple[int, int] = (5, 4)):
        super().__init__()
        self.w_msa  = SwinBlock(channels, num_heads, window_size, shift=False)
        self.sw_msa = SwinBlock(channels, num_heads, window_size, shift=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sw_msa(self.w_msa(x))

class TFSWAModule(nn.Module):
    """
    One TFSWA module: TSA -> FSA -> SwinPair (via residual branch).
    Input/output: (B, C, F, T)
    """

    def __init__(self, channels: int, num_heads: int = 4,
                 window_size: Tuple[int, int] = (5, 4)):
        super().__init__()
        self.tsa  = TSABlock(channels, num_heads)
        self.fsa  = FSABlock(channels, num_heads)
        self.swin = SwinPair(channels, num_heads, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tsa(x)
        x = self.fsa(x)
        x = x + self.swin(x)    # residual connection around the Swin branch
        return x


class TFSWABottleneck(nn.Module):
    """Four sequential TFSWA modules serving as the U-Net bottleneck."""

    def __init__(self, channels: int, num_heads: int = 4,
                 window_size: Tuple[int, int] = (5, 4)):
        super().__init__()
        self.tfswa_blocks = nn.ModuleList(
            [TFSWAModule(channels, num_heads, window_size) for _ in range(4)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.tfswa_blocks:
            x = block(x)
        return x


class TFSWAResUNet(nn.Module):
    """
    TFSWA-ResUNet for stereo music source separation.

    Encoder channel schedule (in_ch -> out_ch):
      Block 0:   8 ->  16
      Block 1:  16 ->  32
      Block 2:  32 ->  64
      Block 3:  64 -> 128
      Block 4: 128 -> 128   (last encoder block keeps channels fixed)

    Decoder is symmetric:
      Block 0: (128+128) -> 128
      Block 1: (128+ 64) ->  64
      Block 2: ( 64+ 32) ->  32
      Block 3: ( 32+ 16) ->  16
      Block 4: ( 16+  8) ->   8   (back to 8 input channels)
    """

    _ENC_CH: List[Tuple[int, int]] = [
        (8,   16),
        (16,  32),
        (32,  64),
        (64,  128),
        (128, 128),
    ]

    def __init__(self,
                 n_fft: int = 2046,
                 hop_length: int = 512,
                 n_subbands: int = 4,
                 num_heads: int = 4,
                 swin_window: Tuple[int, int] = (5, 4),
                 debug_shapes: bool = False):
        super().__init__()
        assert n_fft % 2 == 0, "n_fft must be even"
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.n_subbands = n_subbands
        self.debug_shapes = debug_shapes
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        # ---- Encoder --------------------------------------------------------
        self.encoders = nn.ModuleList(
            [EncoderBlock(ic, oc) for ic, oc in self._ENC_CH]
        )

        # ---- Bottleneck -----------------------------------------------------
        bottleneck_ch = self._ENC_CH[-1][1]   # 128
        self.bottleneck = TFSWABottleneck(bottleneck_ch, num_heads, swin_window)

        # ---- Decoder --------------------------------------------------------
        enc_out = [oc for _, oc in self._ENC_CH]   # [16,32,64,128,128]
        dec_specs: List[Tuple[int, int, int]] = [
            (enc_out[4], enc_out[3], enc_out[3]),   # (128,128)->128
            (enc_out[3], enc_out[3], enc_out[2]),   # (128, 64)-> 64
            (enc_out[2], enc_out[2], enc_out[1]),   # ( 64, 32)-> 32
            (enc_out[1], enc_out[1], enc_out[0]),   # ( 32, 16)-> 16
            (enc_out[0], enc_out[0],          8),             # ( 16,  8)->  8
        ]
        self.decoders = nn.ModuleList(
            [DecoderBlock(in_ch = ic, skip_ch = sc, out_ch = oc) for ic, sc, oc in dec_specs]
        )
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(16, 8)
        )
        # print(enc_out)
        # print(dec_specs)

    def _shape_log(self, name: str, tensor: torch.Tensor) -> None:
        if self.debug_shapes:
            print(f"[shape-debug] {name}: {tuple(tensor.shape)}", flush=True)

    # ------------------------------------------------------------------
    # STFT / iSTFT
    # ------------------------------------------------------------------

    def stft(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """wav (B,2,S) -> magnitude (B,2,F_pad,T),  phase (B,2,F_pad,T)
        F_pad is F rounded up to the nearest multiple of n_subbands so that
        _split_subbands never fails regardless of n_fft choice.
        """
        B, C, S = wav.shape
        x = wav.reshape(B * C, S)
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window, return_complex=True, center=True)
        spec = spec.view(B, C, *spec.shape[-2:])

        # Pad F dimension so it is an exact multiple of n_subbands
        F_raw = spec.shape[2]
        remainder = F_raw % self.n_subbands
        if remainder != 0:
            pad_bins = self.n_subbands - remainder
            spec = F.pad(spec, (0, 0, 0, pad_bins))  # pad along F axis

        return spec.abs(), torch.angle(spec)

    def istft(self, mag: torch.Tensor, phase: torch.Tensor, length: int) -> torch.Tensor:
        """magnitude + mixture phase -> waveform (B,2,S)"""
        spec = mag * torch.exp(1j * phase)
        B, C, F_, T = spec.shape
        wav = torch.istft(spec.reshape(B * C, F_, T),
                          n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window, length=length, center=True)
        return wav.view(B, C, length)

    # ------------------------------------------------------------------
    # Subband helpers
    # ------------------------------------------------------------------

    def _split_subbands(self, mag: torch.Tensor) -> torch.Tensor:
        """(B,2,F_pad,T) -> (B, 2*n_subbands, F_pad//n_subbands, T)
        F_pad is already a multiple of n_subbands (guaranteed by stft()).
        """
        B, C, F, T = mag.shape
        assert F % self.n_subbands == 0, "F must be divisible by n_subbands"
        sb = F // self.n_subbands
        mag = mag.view(B, C, self.n_subbands, sb, T)
        return mag.reshape(B, C * self.n_subbands, sb, T)

    def _merge_subbands(self, x: torch.Tensor, F_raw: int) -> torch.Tensor:
        """(B, 2*n_subbands, sb, T) -> (B, 2, F_raw, T)
        Reconstructs the full padded spectrogram then crops back to F_raw
        (the true number of STFT bins before any padding was added).
        """
        B, ch, sb, T = x.shape
        C = ch // self.n_subbands          # stereo channels = 2
        F_pad = self.n_subbands * sb
        merged = x.view(B, C, self.n_subbands, sb, T).reshape(B, C, F_pad, T)
        return merged[:, :, :F_raw, :]  # crop back to original F bins

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        length = mixture.shape[-1]
        self._shape_log("input mixture", mixture)
        mix_mag, mix_phase = self.stft(mixture)
        _, _, F_pad, T = mix_mag.shape
        F_raw = self.n_fft // 2 + 1  # true STFT bins before padding
        self._shape_log("stft mix_mag", mix_mag)
        self._shape_log("stft mix_phase", mix_phase)
        x = self._split_subbands(mix_mag)
        self._shape_log("split_subbands", x)
        # Encoder
        skips: List[torch.Tensor] = []
        for idx, enc in enumerate(self.encoders):
            x, skip = enc(x)
            self._shape_log(f"encoder {idx} output", x)
            self._shape_log(f"encoder {idx} skip", skip)
            skips.append(skip)
        # Bottleneck
        x = self.bottleneck(x)
        self._shape_log("bottleneck", x)
        # Decoder
        for idx, (dec, skip) in enumerate(zip(self.decoders, reversed(skips))):
            x = dec(x, skip)
            self._shape_log(f"decoder {idx} output", x)
        # x = self.final_up(x)
        # print(f"Upsample: {x.shape}")

        # Merge subbands -> target magnitude; crop F back to F_raw
        target_mag = self._merge_subbands(x, F_raw)
        target_mag = torch.relu(target_mag)  # magnitude >= 0
        self._shape_log("target_mag", target_mag)

        # Also crop phase to match (in case stft() padded F)
        mix_phase_crop = mix_phase[:, :, :F_raw, :]
        out_wav = self.istft(target_mag, mix_phase_crop, length)
        self._shape_log("output waveform", out_wav)

        return {
            "waveform":   out_wav,
            "target_mag": target_mag,
            "mix_mag":    mix_mag,
            "mix_phase":  mix_phase,
        }

# =============================================================================
# 3.  Loss + metrics
# =============================================================================

class stftL1Loss(nn.Module):
    """
    L1 loss computed on the stft domain.
    Both `pred` (model output dict)
    are accepted in two calling conventions:

        loss_fn(output_dict, target_wav)   # target_wav: (B, 2, S)
        loss_fn(pred_wav,    target_wav)   # both tensors
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self,
                pred: "dict | torch.Tensor") -> torch.Tensor:
        if isinstance(pred, dict):
            mix_stft = pred["mix_mag"]  # pull the waveform from the model output dict
            target_wav = pred["target_mag"]
        return self.l1(mix_stft, target_wav)

class WaveformL1Loss(nn.Module):
    """
    L1 loss computed on the waveform domain.
    Both `pred` (model output dict) and `target` (ground-truth waveform tensor)
    are accepted in two calling conventions:

        loss_fn(output_dict, target_wav)   # target_wav: (B, 2, S)
        loss_fn(pred_wav,    target_wav)   # both tensors
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self,
                pred: "dict | torch.Tensor",
                target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, dict):
            pred = pred["waveform"]  # pull the waveform from the model output dict
        return self.l1(pred, target)


def build_optimizer_and_scheduler(
        model: nn.Module,
        lr: float = 1e-3,
        lr_decay: float = 0.9,
        decay_per_epochs: float = 1.5,
        steps_per_epoch: int = 100,
) -> tuple:
    """
    Returns (optimizer, scheduler).

    Parameters
    ----------
    model            : the network whose parameters will be optimised
    lr               : initial learning rate (default 0.001)
    lr_decay         : multiplicative factor applied each decay step (default 0.9)
    decay_per_epochs : how many epochs between each lr decay (default 1.5)
    steps_per_epoch  : number of optimizer steps (batches) per epoch — must be
                       computed from your DataLoader: len(train_loader)

    Scheduler notes
    ---------------
    StepLR decays lr by `gamma` every `step_size` *optimizer steps*.
    We want a decay every 1.5 epochs, so:
        step_size = ceil(1.5 * steps_per_epoch)
    Call scheduler.step() after every optimizer.step().
    """
    optimizer = Adam(model.parameters(), lr=lr)

    step_size = math.ceil(decay_per_epochs * steps_per_epoch)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_decay)

    return optimizer, scheduler


def print_training_setup(
    loss_fn: nn.Module,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> None:
    optimizer_group = optimizer.param_groups[0]
    print("[train] loss function:", flush=True)
    print(f"[train]   {loss_fn.__class__.__name__}", flush=True)
    print("[train] optimizer:", flush=True)
    print(f"[train]   {optimizer.__class__.__name__}", flush=True)
    print(f"[train]   lr={optimizer_group.get('lr')}", flush=True)
    print(f"[train]   betas={optimizer_group.get('betas')}", flush=True)
    print(f"[train]   eps={optimizer_group.get('eps')}", flush=True)
    print(f"[train]   weight_decay={optimizer_group.get('weight_decay')}", flush=True)
    print("[train] scheduler:", flush=True)
    print(f"[train]   {scheduler.__class__.__name__}", flush=True)
    print(f"[train]   step_size={scheduler.step_size}", flush=True)
    print(f"[train]   gamma={scheduler.gamma}", flush=True)
    print("[train] training hyperparameters:", flush=True)
    print(f"[train]   batch_size={args.batch_size}", flush=True)
    print(f"[train]   epochs={args.epochs}", flush=True)
    print(f"[train]   segment_seconds={args.segment}", flush=True)
    print(f"[train]   samples_per_epoch={args.samples_per_epoch}", flush=True)
    print(f"[train]   steps_per_epoch={steps_per_epoch}", flush=True)
    print(f"[train]   grad_clip={args.grad_clip}", flush=True)
    print(f"[train]   lr_decay={args.lr_decay}", flush=True)
    print(f"[train]   decay_per_epochs={args.decay_per_epochs}", flush=True)



# =============================================================================
# 4.  Inference (overlap-add)
# =============================================================================

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

@torch.no_grad()
def separate(
    model:            nn.Module,
    wav:              torch.Tensor,   # (C, S)
    *,
    sr:               int,
    segment_seconds:  float = 3.0,
    hop_seconds:      float = 1.5,
    device:           Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Overlap-add separation for an arbitrary-length (channels, samples) waveform.

    Uses a Bartlett (triangular) synthesis window for smooth reconstruction,
    matching the original snippet. Returns estimated source as (C, S) on CPU.
    """
    device = device or next(model.parameters()).device
    model.eval()

    # ensure stereo
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    seg = int(round(segment_seconds * sr))
    hop = int(round(hop_seconds * sr))
    pad = seg - hop                    # reflective context at both edges

    padded = F.pad(wav.unsqueeze(0), (pad, pad + seg)).squeeze(0)  # (C, S+pads)
    total  = padded.shape[-1]

    # triangular synthesis window for smooth overlap-add
    window = torch.bartlett_window(seg, periodic=False, device=device)

    out    = torch.zeros_like(padded, device=device)
    weight = torch.zeros(total, device=device)

    start = 0
    while start + seg <= total:
        chunk = padded[:, start:start + seg].to(device).unsqueeze(0)  # (1, C, seg)
        est   = model(chunk)["waveform"].squeeze(0)                    # (C, seg)
        out[:, start:start + seg] += est * window
        weight[start:start + seg] += window
        start += hop

    out = out / weight.clamp_min(1e-8)
    out = out[:, pad:pad + wav.shape[-1]]
    return out.cpu()

def _bss_eval(
    reference: np.ndarray,   # (channels, samples)
    estimate:  np.ndarray,   # (channels, samples)
    sr:        int,
    win:       float = 1.0,  # evaluation window in seconds (SiSEC/museval default)
    hop:       float = 1.0,  # evaluation hop   in seconds
) -> Dict[str, float]:
    """
    Compute median SDR, SIR, SAR across non-overlapping 1-second frames
    using the museval package (Eq. 6-8).

    museval.evaluate expects shape (n_targets, n_samples, n_channels).

    SDR = 10 log10( ||s_target||^2 / ||e_interf + e_noise + e_artif||^2 )
    SIR = 10 log10( ||s_target||^2 / ||e_interf||^2 )
    SAR = 10 log10( ||s_target + e_interf + e_noise||^2 / ||e_artif||^2 )
    """
    # (1, S, C) — museval convention
    ref_mu = reference.T[np.newaxis]
    est_mu = estimate.T[np.newaxis]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdr_frames, _, sir_frames, sar_frames = museval.evaluate(
            references = ref_mu,
            estimates  = est_mu,
            win        = int(win * sr),
            hop        = int(hop * sr),
        )

    def _median(arr: np.ndarray) -> float:
        vals = arr[np.isfinite(arr)]
        return float(np.median(vals)) if len(vals) else float("nan")

    return {
        "SDR": _median(sdr_frames),
        "SIR": _median(sir_frames),
        "SAR": _median(sar_frames),
    }

@torch.no_grad()
def evaluate_songs(
    model:            nn.Module,
    songs:            Sequence[SongStems],
    target:           str,
    *,
    sr:               int,
    device:           torch.device,
    segment_seconds:  float = 3.0,
    hop_seconds:      float = 1.5,
    win_seconds:      float = 1.0,
    compute_sir_sar:  bool  = True,
) -> Dict[str, float]:
    """
    Run separation on every song and report mean museval SDR / SIR / SAR.

    Parameters
    ----------
    model            : trained separation model
    songs            : list of SongStems (each carries .mixture and .stems)
    target           : stem name to separate, e.g. "vocals"
    sr               : sample rate (Hz)
    device           : inference device
    segment_seconds  : chunk length for overlap-add inference
    hop_seconds      : hop length for overlap-add inference
    win_seconds      : museval frame length (default 1 s = SiSEC standard)
    compute_sir_sar  : True for ablation experiments; False for main results
                       (SDR only, slightly faster)

    Returns
    -------
    dict with keys: "mean_SDR", optionally "mean_SIR" / "mean_SAR", "n_songs"
    """
    model.eval()

    sdrs: List[float] = []
    sirs: List[float] = []
    sars: List[float] = []

    for song in songs:
        # ---- overlap-add separation ----
        est = separate(
            model, song.mixture,
            sr=sr,
            segment_seconds=segment_seconds,
            hop_seconds=hop_seconds,
            device=device,
        ).numpy()                                         # (C, S)

        # ---- align lengths ----
        ref = song.stems[target].numpy()   # torch.Tensor -> np.ndarray
        L   = min(ref.shape[-1], est.shape[-1])
        ref, est = ref[:, :L], est[:, :L]

        # ---- museval SDR / SIR / SAR (Eq. 6-8) ----
        metrics = _bss_eval(ref, est, sr=sr, win=win_seconds, hop=win_seconds)

        sdrs.append(metrics["SDR"])
        sirs.append(metrics["SIR"])
        sars.append(metrics["SAR"])

        sir_str = f"  SIR={metrics['SIR']:6.2f}" if compute_sir_sar else ""
        sar_str = f"  SAR={metrics['SAR']:6.2f}" if compute_sir_sar else ""
        print(
            f"  {song.song_id:30s}"
            f"  SDR={metrics['SDR']:6.2f}"
            f"{sir_str}{sar_str}",
            flush=True,
        )

    def _mean(vals: List[float]) -> float:
        arr = np.array(vals, dtype=float)
        return float(np.nanmean(arr)) if len(arr) else float("nan")

    result: Dict[str, object] = {
        "mean_SDR": _mean(sdrs),
        "n_songs":  len(sdrs),
    }
    if compute_sir_sar:
        result["mean_SIR"] = _mean(sirs)
        result["mean_SAR"] = _mean(sars)

    return result

def run_evaluation(
    model:           nn.Module,
    songs:           Sequence[SongStems],
    target:          str,
    *,
    sr:              int,
    device:          torch.device,
    ablation:        bool  = False,
    segment_seconds: float = 3.0,
    hop_seconds:     float = 1.5,
) -> Dict[str, float]:
    """
    Convenience wrapper called from training scripts.

    ablation=False  →  SDR only  (comparison / test-set evaluation)
    ablation=True   →  SDR + SIR + SAR  (ablation experiments)
    """
    print(f"\n=== Evaluating '{target}' | ablation={ablation} ===")
    results = evaluate_songs(
        model, songs, target,
        sr=sr,
        device=device,
        segment_seconds=segment_seconds,
        hop_seconds=hop_seconds,
        compute_sir_sar=ablation,
    )

    print("\n--- Summary ---")
    for k, v in results.items():
        if isinstance(v, float):
            unit = " dB" if any(m in k for m in ("SDR", "SIR", "SAR")) else ""
            print(f"  {k}: {v:.4f}{unit}")
        else:
            print(f"  {k}: {v}")

    return results


# =============================================================================
# 5.  Training
# =============================================================================


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "mixture": torch.stack([b["mixture"] for b in batch], dim=0),
        "target":  torch.stack([b["target"]  for b in batch], dim=0),
    }


def train_one_epoch(model, loader, loss_fn, optimizer,scheduler, device, *,
                    grad_clip: float = 5.0, log_every: int = 50) -> float:
    model.train()
    running = 0.0
    n = 0
    for step, batch in enumerate(loader):
        mixture = batch["mixture"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        out = model(mixture)
        loss = loss_fn(out, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

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
        loss = loss = loss_fn(out, target)
        total += loss.item() * mixture.size(0)
        n += mixture.size(0)
    return total / max(n, 1)


@torch.no_grad()
def test_model_input_output_shapes(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    debug_shapes: bool = False,
) -> Dict[str, Tuple[int, ...]]:
    """
    Run one pre-training batch through the model and verify the shape contract.

    Expected training batch:
        mixture: (B, 2, samples)
        target:  (B, 2, samples)

    Expected model output:
        dict with waveform: (B, 2, samples)
    """
    was_training = model.training
    old_debug_shapes = getattr(model, "debug_shapes", None)
    if old_debug_shapes is not None:
        model.debug_shapes = debug_shapes
    model.eval()

    try:
        try:
            batch = next(iter(loader))
        except StopIteration as exc:
            raise RuntimeError("Shape check failed: training DataLoader is empty.") from exc

        if not isinstance(batch, dict):
            raise TypeError(f"Shape check failed: expected batch dict, got {type(batch).__name__}.")
        if "mixture" not in batch or "target" not in batch:
            raise KeyError("Shape check failed: batch must contain 'mixture' and 'target'.")

        mixture = batch["mixture"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        if mixture.ndim != 3:
            raise ValueError(f"Shape check failed: mixture must be (B, 2, S), got {tuple(mixture.shape)}.")
        if target.ndim != 3:
            raise ValueError(f"Shape check failed: target must be (B, 2, S), got {tuple(target.shape)}.")
        if mixture.shape[1] != 2:
            raise ValueError(f"Shape check failed: mixture must have 2 channels, got {mixture.shape[1]}.")
        if target.shape != mixture.shape:
            raise ValueError(
                "Shape check failed: target must match mixture shape, "
                f"got target={tuple(target.shape)} mixture={tuple(mixture.shape)}."
            )

        output = model(mixture)
        if not isinstance(output, dict):
            raise TypeError(f"Shape check failed: model output must be a dict, got {type(output).__name__}.")
        if "waveform" not in output:
            raise KeyError("Shape check failed: model output dict must contain 'waveform'.")

        waveform = output["waveform"]
        if waveform.shape != mixture.shape:
            raise ValueError(
                "Shape check failed: output['waveform'] must match mixture shape, "
                f"got waveform={tuple(waveform.shape)} mixture={tuple(mixture.shape)}."
            )

        shapes = {
            "mixture": tuple(mixture.shape),
            "target": tuple(target.shape),
            "waveform": tuple(waveform.shape),
        }
        for key in ("mix_mag", "target_mag", "mix_phase"):
            if key in output:
                shapes[key] = tuple(output[key].shape)

        print("[shape-check] passed", flush=True)
        print(f"[shape-check] mixture={shapes['mixture']} target={shapes['target']}", flush=True)
        print(f"[shape-check] waveform={shapes['waveform']}", flush=True)
        if "mix_mag" in shapes and "target_mag" in shapes:
            print(
                f"[shape-check] mix_mag={shapes['mix_mag']} target_mag={shapes['target_mag']}",
                flush=True,
            )

        return shapes
    finally:
        if old_debug_shapes is not None:
            model.debug_shapes = old_debug_shapes
        if was_training:
            model.train()


def build_model(n_fft: int, hop_length: int, n_subbands: int,
                num_heads: int, swin_window: Tuple[int, int],
                debug_shapes: bool = False) -> TFSWAResUNet:
    return TFSWAResUNet(n_fft=n_fft, hop_length=hop_length, n_subbands=n_subbands,
                 num_heads=num_heads, swin_window= swin_window,
                 debug_shapes=debug_shapes)


def save_checkpoint(path: Path, model: TFSWAResUNet, optimizer, epoch: int,
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
            "n_subbands": args.n_subbands,
            "num_heads": args.num_heads,
            "swin_window": args.swin_window
        },
    }, str(path))


def load_checkpoint(path: Path, device) -> Tuple[TFSWAResUNet, dict]:
    state = torch.load(str(path), map_location=device)
    cfg = state["config"]
    model = TFSWAResUNet(swin_window=cfg["swin_window"], n_subbands=cfg["n_subbands"],
                num_heads=cfg["num_heads"], n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"]).to(device)
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
# 6.  CLI
# =============================================================================


def cmd_train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[train] target={args.target} device={device}", flush=True)

    indices = discover_songs(Path(args.data), sr=args.sr)
    print(f"Total songs found ={len(indices)}", flush=True)

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

    print("[train] segmenting and augmenting...", flush=True)
    train_set = MoisesDBSegments(
        train_songs, target=args.target, sr=args.sr,
        segment_seconds=args.segment, augment = True,
        samples_per_epoch=args.samples_per_epoch)
    val_set = MoisesDBSegments(
        val_songs, target=args.target, sr=args.sr,
        segment_seconds=args.segment, augment = False,
        samples_per_epoch=max(64, args.samples_per_epoch // 20))

    print("[train] Creating DataLoader...", flush=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate, pin_memory=True)

    model = build_model(n_fft=args.n_fft, hop_length=args.hop_length, n_subbands=args.n_subbands,
                        num_heads=args.num_heads, swin_window=args.swin_window,
                        debug_shapes=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] TFSWAResUNet params: {n_params/1e6:.2f}M  ", flush=True)

    if not args.skip_shape_check:
        test_model_input_output_shapes(
            model,
            train_loader,
            device,
            debug_shapes=True,
        )

    loss_fn = WaveformL1Loss()
    steps_per_epoch = len(train_loader)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr=args.lr,
        lr_decay=args.lr_decay,
        decay_per_epochs=args.decay_per_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    print_training_setup(loss_fn, optimizer, scheduler, args, steps_per_epoch)


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -float("inf")  # we now select on val uSDR (higher = better)
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n[epoch {epoch}/{args.epochs}]", flush=True)
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                     scheduler, device, grad_clip=args.grad_clip,
                                     log_every=args.log_every)
        val_loss = validate(model, val_loader, loss_fn, device)

        # Per-epoch SDR on val songs — loss is a poor proxy for SDR.
        val_sdr_metrics = {"mean_SDR": float("nan"), "n_songs": 0}
        if val_songs and (epoch % args.sdr_every == 0 or epoch == args.epochs):
            sdr_subset = val_songs[:args.sdr_max_songs] if args.sdr_max_songs else val_songs
            print(f"[epoch {epoch}] computing SDR on {len(sdr_subset)} val song(s)...",
                  flush=True)
            val_sdr_metrics = run_evaluation(
                model,
                sdr_subset,
                target=args.target,
                sr=args.sr,
                device=device,
                segment_seconds=args.segment,
                hop_seconds=args.segment / 2,
                ablation=False)

        print(f"[epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_SDR={val_sdr_metrics['mean_SDR']:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}",
              flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_SDR": val_sdr_metrics["mean_SDR"],
            "n_songs": val_sdr_metrics["n_songs"]
        })
        # optionally store SIR/SAR if present
        if "mean_SIR" in val_sdr_metrics:
            history[-1]["val_SIR"] = val_sdr_metrics["mean_SIR"]
        if "mean_SAR" in val_sdr_metrics:
            history[-1]["val_SAR"] = val_sdr_metrics["mean_SAR"]

        with (out_dir / "history.json").open("w") as fh:
            json.dump(history, fh, indent=2)

        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, val_loss, args)

        # Use SDR as the main selection metric
        current = val_sdr_metrics["mean_SDR"]
        if not math.isfinite(current):
            current = -val_loss

        if current > best_metric:
            best_metric = current
            no_improve = 0
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, val_loss, args)
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
            indices,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed
        )

    if args.max_songs:
        test_idx = test_idx[:args.max_songs]
    print(f"[evaluate] {len(test_idx)} test songs (loading via moisesdb)...",
          flush=True)

    songs = load_songs(data_path, test_idx, sr=sr)
    metrics = run_evaluation(
        model,
        songs,
        target= cfg["target"],
        sr= sr,
        device= device,
        ablation= True,
        segment_seconds= args.segment,
        hop_seconds = args.hop
    )

    # --- logging ---
    print(f"\n[evaluate] mean SDR     = {metrics['mean_SDR']:.3f} dB")
    print(f"[evaluate] songs scored = {metrics['n_songs']}")

    # optional metrics
    if "mean_SIR" in metrics:
        print(f"[evaluate] mean SIR     = {metrics['mean_SIR']:.3f} dB")
    if "mean_SAR" in metrics:
        print(f"[evaluate] mean SAR     = {metrics['mean_SAR']:.3f} dB")

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
        ap.add_argument("--n-subbands", type=int, default=4)
        ap.add_argument("--num-heads", type=int, default=4)
        ap.add_argument("--swin-window", type=int, nargs=2, default=[5, 4])

    # train ----------------------------------------------------------------
    tr = sub.add_parser("train", help="train a TFSWAResUNet on MoisesDB")
    tr.add_argument("--data", required=True, type=str,
                    help="Path passed to moisesdb.dataset.MoisesDB(data_path=...). "
                         "Typically the directory containing moisesdb_v0.1/.")
    tr.add_argument("--target", choices=TARGETS, required=True)
    tr.add_argument("--out-dir", default="runs/TFSWAResUNet")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--segment", type=float, default=3.0)
    tr.add_argument("--samples-per-epoch", type=int, default=800)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--lr-decay", type=float, default=0.9)
    tr.add_argument("--decay-per-epochs", type=float, default=1.5)
    tr.add_argument("--grad-clip", type=float, default=5.0)
    tr.add_argument("--patience", type=int, default=15)
    tr.add_argument("--sdr-every", type=int, default=5,
                    help="evaluate val SDR every N epochs (1 = every epoch)")
    tr.add_argument("--sdr-max-songs", type=int, default=0,
                    help="cap on val songs used for SDR (0 = all)")
    tr.add_argument("--num-workers", type=int, default=2)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=18)
    tr.add_argument("--val-frac", type=float, default=0.10)
    tr.add_argument("--test-frac", type=float, default=0.10)
    tr.add_argument("--max-songs", type=int, default=0,
                    help="optional cap on the number of songs loaded (for quick runs)")
    tr.add_argument("--log-every", type=int, default=20)
    tr.add_argument("--skip-shape-check", action="store_true",
                    help="skip the one-batch model input/output shape check before training")
    tr.add_argument("--debug-shapes", action="store_true",
                    help="print tensor shapes after each model block during the pre-training shape check")
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
    ev.add_argument("--out-dir", default="runs/TFSWAResUNet")
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
    print('build_args')
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    print("start_main", flush=True)
    main()
