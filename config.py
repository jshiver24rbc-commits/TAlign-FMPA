"""Portable defaults for the frozen EAV TAlign-FMPA V5 experiment."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    data_root: Path = Path("local_data")
    manifest: Path = Path("splits/eav_manifest_subject_wise_4x5s_60_10_30_seed42.csv")
    features_root: Path = Path("local_data/eav_features")
    output_dir: Path = Path("output/eav_v5")
    stats_cache: Optional[Path] = None
    eeg_feature_name: str = "EEG-TDE-16"
    audio_feature_name: str = "audio-frame-lld"
    video_feature_name: str = "EVA-ViT-G-5s"
    model: str = "residual_align_fmpa_v5"
    run: str = "eav"
    seed: int = 42
    epochs: int = 60
    patience: int = 10
    batch_size: int = 16
    eval_batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    grad_clip: float = 1.0
    token_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_classes: int = 5
    temporal_slots: int = 16
    v5_local_radius: int = 1
    v5_residual_gate_init: float = 0.1
    v5_alignment_aux_weight: float = 0.1
    v5_fmpa_aux_weight: float = 0.1
    workers: int = 0
    preload_features: bool = True
    device: str = "cuda:0"
    smoke: bool = False
    limit_per_split: int = -1

    def __post_init__(self):
        for name in ("data_root", "manifest", "features_root", "output_dir", "stats_cache"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value))

    def to_dict(self):
        """Return JSON/checkpoint-safe configuration values."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, values):
        names = {field.name for field in fields(cls)}
        unexpected = set(values) - names
        if unexpected:
            raise ValueError(f"Unknown configuration fields: {sorted(unexpected)}")
        return cls(**values)
