"""EAV manifests, pre-extracted features and train-only normalization."""

from .dataset import (
    AUDIO_SHAPE,
    EEG_SHAPE,
    LABELS,
    LABEL_TO_ID,
    RUN_ACTIVE_MODALITIES,
    VIDEO_SHAPE,
    EAVDataset,
    EAVFeatureDataset,
    active_mask_for_run,
    compute_train_statistics,
    compute_train_stats,
    feature_paths,
    load_checked,
    load_statistics,
    make_loader,
    manifest_identity_sha256,
    read_manifest,
    save_statistics,
    statistics_to_serializable,
    validate_statistics,
)

__all__ = [name for name in globals() if not name.startswith("_")]
