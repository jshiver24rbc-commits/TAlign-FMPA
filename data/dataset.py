"""Frozen V5 data semantics with portable manifests and statistics provenance.

Feature caches are inputs, not products of an extractor supplied by this project.
The data tuple and default PyTorch collation match the original formal runner.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


LABELS = ("neutral", "anger", "calm", "sad", "happy")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
SPLITS = ("train", "val", "test")
EEG_SHAPE = (16, 150)
AUDIO_SHAPE = (64, 90)
VIDEO_SHAPE = (64, 1408)
IDENTITY_FIELDS = (
    "sample_id", "subject_id", "trial_id", "trial_index", "segment_id", "emotion", "split"
)
STATISTIC_KEYS = ("eeg_mean", "eeg_std", "audio_mean", "audio_std")
RUN_ACTIVE_MODALITIES = {
    # The mask uses Video, Audio, EEG, whereas forward inputs use EEG, Audio, Video.
    "video": (True, False, False),
    "audio": (False, True, False),
    "eeg": (False, False, True),
    "av": (True, True, False),
    "ev": (True, False, True),
    "ea": (False, True, True),
    "eav": (True, True, True),
}


def _json_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_identity_sha256(rows):
    """Hash public identity/label/split columns, preserving manifest row order."""
    return _json_sha256([
        {key: str(row[key]) for key in IDENTITY_FIELDS if key in row}
        for row in rows
    ])


def read_manifest(path, limit_per_split=-1):
    """Read a fixed split without resampling; validate full rows before limiting."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = set(reader.fieldnames or ())
        required = {"sample_id", "subject_id", "segment_id", "emotion", "split"}
        if required - names or not ({"trial_index", "trial_id"} & names):
            raise ValueError("Manifest requires sample, subject, trial, segment, emotion and split columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    trial_splits = {}
    segments = set()
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id or Path(sample_id).name != sample_id or any(c in sample_id for c in ("/", "\\", ":")):
            raise ValueError(f"Invalid sample_id: {sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if row["split"] not in SPLITS:
            raise ValueError(f"Unknown split for {sample_id}: {row['split']}")
        if row["emotion"].lower() not in LABEL_TO_ID:
            raise ValueError(f"Unknown emotion for {sample_id}: {row['emotion']}")
        subject_id = int(row["subject_id"])
        trial_id = int(row.get("trial_index") or row["trial_id"])
        segment_id = int(row["segment_id"])
        trial_key = (subject_id, trial_id)
        previous = trial_splits.setdefault(trial_key, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Trial appears in multiple splits: {trial_key}")
        segment_key = (*trial_key, segment_id)
        if segment_key in segments:
            raise ValueError(f"Duplicate trial segment: {segment_key}")
        segments.add(segment_key)
    split_rows = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    if limit_per_split >= 0:
        split_rows = {split: values[:limit_per_split] for split, values in split_rows.items()}
    return split_rows


def feature_paths(features_root, row, audio_feature_name, video_feature_name, eeg_feature_name="EEG-TDE-16"):
    sample_id = row["sample_id"]
    root = Path(features_root)
    return (
        root / eeg_feature_name / f"{sample_id}.npy",
        root / audio_feature_name / f"{sample_id}.npy",
        root / video_feature_name / f"{sample_id}.npy",
    )


def load_checked(path, expected_shape):
    """Load with exactly the original float32 conversion and squeeze order."""
    value = np.load(path, allow_pickle=False).astype(np.float32)
    value = np.squeeze(value)
    if value.shape != tuple(expected_shape):
        raise ValueError(f"Expected {tuple(expected_shape)}, got {value.shape}: {path}")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite feature values: {path}")
    return value


def _statistics_sha256(stats):
    digest = hashlib.sha256()
    for key in STATISTIC_KEYS:
        value = np.asarray(stats[key], dtype="<f4")
        digest.update(key.encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def compute_train_stats(
    rows, features_root, audio_feature_name, video_feature_name, eeg_feature_name,
    eeg_expected_shape=EEG_SHAPE, *, manifest_path=None,
):
    """Compute only on train rows; numerical operations match the formal trainer."""
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("Normalization requires a nonempty list of training rows only")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate samples in training statistics input")
    if manifest_path is not None:
        manifest_train = read_manifest(manifest_path)["train"]
        # Smoke runs may intentionally use the ordered prefix of the train split.
        if manifest_identity_sha256(manifest_train[:len(rows)]) != manifest_identity_sha256(rows):
            raise ValueError("Statistics rows do not match the manifest training split/prefix")
    eeg_values = []
    audio_sum = np.zeros(90, dtype=np.float64)
    audio_square_sum = np.zeros(90, dtype=np.float64)
    audio_count = 0
    for row in rows:
        eeg_path, audio_path, _ = feature_paths(
            features_root, row, audio_feature_name, video_feature_name, eeg_feature_name
        )
        eeg = load_checked(eeg_path, eeg_expected_shape)
        eeg_values.append(eeg.reshape(-1, 150))
        audio = load_checked(audio_path, AUDIO_SHAPE).astype(np.float64)
        audio_sum += audio.sum(axis=0)
        audio_square_sum += np.square(audio).sum(axis=0)
        audio_count += audio.shape[0]
    eeg_values = np.concatenate(eeg_values, axis=0).astype(np.float64)
    eeg_mean = eeg_values.mean(axis=0)
    eeg_std = eeg_values.std(axis=0).clip(1e-6)
    audio_mean = audio_sum / max(audio_count, 1)
    audio_var = audio_square_sum / max(audio_count, 1) - np.square(audio_mean)
    audio_std = np.sqrt(np.maximum(audio_var, 1e-6))
    stats = {
        "eeg_mean": eeg_mean.astype(np.float32),
        "eeg_std": eeg_std.astype(np.float32),
        "audio_mean": audio_mean.astype(np.float32),
        "audio_std": audio_std.astype(np.float32),
    }
    stats["metadata"] = {
        "version": 1,
        "split": "train",
        "provenance_verified": manifest_path is not None,
        "num_samples": len(rows),
        "train_sample_ids_sha256": _json_sha256([row["sample_id"] for row in rows]),
        "train_identity_sha256": manifest_identity_sha256(rows),
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() if manifest_path else None,
        "feature_names": {"eeg": eeg_feature_name, "audio": audio_feature_name, "video": video_feature_name},
        "feature_shapes": {"eeg": list(eeg_expected_shape), "audio": list(AUDIO_SHAPE), "video": list(VIDEO_SHAPE)},
        "normalization": "EEG/audio float32 input, float64 population statistics; video unchanged",
        "statistics_sha256": _statistics_sha256(stats),
    }
    return stats


compute_train_statistics = compute_train_stats


def validate_statistics(stats, rows=None, manifest_path=None, *, allow_legacy=False):
    """Normalize checkpoint/list/NumPy values and verify available train provenance.

    Legacy four-array caches require an explicit opt-in. Their provenance stays
    unverified even when a caller supplies a manifest filename.
    """
    checked = {}
    for key, size in (("eeg_mean", 150), ("eeg_std", 150), ("audio_mean", 90), ("audio_std", 90)):
        value = stats[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid normalization statistic {key}: expected finite ({size},)")
        if key.endswith("_std") and np.any(value <= 0):
            raise ValueError(f"Normalization standard deviation must be positive: {key}")
        checked[key] = value
    metadata = stats.get("metadata")
    if metadata is None:
        if not allow_legacy:
            raise ValueError("Statistics lack train provenance; explicitly allow legacy statistics to proceed")
        metadata = {"provenance_verified": False, "reason": "legacy_statistics_without_provenance"}
    else:
        metadata = dict(metadata)
    if metadata.get("reason") == "legacy_statistics_without_provenance":
        if not allow_legacy:
            raise ValueError("Legacy statistics require allow_legacy=True")
    else:
        if metadata.get("split") != "train":
            raise ValueError("Statistics were not identified as train-only")
        if metadata.get("statistics_sha256") != _statistics_sha256(checked):
            raise ValueError("Normalization statistics content hash mismatch")
        if rows is not None:
            if any(row.get("split") != "train" for row in rows):
                raise ValueError("Statistics identity validation requires train rows")
            if metadata.get("train_identity_sha256") != manifest_identity_sha256(rows):
                raise ValueError("Normalization statistics belong to different training rows")
        if manifest_path is not None:
            expected = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
            if metadata.get("manifest_sha256") != expected:
                raise ValueError("Normalization statistics belong to a different manifest")
    checked["metadata"] = metadata
    return checked


def statistics_to_serializable(stats):
    """Return plain types for JSON and torch.load(weights_only=True) checkpoints."""
    checked = validate_statistics(stats, allow_legacy=True)
    return {**{key: checked[key].tolist() for key in STATISTIC_KEYS}, "metadata": checked["metadata"]}


def save_statistics(stats, path):
    checked = validate_statistics(stats, allow_legacy=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the file explicitly so NumPy does not silently append '.npz'.
    with path.open("wb") as handle:
        np.savez(handle, **{key: checked[key] for key in STATISTIC_KEYS}, metadata_json=json.dumps(checked["metadata"], sort_keys=True))


def load_statistics(path, *, allow_legacy=False):
    with np.load(path, allow_pickle=False) as archive:
        stats = {key: archive[key] for key in STATISTIC_KEYS}
        if "metadata_json" in archive:
            stats["metadata"] = json.loads(str(archive["metadata_json"].item()))
    return validate_statistics(stats, allow_legacy=allow_legacy)


class EAVFeatureDataset(Dataset):
    """EEG/audio standardization and unmodified video; original six-item tuple."""

    def __init__(
        self, rows, features_root, audio_feature_name, video_feature_name,
        eeg_feature_name, eeg_expected_shape, stats, preload=False, split_name="",
    ):
        self.rows = rows
        self.features_root = Path(features_root)
        self.audio_feature_name = audio_feature_name
        self.video_feature_name = video_feature_name
        self.eeg_feature_name = eeg_feature_name
        self.eeg_expected_shape = tuple(eeg_expected_shape)
        checked = validate_statistics(stats, allow_legacy=True)
        self.eeg_mean = checked["eeg_mean"]
        self.eeg_std = checked["eeg_std"]
        self.audio_mean = checked["audio_mean"]
        self.audio_std = checked["audio_std"]
        self.preloaded = None
        if preload:
            print(f"Preloading {split_name or 'dataset'} features: {len(rows)} samples")
            self.preloaded = []
            for index in range(len(rows)):
                self.preloaded.append(self._load_item(index))
                if (index + 1) % 1000 == 0 or index + 1 == len(rows):
                    print(f"[preload:{split_name or 'data'}] {index + 1}/{len(rows)}", flush=True)

    def __len__(self):
        return len(self.rows)

    def _load_item(self, index):
        row = self.rows[index]
        eeg_path, audio_path, video_path = feature_paths(
            self.features_root, row, self.audio_feature_name, self.video_feature_name, self.eeg_feature_name
        )
        eeg = (load_checked(eeg_path, self.eeg_expected_shape) - self.eeg_mean) / self.eeg_std
        audio = (load_checked(audio_path, AUDIO_SHAPE) - self.audio_mean) / self.audio_std
        video = load_checked(video_path, VIDEO_SHAPE)
        return (
            torch.from_numpy(eeg), torch.from_numpy(audio), torch.from_numpy(video),
            torch.tensor(LABEL_TO_ID[row["emotion"].lower()], dtype=torch.long),
            torch.tensor(int(row["subject_id"]), dtype=torch.long), row["sample_id"],
        )

    def __getitem__(self, index):
        return self.preloaded[index] if self.preloaded is not None else self._load_item(index)


EAVDataset = EAVFeatureDataset


def make_loader(dataset, batch_size, shuffle, workers, pin_memory, subject_balanced=False, subjects_per_batch=4, seed=42):
    """Formal sampling: PyTorch shuffle, every sample once, default collation.

    The formal runner never enables the original optional subject-balanced
    sampler; rejecting that option keeps this independent release on protocol.
    """
    if subject_balanced:
        raise ValueError("Subject-balanced sampling is outside the frozen V5 formal protocol")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=pin_memory, persistent_workers=workers > 0,
    )


def active_mask_for_run(run, batch_size, device):
    if run not in RUN_ACTIVE_MODALITIES:
        raise ValueError(f"Unknown run: {run}")
    return torch.tensor(RUN_ACTIVE_MODALITIES[run], dtype=torch.bool, device=device).unsqueeze(0).expand(batch_size, -1)
