"""Self-contained checkpoints and explicitly trusted historical migration.

New checkpoints contain tensors and ordinary Python values and are loaded with
``weights_only=True``. Historical pickle checkpoints require the caller's explicit
trust; the compatibility unpickler keeps Linux paths as inert PurePosixPath values.
"""
from __future__ import annotations

import argparse
import pickle
import types
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import torch


CHECKPOINT_FORMAT = "talign_fmpa_eav_v5"
CHECKPOINT_VERSION = 1
ACTIVE_PREFIXES = (
    "v5_frontend", "v5_fmpa", "mlce_classifier",
    "v5_alignment_aux_classifier", "v5_fmpa_aux_classifier",
)
# These modules are registered by the old all-experiment model, but are never
# executed by residual_align_fmpa_v5. No other unexpected keys are discarded.
KNOWN_INACTIVE_PREFIXES = frozenset({
    "eeg_tokenizer", "eeg_channel_tokenizer", "audio_projector",
    "video_projector", "temporal_frontend", "mlce", "fmpa",
    "av_anchor_fmpa", "concat_classifier", "temporal_attention_classifier",
    "temporal_cross_attention_classifier",
})


class _TrustedLegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return PurePosixPath
        return super().find_class(module, name)


def _legacy_pickle_module():
    module = types.ModuleType("trusted_legacy_pickle")
    module.Unpickler = _TrustedLegacyUnpickler
    module.Pickler = pickle.Pickler
    module.load = pickle.load
    module.loads = pickle.loads
    module.dump = pickle.dump
    module.dumps = pickle.dumps
    return module


def load_checkpoint(path, *, trusted_legacy=False):
    """Read without modifying the checkpoint; never silently unpickle a legacy file."""
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError, AttributeError) as exc:
        if not trusted_legacy:
            raise ValueError(
                "The checkpoint could not be read with weights_only=True. "
                "For a historical checkpoint you trust, explicitly pass "
                "--trusted-legacy-checkpoint (pickle can execute code)."
            ) from exc
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False,
            pickle_module=_legacy_pickle_module(),
        )
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("Expected a checkpoint dictionary containing a 'model' state dictionary")
    modern = checkpoint.get("format") == CHECKPOINT_FORMAT
    if modern:
        if checkpoint.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version: {checkpoint.get('format_version')!r}")
        required = {"config", "labels", "stats", "source_provenance", "split_provenance"}
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(f"Self-contained checkpoint is missing: {sorted(missing)}")
    elif not trusted_legacy:
        raise ValueError("Historical checkpoints require --trusted-legacy-checkpoint")
    return dict(checkpoint), not modern


def checkpoint_config(checkpoint, *, legacy):
    value = checkpoint.get("args" if legacy else "config", {})
    if isinstance(value, argparse.Namespace):
        value = vars(value)
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint configuration must be a dictionary or argparse.Namespace")
    config = dict(value)
    if config.get("model", "residual_align_fmpa_v5") != "residual_align_fmpa_v5":
        raise ValueError("Only the complete residual_align_fmpa_v5 model is supported")
    if config.get("run", "eav") != "eav":
        raise ValueError("Only the complete three-modality EAV experiment is supported")
    if config.get("disable_self_attention", False) or config.get("disable_cross_attention", False):
        raise ValueError("Historical attention ablations are not supported")
    return config


def load_active_model_state(model, state, *, legacy=False):
    expected = model.state_dict()
    unsupported_model_keys = [k for k in expected if k.split(".")[0] not in ACTIVE_PREFIXES]
    if unsupported_model_keys:
        raise ValueError(f"Model has unsupported active modules: {unsupported_model_keys[:5]}")
    selected = {}
    ignored = []
    unknown = []
    for key, value in state.items():
        prefix = key.split(".")[0]
        if prefix in ACTIVE_PREFIXES:
            if key not in expected:
                unknown.append(key)
            else:
                selected[key] = value
        elif legacy and prefix in KNOWN_INACTIVE_PREFIXES:
            ignored.append(key)
        else:
            unknown.append(key)
    missing = sorted(set(expected) - set(selected))
    mismatched = []
    for key, value in selected.items():
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(expected[key].shape):
            actual = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            mismatched.append(f"{key}: expected {tuple(expected[key].shape)}, got {actual}")
        elif value.dtype != expected[key].dtype:
            mismatched.append(f"{key}: expected dtype {expected[key].dtype}, got {value.dtype}")
    if unknown or missing or mismatched:
        raise ValueError(
            "Incompatible checkpoint state: "
            f"unknown={sorted(unknown)[:10]}, missing={missing[:10]}, "
            f"shape_or_dtype_mismatch={mismatched[:10]}"
        )
    model.load_state_dict(selected, strict=True)
    return {"loaded_tensors": len(selected), "ignored_known_inactive_tensors": len(ignored)}


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
