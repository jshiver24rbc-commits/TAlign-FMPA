"""Replay CPU draws removed with unused source modules, without building them.

The formal source constructed unused networks around the active V5 modules.
Those constructions consumed random numbers even though their parameters were
frozen. Deleting them without this replay changes both initialized V5 weights
and the RNG subsequently used by data shuffling and dropout.

This schedule was recorded from train_eav_clean_mlce_fmpa.py, SHA-256
26520026d07b291c39fc31a8fa90191ae7fe020ab688ed95dbaa7df909a7dc4d,
using the formal defaults (64 legacy tokens, four context blocks, and two
attention layers). Shapes follow token_dim/hidden_dim; no legacy network,
weights, checkpoint, or source-repository import is needed at runtime.

Each uniform call deliberately keeps its original shape and call boundary.
Its bounds cannot affect CPU RNG consumption, so the discarded values use
[0,1]. Gaussian draws retain their original shape and call boundary as well.
Compatibility is for the same PyTorch version, default dtype, and CPU generator
as the source; it is not a cross-version guarantee from PyTorch.
"""

import torch


def _uniform(*shape):
    torch.empty(shape).uniform_(0.0, 1.0)


def _normal(*shape):
    torch.randn(shape)


def _linear(output_dim, input_dim, *, bias=True):
    _uniform(output_dim, input_dim)
    if bias:
        _uniform(output_dim)


def _attention(dim):
    _linear(dim, dim)
    _uniform(3 * dim, dim)


def _classifier(parts, dim, hidden):
    _linear(hidden, parts * dim)
    _linear(5, hidden)


def advance_before_classifier(token_dim, hidden_dim, temporal_slots):
    """Consume the formal draws preceding the active mlce_classifier."""
    d, h = token_dim, hidden_dim
    _linear(d, 5)
    _normal(1, 30, d)
    _normal(1, 64, d)
    _attention(d)
    _linear(d, 5)
    _normal(1, 30, d)
    _linear(d, 90)
    _normal(1, 64, d)
    _linear(d, 1408)
    _normal(1, 64, d)
    for input_dim in (150, 90, 1408):
        _linear(d, input_dim)
    _normal(1, temporal_slots, d)
    _normal(1, 3, 1, d)
    for _ in range(12):
        _uniform(d, d, 3)
        _uniform(d)
    _linear(d, 3 * d)
    _linear(3, d)
    _linear(d, d)
    for _ in range(2):
        _attention(d)
        _attention(d)
        _linear(1, d, bias=False)
        _classifier(4, d, h)
    _classifier(4, d, h)
    _linear(h, 4 * d + 5)
    _linear(1, h)
    _classifier(1, d, h)
    _classifier(3, d, h)


def advance_after_auxiliary_heads(token_dim, hidden_dim):
    """Consume the formal draws after the last active auxiliary head."""
    d, h = token_dim, hidden_dim
    _attention(d)
    _linear(h, d)
    _linear(d, h)
    _linear(1, d, bias=False)
    _classifier(1, d, h)
    _normal(1, 1, d)
    _attention(d)
    _classifier(1, d, h)
