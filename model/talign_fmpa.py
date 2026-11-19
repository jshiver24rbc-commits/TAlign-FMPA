"""The complete EAV residual_align_fmpa_v5 model, extracted without baselines.

V5 building blocks and their forward/diagnostic logic are copied verbatim from
Emotion-LLaMA-v2/scripts/train_eav_clean_mlce_fmpa.py (SHA-256
26520026d07b291c39fc31a8fa90191ae7fe020ab688ed95dbaa7df909a7dc4d).
The wrapper selects the full aligned EAV V5 path only. Its initialization draw
schedule preserves the original formal defaults without unused networks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._formal_initialization import (
    advance_after_auxiliary_heads,
    advance_before_classifier,
)


LABELS = ("neutral", "anger", "calm", "sad", "happy")


class AlignedFeatureProjector(nn.Module):
    def __init__(self, input_dim, token_dim, dropout):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        return self.projection(self.input_norm(values))


class DynamicResidualGate(nn.Module):
    """Blend a candidate update into a stable base through a bounded local gate."""

    def __init__(self, token_dim, initial_probability):
        super().__init__()
        if not 0 < initial_probability < 1:
            raise ValueError("Residual gate initialization must be in (0, 1)")
        self.norm = nn.LayerNorm(token_dim * 3)
        self.score = nn.Linear(token_dim * 3, 1)
        nn.init.zeros_(self.score.weight)
        initial_bias = math.log(
            initial_probability / (1.0 - initial_probability)
        )
        nn.init.constant_(self.score.bias, initial_bias)
        self.last_gate = None

    def forward(self, base, candidate):
        delta = candidate - base
        gate_input = torch.cat([base, candidate, delta.abs()], dim=-1)
        gate = torch.sigmoid(self.score(self.norm(gate_input)))
        self.last_gate = gate.detach()
        return base + gate * delta, gate


class GroupedResidualAttentionResamplerV5(nn.Module):
    """Refine each physical four-token group around its information-safe mean."""

    def __init__(self, token_dim, temporal_slots, gate_init):
        super().__init__()
        self.temporal_slots = temporal_slots
        self.score_norm = nn.LayerNorm(token_dim)
        self.score = nn.Linear(token_dim, 1, bias=False)
        # Start as the exact group mean so the learnable resampler cannot
        # destroy the common base at initialization.
        nn.init.zeros_(self.score.weight)
        self.residual_gate = DynamicResidualGate(token_dim, gate_init)
        self.last_attention = None
        self.last_gate = None

    def forward(self, native_tokens):
        expected = self.temporal_slots * 4
        if native_tokens.ndim != 3 or native_tokens.size(1) != expected:
            raise ValueError(
                f"V5 grouped resampling expects [B,{expected},D], "
                f"got {tuple(native_tokens.shape)}"
            )
        grouped = native_tokens.reshape(
            native_tokens.size(0),
            self.temporal_slots,
            4,
            native_tokens.size(-1),
        )
        group_mean = grouped.mean(dim=2)
        scores = self.score(self.score_norm(grouped)).squeeze(-1)
        attention = torch.softmax(scores, dim=2)
        attended = (grouped * attention.unsqueeze(-1)).sum(dim=2)
        output, gate = self.residual_gate(group_mean, attended)
        self.last_attention = attention.detach()
        self.last_gate = gate.detach()
        return output


class ResidualLocalTemporalResamplerV5(nn.Module):
    """Common native backbone plus an optional residual 16-slot resampler."""

    MODALITY_NAMES = ("video", "audio", "eeg")

    def __init__(self, token_dim, temporal_slots, dropout, gate_init):
        super().__init__()
        if temporal_slots != 16:
            raise ValueError("V5 residual alignment currently requires 16 slots")
        self.temporal_slots = temporal_slots
        self.video_projector = AlignedFeatureProjector(1408, token_dim, 0.0)
        self.audio_projector = AlignedFeatureProjector(90, token_dim, 0.0)
        self.eeg_projector = AlignedFeatureProjector(150, token_dim, 0.0)

        self.native_positions = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, 64, token_dim) * 0.02),
                nn.Parameter(torch.randn(1, 64, token_dim) * 0.02),
                nn.Parameter(torch.randn(1, temporal_slots, token_dim) * 0.02),
            ]
        )
        self.shared_position = nn.Parameter(
            torch.randn(1, temporal_slots, token_dim) * 0.02
        )
        self.modality_embedding = nn.Parameter(
            torch.randn(1, 3, 1, token_dim) * 0.02
        )
        self.native_norms = nn.ModuleList(
            [nn.LayerNorm(token_dim) for _ in range(3)]
        )
        self.aligned_norms = nn.ModuleList(
            [nn.LayerNorm(token_dim) for _ in range(3)]
        )
        self.native_dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(3)]
        )
        self.aligned_dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(3)]
        )
        self.group_resamplers = nn.ModuleDict(
            {
                "video": GroupedResidualAttentionResamplerV5(
                    token_dim, temporal_slots, gate_init
                ),
                "audio": GroupedResidualAttentionResamplerV5(
                    token_dim, temporal_slots, gate_init
                ),
            }
        )
        # Fusion, video, audio, EEG summaries each retain their native base.
        self.alignment_representation_gates = nn.ModuleList(
            [DynamicResidualGate(token_dim, gate_init) for _ in range(4)]
        )
        self.last_resampler_attention = None
        self.last_resampler_gates = None
        self.last_alignment_representation_gates = None

    def freeze_alignment_branch(self):
        for module in (
            self.group_resamplers,
            self.aligned_norms,
            self.alignment_representation_gates,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.shared_position.requires_grad = False

    @staticmethod
    def _masked_fusion_tokens(streams, active_mask):
        values = torch.stack(streams, dim=2)
        weights = active_mask.to(values.dtype)
        return (
            values * weights[:, None, :, None]
        ).sum(dim=2) / weights.sum(dim=1).clamp_min(1.0)[:, None, None]

    @staticmethod
    def _masked_summaries(streams, active_mask):
        summaries = torch.stack(
            [tokens.mean(dim=1) for tokens in streams], dim=1
        )
        weights = active_mask.to(summaries.dtype)
        fusion = (
            summaries * weights.unsqueeze(-1)
        ).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        modal = [
            summaries[:, index]
            * weights[:, index].unsqueeze(-1)
            for index in range(3)
        ]
        return fusion, modal

    def forward(self, eeg, audio, video, active_mask, align):
        if eeg.ndim != 3 or eeg.shape[1:] != (self.temporal_slots, 150):
            raise ValueError(
                f"V5 expects EEG [B,{self.temporal_slots},150], "
                f"got {tuple(eeg.shape)}"
            )
        if audio.ndim != 3 or audio.shape[1:] != (64, 90):
            raise ValueError(f"V5 expects audio [B,64,90], got {tuple(audio.shape)}")
        if video.ndim != 3 or video.shape[1:] != (64, 1408):
            raise ValueError(
                f"V5 expects video [B,64,1408], got {tuple(video.shape)}"
            )

        projected = (
            self.video_projector(video),
            self.audio_projector(audio),
            self.eeg_projector(eeg),
        )
        native_tokens = tuple(
            self.native_norms[index](
                self.native_dropouts[index](
                    values
                    + self.native_positions[index]
                    + self.modality_embedding[:, index]
                )
            )
            for index, values in enumerate(projected)
        )
        native_fusion, native_modal = self._masked_summaries(
            native_tokens, active_mask
        )
        if not align:
            self.last_resampler_attention = None
            self.last_resampler_gates = None
            self.last_alignment_representation_gates = None
            return {
                "native_tokens": native_tokens,
                "aligned_tokens": None,
                "fusion_tokens": native_fusion.unsqueeze(1),
                "base_representations": [native_fusion] + native_modal,
                "alignment_aux_representations": None,
            }

        aligned_pre_position = (
            self.group_resamplers["video"](projected[0]),
            self.group_resamplers["audio"](projected[1]),
            projected[2],
        )
        aligned_tokens = tuple(
            self.aligned_norms[index](
                self.aligned_dropouts[index](values)
                + self.shared_position
                + self.modality_embedding[:, index]
            )
            for index, values in enumerate(aligned_pre_position)
        )
        aligned_fusion_tokens = self._masked_fusion_tokens(
            aligned_tokens, active_mask
        )
        aligned_fusion, aligned_modal = self._masked_summaries(
            aligned_tokens, active_mask
        )
        residual_representations = []
        representation_gates = []
        for index, (base, candidate) in enumerate(
            zip(
                [native_fusion] + native_modal,
                [aligned_fusion] + aligned_modal,
            )
        ):
            refined, gate = self.alignment_representation_gates[index](
                base, candidate
            )
            if index > 0:
                active = active_mask[:, index - 1].to(refined.dtype).unsqueeze(-1)
                refined = refined * active
                gate = gate * active
            residual_representations.append(refined)
            representation_gates.append(gate.detach())

        self.last_resampler_attention = torch.stack(
            [
                self.group_resamplers[name].last_attention
                for name in ("video", "audio")
            ],
            dim=1,
        )
        self.last_resampler_gates = torch.stack(
            [
                self.group_resamplers[name].last_gate
                for name in ("video", "audio")
            ],
            dim=1,
        )
        self.last_alignment_representation_gates = torch.stack(
            representation_gates, dim=1
        )
        return {
            "native_tokens": native_tokens,
            "aligned_tokens": aligned_tokens,
            "fusion_tokens": aligned_fusion_tokens,
            "base_representations": residual_representations,
            "alignment_aux_representations": (
                [aligned_fusion] + aligned_modal
            ),
        }


class HighResolutionResidualFMPAV5(nn.Module):
    """Query native high-resolution evidence and fuse it at representation level."""

    MODALITY_NAMES = ("video", "audio", "eeg")

    def __init__(
        self,
        token_dim,
        temporal_slots,
        num_heads,
        dropout,
        local_radius,
        gate_init,
    ):
        super().__init__()
        if local_radius < 0 or local_radius >= temporal_slots:
            raise ValueError("V5 local radius must be in [0, temporal_slots)")
        self.temporal_slots = temporal_slots
        self.local_radius = local_radius
        self.query_norms = nn.ModuleList(
            [nn.LayerNorm(token_dim) for _ in range(3)]
        )
        self.kv_norms = nn.ModuleList(
            [nn.LayerNorm(token_dim) for _ in range(3)]
        )
        # Preserve the original FMPA sequence: modality-internal
        # contextualization followed by fusion-guided cross-retrieval. The
        # same module is used in C01 and C11; only the query grid/mask changes.
        self.modal_self_attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_output_norms = nn.ModuleList(
            [nn.LayerNorm(token_dim) for _ in range(3)]
        )
        self.self_dropout = nn.Dropout(dropout)
        self.modality_query_embedding = nn.Parameter(
            torch.randn(1, 3, 1, token_dim) * 0.02
        )
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    token_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(3)
            ]
        )
        self.candidate_norms = nn.ModuleList(
            [nn.LayerNorm(token_dim) for _ in range(3)]
        )
        self.representation_gates = nn.ModuleList(
            [DynamicResidualGate(token_dim, gate_init) for _ in range(3)]
        )

        query_slots = torch.arange(temporal_slots)[:, None]
        native_64_slots = torch.arange(temporal_slots * 4)[None, :] // 4
        native_16_slots = torch.arange(temporal_slots)[None, :]
        self.register_buffer(
            "local_mask_64",
            (query_slots - native_64_slots).abs() > local_radius,
            persistent=False,
        )
        self.register_buffer(
            "local_mask_16",
            (query_slots - native_16_slots).abs() > local_radius,
            persistent=False,
        )
        self.last_representation_gates = None
        self.last_cross_attention = None
        self.last_cross_attention_key_lengths = None
        self.last_cross_attention_query_length = None

    def forward(
        self,
        fusion_query,
        native_tokens,
        base_representations,
        active_mask,
        aligned_query,
    ):
        modal_representations = []
        evidence_representations = []
        gates = []
        attentions = []
        key_lengths = []
        for index, tokens in enumerate(native_tokens):
            query = self.query_norms[index](
                fusion_query + self.modality_query_embedding[:, index]
            )
            values = self.kv_norms[index](tokens)
            contextualized, _ = self.modal_self_attention(
                values,
                values,
                values,
                need_weights=False,
            )
            values = self.self_output_norms[index](
                values + self.self_dropout(contextualized)
            )
            if aligned_query:
                attention_mask = (
                    self.local_mask_16 if index == 2 else self.local_mask_64
                )
            else:
                attention_mask = None
            cross, attention = self.cross_attentions[index](
                query,
                values,
                values,
                attn_mask=attention_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            candidate = self.candidate_norms[index](cross).mean(dim=1)
            active = active_mask[:, index].to(candidate.dtype).unsqueeze(-1)
            candidate = candidate * active
            refined, gate = self.representation_gates[index](
                base_representations[index + 1], candidate
            )
            refined = refined * active
            gate = gate * active
            modal_representations.append(refined)
            evidence_representations.append(candidate)
            gates.append(gate.detach())
            attentions.append(attention.detach())
            key_lengths.append(values.size(1))

        max_queries = max(values.size(-2) for values in attentions)
        max_keys = max(values.size(-1) for values in attentions)
        padded = [
            F.pad(
                values,
                (
                    0,
                    max_keys - values.size(-1),
                    0,
                    max_queries - values.size(-2),
                ),
            )
            for values in attentions
        ]
        self.last_representation_gates = torch.stack(gates, dim=1)
        self.last_cross_attention = torch.stack(padded, dim=1)
        self.last_cross_attention_key_lengths = key_lengths
        self.last_cross_attention_query_length = fusion_query.size(1)
        return (
            [base_representations[0]] + modal_representations,
            [base_representations[0]] + evidence_representations,
        )


class LightweightRepresentationAuxClassifier(nn.Module):
    """Training-only linear supervision over four branch representations."""

    def __init__(self, token_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim * 4),
            nn.Linear(token_dim * 4, len(LABELS)),
        )

    def forward(self, representations):
        return self.classifier(torch.cat(representations, dim=-1))


class MeanConcatClassifier(nn.Module):
    def __init__(self, token_dim, hidden_dim, dropout, include_fusion):
        super().__init__()
        self.include_fusion = include_fusion
        parts = 4 if include_fusion else 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim * parts),
            nn.Linear(token_dim * parts, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(LABELS)),
        )

    def forward(self, video, audio, eeg, active_mask, fusion=None):
        representations = []
        if self.include_fusion:
            representations.append(fusion.mean(dim=1))
        for index, tokens in enumerate([video, audio, eeg]):
            active = active_mask[:, index].to(tokens.dtype).unsqueeze(-1)
            representations.append(tokens.mean(dim=1) * active)
        return self.classifier(torch.cat(representations, dim=-1))


class TAlignFMPA(nn.Module):
    """Full V5 EAV classifier with formal-source initialization compatibility.

    ``config`` supplies token_dim, temporal_slots, num_heads, hidden_dim,
    dropout, v5_local_radius, and v5_residual_gate_init. Inputs are EEG
    [B,16,150], audio [B,64,90], and video [B,64,1408]. ``active_mask`` has
    shape [B,3], in video/audio/EEG order, and defaults to all three active.
    Training-only auxiliary logits and detached attention/gate diagnostics
    remain available on the same attributes as the formal source.
    """

    def __init__(self, config):
        super().__init__()
        self.model_type = "residual_align_fmpa_v5"
        self.v5_enabled = True
        self.v5_alignment_enabled = True
        self.v5_fmpa_enabled = True
        self.last_eeg_tokens = None
        if config.temporal_slots != 16:
            raise ValueError("V5 residual alignment currently requires 16 slots")
        if not 0 <= config.v5_local_radius < config.temporal_slots:
            raise ValueError("V5 local radius must be in [0, temporal_slots)")
        if not 0 < config.v5_residual_gate_init < 1:
            raise ValueError("Residual gate initialization must be in (0, 1)")

        advance_before_classifier(
            config.token_dim, config.hidden_dim, config.temporal_slots
        )
        self.mlce_classifier = MeanConcatClassifier(
            config.token_dim, config.hidden_dim, config.dropout, include_fusion=True
        )
        self.v5_frontend = ResidualLocalTemporalResamplerV5(
            config.token_dim,
            config.temporal_slots,
            config.dropout,
            config.v5_residual_gate_init,
        )
        self.v5_fmpa = HighResolutionResidualFMPAV5(
            config.token_dim,
            config.temporal_slots,
            config.num_heads,
            config.dropout,
            config.v5_local_radius,
            config.v5_residual_gate_init,
        )
        self.v5_alignment_aux_classifier = LightweightRepresentationAuxClassifier(
            config.token_dim
        )
        self.v5_fmpa_aux_classifier = LightweightRepresentationAuxClassifier(
            config.token_dim
        )
        self.last_v5_alignment_aux_logits = None
        self.last_v5_fmpa_aux_logits = None
        advance_after_auxiliary_heads(config.token_dim, config.hidden_dim)

    def _forward_v5(self, eeg, audio, video, active_mask):
        branch = self.v5_frontend(
            eeg,
            audio,
            video,
            active_mask,
            align=self.v5_alignment_enabled,
        )
        self.last_eeg_tokens = branch["native_tokens"][2]
        if branch["alignment_aux_representations"] is not None:
            self.last_v5_alignment_aux_logits = (
                self.v5_alignment_aux_classifier(
                    branch["alignment_aux_representations"]
                )
            )
        representations = branch["base_representations"]
        if self.v5_fmpa_enabled:
            representations, evidence_representations = self.v5_fmpa(
                branch["fusion_tokens"],
                branch["native_tokens"],
                representations,
                active_mask,
                aligned_query=self.v5_alignment_enabled,
            )
            self.last_v5_fmpa_aux_logits = self.v5_fmpa_aux_classifier(
                evidence_representations
            )
        return self.mlce_classifier(
            representations[1].unsqueeze(1),
            representations[2].unsqueeze(1),
            representations[3].unsqueeze(1),
            active_mask,
            representations[0].unsqueeze(1),
        )

    def forward(self, eeg, audio, video, active_mask=None):
        self.last_v5_alignment_aux_logits = None
        self.last_v5_fmpa_aux_logits = None
        if active_mask is None:
            active_mask = torch.ones(
                eeg.size(0), 3, dtype=torch.bool, device=eeg.device
            )
        return self._forward_v5(eeg, audio, video, active_mask)
