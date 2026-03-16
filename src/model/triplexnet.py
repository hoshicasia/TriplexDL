"""TriplexNet model for nucleotide-level triplex prediction.

Architecture:
- multi-scale Conv1d stem,
- dilated residual tower with SE blocks,
- FiLM conditioning from omics features,
- per-position prediction head (+ optional sequence auxiliary head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth for residual branches."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        s = self.pool(x).view(b, c)
        e = self.fc(s).view(b, c, 1)
        return x * e


class DilatedResBlock(nn.Module):
    """Residual block with dilated Conv1d, GroupNorm and DropPath."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        n_groups: int = 8,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size, padding=padding, dilation=dilation
            ),
            nn.GroupNorm(n_groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.block(x))


class TriplexNet(nn.Module):
    """
    Dilated CNN with SE attention and FiLM conditioning for triplex prediction.

    Inputs:
        sequence       [batch, seq_len, 4]   one-hot encoded DNA
        omics_features [batch, n_omics]      global chromatin/epigenomic features

    Output:
        {"logits": Tensor}
            nucleotide_level=True  → [batch, seq_len]
            nucleotide_level=False → [batch]
    """

    def __init__(
        self,
        n_omics_features: int = 221,
        n_kmer_features: int = 0,
        dropout: float = 0.2,
        n_channels: int = 128,
        n_dilated_blocks: int = 7,
        kernel_size: int = 3,
        se_reduction: int = 4,
        drop_path_rate: float = 0.15,
        aux_loss_weight: float = 0.15,
        nucleotide_level: bool = True,
        kmer_proj_dim: int = 64,
    ):
        super().__init__()
        self.nucleotide_level = nucleotide_level
        self.aux_loss_weight = aux_loss_weight
        self.n_kmer_features = int(n_kmer_features)
        self.kmer_proj_dim = int(kmer_proj_dim) if self.n_kmer_features > 0 else 0
        self.head_input_channels = n_channels * 2 + self.kmer_proj_dim

        n_groups = min(8, n_channels)
        stem_ch = n_channels // 3
        stem_rem = n_channels - 2 * stem_ch

        self.stem_k7 = nn.Sequential(
            nn.Conv1d(4, stem_ch, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.stem_k11 = nn.Sequential(
            nn.Conv1d(4, stem_ch, kernel_size=11, padding=5),
            nn.GELU(),
        )
        self.stem_k19 = nn.Sequential(
            nn.Conv1d(4, stem_rem, kernel_size=19, padding=9),
            nn.GELU(),
        )
        self.stem_proj = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
            nn.GroupNorm(n_groups, n_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        dp_rates = [
            drop_path_rate * i / max(n_dilated_blocks - 1, 1)
            for i in range(n_dilated_blocks)
        ]
        self.tower = nn.ModuleList()
        for i in range(n_dilated_blocks):
            dilation = 2**i
            self.tower.append(
                nn.Sequential(
                    DilatedResBlock(
                        n_channels,
                        kernel_size,
                        dilation,
                        dropout,
                        drop_path=dp_rates[i],
                        n_groups=n_groups,
                    ),
                    SEBlock(n_channels, se_reduction),
                )
            )

        self.film_mid_idx = n_dilated_blocks // 2

        self.film_gamma_mid = nn.Sequential(
            nn.Linear(n_omics_features, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels),
            nn.Sigmoid(),
        )
        self.film_beta_mid = nn.Sequential(
            nn.Linear(n_omics_features, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels),
        )

        self.film_gamma = nn.Sequential(
            nn.Linear(n_omics_features, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels),
            nn.Sigmoid(),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(n_omics_features, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels),
        )

        self.attn_pool_layer = nn.Conv1d(n_channels, 1, kernel_size=1)

        if self.n_kmer_features > 0:
            self.kmer_proj = nn.Sequential(
                nn.Linear(self.n_kmer_features, self.kmer_proj_dim),
                nn.LayerNorm(self.kmer_proj_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
        else:
            self.kmer_proj = None

        global_feature_dim = n_channels + n_omics_features + self.kmer_proj_dim

        if nucleotide_level:
            self.head = nn.Sequential(
                nn.Conv1d(self.head_input_channels, n_channels, kernel_size=1),
                nn.GroupNorm(n_groups, n_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(n_channels, 1, kernel_size=1),
            )

            self.aux_head = nn.Sequential(
                nn.Linear(global_feature_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(global_feature_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _attn_pool(self, x, mask):
        """Attention-weighted pooling: learns which positions matter most
        for the global sequence representation.

        Args:
            x:    [B, C, L]
            mask: [B, L] with 1=real, 0=padding, or None
        Returns:
            [B, C]
        """
        attn_logits = self.attn_pool_layer(x).squeeze(1)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.bool(), float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=1)
        return (x * attn_weights.unsqueeze(1)).sum(dim=2)

    @staticmethod
    def _masked_mean_pool(x, mask):
        if mask is None:
            return x.mean(dim=2)

        weights = mask.unsqueeze(1).to(dtype=x.dtype)
        denom = weights.sum(dim=2).clamp(min=1.0)
        return (x * weights).sum(dim=2) / denom

    def forward(
        self,
        sequence,
        omics_features,
        kmer_features=None,
        label=None,
        tissue_ids=None,
        mask=None,
    ):
        x = sequence.transpose(1, 2)

        x = torch.cat(
            [
                self.stem_k7(x),
                self.stem_k11(x),
                self.stem_k19(x),
            ],
            dim=1,
        )
        x = self.stem_proj(x)

        for i, block in enumerate(self.tower):
            x = block(x)
            if i == self.film_mid_idx:
                gm = self.film_gamma_mid(omics_features).unsqueeze(-1) * 2
                bm = self.film_beta_mid(omics_features).unsqueeze(-1)
                x = gm * x + bm

        gamma = self.film_gamma(omics_features).unsqueeze(-1) * 2
        beta = self.film_beta(omics_features).unsqueeze(-1)
        x = gamma * x + beta

        local_kmer_map = None
        pooled_kmer = None
        if self.kmer_proj is not None:
            if kmer_features is None:
                kmer_features = omics_features.new_zeros(
                    omics_features.size(0), 1, self.n_kmer_features
                )
            projected_kmers = self.kmer_proj(kmer_features)
            local_kmer_map = projected_kmers.transpose(1, 2)
            local_kmer_map = F.interpolate(
                local_kmer_map,
                size=x.size(-1),
                mode="linear",
                align_corners=False,
            )
            pooled_kmer = self._masked_mean_pool(local_kmer_map, mask)

        global_features = [omics_features]
        if pooled_kmer is not None:
            global_features.append(pooled_kmer)

        result = {}
        if self.nucleotide_level:
            x_global = self._attn_pool(x, mask)
            x_global_bc = x_global.unsqueeze(-1).expand_as(x)
            head_inputs = [x, x_global_bc]
            if local_kmer_map is not None:
                head_inputs.append(local_kmer_map)
            x_combined = torch.cat(head_inputs, dim=1)
            logits_pos = self.head(x_combined).squeeze(1)

            aux_input = torch.cat([x_global] + global_features, dim=1)
            seq_logit = self.aux_head(aux_input).squeeze(-1)
            result["logits"] = logits_pos
            result["seq_logit"] = seq_logit

            if self.training and label is not None and self.aux_loss_weight > 0:
                if label.dim() > 1:
                    seq_label = (label.sum(dim=1) > 0).float()
                else:
                    seq_label = label.float()
                aux_loss = F.binary_cross_entropy_with_logits(seq_logit, seq_label)
                result["aux_loss"] = aux_loss * self.aux_loss_weight
        else:
            pooled = self._attn_pool(x, mask)
            combined = torch.cat([pooled] + global_features, dim=1)
            logits = self.head(combined).squeeze(-1)
            result["logits"] = logits

        return result

    def __str__(self):
        n_all = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        s = super().__str__()
        s += f"\nAll parameters:       {n_all:,}"
        s += f"\nTrainable parameters: {n_train:,}"
        return s
