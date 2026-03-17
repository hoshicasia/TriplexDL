import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
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
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            m = mask.unsqueeze(1)
            s = (x * m).sum(dim=2) / (m.sum(dim=2) + 1e-8)
        else:
            s = x.mean(dim=2)
        e = self.fc(s).view(x.size(0), x.size(1), 1)
        return x * e


class DilatedResBlock(nn.Module):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        out = x + self.drop_path(self.block(x))
        if mask is not None:
            out = out * mask.unsqueeze(1)
        return out


class TriplexNet(nn.Module):
    def __init__(
        self,
        n_omics_features: int = 221,
        dropout: float = 0.2,
        n_channels: int = 128,
        n_dilated_blocks: int = 7,
        kernel_size: int = 3,
        se_reduction: int = 4,
        drop_path_rate: float = 0.15,
        aux_loss_weight: float = 0.15,
        nucleotide_level: bool = True,
        positional_omics: bool = False,
    ):
        super().__init__()
        self.nucleotide_level = nucleotide_level
        self.aux_loss_weight = aux_loss_weight
        self.positional_omics = positional_omics

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

        if positional_omics:
            omics_ch = n_channels // 3
            self.omics_encoder = nn.Sequential(
                nn.Conv1d(n_omics_features, omics_ch * 2, kernel_size=1),
                nn.GroupNorm(min(n_groups, omics_ch * 2), omics_ch * 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Conv1d(omics_ch * 2, omics_ch, kernel_size=1),
                nn.GroupNorm(min(n_groups, omics_ch), omics_ch),
                nn.GELU(),
            )
            stem_proj_in = n_channels + omics_ch
            film_input_dim = omics_ch
        else:
            self.omics_encoder = None
            stem_proj_in = n_channels
            film_input_dim = n_omics_features

        self._film_input_dim = film_input_dim

        self.stem_proj = nn.Sequential(
            nn.Conv1d(stem_proj_in, n_channels, kernel_size=1),
            nn.GroupNorm(n_groups, n_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        dilation_cycle = [1, 2, 4, 8, 16, 32, 64, 128]
        dp_rates = [
            drop_path_rate * i / max(n_dilated_blocks - 1, 1)
            for i in range(n_dilated_blocks)
        ]
        self.tower = nn.ModuleList()
        for i in range(n_dilated_blocks):
            dilation = dilation_cycle[i % len(dilation_cycle)]
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
            nn.Linear(film_input_dim, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels),
            nn.Sigmoid(),
        )
        self.film_beta_mid = nn.Sequential(
            nn.Linear(film_input_dim, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels),
        )

        self.film_gamma = nn.Sequential(
            nn.Linear(film_input_dim, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels),
            nn.Sigmoid(),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(film_input_dim, n_channels),
            nn.GELU(),
            nn.Linear(n_channels, n_channels),
        )

        attn_mid = max(n_channels // 4, 16)
        self.attn_pool_layer = nn.Sequential(
            nn.Conv1d(n_channels, attn_mid, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attn_mid, 1, kernel_size=1),
        )

        if nucleotide_level:
            self.head = nn.Sequential(
                nn.Conv1d(n_channels * 2, n_channels, kernel_size=5, padding=2),
                nn.GroupNorm(n_groups, n_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(n_channels, 1, kernel_size=1),
            )
            self.aux_head = nn.Sequential(
                nn.Linear(n_channels, n_channels // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(n_channels // 2, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(n_channels + film_input_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _masked_mean_pool(self, x, mask):
        if mask is None:
            return x.mean(dim=2)
        m = mask.unsqueeze(1)
        return (x * m).sum(dim=2) / (m.sum(dim=2) + 1e-8)

    def _attn_pool(self, x, mask):
        attn_logits = self.attn_pool_layer(x).squeeze(1)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.bool(), float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=1)
        return (x * attn_weights.unsqueeze(1)).sum(dim=2)

    def forward(self, sequence, omics_features, label=None, tissue_ids=None, mask=None):
        x = sequence.transpose(1, 2)

        x = torch.cat(
            [
                self.stem_k7(x),
                self.stem_k11(x),
                self.stem_k19(x),
            ],
            dim=1,
        )

        if self.positional_omics and self.omics_encoder is not None:
            omics_pos = omics_features.transpose(1, 2)
            omics_encoded = self.omics_encoder(omics_pos)
            if mask is not None:
                omics_encoded = omics_encoded * mask.unsqueeze(1)
            omics_global = self._masked_mean_pool(omics_encoded, mask)
            x = torch.cat([x, omics_encoded], dim=1)
        else:
            omics_global = omics_features

        x = self.stem_proj(x)

        if mask is not None:
            x = x * mask.unsqueeze(1)

        for i, block in enumerate(self.tower):
            dilated_res, se = block[0], block[1]
            x = dilated_res(x, mask=mask)
            x = se(x, mask=mask)
            if i == self.film_mid_idx:
                gm = self.film_gamma_mid(omics_global).unsqueeze(-1) * 2
                bm = self.film_beta_mid(omics_global).unsqueeze(-1)
                x = gm * x + bm
                if mask is not None:
                    x = x * mask.unsqueeze(1)

        gamma = self.film_gamma(omics_global).unsqueeze(-1) * 2
        beta = self.film_beta(omics_global).unsqueeze(-1)
        x = gamma * x + beta
        if mask is not None:
            x = x * mask.unsqueeze(1)

        result = {}
        if self.nucleotide_level:
            x_global = self._attn_pool(x, mask)
            x_global_bc = x_global.unsqueeze(-1).expand_as(x)
            x_combined = torch.cat([x, x_global_bc], dim=1)
            logits_pos = self.head(x_combined).squeeze(1)

            seq_logit = self.aux_head(x_global).squeeze(-1)
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
            combined = torch.cat([pooled, omics_global], dim=1)
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
