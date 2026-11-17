# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains the masked LPIPS loss
(taken from the tamrfsits project, unchanged)
"""
import torch
from einops import repeat
from piq import LPIPS  # type: ignore
from piq.utils import _reduce  # type: ignore
from torch import nan_to_num


def mask_to_nan(data: torch.Tensor, mask: torch.Tensor):
    """
    Set masked pixels to nan
    """
    broadcast_mask = repeat(mask, "b w h -> b 3 w h")
    return torch.masked_fill(data, broadcast_mask, torch.nan)


class MaskedLPIPSLoss(torch.nn.Module):
    """
    This class implements masking on top of the lpips loss
    """

    def __init__(self, lpips: LPIPS):
        super().__init__()
        self.lpips = lpips

    def forward(
        self, pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor | None:
        if pred.shape[0] == 0:
            return None

        assert pred.shape == ref.shape
        assert mask.dim() == 3
        for i, j in enumerate((0, 2, 3)):
            assert mask.shape[i] == pred.shape[j]

        pred_features = self.lpips.get_features(pred)

        # First, we set masked data to NaN
        # Here we assume that only the reference needs to be masked
        with torch.no_grad():
            ref_masked = mask_to_nan(ref, ~mask)
            ref_features = self.lpips.get_features(ref_masked)

            # We used masked data to obtain a validity mask of vgg features
            validity_mask = [~torch.isnan(f) for f in ref_features]

            nb_valid_features = sum(v.sum() for v in validity_mask)
            total_nb_features = sum(v.numel() for v in validity_mask)

            if nb_valid_features == 0:
                return None

            # Now, set nan in ref_features to 0
            ref_features = [nan_to_num(f) for f in ref_features]

        for f in ref_features:
            assert torch.all(~torch.isnan(f))

        # Compute distances
        distances = self.lpips.compute_distance(pred_features, ref_features)

        # Same thing as in original piq code, but weighted by validity_mask
        loss = torch.cat(
            [
                (w.to(d) * d * m).sum(dim=(2, 3))
                / torch.maximum(m.sum(dim=(2, 3)), torch.tensor(1e-6, device=d.device))
                for d, w, m in zip(distances, self.lpips.weights, validity_mask)
            ],
            dim=1,
        ).sum(dim=1)

        # This scaling ensures that fewer valid features does not automatically
        # yield lower loss values
        return (total_nb_features / nb_valid_features) * _reduce(
            loss, self.lpips.reduction
        )
