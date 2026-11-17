#!/usr/bin/env python
"""Pool of past fakes: the discriminator sees a mix of new and older fakes, so it keeps a memory of the generator."""

import torch                                     # tensors and random draws


class ImagePool:
    """Up to size fakes with their masks; size 0 = off, every query returns its input."""

    def __init__(self, size: int = 50):
        """Empty pool."""
        self.size = size
        self.images: list[torch.Tensor] = []     # one c h w fake per entry
        self.masks: list[torch.Tensor] = []      # one 1 h w mask per entry

    def query(self, images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """For each fake of the batch: kept while the pool fills, then half the time swapped with an old one."""
        if self.size == 0:
            return images, masks
        out_images, out_masks = [], []
        for image, mask in zip(images.detach(), masks):
            if len(self.images) < self.size:                       # pool not full: store and use the new one
                self.images.append(image.clone())
                self.masks.append(mask.clone())
                out_images.append(image)
                out_masks.append(mask)
            elif torch.rand(1).item() < 0.5:                       # use an old one, store the new one in its place
                k = int(torch.randint(0, self.size, (1,)))
                out_images.append(self.images[k])
                out_masks.append(self.masks[k])
                self.images[k], self.masks[k] = image.clone(), mask.clone()
            else:                                                  # use the new one
                out_images.append(image)
                out_masks.append(mask)
        return torch.stack(out_images), torch.stack(out_masks)
