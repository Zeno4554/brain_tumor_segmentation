"""Training objectives; the default remains the original Dice+BCE objective."""
import torch
import segmentation_models_pytorch as smp


class SegmentationObjective(torch.nn.Module):
    def __init__(self, name='dice_bce'):
        super().__init__()
        if name == 'dice_bce':
            self.overlap = smp.losses.DiceLoss('binary')
        elif name == 'tversky_bce':
            self.overlap = smp.losses.TverskyLoss('binary', alpha=0.3, beta=0.7)
        else:
            raise ValueError(f'Unknown segmentation objective: {name}')
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, masks):
        return self.overlap(logits, masks) + self.bce(logits, masks)
