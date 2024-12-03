import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Pretraining Loss Class
class PretrainingLoss(nn.Module):
    def __init__(self, encoder):
        super(PretrainingLoss, self).__init__()
        self.encoder = encoder

    def channel_discrimination_loss(self, seg1, seg2):
        """
        Calculates channel discrimination loss between two segments.
        """
        emb1 = self.encoder(seg1)
        emb2 = self.encoder(seg2)
        return F.mse_loss(emb1, emb2)

    def context_swapping_loss(self, seg1, seg2, swapped):
        """
        Calculates context swapping loss for each element in the batch.
        Only applies the loss if `swapped` is True for that specific element.
        """
        batch_size = seg1.size(0)
        loss = 0.0
        for i in range(batch_size):
            emb1 = self.encoder(seg1[i].unsqueeze(0))
            emb2 = self.encoder(seg2[i].unsqueeze(0))
            if swapped[i]:
                loss += torch.mean((emb1 - emb2) ** 2)
        return loss / batch_size  # Average over the batch

    def forward(self, seg1, seg2, swapped):
        """
        Calculates the combined pretraining loss (CD + CS loss).
        """
        # Compute pretraining losses
        cd_loss = self.channel_discrimination_loss(seg1, seg2)
        cs_loss = self.context_swapping_loss(seg1, seg2, swapped)
        return cd_loss + cs_loss
