import torch
import torch.nn as nn
import torch.nn.functional as F

class PretrainingLoss(nn.Module):
    def __init__(self, encoder):
        super(PretrainingLoss, self).__init__()
        self.encoder = encoder

    def channel_discrimination_loss(self, seg1, seg2):
        emb1 = self.encoder(seg1)
        emb2 = self.encoder(seg2)
        return F.mse_loss(emb1, emb2)

    def context_swapping_loss(self, seg1, seg2, swapped):
        batch_size = seg1.size(0)
        loss = 0.0
        for i in range(batch_size):
            emb1 = self.encoder(seg1[i].unsqueeze(0))
            emb2 = self.encoder(seg2[i].unsqueeze(0))
            if swapped[i]:
                loss += torch.mean((emb1 - emb2) ** 2)
        return loss / batch_size

    def forward(self, seg1, seg2, swapped):
        cd_loss = self.channel_discrimination_loss(seg1, seg2)
        cs_loss = self.context_swapping_loss(seg1, seg2, swapped)
        return cd_loss + cs_loss
