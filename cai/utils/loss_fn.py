import torch
import torch.nn as nn

class SmoothMacroF1Loss(nn.Module):
    def __init__(self, threshold=0.5, beta=1.0, eta=0.0):
        super().__init__()
        self.beta = beta
        self.eta = eta
        self.threshold = threshold
        self.sigmoid = lambda x : torch.special.expit(self.beta * x + self.eta)

    def forward(self, logits_, labels):
        y_hat = self.sigmoid(logits_)

        tp2 = torch.sum(y_hat * y, dim=0) * 2
        fp = torch.sum(y_hat * (1-y), dim=0)
        fn = torch.sum((1-y_hat) * y, dim=0)

        f1 = tp2 / (tp2 + fp + fn + torch.finfo(torch.float32).eps)
        loss = 1 - f1
        loss = loss.mean()
        
        return loss

