import torch
import math


class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super().__init__()

        self.sigma = sigma
        self.eps = 1e-8

    def median(self, tensor):
        tensor = tensor.flatten().sort()[0]
        length = tensor.shape[0]

        if length % 2 == 0:
            szh = length // 2
            kth = [szh - 1, szh]
        else:
            kth = [(length - 1) // 2]
        return tensor[kth].mean()

    def forward(self, X, Y):
        XX = X.matmul(X.T)
        XY = X.matmul(Y.T)
        YY = Y.matmul(Y.T)

        norm = XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0) - 2 * XY

        # Apply median heuristic as PyTorch does not give true median
        if self.sigma is None:
            h = self.median(norm.detach()) / (2 * torch.tensor(math.log(X.size(0) + 1)))
        else:
            h = self.sigma**2

        gamma = 1.0 / (2 * h + self.eps)
        K_XY = (-gamma * norm).exp()

        return K_XY
