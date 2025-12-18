import sys, os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from setup_env import PROJECT_ROOT
from util.readDatas import ReadDatas

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

def grid_to_video(x, frame_size=24):
    """
    x: [B, C, H, W]  where H=W=N*frame_size
    returns: [B, T, C, frame_size, frame_size]
    """
    B, C, H, W = x.shape
    assert H % frame_size == 0 and W % frame_size == 0

    gh = H // frame_size
    gw = W // frame_size

    x = x.view(B, C, gh, frame_size, gw, frame_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.contiguous().view(B, gh * gw, C, frame_size, frame_size)

    return x
class VideoEncoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, z_dim, kernel_size=3, stride=(2, 2, 2), padding=1),
        )

    def forward(self, video):
        # video: [B, T, C, H, W]
        x = video.permute(0, 2, 1, 3, 4)   # [B, C, T, H, W]
        z = self.net(x)                   # [B, D, T', H', W']
        z = z.mean(dim=[3, 4])            # spatial pool
        z = z.permute(0, 2, 1)             # [B, T', D]
        return z
class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=32, dim=128, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, dim)
        self.codebook.weight.data.uniform_(-1 / num_codes, 1 / num_codes)
        self.beta = beta

    def forward(self, z):
        # z: [B, T, D]
        z_flat = z.reshape(-1, z.size(-1))

        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )

        indices = dist.argmin(dim=1)
        z_q = self.codebook(indices).view_as(z)

        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()

        # stats
        used = torch.unique(indices).numel()
        perplexity = torch.exp(
            -torch.sum(
                torch.bincount(indices, minlength=self.codebook.num_embeddings).float()
                / indices.numel()
                * torch.log(
                    torch.bincount(indices, minlength=self.codebook.num_embeddings).float()
                    / indices.numel()
                    + 1e-10
                )
            )
        )

        return z_q, loss, perplexity, used
class WeakDecoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 24 * 24)
        )

    def forward(self, z):
        return self.net(z).view(-1, 3, 24, 24)
class TemporalPredictor(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim)
        )

    def forward(self, z):
        return self.net(z)
class RGBVQVAE(nn.Module):
    def __init__(self, device,z_dim=128):
        super().__init__()
        self.encoder = VideoEncoder(z_dim)
        self.vq = VectorQuantizer(32, z_dim)
        self.decoder = WeakDecoder(z_dim)
        self.temporal = TemporalPredictor(z_dim)
        self.to(device)

    def forward(self, grid):
        # grid: [B, 3, 288, 288]
        video = grid_to_video(grid)          # [B, T, 3, 24, 24]
        z_e = self.encoder(video)             # [B, T', D]
        z_q, vq_loss, perplexity, used = self.vq(z_e)

        # Temporal Δ prediction
        pred = self.temporal(z_q[:, :-1])
        target = z_q[:, 1:].detach()
        L_temp = F.mse_loss(pred, target - z_q[:, :-1])

        # Weak reconstruction (middle frame only)
        mid = z_q.size(1) // 2
        recon = self.decoder(z_q[:, mid])
        target_img = video[:, video.size(1)//2]

        L_recon = F.mse_loss(recon, target_img)

        loss = L_recon + vq_loss + L_temp

        return {
            "loss": loss,
            "recon": L_recon.detach(),
            "vq": vq_loss.detach(),
            "temp": L_temp.detach(),
            "perplexity": perplexity,
            "used_codes": used,
            "r" :recon
        }

def teste1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGBVQVAE(device)
    video0 = ReadDatas.readData("./",["resultado.npz"],OneHot=True)[0]
    video0 = video0[:3,:,:]
    test = video0.unsqueeze(0).float()
    test = test.to(device)
    out = model(test)
    print(out['r'].shape)
    


if __name__ == "__main__":
    teste1()