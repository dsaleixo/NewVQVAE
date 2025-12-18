import sys, os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from setup_env import PROJECT_ROOT

import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from rgb2.rgb22 import RGB
from util.Viewer import Viewer
from util.analysis import Analysis

from util.readDatas import ReadDatas
from torch.nn import functional as F

from newRGB.newRGBModel import RGBVQVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device)",device)

model = RGBVQVAE(device,z_dim=128)
trainLoader,testLoader,valLoader=ReadDatas.loadDataLoader()

optimizer = optim.Adam(
    model.parameters(),
    lr=3e-4
)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()

    running = {
        "loss": 0.0,
        "recon": 0.0,
        "vq": 0.0,
        "temp": 0.0,
        "perp": 0.0,
        "used": 0.0
    }

    for i, grid in enumerate(trainLoader):
        grid = grid.to(device)
        grid = grid[:,:3,:,:]
        out = model(grid)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()

        # evita instabilidade no VQ
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # logging
        running["loss"] += loss.item()
        running["recon"] += out["recon"].item()
        running["vq"] += out["vq"].item()
        running["temp"] += out["temp"].item()
        running["perp"] += out["perplexity"].item()
        running["used"] += out["used_codes"]

        if i % 1 == 0:
            print(
                f"[Epoch {epoch} | Batch {i}] "
                f"Loss {loss.item():.4f} | "
                f"Recon {out['recon'].item():.4f} | "
                f"VQ {out['vq'].item():.4f} | "
                f"Temp {out['temp'].item():.4f} | "
                f"Perp {out['perplexity'].item():.2f} | "
                f"Used {out['used_codes']}/32"
            )

    # médias por época
    n = len(trainLoader)
    print(
        f"\n== Epoch {epoch} Summary ==\n"
        f"Loss: {running['loss']/n:.4f}\n"
        f"Recon: {running['recon']/n:.4f}\n"
        f"VQ: {running['vq']/n:.4f}\n"
        f"Temp: {running['temp']/n:.4f}\n"
        f"Perplexity: {running['perp']/n:.2f}\n"
        f"Used Codes: {running['used']/n:.2f}/32\n"
    )
