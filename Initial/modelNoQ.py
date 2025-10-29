import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from modelBase import ModelBase
from util.Viewer import Viewer
from util.readDatas import ReadDatas

from torch import nn, optim


import os
import wandb
os.environ["WANDB_API_KEY"] = "e6dd69e5ba37b74ef8d3ef0fa9dd28a33e4eeb6e"

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, D]
        return x

class ViTEncoder(nn.Module):
    def __init__(self, in_channels=3, img_size=288, patch_size=24, emb_dim=128, n_layers=2, n_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.img_size = img_size

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        B, N, D = x.shape
        H_out = W_out = self.img_size // self.patch_size
        x = x.transpose(1,2).reshape(B, D, H_out, W_out)  # mapa 2D latente
        return x  # [B, D, H', W']



class TransformerDecoder(nn.Module):
    def __init__(self, emb_dim: int = 128, img_size: int = 288, patch_size: int = 24, n_layers: int = 2, n_heads: int = 8, out_channels: int = 3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(emb_dim, patch_size * patch_size * out_channels)
        decoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        B, D, H, W = z_q.shape
        tokens = z_q.flatten(2).permute(0, 2, 1)  # [B, N, D]
        tokens = self.transformer(tokens)
        x = self.proj(tokens).view(B, self.img_size // self.patch_size, self.img_size // self.patch_size, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, self.img_size, self.img_size)
        return torch.tanh(x)



class HybridTransformerDecoder(nn.Module):
    def __init__(
        self,
        emb_dim: int = 128,
        img_size: int = 288,
        patch_size: int = 24,
        n_layers: int = 2,
        n_heads: int = 8,
        out_channels: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # --- Transformer global ---
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        # --- Projeção para patch ---
        self.proj = nn.Linear(emb_dim, patch_size * patch_size * out_channels)

        # --- CNN refinadora para detalhes locais ---
        self.refiner = nn.Sequential(
            nn.Conv2d(out_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        B, D, H, W = z_q.shape

        # --- Flatten para transformer ---
        tokens = z_q.flatten(2).permute(0, 2, 1)  # [B, N, D]
        tokens = self.transformer(tokens)         # [B, N, D]

        # --- Reprojetar para patches ---
        x = self.proj(tokens).view(
            B,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.patch_size,
            self.patch_size,
            -1,
        )  # [B, H_p, W_p, ph, pw, C]

        # --- Rearranjo para imagem completa ---
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(
            B, -1, self.img_size, self.img_size
        )  # [B, C, H, W]

        # --- Refinador CNN ---
        
        x = self.refiner(x)

        return torch.tanh(x)  # normalizado [-1, 1]


class ModelNoQ(ModelBase):
    def __init__(self,device):
        super().__init__()   
        self.device = device
   
        self.encoder =ViTEncoder()
 
        self.decoder = HybridTransformerDecoder()
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
    
     
        out = self.decoder(out) 
        return out



    

    def getFeature(self,x):

        return None
    
    def getnNumEmbeddings(self):
        return None
    
    def getEmbeddingDim(self):
        return None
    
    def getVector(self,i:int):
        return None


    def prepareInputData(self,x):
        x=x.to(self.device) 
        x = x[:,:3,:,:]
        return x



def teste0():
    video0 = ReadDatas.readData("./",["resultado.npz"])[0]
    video0 = video0[:3,:,:]
    test = video0.unsqueeze(0).float()
    print(video0.shape)
    model =ViTEncoder()
    out = model(test)
    print(out.shape)

 

    decoder = TransformerDecoder()
    x_rec = decoder(out) 
    print(x_rec.shape)

def teste1():
    model = ModelNoQ()
    video0 = ReadDatas.readData("./",["resultado.npz"])[0]
    video0 = video0[:3,:,:]
    test = video0.unsqueeze(0).float()
    out = model(test)[0]
    print(out.shape)
    





    





if __name__ == "__main__":
    teste0()