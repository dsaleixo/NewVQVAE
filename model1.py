import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Viewer import Viewer
from readDatas import ReadDatas

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


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99, eps: float = 1e-5):
        """
        Vector Quantizer with Exponential Moving Average (EMA) updates.
        Args:
            num_embeddings: number of codebook entries (K)
            embedding_dim: dimension of each embedding (D)
            decay: EMA decay rate
            eps: small epsilon for numerical stability
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        # Inicializa codebook e buffers para EMA
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("_embedding", embed)
        self.register_buffer("_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_embedding_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: input tensor [B, D, H, W]
        Returns:
            z_q: quantized tensor [B, D, H, W]
            loss: commitment loss
            indices: selected code indices [B, H, W]
            perplexity: codebook perplexity
            used_codes: binary mask [K], True if code used
        """
        B, D, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, D]
        flat_z = z_perm.view(-1, D)  # [B*H*W, D]

        # Distância L2 entre vetores e embeddings
        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            + self._embedding.pow(2).sum(1)
            - 2 * torch.matmul(flat_z, self._embedding.t())
        )

        # Índices do vetor mais próximo
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)

        # Quantização
        quantized = torch.matmul(encodings, self._embedding)
        quantized = quantized.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # --- EMA updates ---
        if self.training:
            cluster_size = encodings.sum(0)
            embed_sum = torch.matmul(encodings.t(), flat_z)

            self._cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self._embedding_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self._cluster_size.sum()
            cluster_size = (
                (self._cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps)
                * n
            )
            self._embedding.data.copy_(self._embedding_avg / cluster_size.unsqueeze(1))

        # Straight-through estimator
        z_q = z + (quantized - z).detach()

        # --- Métricas extras ---
        # 1. Perda de compromisso
        loss = F.mse_loss(z_q.detach(), z)

        # 2. Índices reshape
        indices = encoding_indices.view(B, H, W)

        # 3. Perplexity
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # 4. used_codes (máscara binária de códigos utilizados)
        used_codes = (encodings.sum(0) > 0).float()

        return z_q, loss, indices, perplexity, used_codes



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


class Model1(nn.Module):
    def __init__(self):
        super().__init__()   
        self.encoder =ViTEncoder()
        self.quantizer = VectorQuantizerEMA(num_embeddings=30, embedding_dim=128)
        self.decoder = HybridTransformerDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        z_q, loss, indices, perplexity, used_codes = self.quantizer(out)
     
        out = self.decoder(z_q) 
        return out, loss, indices, perplexity, used_codes 




def teste0():
    video0 = ReadDatas.readData("./",["resultado.npz"])[0]
    video0 = video0[:3,:,:]
    test = video0.unsqueeze(0).float()
    print(video0.shape)
    model =ViTEncoder()
    out = model(test)
    print(out.shape)
    quantizer = VectorQuantizerEMA(num_embeddings=30, embedding_dim=128)
    z_q, loss, indices, perplexity, used_codes = quantizer(out)
    print(z_q.shape)
    decoder = TransformerDecoder()
    x_rec = decoder(z_q) 
    print(x_rec.shape)

def teste1():
    model = Model1()
    video0 = ReadDatas.readData("./",["resultado.npz"])[0]
    video0 = video0[:3,:,:]
    test = video0.unsqueeze(0).float()
    out = model(test)[0]
    print(out.shape)
    


def teste2():


    wandb.init(
    project="VQVAE",
    name = "X1_2ee",
    config={
     "test": 1,

        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device)",device)
    model = Model1().to(device)
    trainLoader,testLoader,valLoader=ReadDatas.loadDataLoader()

    num_epochs = 100000000000
    

    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-5
    )
    
    
    x = first_batch = next(iter(valLoader))
    Viewer.saveTensorAsImg(x,"OriginalImg","trainImagens")
    Viewer.saveTensorAsGIF(x,"OriginalVideo","trainVideo")
    x = x[:3,:,:].unsqueeze(0).to(device)

    print(x.shape)
    
    for epoch in range(num_epochs):
        model.train()

         # [B, C, H, W]
        running_loss = 0.0
        running_perplexity = 0.0

        
        
        
        optimizer.zero_grad()
        
        # --- Forward ---
        

        x_rec, vq_loss, indices, perplexity, used_codes = model(x)  
      
        if epoch%1000==0:
            Viewer.saveTensorAsImg(x_rec.squeeze(),"recontructionImg","trainImagens")
            Viewer.saveTensorAsGIF(x_rec.squeeze(),"recontructioVideo","trainVideo")
       
        # --- Loss ---
        recon_loss = F.mse_loss(x_rec, x)
        loss = recon_loss +vq_loss
        
        # --- Backprop ---
        loss.backward()
        optimizer.step()
        
        # --- Logging ---
        running_loss += loss.item()
        running_perplexity += perplexity.item()
        
        wandb.log({
              
                "Train/Recon Loss": loss.item(),
         
                "Train/VQ Loss": vq_loss.item(),
                "Train/VQ Perplexity": perplexity.item(),
                "Train/VQ Used Codes": used_codes.sum().item(),
                
                
            })
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Batch [{0}], "
            f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
            f"VQ Loss: {vq_loss.item():.4f}, Perplexity: {perplexity.item():.2f}, "
            f"Used Codes: {used_codes.sum().item()}/{model.quantizer.num_embeddings}"
        )


def teste3():



    def validation(model, val_loader: DataLoader, device='cuda',): 
        model.eval()
        total_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        perplexity_loss_epoch = 0.0
        used_codes_loss_epoch = 0.0
        vq_loss_loss_epoch = 0.0
        for batch in val_loader:
            x = batch[:,:3,:,:].to(device)  # [B, C, H, W]

            x_rec, vq_loss, indices, perplexity, used_codes = model(x)              
            # --- Loss ---
            recon_loss = F.mse_loss(x_rec, x)
            loss = recon_loss*10 + vq_loss

            total_loss_epoch += loss.item()
            recon_loss_epoch += recon_loss.item()
            perplexity_loss_epoch +=perplexity.item()/x.shape[0]
            used_codes_loss_epoch +=(used_codes.sum().item())/x.shape[0]
            vq_loss_loss_epoch +=vq_loss.item()
        print(
                 
                    f"Test  ===>   Loss: {total_loss_epoch:.4f}, Recon: {recon_loss_epoch:.4f}, "
                    f"VQ Loss: {vq_loss_loss_epoch:.4f}, Perplexity: {perplexity_loss_epoch:.2f}, "
                    f"Used Codes: {used_codes_loss_epoch}/{model.quantizer.num_embeddings}\n,"
                    
                )
        wandb.log({
            
            "Test/Recon Loss": recon_loss_epoch,
            "Test/Loss": total_loss_epoch,
            "Test/VQ Loss": vq_loss_loss_epoch,
            "Test/VQ Perplexity": perplexity_loss_epoch,
            "Test/VQ Used Codes": used_codes_loss_epoch,
            
            
        })
          
        return total_loss_epoch

    def initialProcess(model,valLoader,device):
        model.eval()
        #for i in range(len(valLoader)):
        i=0
        x = valLoader[i][:3,:,:].unsqueeze(0).to(device)
        x_rec, vq_loss, indices, perplexity, used_codes = model(x)   
        imgs = [x.squeeze(),x_rec.squeeze()]
        Viewer.saveListTensorAsImg(imgs,f"RecImagemVal{i}",f"match{i}")
        Viewer.saveTensorAsGIF(imgs,f"RecVideoVal{i}",f"match{i}")


    wandb.init(
    project="VQVAE",
    name = "Loopdevagar e sempre",
    #mode ="disabled",
    resume=False,
    config={
     "test": 1,

        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device)",device)
    model = Model1().to(device)
    trainLoader,testLoader,valLoader=ReadDatas.loadDataLoader()

    num_epochs = 100000000000000
    

    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-5
    )
    
    
    
    #initialProcess(model,valLoader,device)
    initialProcess(model,valLoader,device)
    bestModelVal = validation(model,testLoader)
    for epoch in range(num_epochs):

    
    
        running_loss = 0.0
        running_perplexity = 0.0

        for batch_idx, batch in enumerate(trainLoader):
            # Supondo que batch = (x, y, z) ou apenas imagens x
            x = batch[:,:3,:,:].to(device)  # [B, C, H, W]
            model.train()
            optimizer.zero_grad()
            
            # --- Forward ---
       
            x_rec, vq_loss, indices, perplexity, used_codes = model(x)              
            # --- Loss ---
            recon_loss = F.mse_loss(x_rec, x)
            loss = recon_loss*10 + vq_loss
            
            # --- Backprop ---
            loss.backward()
            optimizer.step()
            
            # --- Logging ---
            running_loss += loss.item()
            running_perplexity += perplexity.item()
            
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], "
                    f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                    f"VQ Loss: {vq_loss.item():.4f}, Perplexity: {perplexity.item():.2f}, "
                    f"Used Codes: {used_codes.sum().item()}/{model.quantizer.num_embeddings}\n"
                )
            wandb.log({
              
                "Train/Recon Loss": recon_loss.item(),
                "Train/Loss": loss.item(),
                "Train/VQ Loss": vq_loss.item(),
                "Train/VQ Perplexity": perplexity.item(),
                "Train/VQ Used Codes": used_codes.sum().item(),
                
                
            })
        modelVal = validation(model,testLoader)
        if bestModelVal-modelVal>0.0000001:
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxUpdadtexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            bestModelVal=modelVal
            torch.save(model.state_dict(), f"BestTEstModelBest.pth")
            torch.save(model.state_dict(), f"BestTEstModel{epoch}.pth")
            wandb.save("BestTEstModelBest.pth")
            wandb.save(f"BestTEstModel{epoch}.pth")
            initialProcess(model,valLoader,device)  
            wandb.log({"Updade":1})
        else:
            wandb.log({"Updade":0})




if __name__ == "__main__":
    teste3()