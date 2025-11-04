import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from setup_env import PROJECT_ROOT


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from modelBase import ModelBase
from util.Viewer import Viewer
from util.readDatas import ReadDatas

from torch import nn, optim
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
os.environ["WANDB_API_KEY"] = "e6dd69e5ba37b74ef8d3ef0fa9dd28a33e4eeb6e"


# ---------- utilitários para split/join (grid) ----------
def split_grid(x: torch.Tensor, frame_size: int) -> List[torch.Tensor]:
    """
    Divide uma imagem-grid [B, C, H_total, W_total] em lista de frames [B, C, h, w].
    Ordem: row-major (top-left → right → next row).
    """
    B, C, H, W = x.shape
    n_rows = H // frame_size
    n_cols = W // frame_size
    frames: List[torch.Tensor] = []
    for i in range(n_rows):
        for j in range(n_cols):
            f = x[:, :, i * frame_size:(i + 1) * frame_size, j * frame_size:(j + 1) * frame_size]
            frames.append(f)
    return frames


def join_grid(frames: List[torch.Tensor], n_rows: int, n_cols: int) -> torch.Tensor:
    """
    Junta lista de frames [B, C, h, w] em uma única imagem [B, C, n_rows*h, n_cols*w].
    n_rows, n_cols devem corresponder à divisão original.
    """
    B, C, h, w = frames[0].shape
    out = torch.zeros(B, C, n_rows * h, n_cols * w, device=frames[0].device, dtype=frames[0].dtype)
    for idx, f in enumerate(frames):
        i, j = divmod(idx, n_cols)
        out[:, :, i * h:(i + 1) * h, j * w:(j + 1) * w] = f
    return out


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
    def __init__(self, in_channels=7, img_size=288, patch_size=24, emb_dim=128, n_layers=3, n_heads=16):
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

    @torch.no_grad()
    def init_from_features(self, z: torch.Tensor) -> None:
        """
        Inicializa o codebook a partir de vetores de features reais.
        Espera um tensor [N, D].
        """
        assert z.shape[1] == self.embedding_dim, "Dimensão incompatível."
        n = min(z.shape[0], self.num_embeddings)
        idx = torch.randperm(z.shape[0])[:n]
        self._embedding[:n].copy_(z[idx])
        self._embedding_avg.copy_(self._embedding)
        print(f"✅ Codebook inicializado com {n} features reais.")

    def _init_codebook(self, method: str) -> torch.Tensor:
        if method == "uniform":
            embed = torch.empty(self.num_embeddings, self.embedding_dim)
            nn.init.uniform_(embed, -1 / self.num_embeddings, 1 / self.num_embeddings)
        elif method == "normalized":
            embed = torch.randn(self.num_embeddings, self.embedding_dim)
            embed = F.normalize(embed, dim=1)
        else:  # "normal_small"
            embed = torch.randn(self.num_embeddings, self.embedding_dim) * 0.01
        return embed




    @torch.no_grad()
    def init_from_dataloader(self, encoder: nn.Module, dataloader, device, max_samples: int = 50_000):
        """
        Inicializa o codebook com features amostradas de todo o dataset.
        Usa memória constante, coleta batches pequenos e acumula até atingir max_samples.
        """
        encoder.eval()
        feats = []
        for batch in dataloader:
            x = batch.to(device)
            #print(x.shape)
            x= x[:,:7,:,:]
            z = encoder(x)
            z = z.reshape(-1, self.embedding_dim).cpu()

            # Amostragem aleatória
            if z.shape[0] > 2048:
                idx = torch.randperm(z.shape[0])[:2048]
                z = z[idx]

            feats.append(z)
            if sum(len(f) for f in feats) > max_samples:
                break

        feats = torch.cat(feats, dim=0)[:max_samples]
        print(f"🧠 Coletados {len(feats)} vetores para inicializar o codebook.")
        self.init_from_features(feats)


    

    @torch.no_grad()
    def init_kmeans_streaming(
        self,
        encoder: nn.Module,
        dataloader,
        device,
        n_iter: int = 100,
        batch_size_kmeans: int = 2048,
    ):
        """
        Executa K-Means incremental usando batches do dataset.
        Extremamente eficiente em memória.
        """
        encoder.eval()
        kmeans = MiniBatchKMeans(n_clusters=self.num_embeddings, batch_size=batch_size_kmeans, n_init=1, max_iter=n_iter)
        for _ in range(5):
            for i, batch in enumerate(dataloader):
                x = batch.to(device)
                #print(x.shape)
                x= x[:,:7,:,:]
                z = encoder(x).reshape(-1, self.embedding_dim).cpu().numpy()
                kmeans.partial_fit(z)
                if i % 10 == 0:
                    print(f"Iteração {i}, centróides atualizados.")

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
        self._embedding.copy_(centers)
        self._embedding_avg.copy_(centers)
        print(f"✅ Codebook inicializado via streaming K-Means ({n_iter} iterações totais).")


    @torch.no_grad()
    def init_from_dataset_ema(self, encoder: nn.Module, dataloader, device, decay: float = 0.99):
        """
        Inicializa o codebook usando médias exponenciais dos features do dataset.
        """
        encoder.eval()
        embed_sum = torch.zeros_like(self._embedding)
        cluster_size = torch.zeros(self.num_embeddings, device=device)
        
        for batch in dataloader:
            x = batch.to(device)
            #print(x.shape)
            x= x[:,:7,:,:]
            z = encoder(x).reshape(-1, self.embedding_dim)
            idx = torch.randint(0, self.num_embeddings, (z.size(0),), device=device)
            embed_sum.index_add_(0, idx, z)
            cluster_size.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))

        self._embedding.copy_(embed_sum / (cluster_size.unsqueeze(1) + 1e-5))
        self._embedding_avg.copy_(self._embedding)
        print("✅ Codebook inicializado via EMA sobre o dataset.")

'''
#---------- Decoder temporal que usa um único z_q ----------
class TemporalDecoderSingleZq(nn.Module):
    def __init__(self, z_dim: int=128, frame_channels: int = 7, frame_size: int = 24, hidden: int = 128):
        """
        Decoder que, para cada passo t, usa o mesmo z_q (upsampleado) + frame anterior para produzir o frame t.
        Args:
            z_dim: canais do z_q (D)
            frame_channels: canais de saída / entrada de frames (ex: 7)
            frame_size: tamanho espacial dos frames (h == w == frame_size)
            hidden: canais internos do decoder
        """
        super().__init__()
        self._z_dim = z_dim
        self._frame_channels = frame_channels
        self._frame_size = frame_size

        # projeta z_q (após upsample) para um número razoável de canais
        self.z_proj = nn.Conv2d(z_dim, hidden, kernel_size=1)

        # bloco que recebe [z_proj, x_prev]
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(hidden + frame_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, frame_channels, kernel_size=1),
        )

    def forward(self, z_q: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: [B, D, H_z, W_z]  # same z_q used for all frames
            x_prev: [B, C, h, w]   # frame previous (recon or ground-truth if teacher forcing)
        Returns:
            x_t: [B, C, h, w]      # predicted frame
        """
        # Upsample z_q para o tamanho do frame
        z_up = F.interpolate(z_q, size=(self._frame_size, self._frame_size), mode="bilinear", align_corners=False)
        z_feat = self.z_proj(z_up)  # [B, hidden, h, w]
        # concatena com frame anterior e processa
        x = torch.cat([z_feat, x_prev], dim=1)
        x = self.conv_blocks(x)
        return x

'''
class TemporalDecoderSingleZq(nn.Module):
    def __init__(self, 
                 z_dim: int = 128, 
                 frame_channels: int = 7, 
                 frame_size: int = 24, 
                 hidden: int = 128):
        """
        Decoder totalmente conectado que usa o mesmo z_q + frame anterior para prever o próximo frame.
        Args:
            z_dim: canais do z_q (D)
            frame_channels: canais do frame (ex: 7)
            frame_size: tamanho espacial (h == w == frame_size)
            hidden: tamanho das camadas escondidas
        """
        super().__init__()
        self._z_dim = z_dim
        self._frame_channels = frame_channels
        self._frame_size = frame_size
        self._flat_frame_dim = frame_channels * frame_size * frame_size
        self._flat_z_dim = z_dim * frame_size * frame_size

        # Projeção de z_q (upsampleado)
        self.z_proj = nn.Linear(self._flat_z_dim, hidden)

        # MLP com duas camadas escondidas
        self.mlp = nn.Sequential(
            nn.Linear(hidden + self._flat_frame_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self._flat_frame_dim),
            nn.Sigmoid()  # ou Tanh, dependendo da escala da sua saída
        )

    def forward(self, z_q: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        B = z_q.size(0)

        # Upsample z_q → mesmo tamanho do frame
        z_up = F.interpolate(z_q, size=(self._frame_size, self._frame_size),
                            mode="bilinear", align_corners=False)
        
        # Achata e projeta
        z_flat = z_up.reshape(B, -1)
        z_feat = self.z_proj(z_flat)

        # Frame anterior achatado
        x_prev_flat = x_prev.reshape(B, -1)

        # Concatena z + frame anterior
        x_in = torch.cat([z_feat, x_prev_flat], dim=1)

        # Passa pela MLP
        x_out = self.mlp(x_in)

        # Reconstrói o frame [B, C, H, W]
        x_t = x_out.view(B, self._frame_channels, self._frame_size, self._frame_size)
        return x_t


# ---------- Modelo completo que recebe imagem-grid e usa UM z_q ----------
class ModelGridTemporalVQVAE(nn.Module):
    def __init__(
        self,
        device,
        frame_size: int = 24,
    ):
        """
        Args:
            encoder: recebe a imagem inteira [B, C, H_total, W_total] e retorna z: [B, D, H_z, W_z]
            quantizer: VectorQuantizerEMA que recebe z e retorna (z_q, loss, indices, perplexity, used_codes)
            decoder: TemporalDecoderSingleZq que usa z_q + x_prev → x_t
            frame_size: tamanho do frame (assume square)
        """
        super().__init__()
        self.device = device
        self.encoder = ViTEncoder()
        self.quantizer = VectorQuantizerEMA(num_embeddings=30, embedding_dim=128)
        self.decoder = TemporalDecoderSingleZq()
        self._frame_size = frame_size
        self.to(device)

    def forward(
        self,
        img_grid: torch.Tensor,
        turnOnQuantization=True,
        teacher_forcing: bool = True,
        gt_frames: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            img_grid: [B, C, H_total, W_total]  (entrada única)
            teacher_forcing: durante treino, se True usar frame real anterior (gt_frames)
            gt_frames: opcional, lista de frames reais [B, C, h, w], required se teacher_forcing=True
        Returns:
            out_grid: [B, C, H_total, W_total] (reconstrução completa em grid)
            loss_avg: tensor escalar (média das perdas de quantização ao longo dos passos - aqui é same z_q reused)
            indices, perplexity, used_codes: métricas do quantizer (geralmente referentes ao z_q)
        """
        B, C, H_total, W_total = img_grid.shape
        n_rows = H_total // self._frame_size
        n_cols = W_total // self._frame_size
        n_frames = n_rows * n_cols

        # 1) Encoder na imagem inteira -> um único z (mapa 2D)
        z = self.encoder(img_grid)  # espera [B, D, H_z, W_z]
        # 2) Quantize (single z_q)
        if turnOnQuantization:
            z_q, q_loss, indices, perplexity, used_codes = self.quantizer(z)
        else:
            # ainda não quantiza — apenas passa direto
            z_q = z
            q_loss = torch.tensor(0.0, device=img_grid.device)
            indices = None
            perplexity = torch.zeros(1, device=img_grid.device, dtype=img_grid.dtype)-1
            used_codes = torch.zeros(1, device=img_grid.device, dtype=img_grid.dtype)-1

        # 3) Split GT frames (se fornecido) para teacher forcing
        frames_gt = None
        if teacher_forcing:
            if gt_frames is None:
                # tenta extrair dos próprios dados (caso img_grid contenha GT frames)
                frames_gt = split_grid(img_grid, self._frame_size)
            else:
                frames_gt = gt_frames  # assume lista de tensors
            # frames_gt tem n_frames elementos
            assert len(frames_gt) == n_frames, f"Esperava {n_frames} frames, recebido {len(frames_gt)}"

        # 4) reconstrução passo a passo usando o mesmo z_q
        recons: List[torch.Tensor] = []
        # frame inicial: zeros (pode trocar por frame preto ou outro condicional)
        x_prev = torch.zeros(B, C, self._frame_size, self._frame_size, device=img_grid.device, dtype=img_grid.dtype)

        for t in range(n_frames):
            # se teacher forcing: usa frame real anterior (frames_gt[t-1]) quando t>0
            if teacher_forcing and t > 0:
                x_prev = frames_gt[t - 1]

            x_t = self.decoder(z_q, x_prev)  # usa o mesmo z_q em cada passo
            recons.append(x_t)
            # atualiza x_prev para próxima iteração usando a predição
            x_prev = x_t.detach()  # detach evita gradientes recorrentes indesejados

        # 5) junta grid e retorna
        out_grid = join_grid(recons, n_rows, n_cols)
        # q_loss é o loss da quantização (é o mesmo pois z_q é único); manter assim para compatibilidade
        return out_grid, q_loss, indices, perplexity, used_codes


    

    def getFeature(self,x):
        out = self.encoder(x)
        z_q, loss, indices, perplexity, used_codes = self.quantizer(out)
        return indices
    
    def getnNumEmbeddings(self):
        return self.num_embeddings
    
    def getEmbeddingDim(self):
        return self.embedding_dim
    
    def getVector(self,i:int):
        return self.quantizer._embedding[i]


    def prepareInputData(self,x):
        x=x.to(self.device) 
        x = x[:,:7,:,:]
        return x

    def initializeWeights(self,opEnc:int= -1,opVQ:int=-1,data=None):
        if opEnc ==0:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
        elif opEnc ==1:
            nn.init.normal_(self.encoder.patch_embed.proj.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.encoder.patch_embed.proj.bias, 0.0)
        elif opEnc ==2:
            with torch.no_grad():
                w = torch.randn_like(self.encoder.patch_embed.proj.weight)
                w = F.normalize(w, dim=1) * 0.02  # controla magnitude
                self.encoder.patch_embed.proj.weight.copy_(w)
                nn.init.constant_(self.encoder.patch_embed.proj.bias, 0.0)
        elif opEnc ==3:
            pass


        if opVQ ==0:
            self.quantizer._init_codebook("normal_small")
        elif opVQ ==1:
            self.quantizer._init_codebook("uniform")
        elif opVQ==2:
            self.quantizer._init_codebook("normalized")
        elif opVQ==3:
            self.quantizer.init_from_dataloader(self.encoder,data,self.device)
        elif opVQ==4:
            self.quantizer.init_kmeans_streaming(self.encoder,data,self.device,10)
        elif opVQ==5:
            self.quantizer.init_from_dataset_ema(self.encoder,data,self.device)
            

def teste0():
    print("Projeto localizado em:", PROJECT_ROOT)
    video0 = ReadDatas.readData("./",["resultado.npz"],OneHot=True)[0]
    video0 = video0[:7,:,:]
    test = video0.unsqueeze(0).float()
    print(video0.shape)
    model =ViTEncoder()
    out = model(test)
    print(out.shape)
    quantizer = VectorQuantizerEMA(num_embeddings=30, embedding_dim=128)
    z_q, loss, indices, perplexity, used_codes = quantizer(out)
    print(z_q.shape)
    decoder = TemporalDecoderSingleZq()
    frames_gt = None

    frames_gt = split_grid(test, 24)
    n_frames=144
    assert len(frames_gt) == n_frames, f"Esperava {n_frames} frames, recebido {len(frames_gt)}"

        # 4) reconstrução passo a passo usando o mesmo z_q
    recons: List[torch.Tensor] = []
    # frame inicial: zeros (pode trocar por frame preto ou outro condicional)
    x_prev = torch.zeros(1, 7, 24, 24, device=video0.device, dtype=video0.dtype)
    
    for t in range(n_frames):
            # se teacher forcing: usa frame real anterior (frames_gt[t-1]) quando t>0
            if t > 0:
                x_prev = frames_gt[t - 1]

            x_t = decoder(z_q, x_prev)  # usa o mesmo z_q em cada passo
            recons.append(x_t)
            # atualiza x_prev para próxima iteração usando a predição
            x_prev = x_t.detach()  # detach evita gradientes recorrentes indesejados

    # 5) junta grid e retorna
    out_grid = join_grid(recons, 12, 12)
    print("fim",out_grid.shape)
def teste1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelGridTemporalVQVAE(device)
    video0 = ReadDatas.readData("./",["resultado.npz"],OneHot=True)[0]
    video0 = video0[:7,:,:]
    test = video0.unsqueeze(0).float()
    out = model(test)[0]
    print(out.shape)
    





    





if __name__ == "__main__":
    teste1()