import torch
import os
import wandb
os.environ["WANDB_API_KEY"] = "e6dd69e5ba37b74ef8d3ef0fa9dd28a33e4eeb6e"
from readDatas import ReadDatas


import tempfile
import imageio

class Viewer():



    def saveTensorAsImg(img,name,caption):
        def add_white_grid_torch(
            img: torch.Tensor,
            cell_size: int,
            gap: int = 1,
            color: float | tuple[float, float, float] = 1.0
        ) -> torch.Tensor:
            """
                Insere linhas brancas (ou coloridas) entre blocos da imagem, expandindo o tamanho.

                Args:
                    img: tensor (C, H, W), com valores em [0,1] ou [0,255]
                    cell_size: tamanho de cada célula (ex: 24)
                    gap: espessura da linha
                    color: valor ou tupla RGB (no mesmo range de img)
                
                Retorna:
                    Novo tensor (C, H_out, W_out) com as linhas inseridas.
            """
            assert img.ndim == 3, f"Esperado tensor (C,H,W), recebido {img.shape}"
            C, H, W = img.shape

            # número de células
            n_rows = H // cell_size
            n_cols = W // cell_size

            # tamanhos novos
            H_out = H + (n_rows - 1) * gap
            W_out = W + (n_cols - 1) * gap

            # cria imagem nova preenchida com a cor do grid
            if isinstance(color, (int, float)):
                color = torch.tensor([color] * C, dtype=img.dtype, device=img.device)
            else:
                color = torch.tensor(color, dtype=img.dtype, device=img.device)
            out = color.view(C, 1, 1).expand(C, H_out, W_out).clone()

            # copia blocos originais para dentro da nova imagem
            for i in range(n_rows):
                for j in range(n_cols):
                    y = i * (cell_size + gap)
                    x = j * (cell_size + gap)
                    out[:, y:y+cell_size, x:x+cell_size] = img[
                        :, i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size
                    ]
            return out
        img = img[:3,:,:]
        grid_img = add_white_grid_torch(img, cell_size=24, gap=2, color=1.0)
        wandb.log({
            name: wandb.Image(grid_img, caption=caption)
        })
    


    def saveTensorAsVideo(img,name,caption):
        def image_grid_to_video(img: torch.Tensor, frame_size: int = 24) -> torch.Tensor:
            """
            Converte uma imagem em grade (por exemplo 12x12 blocos de 24x24)
            em uma sequência de frames (vídeo).

            Args:
                img: tensor (C, H, W)
                frame_size: tamanho de cada frame

            Retorna:
                Tensor (T, C, H, W), onde T = n_rows * n_cols
            """
            assert img.ndim == 3, f"Esperado tensor (C,H,W), recebido {img.shape}"
            C, H, W = img.shape
            assert H % frame_size == 0 and W % frame_size == 0, \
                "Altura e largura devem ser múltiplos de frame_size"

            n_rows = H // frame_size
            n_cols = W // frame_size
            frames = []

            for i in range(n_rows):
                for j in range(n_cols):
                    y0, y1 = i * frame_size, (i + 1) * frame_size
                    x0, x1 = j * frame_size, (j + 1) * frame_size
                    frame = img[:, y0:y1, x0:x1]
                    frames.append(frame)

            video = torch.stack(frames, dim=0)  # (T, C, H, W)
            return video
        img = img[:3,:,:]
        video = image_grid_to_video(img,24)
        assert video.ndim == 4, f"Esperado tensor (T,C,H,W), recebido {video.shape}"
        T, C, H, W = video.shape

        # --- normaliza ---
        if video.max() <= 1.0:
            video = (video * 255).clamp(0, 255)
        video = video.to(torch.uint8)

        # --- converte para formato esperado por wandb: (T, H, W, C)
        video_np = video.permute(0, 2, 3, 1).cpu().numpy()

        # --- loga ---
        wandb.log({name: wandb.Video(video_np, fps=10, format="mp4", caption=caption)})


    def saveTensorAsGIF(
        img: torch.Tensor,
        name: str,
        caption: str = "",
        frame_size: int = 24,
        fps: int = 10
    ):
        """
        Converte uma imagem grid (C,H,W) em GIF, ordem:
        esquerda → direita, topo → baixo, e loga no W&B.
        """
        # pega apenas 3 canais
        img = img[:3, :, :]
        C, H, W = img.shape

        n_rows = H // frame_size
        n_cols = W // frame_size
        frames = []

        for i in range(n_rows):      # topo → baixo
            for j in range(n_cols):  # esquerda → direita
                y0, y1 = i * frame_size, (i + 1) * frame_size
                x0, x1 = j * frame_size, (j + 1) * frame_size
                frame = img[:, y0:y1, x0:x1]  # C,H,W

                # normaliza para 0-255 se necessário
                if frame.max() <= 1.0:
                    frame = (frame * 255).clamp(0, 255)
                
                # converte para numpy (H,W,C)
                frame_np = frame.permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
                frames.append(frame_np)

        # salva GIF temporário
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
            gif_path = tmpfile.name

        imageio.mimsave(gif_path, frames, fps=fps)

        # loga no W&B
        wandb.log({name: wandb.Video(gif_path, fps=fps, format="gif", caption=caption)})

        # remove arquivo temporário
        os.remove(gif_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mode="disabled",
   
    wandb.init(
    project="VQVAE",
    name = "Visualição",
    config={
     "test": 1,

        }
    )
  
   
    video0 = ReadDatas.readData("./",["resultado.npz"])[0]
    Viewer.saveTensorAsImg(video0,"controlii","trainImagens")
    Viewer.saveTensorAsGIF(video0,"control","trainVideo")

 
    videos = ReadDatas.readData("./data/")
    for i, video in enumerate(videos):
        Viewer.saveTensorAsImg(video,"train",f"match{i}")
        Viewer.saveTensorAsGIF(video,"trainV",f"Video{i}")
   

    videos = ReadDatas.readData("./dataValidation/")
    for i, video in enumerate(videos):
        print("val",i)
        Viewer.saveTensorAsImg(video, "Validation", f"matche{i}")
        Viewer.saveTensorAsGIF(video,"ValidationV",f"Video{i}")

  

    