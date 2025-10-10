import torch
import os
import wandb
os.environ["WANDB_API_KEY"] = "e6dd69e5ba37b74ef8d3ef0fa9dd28a33e4eeb6e"
from readDatas import ReadDatas

import numpy as np
import tempfile
import imageio

class Viewer():


    def saveListTensorAsImg(imgs,name,caption):
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
       
       
       
        outs =[]
        for img in imgs:
            img = img[:3,:,:]
            grid_img = add_white_grid_torch(img, cell_size=24, gap=2, color=1.0)
            outs.append(grid_img)
        

        # Dimensões
        C, H, W = outs[0].shape
        line_width = 3  # largura da linha vermelha

        # Cria linha vermelha
        red_line = torch.zeros(C, H, line_width).to(outs[0].device)  # zeros para todos canais
        red_line[0, :, :] = 1.0  # canal R = 1
        # Para vídeos normalizados [-1,1], use red_line[0,:,:]=1 ou red_line[0,:,:]=0.5

        # Intercala imagens e linhas
        imgs_with_lines = []
        for i, img in enumerate(outs):
            imgs_with_lines.append(img)
            if i < len(outs) - 1:  # não colocar linha após a última
                imgs_with_lines.append(red_line)

        newImg = torch.cat(imgs_with_lines, dim=2)

        wandb.log({
            "Imgs/"+name: wandb.Image(newImg, caption=caption)
        },commit=False)
    
 
    '''
    def saveTensorListAsGIF(
    imgs: list[torch.Tensor],
    name: str,
    caption: str = "",
    frame_size: int = 24,
    fps: int = 10,
    line_color: tuple[int, int, int] = (255, 0, 0),  # vermelho
    line_width: int = 2
) -> None:
        """
        Recebe uma lista de grids (C,H,W), fatia cada uma em vídeos (frame_size×frame_size)
        e concatena os vídeos lado a lado, adicionando linhas vermelhas entre eles.
        """

        all_videos = []

        # --- fatia cada imagem em uma sequência de frames ---
        for img in imgs:
            img = img[:3, :, :]  # mantém só RGB
            C, H, W = img.shape
            n_rows = H // frame_size
            n_cols = W // frame_size

            frames = []
            for i in range(n_rows):
                for j in range(n_cols):
                    y0, y1 = i * frame_size, (i + 1) * frame_size
                    x0, x1 = j * frame_size, (j + 1) * frame_size
                    frame = img[:, y0:y1, x0:x1]
                    if frame.max() <= 1.0:
                        frame = (frame * 255).clamp(0, 255)
                    frame_np = frame.permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
                    frames.append(frame_np)
            all_videos.append(frames)

        # --- garante mesmo número de frames ---
        min_len = min(len(v) for v in all_videos)
        all_videos = [v[:min_len] for v in all_videos]

        # --- cria linha vermelha ---
        H, W, _ = all_videos[0][0].shape
        line = np.full((H, line_width, 3), np.array(line_color, dtype=np.uint8))

        # --- concatena lado a lado ---
        frames_concat = []
        for t in range(min_len):
            parts = []
            for i, video in enumerate(all_videos):
                parts.append(video[t])
                if i < len(all_videos) - 1:
                    parts.append(line)  # adiciona linha vermelha
            frame_cat = np.concatenate(parts, axis=1)
            frames_concat.append(frame_cat)

        # --- salva GIF ---
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
            gif_path = tmpfile.name
        imageio.mimsave(gif_path, frames_concat, fps=fps)

        # --- loga no W&B ---
        wandb.log({name: wandb.Video(gif_path, fps=fps, format="gif", caption=caption)}, commit=False)

        # --- remove temporário ---
        os.remove(gif_path)
        '''
      
    def saveTensorAsGIF(
            imgs: torch.Tensor,
            name: str,
            caption: str = "",
            frame_size: int = 24,
            fps: int = 5
        ):
            """
            Converte uma imagem grid (C,H,W) em GIF, ordem:
            esquerda → direita, topo → baixo, e loga no W&B.
            """

            line_color: tuple[int, int, int] = (255, 0, 0),  # vermelho
            line_width: int = 2

            # pega apenas 3 canais
            all_videos = []
            for img in imgs:
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
                all_videos.append(frames)
            min_len = min(len(v) for v in all_videos)
        
            # --- garante mesmo número de frames ---
            min_len = min(len(v) for v in all_videos)
            all_videos = [v[:min_len] for v in all_videos]

            # --- cria linha vermelha ---
            H, W, _ = all_videos[0][0].shape
            line = np.full((H, line_width, 3), np.array(line_color, dtype=np.uint8))

            # --- concatena lado a lado ---
            frames_concat = []
            for t in range(min_len):
                parts = []
                for i, video in enumerate(all_videos):
                    parts.append(video[t])
                    if i < len(all_videos) - 1:
                        parts.append(line)  # adiciona linha vermelha
                frame_cat = np.concatenate(parts, axis=1)
                frames_concat.append(frame_cat)

            # --- salva GIF ---
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                gif_path = tmpfile.name
            imageio.mimsave(gif_path, frames_concat, fps=fps, loop=0)

            # --- loga no W&B ---
            wandb.log({"VideoRec/"+name: wandb.Video(gif_path, format="gif", caption=caption)}, commit=False)

            # --- remove temporário ---
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

  

    