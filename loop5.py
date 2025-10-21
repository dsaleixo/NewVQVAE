

import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from Viewer import Viewer
from analysis import Analysis
from model1 import Model1
from readDatas import ReadDatas
from torch.nn import functional as F
palette = torch.tensor([
                [255,255,255],
                [200,200,200],
                [255,100,10],
                [255,255,0],
                [0, 255, 255],
                [0,0,0],
                [127,127,127],
            ], dtype=torch.float32)
palette/=255


def quantize_colors(video: torch.Tensor, ) -> torch.Tensor:

    C, H, W = video.shape
    assert C == 3, "Esperado 3 canais RGB"
    flat = video.permute(1,2,0).reshape(-1,3)  # (N,3)
    dists = torch.cdist(flat, palette)  # L2 distance
    indices = torch.argmin(dists, dim=1)  # (N,)
    quantized_flat = palette[indices]  # (N,3)
    quantized = quantized_flat.view(H,W,3).permute(2,0,1)  # (3,T,H,W)

    return quantized


def closest_palette_loss(pred_rgb, target_rgb, palette):
    """
    pred_rgb: (B, 3, T, H, W)
    target_rgb: (B, 3, T, H, W)
    palette: (7, 3)
    """
    #print("sahpe_pred",pred_rgb.shape)
    #print("target_rgb",target_rgb.shape)
    target_rgb = target_rgb[:, :3, :, :] 
    pred_rgb = pred_rgb[:, :3, :, :] 
    device = pred_rgb.device

    B, _, H, W = pred_rgb.shape
    N = B  * H * W

    # Flatten (N,3)
    pred_flat = pred_rgb.permute(0,2,3,1).reshape(N,3)
    target_flat = target_rgb.permute(0,2,3,1).reshape(N,3)

    # Distâncias da predição para todas cores da paleta
    pred_dists = torch.cdist(pred_flat, palette)   # (N,7)
    pred_closest_idx = torch.argmin(pred_dists, dim=1)  # (N,)

    # Distâncias do target para todas cores da paleta
    target_dists = torch.cdist(target_flat, palette)  # (N,7)
    target_closest_idx = torch.argmin(target_dists, dim=1)  # (N,)

    # Máscara de erro
    mask_wrong = pred_closest_idx != target_closest_idx

    # Distância entre a predição e a cor-alvo da paleta
    target_palette_colors = palette[target_closest_idx]  # (N,3)
    penalization_dist = torch.norm(pred_flat - target_palette_colors, dim=1)

    if mask_wrong.any():
        loss = penalization_dist[mask_wrong].mean()
    else:
        loss = torch.tensor(0.0, device=device)

    return loss 


def validation(model, val_loader: DataLoader, device='cuda',): 
        model.eval()
        total_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        J_loss_epoch = 0.0
        perplexity_loss_epoch = 0.0
        used_codes_loss_epoch = 0.0
        vq_loss_loss_epoch = 0.0
        for batch in val_loader:
            x = batch[:,:3,:,:].to(device)  # [B, C, H, W]

            x_rec, vq_loss, indices, perplexity, used_codes = model(x)              
            # --- Loss ---
            recon_loss = F.mse_loss(x_rec, x)
            loss_J =closest_palette_loss(x_rec, x,palette)
            loss = recon_loss + vq_loss+loss_J

            total_loss_epoch += loss.item()
            recon_loss_epoch += recon_loss.item()
            J_loss_epoch += loss_J.item()
            perplexity_loss_epoch +=perplexity.item()/x.shape[0]
            used_codes_loss_epoch +=(used_codes.sum().item())/x.shape[0]
            vq_loss_loss_epoch +=vq_loss.item()
        print(
                 
                    f"Test  ===>   Loss: {total_loss_epoch:.4f}, Recon: {recon_loss_epoch:.4f}, LossJ: {J_loss_epoch:.4f}, "
                    f"VQ Loss: {vq_loss_loss_epoch:.4f}, Perplexity: {perplexity_loss_epoch:.2f}, "
                    f"Used Codes: {used_codes_loss_epoch}/{model.quantizer.num_embeddings}\n,"
                    
                )
        wandb.log({
            "Test/LossJ": J_loss_epoch,
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
        x_rec = x_rec.squeeze()
        x_rec_q = quantize_colors(x_rec)
        imgs = [x.squeeze(),x_rec,x_rec_q]
        Viewer.saveListTensorAsImg(imgs,f"RecImagemVal{i}",f"match{i}")
        Viewer.saveTensorAsGIF(imgs,f"RecVideoVal{i}",f"match{i}")


if __name__ == "__main__":



    



    wandb.init(
    project="VQVAE",
    name = "loop5",
    #mode ="disabled",
    resume=False,
    config={
     "test": 1,

        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device)",device)
    palette = palette.to(device)
    model = Model1(device)
    trainLoader,testLoader,valLoader=ReadDatas.loadDataLoader()

    num_epochs = 100000000000000
    

    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-5
    )
    
    
    model.initializeWeights(0,-1)
    #initialProcess(model,valLoader,device)
    initialProcess(model,valLoader,device)
    bestModelVal = validation(model,testLoader)
    epochVQturnOn = 30
    nextEpoch= 30
    for epoch in range(num_epochs):
        if epoch == epochVQturnOn:
            model.initializeWeights(-1,5,trainLoader)
             
    
    
        running_loss = 0.0
        running_perplexity = 0.0

        for batch_idx, batch in enumerate(trainLoader):
            # Supondo que batch = (x, y, z) ou apenas imagens x
            #print(batch.shape)
            x = batch[:,:3,:,:].to(device)  # [B, C, H, W]
            model.train()
            optimizer.zero_grad()
            
            # --- Forward ---
       
            x_rec, vq_loss, indices, perplexity, used_codes = model(x,epoch>epochVQturnOn)              
            # --- Loss ---
            recon_loss = F.mse_loss(x_rec, x)
            loss_J =closest_palette_loss(x_rec, x,palette)
            loss = recon_loss + vq_loss*0.1+loss_J
            
            # --- Backprop ---
            loss.backward()
            optimizer.step()
            
            # --- Logging ---
            running_loss += loss.item()

            running_perplexity += perplexity.item()
            
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], "
                    f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, LossJ: {loss_J.item():.4f} "
                    f"VQ Loss: {vq_loss.item():.4f}, Perplexity: {perplexity.item():.2f}, "
                    f"Used Codes: {used_codes.sum().item()}/{model.quantizer.num_embeddings}\n"
                )
            wandb.log({
              
                "Train/Recon Loss": recon_loss.item(),
                 "Train/loosJ": loss_J.item(),
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
            if epoch>epochVQturnOn and nextEpoch<epoch:
                nextEpoch+=30
                analise = Analysis()
                analise.analiseCodebook(model,trainLoader,"Train")
                analise.analiseCodebook(model,testLoader,"Test")
                analise.analiseCodebookIndividual(model)
        else:
            wandb.log({"Updade":0})

