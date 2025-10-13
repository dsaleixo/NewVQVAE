
import torch
import wandb
from torch import nn, optim

from Viewer import Viewer
from model1 import Model1
from readDatas import ReadDatas
from torch.nn import functional as F

def initialProcess(model,valLoader,device):
        model.eval()
        #for i in range(len(valLoader)):
        i=0
        x = valLoader[i][:3,:,:].unsqueeze(0).to(device)
        x_rec, vq_loss, indices, perplexity, used_codes = model(x)   
        imgs = [x.squeeze(),x_rec.squeeze()]
        Viewer.saveListTensorAsImg(imgs,f"RecImagemVal{i}",f"match{i}")
        Viewer.saveTensorAsGIF(imgs,f"RecVideoVal{i}",f"match{i}")


if __name__ == "__main__":
    wandb.init(
        project="VQVAE",
        name = "X1",
        config={
        "test": 1,

            }
        )
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device)",device)
    model = Model1(device)
    trainLoader,testLoader,valLoader=ReadDatas.loadDataLoader()

    num_epochs = 100000000000
    

    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-5
    )
    
    
    x = first_batch = valLoader[0]
    print(x.shape)
    initialProcess(model,valLoader,device)
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
            initialProcess(model,valLoader,device)
       
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