import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from setup_env import PROJECT_ROOT


import torch
import wandb
from torch import nn, optim

from oneHot.modelOH0 import ModelOH1
from util.Viewer import Viewer

from util.readDatas import ReadDatas
from torch.nn import functional as F



def focal_loss(logits, labels, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits, labels, reduction='none', weight=weight)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()
def compute_class_weights(dataloader, num_classes=7):
    counts = torch.zeros(num_classes)

    for imgs in dataloader:
        labels = torch.argmax(imgs, dim=1)  # (B,H,W)
        for c in range(num_classes):
            counts[c] += (labels == c).sum()

    # peso inversamente proporcional à frequência
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # normaliza agradavelmente

    return weights

def initialProcess(model,valLoader,device):
        model.eval()
        #for i in range(len(valLoader)):
        i=0
        x = valLoader[i][:7,:,:].unsqueeze(0).to(device)
        x_rec, vq_loss, indices, perplexity, used_codes = model(x)   
        x_rec = x_rec.squeeze()  
        imgs = [x.squeeze(),x_rec]
        Viewer.saveListTensorAsImg(imgs,f"RecImagemVal{i}",f"match{i}")
        Viewer.saveTensorAsGIF(imgs,f"RecVideoVal{i}",f"match{i}")

#mode="disabled",
if __name__ == "__main__":
    wandb.init(
        project="VQVAE",
        name = "X1_OH",
       
        config={
        "test": 1,

            }
        )
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device)",device)
  
    model = ModelOH1(device)
    trainLoader,testLoader,valLoader=ReadDatas.loadDataLoader(True)
    model.initializeWeights(1,3,trainLoader)
    num_epochs = 100000000000
    

    optimizer = optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-6  # valor comum; ajuste conforme necessário
    )
    
    weights = compute_class_weights(trainLoader)  # tensor shape (7,)
    print("Pesos das classes:", weights)

    x = first_batch = valLoader[0]
    print(x.shape)
    initialProcess(model,valLoader,device)
    x = x[:7,:,:].unsqueeze(0).to(device)

    print(x.shape)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    #criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()

         # [B, C, H, W]
        running_loss = 0.0
        running_perplexity = 0.0

        
        
        labels = torch.argmax(x, dim=1)
        
        
        # --- Forward ---
        

        logits, vq_loss, indices, perplexity, used_codes = model(x)  
      
        if epoch%1000==0:
            initialProcess(model,valLoader,device)
       
        # --- Loss ---
        loss_focal = focal_loss(logits, labels, gamma=2.0)*20
        recon_loss =criterion(logits, labels)*10
        loss = recon_loss +vq_loss+loss_focal
       
        
        # --- Backprop ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Logging ---
        running_loss += loss.item()
        running_perplexity += perplexity.item()
        
        wandb.log({
              
                "Train/Recon Loss": loss.item(),
                "Train/Recon Loss_focal": loss_focal.item(),
                "Train/VQ Loss": vq_loss.item(),
                "Train/VQ Perplexity": perplexity.item(),
                "Train/VQ Used Codes": used_codes.sum().item(),
                
                
            })
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Batch [{0}], "
            f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
            f" Loss_focal: {loss_focal.item():.4f}, "
            f"VQ Loss: {vq_loss.item():.4f}, Perplexity: {perplexity.item():.2f}, "
            f"Used Codes: {used_codes.sum().item()}/{model.quantizer.num_embeddings}"
        )
