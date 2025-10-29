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


def prob_truth(pred_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    pred_probs: [7, H, W] já normalizado com softmax
    labels: [H, W] com valores entre 0 e num_classes-1
    retorna: [H, W] probabilidade atribuída à classe correta em cada pixel
    """
    assert pred_probs.dim() == 3, "Esperado shape [C, H, W]"
    assert labels.dim() == 2, "Esperado shape [H, W]"
    pred_probs = F.softmax(pred_probs)
    C, H, W = pred_probs.shape
    probs_truth = pred_probs.permute(1, 2, 0)    # vira [H, W, C]
    probs_truth = probs_truth[torch.arange(H).unsqueeze(1),
                              torch.arange(W),
                              labels]
    return probs_truth.unsqueeze(0).repeat(3, 1, 1)

def initialProcess(model,valLoader,device):
        model.eval()
        #for i in range(len(valLoader)):
        i=0
        x = valLoader[i][:7,:,:].unsqueeze(0).to(device)
        labels = torch.argmax(x, dim=1).squeeze()
        
        x_rec, vq_loss, indices, perplexity, used_codes = model(x)   
        x_rec = x_rec.squeeze()  
        pt = prob_truth(x_rec,labels)
        print('ttt',pt.shape)
        
        imgs = [x.squeeze(),x_rec,pt]
        Viewer.saveListTensorAsImg(imgs,f"RecImagemVal{i}",f"match{i}")
        Viewer.saveTensorAsGIF(imgs,f"RecVideoVal{i}",f"match{i}")

#mode="disabled",
if __name__ == "__main__":
    wandb.init(
        project="VQVAE",
        name = "X1_OH",
        mode="disabled",
        config={
        "test": 1,

            }
        )
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device)",device)
  
    model = ModelOH1(device)
    trainLoader,testLoader,valLoader=ReadDatas.loadDataLoader(True)
    model.initializeWeights(1,2,trainLoader)
    num_epochs = 100000000000
    

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-6
    )

    # Scheduler que reage ao desempenho
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",         # optimizar val_loss menor
        factor=0.5,         # reduz LR pela metade
        patience=1000,         # espera 4 epochs sem melhora
        min_lr=1e-6,        # piso de segurança

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
        total_train_loss = 0.0
         # [B, C, H, W]
        running_loss = 0.0
        running_perplexity = 0.0

        
        
        labels = torch.argmax(x, dim=1)
        
        
        # --- Forward ---
        

        logits, vq_loss, indices, perplexity, used_codes = model(x,False)  
      
        if epoch%1000==0:
            initialProcess(model,valLoader,device)
       
        # --- Loss ---
        loss_focal = focal_loss(logits, labels, gamma=5.0)*20
        recon_loss =criterion(logits, labels)*10
        loss = recon_loss +vq_loss*0.1+loss_focal
       
        
        # --- Backprop ---
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        
        # --- Logging ---
        running_loss += loss.item()
        running_perplexity += perplexity.item()
        total_train_loss += loss.item()
        
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
            f" LR: {optimizer.param_groups[0]['lr']:.6e}"
        )
        avg_train_loss = total_train_loss

        # Scheduler escuta a música do treino
        #scheduler.step(avg_train_loss)
   