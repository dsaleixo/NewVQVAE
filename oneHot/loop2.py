import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from setup_env import PROJECT_ROOT

from torch.utils.data import DataLoader
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


def visu0(pred_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    pred_probs: [7, H, W] já normalizado com softmax
    labels: [H, W] com valores entre 0 e num_classes-1
    retorna: [H, W] probabilidade atribuída à classe correta em cada pixel
    """
    assert pred_probs.dim() == 3, "Esperado shape [C, H, W]"
    assert labels.dim() == 2, "Esperado shape [H, W]"
    labels2 = torch.argmax(pred_probs, dim=0)
    pred_probs = F.softmax(pred_probs)
    C, H, W = pred_probs.shape
    probs_truth = pred_probs.permute(1, 2, 0)    # vira [H, W, C]
    probs_truth = probs_truth[torch.arange(H).unsqueeze(1),
                              torch.arange(W),
                              labels]
    probs_truth = probs_truth * (labels2 != 5)
    return probs_truth.unsqueeze(0).repeat(3, 1, 1)

def visu1(pred_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    pred_probs: [7, H, W] já normalizado com softmax
    labels: [H, W] com valores entre 0 e num_classes-1
    retorna: [H, W] probabilidade atribuída à classe correta em cada pixel
    """
    assert pred_probs.dim() == 3, "Esperado shape [C, H, W]"
    assert labels.dim() == 2, "Esperado shape [H, W]"
    labels2 = torch.argmax(pred_probs, dim=0)
    pred_probs = F.softmax(pred_probs)
    C, H, W = pred_probs.shape
    probs_truth = pred_probs.permute(1, 2, 0)    # vira [H, W, C]
    probs_truth = probs_truth[torch.arange(H).unsqueeze(1),
                              torch.arange(W),
                              labels2]
    probs_truth = probs_truth * (labels2 != 5)
    return probs_truth.unsqueeze(0).repeat(3, 1, 1)

def visu2(pred_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    pred_probs: [C, H, W] - logits ou probabilidades por classe
    labels: [H, W] - mapa de classes
    Retorna: [3, H, W] - imagem RGB
              verde -> acerto
              vermelho -> erro
              preto -> classe 5 (ignorada)
    """
    assert pred_probs.dim() == 3, "Esperado shape [C, H, W]"
    assert labels.dim() == 2, "Esperado shape [H, W]"

    # Predição da classe (maior probabilidade)
    labels_pred = torch.argmax(pred_probs, dim=0)

    # Máscaras
    mask_ignore = labels == 5
    mask_correct = (labels_pred == labels) & (~mask_ignore)
    mask_wrong = (labels_pred != labels) & (~mask_ignore)

    # Cria canais RGB
    red = mask_wrong.float()
    green = mask_correct.float()
    blue = torch.zeros_like(red)

    # Combina em [3, H, W]
    rgb = torch.stack([red, green, blue], dim=0)

    return rgb

def visu3(pred_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    pred_probs: [C, H, W] - logits ou probabilidades por classe
    labels: [H, W] - mapa de classes
    Retorna: [3, H, W] - imagem RGB
              verde -> acerto
              vermelho -> erro
              preto -> classe 5 (ignorada)
    """
    assert pred_probs.dim() == 3, "Esperado shape [C, H, W]"
    assert labels.dim() == 2, "Esperado shape [H, W]"

    # Predição da classe (maior probabilidade)
    labels_pred = torch.argmax(pred_probs, dim=0)

    # Máscaras
    mask_ignore = labels_pred == 5
    mask_correct = (labels_pred == labels) & (~mask_ignore)
    mask_wrong = (labels_pred != labels) & (~mask_ignore)

    # Cria canais RGB
    red = mask_wrong.float()
    green = mask_correct.float()
    blue = torch.zeros_like(red)

    # Combina em [3, H, W]
    rgb = torch.stack([red, green, blue], dim=0)

    return rgb

def initialProcess(model,valLoader,device):
        model.eval()
        #for i in range(len(valLoader)):
        i=0
        x = valLoader[i][:7,:,:].unsqueeze(0).to(device)
        labels = torch.argmax(x, dim=1).squeeze()
        
        x_rec, vq_loss, indices, perplexity, used_codes = model(x)   
        x_rec = x_rec.squeeze()  
        pt = visu0(x_rec,labels)
        pt0 = visu1(x_rec,labels)
        pt1 = visu2(x_rec,labels)
        pt2 = visu3(x_rec,labels)
        print('ttt',pt.shape)
        
        imgs = [x.squeeze(),x_rec,pt2,pt1,pt0,pt]
        Viewer.saveListTensorAsImg(imgs,f"RecImagemVal{i}",f"match{i}")
        Viewer.saveTensorAsGIF(imgs,f"RecVideoVal{i}",f"match{i}")

def validation(model, val_loader: DataLoader,criterion, device='cuda',): 
        model.eval()
        total_loss_epoch = 0.0
        
        for batch in val_loader:
            x = batch[:,:7,:,:].to(device)  # [B, C, H, W]
            labels = torch.argmax(x, dim=1)
            logits= model(x)[0]              
            # --- Loss ---
            # --- Loss ---
            loss_focal = focal_loss(logits, labels, gamma=5.0)*20
            recon_loss =criterion(logits, labels)*10
            loss = recon_loss +loss_focal

            total_loss_epoch += loss.item()
            
        print(
                 
                    f"Test  ===>   Loss: {total_loss_epoch:.4f}, "
 
                )
        wandb.log({  
            "Test/Loss": total_loss_epoch, 
        })
          
        return total_loss_epoch


#mode="disabled",
if __name__ == "__main__":
    wandb.init(
        project="VQVAE",
        name = "loop2",
     
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
        lr=1e-4,
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

    
    initialProcess(model,valLoader,device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    #criterion = nn.CrossEntropyLoss()
    bestModelVal = validation(model,testLoader,criterion)
    for epoch in range(num_epochs):

        total_train_loss = 0.0
         # [B, C, H, W]
        running_loss = 0.0
        running_perplexity = 0.0
        for batch_idx, batch in enumerate(trainLoader):
            model.train()
            

            x = batch[:,:7,:,:].to(device) 
            
            labels = torch.argmax(x, dim=1)
            
            
            # --- Forward ---
            

            logits, vq_loss, indices, perplexity, used_codes = model(x)  
        
           
        
            # --- Loss ---
            loss_focal = focal_loss(logits, labels, gamma=5.0)*20
            recon_loss =criterion(logits, labels)*10
            loss = recon_loss +vq_loss*0.1+loss_focal
        
            
            # --- Backprop ---
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            
          
            
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
        modelVal = validation(model,testLoader,criterion)
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

        # Scheduler escuta a música do treino
        #scheduler.step(avg_train_loss)
   