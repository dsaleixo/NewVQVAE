
import torch
from torch.utils.data import DataLoader
import wandb

from modelBase import ModelBase

from util.readDatas import ReadDatas
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class Analysis:

    def analiseCodebookIndividual(self,model:ModelBase):
        def plotDistancias(dist_matrix,name):
            n = dist_matrix.shape[0]

            plt.figure(figsize=(8, 6))
            plt.imshow(dist_matrix, cmap="viridis", interpolation="nearest")
            plt.colorbar(label="Distância")
            plt.title("Heatmap de Distâncias "+name)
            plt.xlabel("Vetores")
            plt.ylabel("Vetores")

            # Opcional: valores dentro de cada célula
            for i in range(n):
                for j in range(n):
                    plt.text(j, i, f"{dist_matrix[i, j]:.2f}", ha='center', va='center', fontsize=8, color='white')

            plt.tight_layout()

            # Loga no wandb
            wandb.log({"Heatmap_"+name: wandb.Image(plt)})
            plt.close()

        def distancias(vetores) :
                # Empilha os vetores em uma matriz 2D
                X = np.vstack(vetores)  # shape: (num_vectors, dim)
                
                # Calcula a distância do cosseno entre todos os pares
                dist_matrixCos = squareform(pdist(X, metric='cosine'))
                dist_matrixL1 = squareform(pdist(X, metric='cityblock'))
                plotDistancias(dist_matrixCos,"Cos")
                plotDistancias(dist_matrixL1,"L1")


        ne = model.getnNumEmbeddings()
        plt.figure(figsize=(12, 3*ne))  # altura proporcional ao número de embeddings
        vetores = []
        for i in range(ne):
            vector = model.getVector(i)
            
            x_np = vector.cpu().numpy()
            vetores.append(x_np)

            #distancias(vetores)
        
            plt.subplot(ne, 1, i+1)       # ne linhas, 1 coluna, posição i+1
            plt.plot(x_np, label=f"Embedding {i}")
            plt.title(f"Gráfico de linha {i}")
            plt.xlabel("Index")
            plt.ylabel("Valor")
            plt.grid(True)

        plt.tight_layout()
        wandb.log({"AllEmbeddings": wandb.Image(plt)})
        plt.close()
        

    def analiseCodebook(self,model:ModelBase,data : DataLoader,name:str):
        


        model.eval()
        ne = model.getnNumEmbeddings()
        contVextorUsed = np.zeros(ne)
        x = torch.rand(1, 6, 288, 288)
        x = model.prepareInputData(x) 
        print(x.shape)
        indices = model.getFeature(x).view(-1)
        contVextorUsedBySpot = np.zeros((indices.shape[0],ne))
        print(contVextorUsedBySpot.shape)
        print(contVextorUsed.shape)
        count  =0
        for batch_idx, batch in enumerate(data):
            # Supondo que batch = (x, y, z) ou apenas imagens x
            
            x = model.prepareInputData(batch) 
            B,C,_,_=batch.shape
            count+=B
            indices = model.getFeature(x).view(B,-1)
        
            for b in range(B):
                for i,ind in enumerate(indices[b]):
                    contVextorUsed[ind]+=1
                    contVextorUsedBySpot[i][ind]+=1  
            contVextorUsed = contVextorUsed/contVextorUsed.sum()
            #contVextorUsedBySpot/=count              


        # Cria uma tabela para o gráfico
        # Cria uma tabela W&B (nomes das colunas devem coincidir com os usados no plot)
        data = [[str(i), float(v)] for i, v in enumerate(contVextorUsed)]
        table = wandb.Table(data=data, columns=["Vector", "Values"])

        # Loga o gráfico de barras
        wandb.log({

            "Analysis/contVextorUsed": wandb.plot.bar(
                table,
                "Vector",   # deve ser idêntico ao nome da coluna
                "Values",
                title="Distribuição de valores"
            )
        })





        # Cria o heatmap
        plt.figure(figsize=(12,6))
        plt.imshow(contVextorUsedBySpot, aspect='auto', cmap='viridis')
        plt.colorbar(label='Quantidade de demissões')
        plt.xlabel('Codebook / coluna')
        plt.ylabel('Time / linha')
        plt.title('Heatmap de demissões')

        # Loga como imagem
        wandb.log({"Heatmap/Demissoes": wandb.Image(plt)})
        plt.close()



if __name__ == "__main__":


    wandb.init(
        project="VQVAE",
        name = "inialização23",
        #mode ="disabled",
        resume=False,
        config={
        "test": 1,

            }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device)",device)
    model : ModelBase = Model1(device)
    trainLoader,testLoader,valLoader=ReadDatas.loadDataLoader()
    model.initializeWeights(2,1,trainLoader)
    analise = Analysis()
    analise.analiseCodebook(model,testLoader,"Test")
    analise.analiseCodebookIndividual(model)
