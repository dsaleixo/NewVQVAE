import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import random_split


class ReadDatas:
    def compress(img):
         
        flat = img.reshape(-1, 6)  # cada pixel é um vetor RGB

        # guarda apenas pixels != 0
        mask = np.any(flat != 0, axis=1)
        indices = np.where(mask)[0]
        values = flat[mask]
        return indices, values

    def video_to_grid(video: np.ndarray) -> np.ndarray:
        T, C, H, W = video.shape
        n = int(np.floor(np.sqrt(T)))  # usa o maior quadrado perfeito sem sobras
        T_used = n * n
        video = video[:T_used]  # descarta frames extras

        # reorganiza para (n, n, C, H, W)
        video = video.reshape(n, n, C, H, W)

        # monta o grid
        grid = np.block([[video[i, j] for j in range(n)] for i in range(n)])  # (C, n*H, n*W)

        # move canais para o final
        grid = np.moveaxis(grid, 0, -1)  # (n*H, n*W, C)

        return grid

    def convertDatasRawforCompress():
        size=144
        data = [ ]
        folderInput = './datas3Original/'
        folderOut = './data/'
        folder_path = Path(folderInput)
        files =  [f.name for f in folder_path.iterdir() if f.is_file()]
        #files =  ['resultado.npy']
        print(len(files))
        cont=0
        for arq in files:
            cont+=1
            if cont%100==0:
                    print(cont//100)
            loaded_data = np.load(folderInput+arq,allow_pickle=True)
            shape = loaded_data.shape
            #print(shape,len(dados))
            aux = [ loaded_data]
            C, H, W = loaded_data.shape[1:]
            for _ in range(size-shape[0]):
                black_frame = np.zeros((1, C, H, W), dtype=loaded_data.dtype)
                aux.append(black_frame)

            loaded_data2  = np.concatenate(aux, axis=0)
            loaded_data2 = loaded_data2[0:size,:,:,:]
            print(loaded_data2.shape)
            img = ReadDatas.video_to_grid(loaded_data2)
            indices,values = ReadDatas.compress(img)
            np.savez_compressed(folderOut+arq[:-3]+"npz", indices=indices, values=values, shape=img.shape)


    def readData(folder,files=None)->torch.tensor:
        folder_path = Path(folder)
        if files ==None:
            files =  [f.name for f in folder_path.iterdir() if f.is_file()]
   
        cont=0
        imgs = []
        for arq in files:
            if cont>100:
                break
            data = np.load(folder+arq)
            indices = data["indices"]
            values = data["values"]
            shape = tuple(data["shape"])

            flat = np.zeros((np.prod(shape[:2]), 6), dtype=values.dtype)
            flat[indices] = values
            img = flat.reshape(shape)
            imgs.append(torch.tensor(img,device="cpu").permute(2,0,1).float())
            cont+=1
        return imgs
     


    def loadDataLoader()->tuple[DataLoader,DataLoader]:


        control = ReadDatas.readData("./",["resultado.npz"])[0]
   

    
        data = ReadDatas.readData("./data/")

    

        dataValidation = ReadDatas.readData("./dataValidation/")
        dataValidation.insert(0,control)
        

        print("load complete",len(data) )
        
        total_size = len(data) 
        print("data shape ",data[0].shape)
        train_size = int(0.8 * total_size)  # 80 amostras para treino
        test_size = total_size - train_size  # 20 amostras para teste
        generator = torch.Generator().manual_seed(42)

        train_set, test_set = random_split(data, [train_size, test_size], generator=generator)

        train_loader = DataLoader(train_set, batch_size=32)
        test_loader = DataLoader(test_set, batch_size=32, )
        return train_loader,test_loader,dataValidation
      
  
if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #data = ReadDatas.readData('./data/',device)
   #print(len(data),data[0].shape)
   x,y,z=ReadDatas.loadDataLoader()
   print(len(x),len(y),len(z))

      