import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
import scipy.sparse

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
        folderInput = './datas3/'
        folderOut = './data/'
        folder_path = Path(folderInput)
        files =  [f.name for f in folder_path.iterdir() if f.is_file()]
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
            for _ in range(size-shape[0]):
                aux.append( np.expand_dims(loaded_data[-1].copy(), axis=0))

            loaded_data2  = np.concatenate(aux, axis=0)
            loaded_data2 = loaded_data2[0:size,:,:,:]
            print(loaded_data2.shape)
            img = ReadDatas.video_to_grid(loaded_data2)
            indices,values = ReadDatas.compress(img)
            np.savez_compressed(folderOut+arq[:-3]+"npz", indices=indices, values=values, shape=img.shape)


    def readData(folder):
        folder_path = Path(folder)
        files =  [f.name for f in folder_path.iterdir() if f.is_file()]
        print(len(files))
        cont=0
        imgs = []
        for arq in files:
            data = np.load(folder+arq)
            indices = data["indices"]
            values = data["values"]
            shape = tuple(data["shape"])

            flat = np.zeros((np.prod(shape[:2]), 6), dtype=values.dtype)
            flat[indices] = values
            img = flat.reshape(shape)
            imgs.append(img)
        return imgs
     
if __name__ == "__main__":

    data = ReadDatas.readData('./data/')
    print(len(data))
  
      