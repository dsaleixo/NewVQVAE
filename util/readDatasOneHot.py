import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
    

class ReadDatasOneHot:
    palette_np = np.array([
                [255,255,255],
                [200,200,200],
                [255,100,10],
                [255,255,0],
                [0, 255, 255],
                [0,0,0],
                [127,127,127],
        ])
    
    palette = torch.tensor(palette_np / 255.0, dtype=torch.float32, device="cpu")
    


    def image_to_onehot(img: torch.Tensor) -> torch.Tensor:
        """
        img: (H, W, 6) torch tensor
        palette: (7, 3)
        retorna: (14, H, W)
        """
        assert img.shape[-1] == 6

        img = img.float()

        img1 = img[:, :, :3]  # (H,W,3)
        img2 = img[:, :, 3:]  # (H,W,3)

        def encode(group):
            group = group.unsqueeze(2)  # (H,W,1,3)
            pal = ReadDatasOneHot.palette.view(1,1,7,3)
            dist = torch.norm(group - pal, dim=-1)   # (H,W,7)
            idx = dist.argmin(dim=-1)   # (H,W)
            oh = torch.nn.functional.one_hot(idx, 7) # (H,W,7)
            return oh.permute(2,0,1).float()         # (7,H,W)

        return torch.cat((encode(img1), encode(img2)), dim=0) # (14,H,W)


    def readData(folder,files=None)->torch.tensor:
        folder_path = Path(folder)
        if files ==None:
            files =  [f.name for f in folder_path.iterdir() if f.is_file()]
   
        cont=0
        imgs = []
        for arq in files:
            if cont>100:
                pass
                #break
            data = np.load(folder+arq)
            indices = data["indices"]
            values = data["values"]
            shape = tuple(data["shape"])

            flat = np.zeros((np.prod(shape[:2]), 6), dtype=values.dtype)
            flat[indices] = values
            img = flat.reshape(shape)
            img = torch.from_numpy(img)
            oh_img = ReadDatasOneHot.image_to_onehot(img)  # (14,H,W)
      
            imgs.append(oh_img)
            cont+=1
        return imgs
     

   
    def loadDataLoader()->tuple[DataLoader,DataLoader]:


        control = ReadDatasOneHot.readData("./",["resultado.npz"])[0]
   

    
        data = ReadDatasOneHot.readData("./data/")

    

        dataValidation = ReadDatasOneHot.readData("./dataValidation/")
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
    data = ReadDatasOneHot.loadDataLoader()