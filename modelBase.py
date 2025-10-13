from __future__ import annotations
from abc import ABC, abstractmethod
import torch.nn as nn


class ModelBase(ABC,nn.Module):

    @abstractmethod
    def getFeature(x):
        raise NotImplementedError
    
    @abstractmethod
    def getnNumEmbeddings(self,):
        raise NotImplementedError
    
    @abstractmethod
    def getEmbeddingDim(self):
        raise NotImplementedError
    
    @abstractmethod
    def getVector(self,i:int):
        raise NotImplementedError

    @abstractmethod
    def prepareInputData(self,x):
        raise NotImplementedError

