import numpy as np
import torch
import torch.nn as nn

from rgb2.rgb22 import RGB
from rgb2.rgbBase import RGBbase

class RGBInference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.quantizer = model.quantizer

    def forward(self, x):
        z = self.encoder(x)
        z = torch.nn.functional.avg_pool2d(z, 2)

        z_q, loss, indices, perplexity, used_codes = self.quantizer(z)

        return indices
    

model = RGBbase(device="cpu")
#model.load_state_dict(torch.load("BestRGB0.pth"))

infer_model = RGBInference(model)
infer_model.eval()

dummy = torch.randn(1,3,288,288)

torch.onnx.export(
    infer_model,
    dummy,
    "rgb_encoder.onnx",
    input_names=["image"],
    output_names=["tokens"],
    opset_version=17
)
np.savetxt("codebook.txt", infer_model.quantizer._embedding.cpu().numpy())


