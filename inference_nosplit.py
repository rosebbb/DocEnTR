import torch
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vit_pytorch import ViT
from models.binae import BinModel
from einops import rearrange
import time
import glob
import os
from math import ceil

THRESHOLD = 0.5 ## binarization threshold after the model output

SPLITSIZE =  256  ## your image will be divided into patches of 256x256 pixels
setting = "base"  ## choose the desired model size [small, base or large], depending on the model you want to use
patch_size = 16 ## choose your desired patch size [8 or 16], depending on the model you want to use
image_size =  (SPLITSIZE,SPLITSIZE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

if setting == 'base':
    encoder_layers = 6
    encoder_heads = 8
    encoder_dim = 768

if setting == 'small':
    encoder_layers = 3
    encoder_heads = 4
    encoder_dim = 512

if setting == 'large':
    encoder_layers = 12
    encoder_heads = 16
    encoder_dim = 1024

v = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = encoder_dim,
    depth = encoder_layers,
    heads = encoder_heads,
    mlp_dim = 2048
)
model = BinModel(
    encoder = v,
    decoder_dim = encoder_dim,      
    decoder_depth = encoder_layers,
    decoder_heads = encoder_heads       
)

model = model.to(device)

# model_path = "./weights/best-model_8_2017base_256_8.pt"
model_path = './weights/whiteboard_8_base_256_8.pt'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
# model.load_state_dict(torch.load(model_path, map_location=device))

image_file = '/data/Datasets/Binarization/Accessmath/data_for_DocEnTr/test/13219_512_1024.png'
# deg_folder = '/data/Datasets/Binarization/test/'
image_name = os.path.basename(image_file)

# read image
deg_image = cv2.imread(image_file) / 255

# Normalize the image
h, w, _ = deg_image.shape
deg_image_norm = np.ones((h,w,3))
for i in range(3):
    deg_image_norm[:,:,i] = (deg_image[:,:,i] - mean[i]) / std[i]

# transpose
p = deg_image_norm.transpose(2,0,1)


# inference
result = []
p = np.array(p, dtype='float32')
train_in = torch.from_numpy(p)

with torch.no_grad():
    train_in = train_in.view(1,3,SPLITSIZE,SPLITSIZE).to(device)
    _ = torch.rand((train_in.shape)).to(device)
    _,_, pred_pixel_values = model(train_in,_)
    rec_image = torch.squeeze(rearrange(pred_pixel_values, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size))
    impred = rec_image.cpu().numpy()
    impred = np.transpose(impred, (1, 2, 0))
    for ch in range(3):
        impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]
    impred[np.where(impred>1)] = 1
    impred[np.where(impred<0)] = 0

impred = (impred>THRESHOLD)*255

output_dir = pathlib.Path('./demo/cleaned_patch')
output_dir.mkdir(exist_ok=True)

model_name = pathlib.Path(model_path).stem
image_path = pathlib.Path(image_name)
output_path = output_dir.joinpath(f'{image_path.stem}__{model_name}{image_path.suffix}')

cv2.imwrite(str(output_path), impred)
print(f'created file: {output_path}')