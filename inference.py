import torch
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vit_pytorch import ViT
from models.binae import BinModel
from einops import rearrange

THRESHOLD = 0.5 ## binarization threshold after the model output

SPLITSIZE =  256  ## your image will be divided into patches of 256x256 pixels
setting = "base"  ## choose the desired model size [small, base or large], depending on the model you want to use
patch_size = 8 ## choose your desired patch size [8 or 16], depending on the model you want to use
image_size =  (SPLITSIZE,SPLITSIZE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model_path = "./weights/best-model_8_2018base_256_8.pt"
model.load_state_dict(torch.load(model_path, map_location=device))

deg_folder = '/data/Datasets/TextDetection/AccessMath/AccessMath_ICDAR_2017_data/annotations/AccessMath2015_lecture_01/keyframes/'
image_name = '10423.png'
deg_image = cv2.imread(deg_folder+image_name) / 255

plt.imshow(deg_image[:, :, [2, 1, 0]]) # Show image 