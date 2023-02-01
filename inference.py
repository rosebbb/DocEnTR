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

deg_folder = '/data/Datasets/Binarization/Accessmath/data_for_DocEnTr/test/'
# deg_folder = '/data/Datasets/Binarization/test/'
for image_file in glob.glob(os.path.join(deg_folder, '*.png')):
    image_name = os.path.basename(image_file)
    if image_name != '13219_0_256.png':
        continue
    deg_image = cv2.imread(image_file) / 255

    # plt.imshow(deg_image[:, :, [2, 1, 0]]) # Show image 
    def split(im,h,w):
        patches=[]
        nsize1=SPLITSIZE
        nsize2=SPLITSIZE
        for ii in range(0,h,nsize1): #2048
            for iii in range(0,w,nsize2): #1536
                patches.append(im[ii:ii+nsize1,iii:iii+nsize2,:].transpose(2,0,1))
        return patches
    
    def merge_image(splitted_images, h,w):
        image=np.zeros(((h,w,3)))
        nsize1=SPLITSIZE
        nsize2=SPLITSIZE
        ind =0
        for ii in range(0,h,nsize1):
            for iii in range(0,w,nsize2):
                image[ii:ii+nsize1,iii:iii+nsize2,:]=splitted_images[ind]
                ind += 1
        return image  

    start = time.time()
    ## Split the image intop patches, an image is padded first to make it dividable by the split size
    h =  (ceil(deg_image.shape[0] / 256))*256 
    w =  (ceil(deg_image.shape[1] / 256))*256
    deg_image_padded = np.ones((h,w,3))
    deg_image_padded[:deg_image.shape[0],:deg_image.shape[1],:] = deg_image

    # Normalize the image
    deg_image_norm = np.ones((h,w,3))
    for i in range(3):
        deg_image_norm[:,:,i] = (deg_image_padded[:,:,i] - mean[i]) / std[i]

    patches_norm = split(deg_image_norm, deg_image.shape[0], deg_image.shape[1])
        

    result = []
    for patch_idx, p in enumerate(patches_norm):
        print(f"({patch_idx} / {len(patches_norm) - 1}) processing patch...")
        p = np.array(p, dtype='float32')
        train_in = torch.from_numpy(p)

        with torch.no_grad():
            train_in = train_in.view(1,3,SPLITSIZE,SPLITSIZE).to(device)
            _ = torch.rand((train_in.shape)).to(device)
            _,_, pred_pixel_values = model(train_in,_)
            rec_patches = pred_pixel_values
            rec_image = torch.squeeze(rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size))
            impred = rec_image.cpu().numpy()
            impred = np.transpose(impred, (1, 2, 0))
            for ch in range(3):
                impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]
            impred[np.where(impred>1)] = 1
            impred[np.where(impred<0)] = 0
        result.append(impred)

    clean_image = merge_image(result, deg_image_padded.shape[0], deg_image_padded.shape[1])
    clean_image = clean_image[:deg_image.shape[0], :deg_image.shape[1],:]
    clean_image = (clean_image>THRESHOLD)*255

    endtime = time.time()
    print('time used: ', endtime-start)
    plt.imshow(clean_image)

    output_dir = pathlib.Path('./demo/cleaned_1')
    output_dir.mkdir(exist_ok=True)

    model_name = pathlib.Path(model_path).stem
    image_path = pathlib.Path(image_name)
    output_path = output_dir.joinpath(f'{image_path.stem}__{model_name}{image_path.suffix}')

    cv2.imwrite(str(output_path), clean_image)
    print(f'created file: {output_path}')