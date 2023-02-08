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
import warnings
from threading import Thread
warnings.filterwarnings("ignore")

# Camera setup
hardwareWidth = 1080
hardwareHeight = 1920
cam_id = '/data/Projects/room-video/Whiteboard/video_samples/whiteboard_0720.mp4'
# cam_id = 'videos/camera2_short.mp4'

# Load Pretrained Model
THRESHOLD = 0.5 ## binarization threshold after the model output
SPLITSIZE =  256  ## your image will be divided into patches of 256x256 pixels
setting = "base"  ## choose the desired model size [small, base or large], depending on the model you want to use
patch_size = 16 ## choose your desired patch size [8 or 16], depending on the model you want to use
image_size =  (SPLITSIZE,SPLITSIZE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_layers = 6
encoder_heads = 8
encoder_dim = 768
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

model_path = './weights/whiteboard_8_base_256_8.pt'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0, canvas_h=1080, canvas_w=1920):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, canvas_w)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, canvas_h)
        self.stream.set(cv2.CAP_PROP_FPS, 25)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            print('------------------video')
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True

class ProcessFrame(object):
    def __init__(self, frame, clean_image):            
        self.frame = frame
        self.stopped = False
        print('***************start ')
        self.bi_frame = clean_image

    # method to start thread 
    def start(self):
        Thread(target=self.binarize, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def binarize(self):
        while not self.stopped:
            if self.frame is None:
                pass
            print('*** frame: ', sum(sum(frame)))
            start = time.time()
            deg_image = self.frame / 255
            def split(im,h,w):
                patches=[]
                nsize1=SPLITSIZE
                nsize2=SPLITSIZE
                for ii in range(0,h,nsize1): #2048
                    for iii in range(0,w,nsize2): #1536
                        patches.append(im[ii:ii+nsize1,iii:iii+nsize2,:])
                
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
            h =  ((deg_image.shape[0] // 256) +1)*256 
            w =  ((deg_image.shape[1] // 256 ) +1)*256
            deg_image_padded=np.ones((h,w,3))
            deg_image_padded[:deg_image.shape[0],:deg_image.shape[1],:]= deg_image
            patches = split(deg_image_padded, deg_image.shape[0], deg_image.shape[1])

            ## preprocess the patches (images)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            out_patches=[]
            for p in patches:
                out_patch = np.zeros([3, *p.shape[:-1]])
                for i in range(3):
                    out_patch[i] = (p[:,:,i] - mean[i]) / std[i]
                out_patches.append(out_patch)


            result = []
            for patch_idx, p in enumerate(out_patches):
                p = np.array(p, dtype='float32')
                train_in = torch.from_numpy(p)

                with torch.no_grad():
                    train_in = train_in.view(1,3,SPLITSIZE,SPLITSIZE).to(device)
                    _ = torch.rand((train_in.shape)).to(device)
                    loss,_, pred_pixel_values = model(train_in,_)
                    rec_patches = pred_pixel_values
                    rec_image = torch.squeeze(rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size))
                    impred = rec_image.cpu().numpy()
                    impred = np.transpose(impred, (1, 2, 0))
                    for ch in range(3):
                        impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]
                    impred[np.where(impred>1)] = 1
                    impred[np.where(impred<0)] = 0
                result.append(impred)

            bi_frame  = merge_image(result, deg_image_padded.shape[0], deg_image_padded.shape[1])
            bi_frame  = bi_frame [:deg_image.shape[0], :deg_image.shape[1],:]
            self.bi_frame  = (bi_frame >THRESHOLD)*255

            total_time = time.time() - start
            print('Finished binarize one frame in: ', total_time)
            print('*** self.bi_frame: ', sum(sum(self.bi_frame)))

        # return self.bi_frame

canvas_w = 1920
canvas_h = 1080
# cam_id = './demo/videos/WIN_20230202_12_57_58_Pro.mp4'
cam_id = '/data/Projects/room-video/Whiteboard/video_samples/whiteboard_0720_short.mp4'
clean_image = np.full((canvas_h, canvas_w, 3),
                        255, dtype = np.uint8)
cap = cv2.VideoCapture(cam_id)
(grabbed, frame) = cap.read()
video_processor = ProcessFrame(frame, clean_image).start()

num_processed =0 
while True:
    print('num_processed:------------', num_processed)

    (grabbed, frame) = cap.read()
    if not grabbed or video_processor.stopped:
        video_processor.stop()
        break

    video_processor.frame = frame
    # print(sum(sum(frame)))
    print('&&& receiving biframe', sum(sum(video_processor.bi_frame)))
    merged = np.concatenate((frame, video_processor.bi_frame), axis=1)

    num_processed+=1
    # cv2.imshow('video', merged)
    cv2.imwrite('./demo/videos/whiteboard_0720_short_v2/'+str(num_processed)+'.jpg', merged)
