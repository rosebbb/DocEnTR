import torch.nn as nn
import torch
from vit_pytorch import ViT
from models.binae import BinModel
from einops import rearrange
import onnx
import numpy as np

batch_size = 1    # just a random number
THRESHOLD = 0.5 ## binarization threshold after the model output
SPLITSIZE =  256  ## your image will be divided into patches of 256x256 pixels
setting = "base"  ## choose the desired model size [small, base or large], depending on the model you want to use
patch_size = 16 ## choose your desired patch size [8 or 16], depending on the model you want to use
image_size =  (SPLITSIZE,SPLITSIZE)
device = 'cpu'
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
torch_model = BinModel(
    encoder = v,
    decoder_dim = encoder_dim,      
    decoder_depth = encoder_layers,
    decoder_heads = encoder_heads       
)


torch_model = torch_model.to(device)
model_path = './weights/whiteboard_8_base_256_8.pt'
checkpoint = torch.load(model_path, map_location=device)
torch_model.load_state_dict(checkpoint['model_state_dict'])

def convertONNX():
    # set the model to inference mode
    torch_model.eval()

    # Input to the model
    x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "./onnx_models/whiteboard_8_base_256_8.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['loss', 'patches', 'pred_pixel_values'], # the model's output names
                    # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #                 'output' : {0 : 'batch_size'}}
                    )
    
def test_onnx():
    onnx_model = onnx.load("./onnx_models/whiteboard_8_base_256_8.onnx")
    onnx.checker.check_model(onnx_model)

def run_onnx():
    print('run_onnx')
    import onnxruntime
    x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
    _, _, torch_out = torch_model(x)

    ort_session = onnxruntime.InferenceSession("./onnx_models/whiteboard_8_base_256_8.onnx")
    input_name = ort_session.get_inputs()[0].name
    print("input name", input_name)
    input_shape = ort_session.get_inputs()[0].shape
    print("input shape", input_shape)
    input_type = ort_session.get_inputs()[0].type
    print("input type", input_type)


    for i in range(3):
        output_name = ort_session.get_outputs()[i].name
        print("output name", output_name)
        output_shape = ort_session.get_outputs()[i].shape
        print("output shape", output_shape)
        output_type = ort_session.get_outputs()[i].type
        print("output type", output_type)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    _, _, ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs, rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    test_onnx()

def run_onnx_on_img():
    import glob
    import os
    import cv2
    deg_folder = '/data/Datasets/Binarization/test/'
    for image_file in glob.glob(os.path.join(deg_folder, '*.png')):
        image_name = os.path.basename(image_file)
        deg_image = cv2.imread(image_file) / 255

    # pad image first
    h =  ((deg_image.shape[0] // 256) +1)*256 
    w =  ((deg_image.shape[1] // 256 ) +1)*256
    deg_image_padded=np.ones((h,w,3))
    deg_image_padded[:deg_image.shape[0],:deg_image.shape[1],:]= deg_image

    patches=[]
    nsize1=SPLITSIZE
    nsize2=SPLITSIZE
    for ii in range(0,h,nsize1): #2048
        for iii in range(0,w,nsize2): #1536
            patches.append(deg_image_padded[ii:ii+nsize1,iii:iii+nsize2,:])
    
    ## preprocess the patches (images)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    out_patches=[]
    for p in patches:
        out_patch = np.zeros([3, *p.shape[:-1]])
        for i in range(3):
            out_patch[i] = (p[:,:,i] - mean[i]) / std[i]
        out_patches.append(out_patch)
convertONNX()
run_onnx()