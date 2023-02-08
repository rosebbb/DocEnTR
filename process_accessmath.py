import numpy as np
import os
import random
import cv2
from tqdm import tqdm
from config  import Configs
import glob
import random

def prepare_accessmath_experiment(patches_size, overlap_size, patches_size_valid):
    """
    Prepare the data for training

    Args:
        patches_size (int): patch size for training data
        overlap_size (int): overlapping size between different patches (vertically and horizontally)
        patches_size_valid (int): patch size for validation data
    """
    img_list = glob.glob(data_path+'/input/*.png')
    print(len(img_list))

    random.shuffle(img_list)
    train_list = img_list[:280]
    val_list = img_list[280:315]
    test_list = img_list[315:]

    n_i = 1
    for img_file in train_list:
        print('-train: ', img_file)
        img = cv2.imread(img_file)
        gt_img = cv2.imread(img_file.replace('input', 'gt'))

        for i in range (0,img.shape[0],overlap_size):
            for j in range (0,img.shape[1],overlap_size):
                if i+patches_size<=img.shape[0] and j+patches_size<=img.shape[1]:
                    p = img[i:i+patches_size,j:j+patches_size,:]
                    gt_p = gt_img[i:i+patches_size,j:j+patches_size,:]
                
                elif i+patches_size>img.shape[0] and j+patches_size<=img.shape[1]:
                    p = (np.ones((patches_size,patches_size,3)) - random.randint(0,1) )*255
                    gt_p = np.ones((patches_size,patches_size,3)) *255
                    
                    p[0:img.shape[0]-i,:,:] = img[i:img.shape[0],j:j+patches_size,:]
                    gt_p[0:img.shape[0]-i,:,:] = gt_img[i:img.shape[0],j:j+patches_size,:]
                
                elif i+patches_size<=img.shape[0] and j+patches_size>img.shape[1]:
                    p = (np.ones((patches_size,patches_size,3)) - random.randint(0,1) )*255
                    gt_p = np.ones((patches_size,patches_size,3)) * 255
                    
                    p[:,0:img.shape[1]-j,:] = img[i:i+patches_size,j:img.shape[1],:]
                    gt_p[:,0:img.shape[1]-j,:] = gt_img[i:i+patches_size,j:img.shape[1],:]

                else:
                    p = (np.ones((patches_size,patches_size,3)) - random.randint(0,1) )*255
                    gt_p = np.ones((patches_size,patches_size,3)) * 255
                    
                    p[0:img.shape[0]-i,0:img.shape[1]-j,:] = img[i:img.shape[0],j:img.shape[1],:]
                    gt_p[0:img.shape[0]-i,0:img.shape[1]-j,:] = gt_img[i:img.shape[0],j:img.shape[1],:]
                
                cv2.imwrite(out_data_path+'/train/'+str(n_i)+'.png',p)
                cv2.imwrite(out_data_path+'/train_gt/'+str(n_i)+'.png',gt_p)
                n_i+=1
                    
    for img_file in test_list:
        print('-test: ', img_file)
        img = cv2.imread(img_file)
        gt_img = cv2.imread(img_file.replace('input', 'gt'))

        for i in range (0,img.shape[0],patches_size_valid):
            for j in range (0,img.shape[1],patches_size_valid):

                if i+patches_size_valid<=img.shape[0] and j+patches_size_valid<=img.shape[1]:
                    p = img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                    gt_p = gt_img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                
                elif i+patches_size_valid>img.shape[0] and j+patches_size_valid<=img.shape[1]:
                    p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                    gt_p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                    
                    p[0:img.shape[0]-i,:,:] = img[i:img.shape[0],j:j+patches_size_valid,:]
                    gt_p[0:img.shape[0]-i,:,:] = gt_img[i:img.shape[0],j:j+patches_size_valid,:]
                
                elif i+patches_size_valid<=img.shape[0] and j+patches_size_valid>img.shape[1]:
                    p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                    gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                    
                    p[:,0:img.shape[1]-j,:] = img[i:i+patches_size_valid,j:img.shape[1],:]
                    gt_p[:,0:img.shape[1]-j,:] = gt_img[i:i+patches_size_valid,j:img.shape[1],:]

                else:
                    p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                    gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                    
                    p[0:img.shape[0]-i,0:img.shape[1]-j,:] = img[i:img.shape[0],j:img.shape[1],:]
                    gt_p[0:img.shape[0]-i,0:img.shape[1]-j,:] = gt_img[i:img.shape[0],j:img.shape[1],:]

                img_name = os.path.basename(img_file)    
                cv2.imwrite(out_data_path+'/test/'+img_name.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',p)
                cv2.imwrite(out_data_path+'/test_gt/'+img_name.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',gt_p)

    for img_file in val_list:
        print('-val: ', img_file)
        img = cv2.imread(img_file)
        gt_img = cv2.imread(img_file.replace('input', 'gt'))

        for i in range (0,img.shape[0],patches_size_valid):
            for j in range (0,img.shape[1],patches_size_valid):

                if i+patches_size_valid<=img.shape[0] and j+patches_size_valid<=img.shape[1]:
                    p = img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                    gt_p = gt_img[i:i+patches_size_valid,j:j+patches_size_valid,:]
                
                elif i+patches_size_valid>img.shape[0] and j+patches_size_valid<=img.shape[1]:
                    p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                    gt_p = np.ones((patches_size_valid,patches_size_valid,3)) *255
                    
                    p[0:img.shape[0]-i,:,:] = img[i:img.shape[0],j:j+patches_size_valid,:]
                    gt_p[0:img.shape[0]-i,:,:] = gt_img[i:img.shape[0],j:j+patches_size_valid,:]
                
                elif i+patches_size_valid<=img.shape[0] and j+patches_size_valid>img.shape[1]:
                    p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                    gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                    
                    p[:,0:img.shape[1]-j,:] = img[i:i+patches_size_valid,j:img.shape[1],:]
                    gt_p[:,0:img.shape[1]-j,:] = gt_img[i:i+patches_size_valid,j:img.shape[1],:]

                else:
                    p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                    gt_p = np.ones((patches_size_valid,patches_size_valid,3)) * 255
                    
                    p[0:img.shape[0]-i,0:img.shape[1]-j,:] = img[i:img.shape[0],j:img.shape[1],:]
                    gt_p[0:img.shape[0]-i,0:img.shape[1]-j,:] = gt_img[i:img.shape[0],j:img.shape[1],:]

                img_name = os.path.basename(img_file)    
                cv2.imwrite(data_path+'valid/'+img_name.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',p)
                cv2.imwrite(data_path+'valid_gt/'+img_name.split('.')[0]+'_'+str(i)+'_'+str(j)+'.png',gt_p)


if __name__ == "__main__":
    # get configs 
    cfg = Configs().parse()
    data_path = '/data/Datasets/Binarization/Accessmath/'
    # validation_dataset = cfg.validation_dataset
    # testing_dataset = cfg.testing_dataset
    patch_size =  256
    # augment the training data patch size to allow cropping augmentation later in data loader
    p_size_train = (patch_size+128)
    p_size_valid  = patch_size
    overlap_size = patch_size//2

    out_data_path = '/data/Datasets/Binarization/Accessmath/data_for_DocEnTr/train_round_2'
    # create train/val/test data_paths if theu are not existent
    os.makedirs(out_data_path+'/train/', exist_ok=True)
    os.makedirs(out_data_path+'/train_gt/', exist_ok=True)
    os.makedirs(out_data_path+'/valid/', exist_ok=True)
    os.makedirs(out_data_path+'/valid_gt/', exist_ok=True)
    os.makedirs(out_data_path+'/test/', exist_ok=True)
    os.makedirs(out_data_path+'/test_gt/', exist_ok=True)

    # create your data...
    prepare_accessmath_experiment(p_size_train, overlap_size, p_size_valid)

