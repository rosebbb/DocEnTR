import glob
import os
import random
import shutil

def random_selet_dataset():
    out_dir = '/data/Datasets/Binarization/Accessmath/data_with_person/gt_corrected'
    train_list = []
    root_folder = '/data/Projects/accessmath-icfhr2018/AccessMathVOC/'
    for folder in glob.glob(root_folder + '*'):
        img_folder = os.path.join(folder, 'JPEGImages')

        img_files = glob.glob(img_folder+'/*.png')
        train_list += random.sample(img_files, k=200)

    print(train_list)

    for img_file in train_list:
        shutil.copyfile(img_file, out_dir+'/'+os.path.basename(img_file))


def move_image():
    out_input_dir = '/data/Datasets/Binarization/Accessmath/input1'
    root_folder = '/data/Datasets/Binarization/Accessmath/data_with_person/gt_corrected'
    for img_file in glob.glob(root_folder + '/*.png'):
        input_img_file = img_file.replace('gt_corrected', 'input')
        shutil.copyfile(input_img_file, out_input_dir+'/'+os.path.basename(img_file))

move_image()