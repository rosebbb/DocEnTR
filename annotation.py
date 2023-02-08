import cv2
events = [i for i in dir(cv2) if 'EVENT' in i]
print( events )
import numpy as np
import glob
import os
# drawing = False # true if mouse is pressed
# mode = True # if True, draw rectangle. Press 'm' to toggle to curve
# ix,iy = -1,-1
# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             img[ix:x, iy:y, :] = 255

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         img[ix:x, iy:y, :] = 255

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(gt_img,(ix,iy),(x,y),(255,255,255),-1)
            else:
                cv2.circle(gt_img,(x,y),5,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(gt_img,(ix,iy),(x,y),(255,255,255),-1)
        else:
            cv2.circle(gt_img,(x,y),5,(0,0,255),-1)

            
img_dir = '/data/Datasets/Binarization/Accessmath/data_with_person/gt'
out_dir = '/data/Datasets/Binarization/Accessmath/data_with_person/gt_corrected'
# img_file = '/data/Datasets/Binarization/Accessmath/data_with_person/gt/58__whiteboard_8_base_256_8.png'
cv2.namedWindow('input')
cv2.namedWindow('gt')
c = 0
for img_file in glob.glob(os.path.join(img_dir, '*.png')):
    c+=1
    print(img_file)
    img_name = os.path.basename(img_file)
    input_file = os.path.dirname(img_file).replace('gt', 'input')+'/'+img_name
    print(input_file)
    input_img = cv2.imread(input_file)
    input_img = cv2.resize(input_img, (192*3, 108*3))
    gt_img = cv2.imread(img_file)
    cv2.setMouseCallback('gt',draw_circle)
    while(1):
        cv2.imshow('input', input_img)
        cv2.imshow('gt',gt_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.imwrite(os.path.join(out_dir, img_name+'.png'), gt_img)
            break
cv2.destroyAllWindows()

