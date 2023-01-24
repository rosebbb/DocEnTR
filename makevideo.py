import cv2
import os

image_folder = '/data/Projects/DocEnTR/result'
video_name = 'test_result_1.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 25, (width,height))

for i in range(1, 779):
    video.write(cv2.imread(os.path.join(image_folder, str(i)+'.jpg')))

cv2.destroyAllWindows()
video.release()