import cv2
import os

image_folder = './demo/videos/whiteboard_0720_short_v2'
video_name = 'whiteboard_bin_realtime_v2.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 25, (width,height))

for i in range(1, 779):
    video.write(cv2.imread(os.path.join(image_folder, str(i)+'.jpg')))

cv2.destroyAllWindows()
video.release()