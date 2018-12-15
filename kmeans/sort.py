from hsv_extract import detect_faces
import glob
from operator import itemgetter
import cv2
import os
import shutil

try:
    shutil.rmtree('dir\\')
except:
    a = 1


avg_color_pic_arr = []
potential_pic_array = []
PicList = []
counter = 0
tmp_array = []

for img in glob.glob('D:\\Shahar\\Project A\\kmeans\\training_set\\*.jpg'):  # The location of training set
    name = img.split("\\")[5]  # Get the picture name
    img_h, img_s, img_v = detect_faces(img, name)  # Get the h,s,v of the dominant color
    if (img_h == 0) or (img_s == 0) or (img_v == 0):
        continue
    avg_hsv = (img_h + img_s + img_v)/3
    line_of_pic_and_hsv = [img, img_h, img_s, img_v, avg_hsv]  # Line of [img_name, h, s, v]
    PicList.append(line_of_pic_and_hsv)  # List of  Lines [img_name, h, s, v]
    tmp_hsv = (img_h, img_s, img_v)
    avg_color_pic_arr.append(tmp_hsv)  # List of hsv colors
    counter = counter + 1

a = 1

PicList.sort(key=itemgetter(1), reverse=False)
counter_2 = 1
length = PicList.__len__()
dirname = "dir"
os.mkdir(dirname)
for i in range(0, length):
    tmp_img = cv2.imread(PicList[i][0])
    name_2 = str(counter_2) + ".jpg"
    cv2.imwrite(os.path.join(dirname, name_2), tmp_img)
    counter_2 = counter_2 + 1




