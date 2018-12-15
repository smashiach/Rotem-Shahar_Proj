from face_extract import detect_faces
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

for img in glob.glob('D:\\Shahar\\Project A\\compare\\training_set\\*.jpg'):  # The location of training set
    PicColorAvg = detect_faces(img)
    if PicColorAvg is None:
        continue
    name = img.split("\\")[5]  # Get the picture name
    tmp_array = [img, PicColorAvg]
    PicList.append(tmp_array)
    avg_color_pic_arr.append(PicColorAvg)
    counter = counter + 1

PicList.sort(key=itemgetter(1), reverse=False)
counter_2 = 1
length = PicList.__len__()
dirname = "dir"
os.mkdir(dirname)
for i in range(0, length-1):
    tmp_img = cv2.imread(PicList[i][0])
    name_2 = str(counter_2) + ".jpg"
    cv2.imwrite(os.path.join(dirname, name_2), tmp_img)
    counter_2 = counter_2 + 1




