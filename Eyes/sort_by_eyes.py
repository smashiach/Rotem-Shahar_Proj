from eye_dimension_extract import eye_dimension_extract
import glob
import numpy as np

right_eye_horizontal_distance = []
right_eye_vertical_distance = []
left_eye_horizontal_distance = []
left_eye_vertical_distance = []
distance_between_eyes = []
photo_path_array = []

for img in glob.glob('D:\\Shahar\\Project A\\color_average\\training_set\\*.jpg'):  # The location of training set
    a, b, c, d, e = eye_dimension_extract(img)
    right_eye_horizontal_distance.append(a)
    right_eye_vertical_distance.append(b)
    left_eye_horizontal_distance.append(c)
    left_eye_vertical_distance.append(d)
    distance_between_eyes.append(e)
    photo_path_array.append(img)  # Get the picture name

num_of_pic = len(right_eye_horizontal_distance)
ratio_eyes = []  # save the ratio according to :

#        this is left eye :
#          ############
#           *  *  *  *
#         *      |      *
#        * a---- | ----  * ---------- c
#         *    b |     *
#           *  *  *  *
# ratio_1 = c/(avg(a_right,a_left)
# ratio_2 = c/(avg(b_right,b_left)
#

for i in range(0, num_of_pic):
    try:
        ratio_1 = distance_between_eyes[i]/((right_eye_horizontal_distance[i] + left_eye_horizontal_distance[i])/2)
    except:
        ratio_1 = "N/A"
    try:
        ratio_2 = distance_between_eyes[i]/((right_eye_vertical_distance[i] + left_eye_vertical_distance[i])/2)
    except:
        ratio_2 = "N/A"
    ratio_eyes.append([(ratio_1, ratio_2), photo_path_array[i]])


a = 1