from get_relevant_points import eye_dimension_extract
from get_face_aligned import align_face
import glob
from sklearn import preprocessing
import numpy as np
import padasip as pa

right_eye_horizontal_distance = []
left_eye_horizontal_distance = []
distance_between_eyes = []
photo_path_array = []
ratio_array = []  # save the ratio according to
ratio_1 = []
ratio_2 = []
ratio_3 = []
ratio_4 = []
weighted_1 = 0.25
weighted_2 = 0.25
weighted_3 = 0.25
weighted_4 = 0.25

for img in glob.glob('D:\\Shahar\\Project A\\color_average\\training_set\\*.jpg'):  # The location of training set
#for img in glob.glob('D:\\Shahar\\Project A\\color_average\\training_set\\person\\*.jpg'):  # The location of training set
    face_aligned = align_face(img)  # Aligned and extract face
    ratio_mouth_chin_nose, ratio_left_eye_nose, ratio_right_eye_nose, ratio_nose_width_middle_eyes, ratio_cheek_nose = eye_dimension_extract(face_aligned)
    if ratio_mouth_chin_nose == 0 or ratio_left_eye_nose == 0 or ratio_right_eye_nose == 0 or ratio_nose_width_middle_eyes == 0 or ratio_cheek_nose == 0:
        continue
    # photo_path_array.append(img)  # Get the picture name
    ratio_eyes_nose = (ratio_left_eye_nose + ratio_right_eye_nose)/2
    ratio_1.append(ratio_mouth_chin_nose)
    ratio_2.append(ratio_eyes_nose)
    ratio_3.append(ratio_nose_width_middle_eyes)
    ratio_4.append(ratio_cheek_nose)
    # weighted_score = ratio_mouth_chin_nose + ratio_eyes_nose + ratio_nose_width_middle_eyes + ratio_cheek_nose
    ratio_array.append([img])

# std_1 = np.std(ratio_1)
# std_2 = np.std(ratio_2)
# std_3 = np.std(ratio_3)
# std_4 = np.std(ratio_4)

norm_ratio_1 = preprocessing.normalize([ratio_1])
norm_ratio_2 = preprocessing.normalize([ratio_2])
norm_ratio_3 = preprocessing.normalize([ratio_3])
norm_ratio_4 = preprocessing.normalize([ratio_4])

norm_ratio_1 = norm_ratio_1 * weighted_1
norm_ratio_2 = norm_ratio_2 * weighted_2
norm_ratio_3 = norm_ratio_3 * weighted_3
norm_ratio_4 = norm_ratio_4 * weighted_4

final_score = []
score_array = []
for i in range(0, ratio_array.__len__()):
    tmp_score = norm_ratio_1[0][i] + norm_ratio_2[0][i] + norm_ratio_3[0][i] + norm_ratio_4[0][i]
    final_score.append([tmp_score, ratio_array[i]])
    score_array.append(tmp_score)

a = 1
size_of_ratio_1 = np.size(ratio_1)

ratio_1 = np.reshape(ratio_1, (size_of_ratio_1, 1))
ratio_2 = np.reshape(ratio_2, (size_of_ratio_1, 1))
ratio_3 = np.reshape(ratio_3, (size_of_ratio_1, 1))
ratio_4 = np.reshape(ratio_4, (size_of_ratio_1, 1))


ratio = [ratio_1, ratio_2, ratio_3, ratio_4]
matrix_size = (size_of_ratio_1, 1)
#desirable_value_1 = 0.73032*(np.ones(matrix_size))
# desirable_value_1 = desirable_value_1*0.73032
#desirable_value_2 = 1.399076*(np.ones(matrix_size))
#desirable_value_3 = 0.63468663*(np.ones(matrix_size))
#desirable_value_4 = 0.878132*(np.ones(matrix_size))
# desirable_value = [0.73032, 1.399076, 0.63468663, 0.878132]
#desirable = [desirable_value_1, desirable_value_2, desirable_value_3, desirable_value_4]
#desirable = np.reshape(desirable_value_1, (29, 1))

# Training the weights with LMS
desirable_value_1 = 0.73032*(np.ones(matrix_size))


etta = 0.01
iterations = 100
# w_t = np.zeros(iterations)
cost = np.zeros(iterations)
threshold = 0.5
w_t = 0
w_t_1 = 1
ratio_1_T = np.transpose(ratio_1)
while abs(w_t_1 - w_t) > threshold:
    gradient_step = -2*np.dot(ratio_1_T, desirable_value_1 - w_t*ratio_1)
    w_tmp = w_t_1
    w_t_1 = w_t - etta*gradient_step
    w_t = w_tmp



#old_error, new_error = 0, 0
# while new_error >= old_error:
#     tmp1 = w_new*ratio_1
#     tmp2 = w_new*desirable_value_1
#     old_error = new_error
#     new_error = abs(tmp1 - tmp2)
#     w_old = w_new
#     w_new = w_new + dw




# f1 = pa.filters.FilterLMS(n=4, mu=0.1, w="random")
# y, e, w = f1.run(desirable, ratio_1)

b = 0





