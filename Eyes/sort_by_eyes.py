from get_relevant_points import eye_dimension_extract
from get_face_aligned import align_face
import glob
from sklearn import preprocessing
import numpy as np
import padasip as pa


# Variables for different geometric ratios
right_eye_horizontal_distance = []
left_eye_horizontal_distance = []
distance_between_eyes = []
photo_path_array = []
ratio_array = []
ratio_1 = []
ratio_2 = []
ratio_3 = []
ratio_4 = []


for img in glob.glob('D:\\Shahar\\Project A\\color_average\\training_set\\*.jpg'):  # The location of training set
    face_aligned = align_face(img)  # Aligned and extract face
    ratio_mouth_chin_nose, ratio_left_eye_nose, ratio_right_eye_nose, ratio_nose_width_middle_eyes, ratio_cheek_nose \
            = eye_dimension_extract(face_aligned)
    if ratio_mouth_chin_nose == 0 or ratio_left_eye_nose == 0 or ratio_right_eye_nose == 0 \
            or ratio_nose_width_middle_eyes == 0 or ratio_cheek_nose == 0:
        continue

    # calculate new ratios and save all in arrays
    ratio_eyes_nose = (ratio_left_eye_nose + ratio_right_eye_nose)/2
    ratio_1.append(ratio_mouth_chin_nose)
    ratio_2.append(ratio_eyes_nose)
    ratio_3.append(ratio_nose_width_middle_eyes)
    ratio_4.append(ratio_cheek_nose)
    ratio_array.append([img])


# normalized ratios
norm_ratio = []
norm_ratio.append(preprocessing.normalize([ratio_1]))
norm_ratio.append(preprocessing.normalize([ratio_2]))
norm_ratio.append(preprocessing.normalize([ratio_3]))
norm_ratio.append(preprocessing.normalize([ratio_4]))


# reshape ratios vectors
size_of_ratio_array = np.size(ratio_1)
ratio_1 = np.reshape(norm_ratio[0], (size_of_ratio_array, 1))
ratio_2 = np.reshape(norm_ratio[1], (size_of_ratio_array, 1))
ratio_3 = np.reshape(norm_ratio[2], (size_of_ratio_array, 1))
ratio_4 = np.reshape(norm_ratio[3], (size_of_ratio_array, 1))
ratio_Matrix = [ratio_1, ratio_2, ratio_3, ratio_4]
ratio_Matrix_T = np.resize(np.transpose(ratio_Matrix), (size_of_ratio_array, 4))
# ratio_Matrix_T = np.resize(ratio_Matrix_T, (size_of_ratio_array, 4))


# specific for roee's image
matrix_size = (size_of_ratio_array, 1)
desirable_value_1 = 0.73032*(np.ones(matrix_size))
desirable_value_2 = 1.399076*(np.ones(matrix_size))
desirable_value_3 = 0.63468663*(np.ones(matrix_size))
desirable_value_4 = 0.878132*(np.ones(matrix_size))
desirable_Matrix = np.reshape(([desirable_value_1, desirable_value_2, desirable_value_3, desirable_value_4]),
                              (size_of_ratio_array, 4))
# desirable_Matrix = np.reshape(desirable_Matrix, (size_of_ratio_array, 4))  # 29 X 4


# Training the weights with Gradient descent
step_length = 0.01
threshold = 0.5
w_t = np.zeros(4)
w_t_1 = np.ones(4)
for i in range(0, 4):
    tmp_ratio_vector_T = np.reshape(ratio_Matrix_T[:, i], (size_of_ratio_array, 1))
    tmp_ratio_vector = np.transpose(tmp_ratio_vector_T)
    tmp_desirable_vector_T = np.reshape(desirable_Matrix[:, i], (size_of_ratio_array, 1))
    while abs(w_t_1[i] - w_t[i]) > threshold:
        gradient_step = -2*np.dot(tmp_ratio_vector, tmp_desirable_vector_T - w_t[i]*tmp_ratio_vector_T)
        w_tmp = w_t_1[i]
        w_t_1[i] = w_t[i] - step_length*gradient_step
        w_t[i] = w_tmp
    norm_ratio[i] *= w_t[i]


final_score = []
score_array = []
norm_ratio = np.reshape(norm_ratio, (4, size_of_ratio_array))
for i in range(0, ratio_array.__len__()):
    tmp_score = norm_ratio[0][i] + norm_ratio[1][i] + norm_ratio[2][i] + norm_ratio[3][i]
    final_score.append([tmp_score, ratio_array[i]])
    score_array.append(tmp_score)

Roee_std = np.std(score_array)


b = 1
