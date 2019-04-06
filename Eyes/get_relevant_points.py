from imutils import face_utils
from fix_rotated_face import fix_ratio_rotated_face
import numpy as np
import imutils
import dlib
from operator import itemgetter
import os
import cv2
import math


def eye_dimension_extract(image_source):
    shape_predictor = os.getcwd() + '\\shape_predictor_68_face_landmarks.dat'

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    # load the input image, resize it, and convert it to grayscale
    # image = cv2.imread(image_source)
    image = image_source
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    LeftEyeCoorXY     = []  # left eye X coordinates array
    RightEyeCoorXY    = []  # right eye X coordinates array
    NoseButtomCoorXY  = []  # array of XY nose coordinates
    JawCoorXY         = []  # array of XY Jaw coordinates
    MouthCoorXY       = []  # array of XY Mouth coordinates
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                if name == "right_eye":
                    LeftEyeCoorXY.append([x, y])
                    # RightEyeCoorY.append(y)
                if name == "left_eye":
                    RightEyeCoorXY.append([x, y])
                    # LeftEyeCoorY.append(y)
                if name == "nose":
                    NoseButtomCoorXY.append([x, y])
                if name == "jaw":
                    JawCoorXY.append([x, y])
                if name == "mouth":
                    MouthCoorXY.append([x,y])
            # extract the ROI of the face region as a separate image
            # (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            # roi = image[y:y + h, x:x + w]
            # roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        # visualize all facial landmarks with a transparent overlay
        # output = face_utils.visualize_facial_landmarks(image, shape)
        # cv2.imshow("Image", output)
        # cv2.waitKey(0)

    # Sorting arrays by x argument
    LeftEyeCoorXY.sort(key=itemgetter(0), reverse=False)
    RightEyeCoorXY.sort(key=itemgetter(0), reverse=False)

    # Sorting arrays by y argument
    NoseButtomCoorXY.sort(key=itemgetter(1), reverse=False)
    MouthCoorXY.sort(key=itemgetter(1), reverse=False)

    # Find the middle point between eyes
    try:
        middle_point_between_eyes = int(np.round((RightEyeCoorXY[0][0] - LeftEyeCoorXY[-1][0])/2)) + LeftEyeCoorXY[-1][0], int(np.round(np.abs(RightEyeCoorXY[0][1] - LeftEyeCoorXY[-1][1])/2)) + min(LeftEyeCoorXY[-1][1], RightEyeCoorXY[0][1])
        middle_point_nose = NoseButtomCoorXY[-1]
        left_point_left_eye = LeftEyeCoorXY[0]
        right_point_right_eye = RightEyeCoorXY[-1]
    except:
        middle_point_between_eyes = 0
        middle_point_nose = 0
        left_point_left_eye = 0
        right_point_right_eye = 0

    # Get relevant points of nose
    try:
        NoseButtomCoorXY.sort(key=itemgetter(0), reverse=False)
        left_point_nose = NoseButtomCoorXY[0]
        right_point_nose = NoseButtomCoorXY[-1]

        # get relevant points of mouth
        mouth_lowest_point = MouthCoorXY[-1]

        # Calculate distance between middle point between eyes and middle point of nose
        distance_eye_nose = math.hypot(middle_point_nose[0] - middle_point_between_eyes[0],
                                       middle_point_nose[1] - middle_point_between_eyes[1])
        distance_left_eye_nose = math.hypot(left_point_left_eye[0] - middle_point_nose[0],
                                            left_point_left_eye[1] - middle_point_nose[1])
        distance_right_eye_nose = math.hypot(right_point_right_eye[0] - middle_point_nose[0],
                                             right_point_right_eye[1] - middle_point_nose[1])

    except:
        left_point_nose = 0
        right_point_nose = 0
        mouth_lowest_point = 0
        distance_eye_nose = 0
        distance_left_eye_nose = 0
        distance_right_eye_nose = 0
    # We understand that the face is rotated and we fix the ratio
    # if distance_left_eye_nose/distance_right_eye_nose > 1.05:
    #     distance_right_eye_nose, distance_left_eye_nose = fix_ratio_rotated_face(distance_right_eye_nose, distance_left_eye_nose)
    # if distance_right_eye_nose/distance_left_eye_nose > 1.05:
    #     distance_left_eye_nose, distance_right_eye_nose = fix_ratio_rotated_face(distance_left_eye_nose, distance_right_eye_nose)

    try:
        nose_width = math.hypot(right_point_nose[0]-left_point_nose[0], right_point_nose[1]-left_point_nose[1])
    except:
        nose_width = 0

    # Find the closest points on the jaw in y axis to the nose's nozzle
    JawCoorXY.sort(key=itemgetter(1), reverse=False)
    try:
        Left_Jaw = JawCoorXY[0]
        Right_Jaw = JawCoorXY[0]
        jaw_lowest_point = JawCoorXY[-1]
        right_noozle_x = right_point_nose[0]
        right_noozle_y = right_point_nose[1]
        left_noozle_x = left_point_nose[0]
        left_noozle_y = left_point_nose[1]
    except:
        Left_Jaw = 0
        Right_Jaw = 0
        jaw_lowest_point = 0
        right_noozle_x = 0
        right_noozle_y = 0
        left_noozle_x = 0
        left_noozle_y = 0

    for jaw in JawCoorXY:
        if np.abs((jaw[1] - left_noozle_y) < np.abs(Left_Jaw[1] - left_noozle_y)) and (jaw[0] < left_noozle_x):
            Left_Jaw = jaw
        if np.abs((jaw[1] - right_noozle_y) < np.abs(Right_Jaw[1] - right_noozle_y)) and (jaw[0] > right_noozle_x):
            Right_Jaw = jaw

    try:
        distance_right_cheek = math.hypot(Right_Jaw[0] - right_noozle_x, Right_Jaw[1] - right_noozle_y)
        distance_left_cheek = math.hypot(Left_Jaw[0] - left_noozle_x, Left_Jaw[1] - left_noozle_y)
    except:
        distance_right_cheek = 0
        distance_left_cheek = 0

    # We understand that the face is rotated and we fix the ratio
    # if distance_left_cheek/distance_right_cheek > 1.05:
    #     distance_right_cheek, distance_left_cheek = fix_ratio_rotated_face(distance_right_cheek, distance_left_cheek)
    # if distance_right_cheek/distance_left_cheek > 1.05:
    #     distance_left_cheek, distance_right_cheek = fix_ratio_rotated_face(distance_left_cheek, distance_right_cheek)

    try:
        avg_cheek_distance = (distance_right_cheek + distance_left_cheek)/2
        distance_mouth_chin = jaw_lowest_point[1] - mouth_lowest_point[1]
    except:
        avg_cheek_distance = 0
        distance_mouth_chin = 0

    #  Section for debug
    # cv2.circle(image, middle_point_between_eyes, 5, (0, 0, 255))
    # cv2.circle(image, (left_point_left_eye[0], left_point_left_eye[1]), 5, (0, 0, 255))
    # cv2.circle(image, (right_point_right_eye[0], right_point_right_eye[1]), 5, (0, 0, 255))
    # cv2.circle(image, (middle_point_nose[0], middle_point_nose[1]), 5, (0, 0, 255))
    # cv2.circle(image, (left_point_nose[0], left_point_nose[1]), 5, (0, 0, 255))
    # cv2.circle(image, (right_point_nose[0], right_point_nose[1]), 5, (0, 0, 255))
    # cv2.circle(image, (Left_Jaw[0], Left_Jaw[1]), 5, (0, 0, 255))
    # cv2.circle(image, (Right_Jaw[0], Right_Jaw[1]), 5, (0, 0, 255))
    # cv2.circle(image, (mouth_lowest_point[0], mouth_lowest_point[1]), 5, (0, 0, 255))
    # cv2.circle(image, (jaw_lowest_point[0], jaw_lowest_point[1]), 5, (0, 0, 255))
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    try:
        ratio_left_eye_nose = distance_left_eye_nose / distance_eye_nose
    except:
        ratio_left_eye_nose = 0
    try:
        ratio_right_eye_nose = distance_right_eye_nose / distance_eye_nose
    except:
        ratio_right_eye_nose = 0
    try:
        ratio_nose_width_middle_eyes = nose_width / distance_eye_nose
    except:
        ratio_nose_width_middle_eyes = 0

    try:
        ratio_cheek_nose = avg_cheek_distance / distance_eye_nose
    except:
        ratio_cheek_nose = 0
    try:
        ratio_mouth_chin_nose = distance_mouth_chin / distance_eye_nose
    except:
        ratio_mouth_chin_nose = 0

    return ratio_mouth_chin_nose, ratio_left_eye_nose, ratio_right_eye_nose, ratio_nose_width_middle_eyes, ratio_cheek_nose


