from imutils import face_utils
import numpy as np
import imutils
import dlib
import os
import cv2


def eye_dimension_extract(image_source):
    shape_predictor = os.getcwd() + '\\shape_predictor_68_face_landmarks.dat'

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_source)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    # detect faces in the grayscale image
    rects = detector(gray, 1)
    LeftEyeCoorX = []  # left eye X coordinates array
    LeftEyeCoorY = []  # left eye Y coordinates array
    RightEyeCoorX = []  # right eye X coordinates array
    RightEyeCoorY = []  # right eye Y coordinates array
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
                    RightEyeCoorX.append(x)
                    RightEyeCoorY.append(y)
                if name == "left_eye":
                    LeftEyeCoorX.append(x)
                    LeftEyeCoorY.append(y)
            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # show the particular face part
            # cv2.imshow("ROI", roi)
            # cv2.imshow("Image", clone)
            # cv2.waitKey(0)

        # visualize all facial landmarks with a transparent overlay
        output = face_utils.visualize_facial_landmarks(image, shape)
        # cv2.imshow("Image", output)
        # cv2.waitKey(0)

    # extract dimensions of right eye
    RightEyeHorizontalDistance = np.abs((np.max(RightEyeCoorX) - np.min(RightEyeCoorX)))
    RightEyeVerticalDistance = np.abs((np.max(RightEyeCoorY) - np.min(RightEyeCoorY)))

    # extract dimensions of left eye
    LeftEyeHorizontalDistance = np.abs((np.max(LeftEyeCoorX) - np.min(LeftEyeCoorX)))
    LeftEyeVerticalDistance = np.abs((np.max(LeftEyeCoorY) - np.min(LeftEyeCoorY)))

    # calculate distance between eyes
    DistanceBetweenEyes = np.abs((np.max(LeftEyeCoorX) - np.min(RightEyeCoorX)))

    return RightEyeHorizontalDistance, RightEyeVerticalDistance, LeftEyeHorizontalDistance, LeftEyeVerticalDistance, DistanceBetweenEyes


