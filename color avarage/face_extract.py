import cv2
import sys
import numpy as np

CASCADE = "D:\\Shahar\\Project A\\part1\\haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE)


def detect_faces(image_path):
    # first part : detect face from image and crop the detected face to new file
    image = cv2.imread(image_path)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting image to grey colors
    faces = FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25), flags=0)  # find face by haarcascade algorithm
    for x, y, w, h in faces:  # crop face from image and convert colors to HSV scale
        sub_img = image[y - 10:y + h + 10, x - 10:x + w + 10]
        sub_image_hsv = cv2.cvtColor(sub_img, cv2.COLOR_RGB2HSV) # convert the face picture to HSV
    # cv2.imwrite('extract.jpg', sub_image_hsv)
    # second part : calculate average colors from the face
    number_of_iteration = 500  # average of 1000 iterations for each face
    tmp_sum = 0
    try:
        height, width = sub_image_hsv.shape[:2]  # extract image dimensions
    except:
        return None
    sample_delta = int((np.maximum(height, width)*0.05))  # adjusting the rectangle sample to the size of the pic (3% from the pic size)
    mu_x = round(width/2)
    mu_y = round(height*1/5)
    sigma_x = 4
    sigma_y = 4
    for i in range(1, number_of_iteration+1):  # calculate random pixel by gaussian segmentation
        x = np.random.normal(mu_x, sigma_x, 1)  # calculate a random x_pixel
        y = np.random.normal(mu_y, sigma_y, 1)  # calculate a random y_pixel
        pixel_x = int(np.round(x))
        pixel_y = int(np.round(y))
        avg = np.mean(sub_image_hsv[pixel_x-sample_delta:pixel_x + sample_delta, pixel_y-sample_delta:pixel_y + sample_delta]) # calculate the average of rectangle
        tmp_sum = tmp_sum + avg
        # cv2.rectangle(sub_image_hsv, (pixel_x - sample_delta, pixel_y - sample_delta),
        #               (pixel_x + sample_delta, pixel_y + sample_delta), (255, 255, 0), 2)
        # cv2.imshow("Faces Found", sub_image_hsv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    ans = tmp_sum / number_of_iteration


    return ans