from face_extract import detect_faces
import glob
import numpy


adjust_std = 1.35
avg_color_pic_arr = []
potential_pic_array = []


for img in glob.glob('D:\\Shahar\\Project A\\compare\\training_set\\*.jpg'): #The location of training set
    PicColorAvg = detect_faces(img)
    name = img.split("\\")[5]  # Get the picture name
    print('The average of pic', name, 'is:', PicColorAvg)
    avg_color_pic_arr.append(PicColorAvg)

avg = numpy.average(avg_color_pic_arr)  # Get the average of all the training pictures
std = numpy.std(avg_color_pic_arr)  # Get the std of all the training pictures

print('The average is:', avg)
print('The std is:', std)
print('The range is from', avg - std, 'to', avg + std)

for pic in glob.glob('D:\\Shahar\\Project A\\compare\\test_set\\*.jpg'):  # The location of testing set
    pic_avg = detect_faces(pic)
    name2 = pic.split("\\")[5]  # Get the picture name
    print('The average of pic', name2, 'is:', pic_avg)
    if (pic_avg >= avg - (std * adjust_std)) and (
            pic_avg <= avg + (std / adjust_std)):  # Add to array only the relevant pictures
        potential_pic_array.append(pic)

print(potential_pic_array)



