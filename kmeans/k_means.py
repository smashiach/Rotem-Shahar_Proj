from sklearn.cluster import KMeans
import utils
import cv2
# import matplotlib.pyplot as plt


def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v




def get_dominant_colors(image, path):
    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # show our image
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(image)


    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))


    # cluster the pixel intensities
    clt = KMeans(n_clusters = 3) # RGB values of HSV picture
    clt.fit(image)


    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = utils.centroid_histogram(clt)
    bar = utils.plot_colors(hist, clt.cluster_centers_)

    max_color = hist.max()

    histogram_size = hist.__len__()
    for i in range(0,histogram_size):
        if hist[i] >= max_color:
            dominant_color_index = i

    h, s, v = clt.cluster_centers_[dominant_color_index]
    # h, s, v = rgb_to_hsv(r, g, b)


    # # show our color bart
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()

    return h, s, v
