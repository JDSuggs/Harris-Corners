import cv2
import copy
from operator import itemgetter


# find corners in an image and save image files with corners displayed and files with corner data created
def harris_corners(image, counter):
    # initialize variables
    cornerness_measurements = []
    section_list = []
    corner_list = []
    height = image.shape[0]
    width = image.shape[1]
    height_section = height / 5
    width_section = width / 5

    # images for different corner calculations
    colored_image = copy.deepcopy(image)
    percent_corners = copy.deepcopy(image)
    n_corners = copy.deepcopy(image)
    partition_corners = copy.deepcopy(image)

    # convert image to grayscale
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gradient in X and Y
    direction_grad_im_x = cv2.Sobel(gray_im, cv2.CV_64F, 1, 0, ksize=3)
    direction_grad_im_y = cv2.Sobel(gray_im, cv2.CV_64F, 0, 1, ksize=3)

    # Determine i_x_x, i_x_y, and i_y_y for calculations
    i_x_x = direction_grad_im_x * direction_grad_im_x
    i_x_y = direction_grad_im_x * direction_grad_im_y
    i_y_y = direction_grad_im_y * direction_grad_im_y

    # Iterate over image
    for j in range(1, height - 1):
        for i in range(1, width - 1):

            # find sums of IxIx, IxIy, and IyIy for each pixel and 8 surrounding ones
            sum_i_x_x = 0
            sum_i_x_y = 0
            sum_i_y_y = 0
            for k in (-1, 0, 1):
                for m in (-1, 0, 1):
                    sum_i_x_x += i_x_x[j+k][i+m]
                    sum_i_x_y += i_x_y[j+k][i+m]
                    sum_i_y_y += i_y_y[j+k][i+m]

            # Use sums to calculate determinant, trace, and r
            det = (sum_i_x_x * sum_i_y_y) - (sum_i_x_y ** 2)
            trace = sum_i_x_x + sum_i_y_y
            r = det - (.05 * trace)

            # create list of cornerness measurements from all pixels locations and r values
            cornerness_measurements.append([i, j, r])

    # sort cornerMeasurements
    cornerness_measurements = sorted(cornerness_measurements, key=itemgetter(2), reverse=True)

    # find the max and min “cornerness” measurements in the image
    max_k = cornerness_measurements[0][2]
    min_k = cornerness_measurements[len(cornerness_measurements) - 1][2]
    pixel_color_divisor = max_k - min_k

    for i in cornerness_measurements:
        if counter == 0:
            # percent of corner max for image 0
            if i[2] >= max_k * .58:
                # add corners to corner_list if above threshold
                corner_list.append(i)
                # make red circles to denote all pixels whose “cornerness” is greater than a certain percentage of max_k
                cv2.circle(percent_corners, (i[0], i[1]), 2, (0, 0, 255), -1)
        if counter == 1:
            # percent of corner max for image 1
            if i[2] >= max_k * .0004:
                # add corners to corner_list if above threshold
                corner_list.append(i)
                # make red circles to denote all pixels whose “cornerness” is greater than a certain percentage of max_k
                cv2.circle(percent_corners, (i[0], i[1]), 2, (0, 0, 255), -1)
        if counter == 2:
            # percent of corner max for image 2
            if i[2] >= max_k * .005:
                # add corners to corner_list if above threshold
                corner_list.append(i)
                # make red circles to denote all pixels whose “cornerness” is greater than a certain percentage of max_k
                cv2.circle(percent_corners, (i[0], i[1]), 2, (0, 0, 255), -1)
        if counter == 3:
            # percent of corner max for image 3
            if i[2] >= max_k * .04:
                # add corners to corner_list if above threshold
                corner_list.append(i)
                # make red circles to denote all pixels whose “cornerness” is greater than a certain percentage of max_k
                cv2.circle(percent_corners, (i[0], i[1]), 2, (0, 0, 255), -1)
        if counter == 4:
            # percent of corner max for image 4
            if i[2] >= max_k * .02:
                # add corners to corner_list if above threshold
                corner_list.append(i)
                # make red circles to denote all pixels whose “cornerness” is greater than a certain percentage of max_k
                cv2.circle(percent_corners, (i[0], i[1]), 2, (0, 0, 255), -1)
        if counter == 5:
            # percent of corner max for image 5
            if i[2] >= max_k * .075:
                # add corners to corner_list if above threshold
                corner_list.append(i)
                # make red circles to denote all pixels whose “cornerness” is greater than a certain percentage of max_k
                cv2.circle(percent_corners, (i[0], i[1]), 2, (0, 0, 255), -1)

    # save and display image
    f_name = "percent_corners_" + str(counter) + ".png"
    cv2.imwrite(f_name, percent_corners)
    cv2.imshow(f_name, percent_corners / 255.0)

    # make blue circles to denote all pixels whose “cornerness” is in the highest 250 (n=250)
    for i in range(250):
        cv2.circle(n_corners, (cornerness_measurements[i][0], cornerness_measurements[i][1]), 2, (255, 0, 0), -1)

    # save and display image
    f_name = "n_corners_" + str(counter) + ".png"
    cv2.imwrite(f_name, n_corners)
    cv2.imshow(f_name, n_corners / 255.0)

    for i in range(5):
        for j in range(5):
            section_list.clear()

            # calculate section coordinates
            x1 = width_section * j
            x2 = width_section * (j + 1)
            y1 = height_section * i
            y2 = height_section * (i + 1)

            # determine if cornerness measurements are in section
            for k in cornerness_measurements:
                if x1 <= k[0] < x2:
                    if y1 <= k[1] < y2:
                        section_list.append(k)

            # draw green circles to denote n pixels with highest cornerness values in neighborhood
            # cornernessMeasurements was already sorted so section_list contains highest cornerness measurements at
            # front of list
            for m in range(50):
                cv2.circle(partition_corners, (section_list[m][0], section_list[m][1]), 2, (0, 255, 0), -1)

    # save and display image
    f_name = "partition_corners" + str(counter) + ".png"
    cv2.imwrite(f_name, partition_corners)
    cv2.imshow(f_name, partition_corners / 255.0)

    # create text file listing all pixels in the image with the highest “cornerness” values as corners
    f_name = "corners_" + str(counter) + ".txt"
    corners = open(f_name, 'w')
    for i in range(corner_list.__len__()):
        corners.write(str(corner_list[i][0]) + ',' + str(corner_list[i][1]) + ',' + str(corner_list[i][2]) + '\n')
    corners.close()

    # colors representing cornerness (extra credit)
    for i in cornerness_measurements:
        # determine the pixel color ratio
        pixel_color_factor = ((i[2] - min_k) / pixel_color_divisor)

        # color the picture based on the ratio
        if pixel_color_factor <= .01:
            colored_image[i[1], i[0]] = [255, 0, 0]
        if .01 < pixel_color_factor <= .375:
            colored_image[i[1], i[0]] = [192, 64, 0]
        if .375 < pixel_color_factor <= .625:
            colored_image[i[1], i[0]] = [0, 255, 255]
        if .625 < pixel_color_factor <= .875:
            colored_image[i[1], i[0]] = [0, 128, 255]
        if .875 < pixel_color_factor <= 1.:
            colored_image[i[1], i[0]] = [0, 0, 255]

    # save and display image
    f_name = "colored_corners_" + str(counter) + ".png"
    cv2.imwrite(f_name, colored_image)
    cv2.imshow(f_name, colored_image / 255.0)


def main():
    image_list = ["checkerboard.png", "square.png", "zeus.png", "puzzle.png", "landscape.png", "league.png"]
    for i in image_list:
        image = cv2.imread(i)
        harris_corners(image, image_list.index(i))
    cv2.waitKey(0)
    # wait until program finishes running, it will create 24 new images


main()
