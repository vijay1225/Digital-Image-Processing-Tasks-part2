import numpy as np
import skimage.io as sk
from pathlib import Path
from matplotlib import pyplot
from skimage import exposure


def vijay_linear_stretch(input_image):
    minimum = np.min(input_image)
    maximum = np.max(input_image)

    scaling_factor = 255/(maximum-minimum)
    stretch_image = np.zeros(np.shape(input_image))
    stretch_image = (scaling_factor * input_image)

    return stretch_image.astype(int)


def vijay_pow_stretch(input_image):
    maximum = np.max(input_image)
    input_image = input_image / maximum
    average = np.mean(input_image)
    exponent = ((np.log(2))/(np.log(1/average)))
    stretch_image = np.power(input_image, exponent)
    maximum1 = np.max(stretch_image)
    return stretch_image * (255/maximum1)


def vijay_histogram_equalization(input_image, clipping=True):
    [freq1, bins] = exposure.histogram(input_image)
    if clipping:
        clip_limit = np.mean(freq1) * 1.5
        count1 = 1
        freq = freq1.copy()
        while count1:
            freq[freq > clip_limit] == clip_limit
            count = np.count_nonzero(freq1)
            diff = freq1 - freq
            count1 = np.count_nonzero(diff)
            distribution_average = np.sum(diff)/count
            freq = freq + distribution_average
            freq[freq1 == 0] == 0

    else:
        freq = freq1
    pmf = freq/np.sum(freq)
    cdf = [sum([i for i in pmf[:x]]) for x in range(1, len(freq)+1)]
    bins = list(bins)

    def mapping(t):
        return cdf[bins.index(t)]

    he_image = np.vectorize(mapping)(input_image)
    return he_image * 255


def vijay_clahe_equalization(input_image):
    size = np.shape(input_image)
    small_window_size = [int(size[0] / 8), int(size[1] / 8)]

    ahe_input_image = np.zeros(size)
    for i in range(0, size[0], small_window_size[0]):
        for j in range(0, size[1], small_window_size[1]):
            ahe_input_image[i:i + small_window_size[0], j:j + small_window_size[1]] = vijay_histogram_equalization(
                input_image[i:i + small_window_size[0], j:j + small_window_size[1]])

    def overlap_mapping(over_pixel, stone_pixel):
        if over_pixel:
            return (over_pixel + stone_pixel) / 2
        else:
            return stone_pixel

    overlap_ahe_input_image = np.zeros(size)
    overlap_percent = [int(small_window_size[0] / 4), int(small_window_size[1] / 4)]
    for i in range(0, size[0], small_window_size[0]):
        for j in range(0, size[1], small_window_size[1]):
            i_low = i - overlap_percent[0]
            i_high = i + small_window_size[0] + overlap_percent[0]
            j_low = j - overlap_percent[1]
            j_high = j + small_window_size[1] + overlap_percent[1]
            if 0 <= i_low <= i_high and 0 <= j_low <= j_high:
                overlap_ahe_input_image[i_low:i_high, j_low:j_high] = np.vectorize(overlap_mapping)(
                    overlap_ahe_input_image[i_low:i_high, j_low:j_high],
                    vijay_histogram_equalization(input_image[i_low:i_high, j_low:j_high]))
            elif 0 >= i_low and 0 <= j_low <= j_high:
                overlap_ahe_input_image[i:i_high, j_low:j_high] = np.vectorize(overlap_mapping)(
                    overlap_ahe_input_image[i:i_high, j_low:j_high],
                    vijay_histogram_equalization(input_image[i:i_high, j_low:j_high]))
            elif 0 <= i_low <= i_high and 0 >= j_low:
                overlap_ahe_input_image[i_low:i_high, j:j_high] = np.vectorize(overlap_mapping)(
                    overlap_ahe_input_image[i_low:i_high, j:j_high],
                    vijay_histogram_equalization(input_image[i_low:i_high, j:j_high]))
            else:
                overlap_ahe_input_image[i:i_high, j:j_high] = vijay_histogram_equalization(
                    input_image[i:i_high, j:j_high])
    return [ahe_input_image, overlap_ahe_input_image]


def vijay_rgb_contrast_stretch(input_image):
    sat_input_image = np.zeros(np.shape(input_image))
    sat_input_image = input_image.copy()

    sat_input_image[:, :, 0] = vijay_saturated_contrast_stretch(input_image[:, :, 0], 5, 40)
    sat_input_image[:, :, 1] = vijay_saturated_contrast_stretch(input_image[:, :, 1], 5, 40)
    sat_input_image[:, :, 2] = vijay_saturated_contrast_stretch(input_image[:, :, 2], 5, 30)

    linearly_stretch_input_image = input_image.copy()
    linearly_stretch_input_image[:, :, 0] = vijay_linear_stretch(input_image[:, :, 0])
    linearly_stretch_input_image[:, :, 1] = vijay_linear_stretch(input_image[:, :, 1])
    linearly_stretch_input_image[:, :, 2] = vijay_linear_stretch(input_image[:, :, 2])

    return [sat_input_image, linearly_stretch_input_image]


def vijay_saturated_contrast_stretch(input_image, low_percentage, high_percentage):
    histogram = exposure.histogram(input_image)
    total_freq = np.sum(histogram[0])
    low_sat_count = total_freq * low_percentage / 100
    high_sat_count = total_freq * high_percentage / 100
    temp = 0
    maximum = len(histogram[1])-1
    minimum = 0
    for i in range(0, len(histogram[1])):
        if temp > low_sat_count:
            minimum = i
            break
        else:
            temp += histogram[0][i]
    temp = 0
    for j in range(len(histogram[1]) - 1, 0, -1):
        if temp > high_sat_count:
            maximum = j
            break
        else:
            temp += histogram[0][j]

    final_image = input_image.copy()
    gain = 254 / histogram[1][maximum]
    final_image = final_image * gain
    final_image[input_image <= histogram[1][minimum]] = 0
    final_image[input_image >= histogram[1][maximum]] = 255

    return final_image


def vijay_resizing_image(input_image, resizing_factor, interpolation_method='bilinear'):
    size = np.shape(input_image)
    new_size = [round(size[0] * resizing_factor) - 1, round(size[1] * resizing_factor) - 1]
    resized_image = np.zeros(new_size)
    if interpolation_method == 'nearest' or resizing_factor <= 1:
        for i in range(new_size[0]):
            for j in range(new_size[1]):
                resized_image[i, j] = input_image[
                    int(np.floor(i / resizing_factor)), int(np.floor(j / resizing_factor))]
    else:
        for i in range(size[0] - 1):
            for j in range(size[1] - 1):
                x1 = int(i * resizing_factor)
                x2 = int((i + 1) * resizing_factor)
                y1 = int(j * resizing_factor)
                y2 = int((j + 1) * resizing_factor)

                co_mat = np.array(
                    [(1, x1, y1, x1 * y1), (1, x1, y2, x1 * y2), (1, x2, y1, x2 * y1), (1, x2, y2, x2 * y2)])
                b_mat = np.array(
                    [input_image[i, j], input_image[i, j + 1], input_image[i + 1, j], input_image[i + 1, j + 1]])
                a_mat = np.matmul(np.linalg.pinv(co_mat), np.transpose(b_mat))
                for m in range(x1, x2):
                    for n in range(y1, y2):
                        resized_image[m, n] = a_mat[0] + a_mat[1] * m + a_mat[2] * n + a_mat[3] * m * n

                [resized_image[x1, y1], resized_image[x1, y2], resized_image[x2, y1], resized_image[x2, y2]] = [
                    input_image[i, j], input_image[i, j + 1], input_image[i + 1, j], input_image[i + 1, j + 1]]

        return resized_image


def vijay_rotate_image(input_image, degree=90, interpolation_method='nearest'):
    theta = np.deg2rad(degree)
    size = np.shape(input_image)
    rotated_image = np.zeros([size[0] * 2, size[1] * 2])
    c_x, c_y = size[0] / 2, size[1] / 2
    if interpolation_method == 'nearest':
        for i in range(2 * size[0]):
            for j in range(2 * size[1]):
                a1 = round(((i - c_x - c_x) * np.cos(theta) - (j - c_y - c_y) * np.sin(theta)) + c_x)
                a2 = round(((i - c_x - c_x) * np.sin(theta) + (j - c_y - c_y) * np.cos(theta)) + c_y)
                if 0 < a1 < size[0] and 0 < a2 < size[1]:
                    rotated_image[i, j] = input_image[a1, a2]
    elif interpolation_method == 'bilinear':
        for i in range(size[0] - 1):
            for j in range(size[1] - 1):
                x1 = round(((i - c_x) * np.cos(theta) + (j - c_y) * np.sin(theta)) + 2 * c_x)
                y1 = round((-(i - c_x) * np.sin(theta) + (j - c_y) * np.cos(theta)) + 2 * c_y)
                x2 = round(((i - c_x + 1) * np.cos(theta) + (j - c_y + 1) * np.sin(theta)) + 2 * c_x)
                y2 = round((-(i - c_x + 1) * np.sin(theta) + (j - c_y + 1) * np.cos(theta)) + 2 * c_y)
                co_mat = np.array(
                    [(1, x1, y1, x1 * y1), (1, x1, y2, x1 * y2), (1, x2, y1, x2 * y1), (1, x2, y2, x2 * y2)])
                b_mat = np.array(
                    [input_image[i, j], input_image[i, j + 1], input_image[i + 1, j], input_image[i + 1, j + 1]])
                a_mat = np.matmul(np.linalg.pinv(co_mat), np.transpose(b_mat))
                for m in range(x1, x2):
                    for n in range(y1, y2):
                        rotated_image[m, n] = a_mat[0] + a_mat[1] * m + a_mat[2] * n + a_mat[3] * m * n

                [rotated_image[x1, y1], rotated_image[x1, y2], rotated_image[x2, y1], rotated_image[x2, y2]] = [
                    input_image[i, j], input_image[i, j + 1], input_image[i + 1, j], input_image[i + 1, j + 1]]
    return rotated_image