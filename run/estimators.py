import sys
import numpy as np
import gzip

# image processing
from PIL import Image
from skimage import color, restoration
import cv2
from ipfml import utils
from ipfml.processing import transform, segmentation, compression
from ipfml.utils import get_entropy
from scipy.signal import medfilt2d, wiener, cwt
import pywt

estimators_list = ['variance', 'l_variance', 'mean', 'l_mean', 'sv_struct', 'sv_noise', 'sobel', 'l_kolmogorov', 'sv_struct_all', 'sv_noise_all', 'l_sv_entropy_blocks', '26_attributes', 'statistics_extended']

def estimate(estimator, arr):

    if estimator == 'variance':
        return np.var(arr)
    
    if estimator == 'l_variance':
        return np.var(transform.get_LAB_L(arr))

    if estimator == 'mean':
        return np.mean(arr)

    if estimator == 'l_mean':
        return np.mean(transform.get_LAB_L(arr))

    if estimator == 'sv_struct_all':
        return transform.get_LAB_L_SVD_s(arr)[0:50]

    if estimator == 'sv_struct':
        return np.mean(transform.get_LAB_L_SVD_s(arr)[0:50])

    if estimator == 'sv_noise_all':
        return transform.get_LAB_L_SVD_s(arr)[50:]

    if estimator == 'sv_noise':
        return np.mean(transform.get_LAB_L_SVD_s(arr)[50:])

    if estimator == 'sobel':

        lab_img = transform.get_LAB_L(arr)

        # 1. extract sobol complexity with kernel 3
        sobelx = cv2.Sobel(lab_img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(lab_img, cv2.CV_64F, 0, 1,ksize=5)

        sobel_mag = np.array(np.hypot(sobelx, sobely), 'uint8')  # magnitude

        return np.std(sobel_mag)

    if estimator == 'l_kolmogorov':

        lab_img = transform.get_LAB_L(arr)

        bytes_data = lab_img.tobytes()
        compress_data = gzip.compress(bytes_data)

        mo_size = sys.getsizeof(compress_data) / 1024.
        go_size = mo_size / 1024.

        return np.float64(go_size)

    if estimator == 'l_sv_entropy_blocks':

        # get L channel
        L_channel = transform.get_LAB_L(arr)

        # split in n block
        blocks = segmentation.divide_in_blocks(L_channel, (20, 20))

        entropy_list = []

        for block in blocks:
            reduced_sigma = compression.get_SVD_s(block)
            reduced_entropy = get_entropy(reduced_sigma)
            entropy_list.append(reduced_entropy)

        return entropy_list

    if estimator == '26_attributes':

        img_width, img_height = 200, 200

        lab_img = transform.get_LAB_L(arr)
        arr = np.array(lab_img)

        # compute all filters statistics
        def get_stats(arr, I_filter):

            e1       = np.abs(arr - I_filter)
            L        = np.array(e1)
            mu0      = np.mean(L)
            A        = L - mu0
            H        = A * A
            E        = np.sum(H) / (img_width * img_height)
            P        = np.sqrt(E)

            return mu0, P
            # return np.mean(I_filter), np.std(I_filter)

        stats = []

        kernel = np.ones((3,3),np.float32)/9
        stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

        kernel = np.ones((5,5),np.float32)/25
        stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 0.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 0.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1.5)))

        stats.append(get_stats(arr, medfilt2d(arr, [3, 3])))

        stats.append(get_stats(arr, medfilt2d(arr, [5, 5])))

        stats.append(get_stats(arr, wiener(arr, [3, 3])))

        stats.append(get_stats(arr, wiener(arr, [5, 5])))

        wave = w2d(arr, 'db1', 2)
        stats.append(get_stats(arr, np.array(wave, 'float64')))

        data = []

        for stat in stats:
            data.append(stat[0])

        for stat in stats:
            data.append(stat[1])
        
        data = np.array(data)

        return data

    if estimator == 'statistics_extended':

        # data = estimate('26_attributes', arr)
        data = np.empty(0)
        # add kolmogorov complexity
        bytes_data = np.array(arr).tobytes()
        compress_data = gzip.compress(bytes_data)

        mo_size = sys.getsizeof(compress_data) / 1024.
        go_size = mo_size / 1024.
        data = np.append(data, go_size)

        lab_img = transform.get_LAB_L(arr)
        arr = np.array(lab_img)

        # add of svd entropy
        svd_entropy = utils.get_entropy(compression.get_SVD_s(arr))
        data = np.append(data, svd_entropy)

        # add sobel complexity (kernel size of 3)
        sobelx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(arr, cv2.CV_64F, 0, 1,ksize=3)

        sobel_mag = np.array(np.hypot(sobelx, sobely), 'uint8')  # magnitude

        data = np.append(data, np.std(sobel_mag))
        
        # add sobel complexity (kernel size of 5)
        sobelx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(arr, cv2.CV_64F, 0, 1,ksize=5)

        sobel_mag = np.array(np.hypot(sobelx, sobely), 'uint8')  # magnitude

        data = np.append(data, np.std(sobel_mag))

        return data

def w2d(arr, mode='haar', level=1):
    #convert to float    
    imArray = arr

    sigma = restoration.estimate_sigma(imArray, average_sigmas=True, multichannel=False)
    imArray_H = restoration.denoise_wavelet(imArray, sigma=sigma, wavelet='db1', mode='hard', 
        wavelet_levels=2, 
        multichannel=False, 
        convert2ycbcr=False, 
        method='VisuShrink', 
        rescale_sigma=True)

    # imArray_H *= 100

    return imArray_H