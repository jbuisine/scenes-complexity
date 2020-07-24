import sys
import numpy as np
import gzip

# image processing
from PIL import Image
import cv2
from ipfml import utils
from ipfml.processing import transform, segmentation

estimators_list = ['variance', 'l_variance', 'mean', 'l_mean', 'sv_struct', 'sv_noise', 'sobel', 'l_kolmogorov', 'sv_struct_all', 'sv_noise_all']

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

        return sys.getsizeof(compress_data)

    if estimator == 'l_sv_entropy':
        pass
