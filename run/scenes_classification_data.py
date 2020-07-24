import os, sys
import argparse
import numpy as np
import gzip

# image processing
from PIL import Image
import cv2
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib.pyplot as plt

from estimators import estimate, estimators_list

data_output = 'data/generated'

def write_progress(progress):
    barWidth = 180

    output_str = "["
    pos = barWidth * progress
    for i in range(barWidth):
        if i < pos:
           output_str = output_str + "="
        elif i == pos:
           output_str = output_str + ">"
        else:
            output_str = output_str + " "

    output_str = output_str + "] " + str(int(progress * 100.0)) + " %\r"
    print(output_str)
    sys.stdout.write("\033[F")


def main():

    parser = argparse.ArgumentParser(description="Check complexity of each zone of scene using estimator during rendering")

    parser.add_argument('--folder', type=str, help='folder where scenes with png scene file are stored')
    parser.add_argument('--output', type=str, help='output data filename', required=True)

    args = parser.parse_args()

    p_folder = args.folder
    p_output = args.output

    estimators = ['sv_noise_all']

    folders = [ f for f in os.listdir(p_folder) if 'min_max' not in f ]

    n_zones = 16

    if not os.path.exists(data_output):
        os.makedirs(data_output)

    datafile = open(os.path.join(data_output, p_output), 'w')

    for i, scene in enumerate(sorted(folders)):

        zones = []
        zones_indices = np.arange(n_zones)

        for _ in zones_indices:
            zones.append([])
            
        scene_folder = os.path.join(p_folder, scene)

        # get all images and extract estimator for each blocks
        images = sorted([ i for i in os.listdir(scene_folder) if '.png' in i ])

        # get reference image
        img_path = os.path.join(scene_folder, images[-1])

        img_arr = np.array(Image.open(img_path))

        blocks = segmentation.divide_in_blocks(img_arr, (200, 200))
        
        for index, b in enumerate(blocks):

            # extract data and write into file
            x = []
            
            for estimator in estimators:
                estimated = estimate(estimator, b)
                
                if not isinstance(estimated, np.float64):
                    for v in estimated:
                        x.append(v)
                else:
                    x.append(estimated)

            line = img_path + ';' + str(index) + ';'
            for v in x:
                line += str(v) + ';'
            line += '\n'

            datafile.write(line)

            write_progress((i * n_zones + index + 1) / (float(len(folders)) * float(n_zones)))


    datafile.close()

if __name__ == "__main__":
    main()