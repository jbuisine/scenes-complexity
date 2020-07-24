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

# ml imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


figures_output = 'data/figs'

def main():

    parser = argparse.ArgumentParser(description="Classify zones using complexity criteria (unsupervised learning)")

    parser.add_argument('--data', type=str, help='required data file', required=True)
    parser.add_argument('--clusters', type=int, help='number of expected clusters', default=2)
    parser.add_argument('--output', type=str, help='output folder name', required=True)

    args = parser.parse_args()

    p_data = args.data
    p_clusters = args.clusters
    p_output = args.output

    x_values = []
    images_path = []
    zones = []

    with open(p_data, 'r') as f:
        for line in f.readlines():
            data = line.split(';')
            del data[-1]

            print(data)
            images_path.append(data[0])
            zones.append(int(data[1]))

            x = []
            for v in data[2:]:
                x.append(float(v))

            x_values.append(x)    

    kmeans = KMeans(init='k-means++', n_clusters=p_clusters, n_init=10)
    kmeans.fit(x_values)

if __name__ == "__main__":
    main()