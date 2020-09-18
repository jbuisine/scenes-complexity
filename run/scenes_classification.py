import os, sys
import argparse
import numpy as np
import gzip

# image processing
from PIL import Image
import cv2
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

# ml imports
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

model_output_folder = 'data/models'

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
    images = []
    zones = []
    scenes = []

    with open(p_data, 'r') as f:
        for line in f.readlines():
            data = line.split(';')
            del data[-1]

            scene = data[0]
            if scene not in scenes:
                scenes.append(scene)

            images_path.append(data[1])
            zones.append(int(data[2]))

            img_arr = segmentation.divide_in_blocks(Image.open(data[1]), (200, 200))[int(data[2])]
            images.append(np.array(img_arr))

            x = []
            for v in data[3:]:
                x.append(float(v))

            x_values.append(x)    

    print(scenes)
    # plt.show()
    # TODO : save kmean model
    kmeans = KMeans(init='k-means++', n_clusters=p_clusters, n_init=10)
    labels = kmeans.fit(x_values).labels_

    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    pca = PCA(n_components=2)
    x_data = pca.fit_transform(x_values)

    # Need to create as global variable so our callback(on_plot_hover) can access
    fig, ax = plt.subplots()
    fig.set_figheight(20)
    fig.set_figwidth(40)

    ax.tick_params(axis='both', which='major', labelsize=20)

    sc = plt.scatter(x_data[:, 0], x_data[:, 1], c=labels, linewidths=10)

    # annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
    #                     bbox=dict(boxstyle="round", fc="w"),
    #                     arrowprops=dict(arrowstyle="->"))
    imagebox = OffsetImage(images[0], zoom=1.2)
    imagebox.image.axes = ax

    annot = AnnotationBbox(imagebox, xy=(0,0),
                    xybox=(-150., 150.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.8,
                    arrowprops=dict(arrowstyle="->"))

    annot.set_visible(False)
    ax.add_artist(annot)

    def update_annot(ind):
        
        imagebox = OffsetImage([images[n] for n in ind["ind"]][0], zoom=1.2)
        imagebox.image.axes = ax
        
        pos = sc.get_offsets()[ind["ind"][0]]

        setattr(annot,'offsetbox', imagebox)
        annot.xy = pos
        # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
        #                     " ".join([images_path[n] for n in ind["ind"]]))
        # annot.text(text)
        # #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        # annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)

    model_path = os.path.join(model_output_folder, p_output + '.joblib')
    joblib.dump(kmeans, model_path)
    
if __name__ == "__main__":
    main()