import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation
from feature_extractors.lbp.extractor import LBP

import skimage


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def hog_extraction(self, img):
        hog_extract = skimage.feature.hog(img)
        return hog_extract

    def hog_extraction_params(self, img):
        hog_extract = skimage.feature.hog(img, orientations=18, pixels_per_cell=(10, 10), cells_per_block=(5, 5),
                                          block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)
        return hog_extract

    def daisy_extraction_params(self, img):
        daisy_extract = skimage.feature.daisy(img, step=2, radius=10, rings=4, histograms=8, orientations=12,
                                              normalization='daisy')
        return daisy_extract

    def daisy_extraction(self, img):
        daisy_extract = skimage.feature.daisy(img)
        return daisy_extract

    def lbp_extraction(self, img):
        radius = 3
        num_points = 8 * radius
        lbp_extract = skimage.feature.local_binary_pattern(img, num_points, radius)
        return lbp_extract

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()
        lbp = LBP()

        cla_d = self.get_annotations(self.annotations_path)

        # Change the following extractors, modify and add your own

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()

        lbp_features_arr = []
        lbp_features_arr_hist = []
        plain_features_arr = []
        y = []
        hog_features_arr = []
        daisy_features_arr = []

        hog_features_arr_params = []
        daisy_features_arr_params = []

        hog_features_arr_sobel = []
        daisy_features_arr_sobel = []

        hog_features_arr_scharr = []
        daisy_features_arr_scharr = []

        hog_features_arr_prewitt = []
        daisy_features_arr_prewitt = []

        lbp_features_arr_sobel = []
        lbp_features_arr_scharr = []
        lbp_features_arr_prewitt = []

        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)

            tmp = im_name.split('/')[-2:][1].replace("\\", "/")
            y.append(cla_d[tmp])

            # Apply some preprocessing here

            # Run the feature extractors
            plain_features = pix2pix.extract(img)
            plain_features_arr.append(plain_features)

            lbp_features = lbp.extract(img)
            lbp_features_arr.append(lbp_features)

            lbp_features_hist = lbp.extract_hist(img)
            lbp_features_arr_hist.append(lbp_features_hist)

            resize = 100
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (resize, resize))

            edge_sobel = skimage.filters.sobel(img)
            edge_scharr = skimage.filters.scharr(img)
            edge_prewitt = skimage.filters.prewitt(img)

            # UNFILTERED
            hog_features = self.hog_extraction(img)
            hog_features_arr.append(hog_features)

            daisy_features = self.daisy_extraction(img)
            daisy_features_arr.append(daisy_features.ravel())

            # EXTRACTORS WITH DIFFERENT PARAMETERS

            hog_features = self.hog_extraction_params(img)
            hog_features_arr_params.append(hog_features)

            daisy_features = self.daisy_extraction_params(img)
            daisy_features_arr_params.append(daisy_features.ravel())

            # SOBEL
            hog_features_sobel = self.hog_extraction(edge_sobel)
            hog_features_arr_sobel.append(hog_features_sobel)

            daisy_features_sobel = self.daisy_extraction(edge_sobel)
            daisy_features_arr_sobel.append(daisy_features_sobel.ravel())

            lbp_features_sobel = self.lbp_extraction(edge_sobel)
            lbp_features_arr_sobel.append(lbp_features_sobel.ravel())
            # SCHARR
            hog_features_scharr = self.hog_extraction(edge_scharr)
            hog_features_arr_scharr.append(hog_features_scharr)

            daisy_features_scharr = self.daisy_extraction(edge_scharr)
            daisy_features_arr_scharr.append(daisy_features_scharr.ravel())

            lbp_features_scharr = self.lbp_extraction(edge_scharr)
            lbp_features_arr_scharr.append(lbp_features_scharr.ravel())
            # PREWITT
            hog_features_prewitt = self.hog_extraction(edge_prewitt)
            hog_features_arr_prewitt.append(hog_features_prewitt)

            daisy_features_prewitt = self.daisy_extraction(edge_prewitt)
            daisy_features_arr_prewitt.append(daisy_features_prewitt.ravel())

            lbp_features_prewitt = self.lbp_extraction(edge_prewitt)
            lbp_features_arr_prewitt.append(lbp_features_prewitt.ravel())

            # print(im_name)

        # Distance matrix Y

        Y_plain = cdist(plain_features_arr, plain_features_arr, 'jensenshannon')

        Y_lbp = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')
        Y_lbp_hist = cdist(lbp_features_arr_hist, lbp_features_arr_hist, 'jensenshannon')

        Y_hog = cdist(hog_features_arr, hog_features_arr, 'jensenshannon')
        Y_daisy = cdist(daisy_features_arr, daisy_features_arr, 'jensenshannon')

        Y_hog_params = cdist(hog_features_arr_params, hog_features_arr_params, 'jensenshannon')
        Y_daisy_params = cdist(daisy_features_arr_params, daisy_features_arr_params, 'jensenshannon')

        Y_hog_sobel = cdist(hog_features_arr_sobel, hog_features_arr_sobel, 'jensenshannon')
        Y_daisy_sobel = cdist(daisy_features_arr_sobel, daisy_features_arr_sobel, 'jensenshannon')
        Y_lbp_sobel = cdist(lbp_features_arr_sobel, lbp_features_arr_sobel, 'jensenshannon')

        Y_hog_scharr = cdist(hog_features_arr_scharr, hog_features_arr_scharr, 'jensenshannon')
        Y_daisy_scharr = cdist(daisy_features_arr_scharr, daisy_features_arr_scharr, 'jensenshannon')
        Y_lbp_scharr = cdist(lbp_features_arr_scharr, lbp_features_arr_scharr, 'jensenshannon')

        Y_hog_prewitt = cdist(hog_features_arr_prewitt, hog_features_arr_prewitt, 'jensenshannon')
        Y_daisy_prewitt = cdist(daisy_features_arr_prewitt, daisy_features_arr_prewitt, 'jensenshannon')
        Y_lbp_prewitt = cdist(lbp_features_arr_prewitt, lbp_features_arr_prewitt, 'jensenshannon')

        # metrics evaluation
        r1 = eval.compute_rank1(Y_plain, y)
        r15 = eval.compute_rank5(Y_plain, y)

        r1_lbp = eval.compute_rank1(Y_lbp, y)
        r1_lbp5 = eval.compute_rank5(Y_lbp, y)

        r1_lbp_hist = eval.compute_rank1(Y_lbp_hist, y)
        r1_lbp_hist5 = eval.compute_rank5(Y_lbp_hist, y)

        r2 = eval.compute_rank1(Y_hog, y)
        r25 = eval.compute_rank5(Y_hog, y)

        r3 = eval.compute_rank1(Y_daisy, y)
        r35 = eval.compute_rank5(Y_daisy, y)

        r5 = eval.compute_rank1(Y_hog_sobel, y)
        r55 = eval.compute_rank5(Y_hog_sobel, y)

        r6 = eval.compute_rank1(Y_daisy_sobel, y)
        r65 = eval.compute_rank5(Y_daisy_sobel, y)

        r1_lbp_sobel = eval.compute_rank1(Y_lbp_sobel, y)
        r5_lbp_sobel = eval.compute_rank5(Y_lbp_sobel, y)

        r7 = eval.compute_rank1(Y_hog_scharr, y)
        r75 = eval.compute_rank5(Y_hog_scharr, y)

        r8 = eval.compute_rank1(Y_daisy_scharr, y)
        r85 = eval.compute_rank5(Y_daisy_scharr, y)

        r1_lbp_scharr = eval.compute_rank1(Y_lbp_scharr, y)
        r5_lbp_scharr = eval.compute_rank5(Y_lbp_scharr, y)

        r9 = eval.compute_rank1(Y_hog_prewitt, y)
        r95 = eval.compute_rank5(Y_hog_prewitt, y)

        r10 = eval.compute_rank1(Y_daisy_prewitt, y)
        r105 = eval.compute_rank5(Y_daisy_prewitt, y)

        r1_lbp_prewitt = eval.compute_rank1(Y_lbp_prewitt, y)
        r5_lbp_prewitt = eval.compute_rank5(Y_lbp_prewitt, y)

        r11 = eval.compute_rank1(Y_hog_params, y)
        r115 = eval.compute_rank5(Y_hog_params, y)

        r12 = eval.compute_rank1(Y_daisy_params, y)
        r125 = eval.compute_rank5(Y_daisy_params, y)

        print('Pix2Pix unfiltered Rank-1[%]', round(r1, 2))
        print('Pix2Pix unfiltered Rank-5[%]', round(r15, 2))

        # LBP
        print('LBP Rank-1[%]', round(r1_lbp, 2))
        print('LBP Rank-5[%]', round(r1_lbp5, 2))

        print('LBP hist Rank-1[%]', round(r1_lbp_hist, 2))
        print('LBP hist Rank-5[%]', round(r1_lbp_hist5, 2))

        print('LBP Sobel Rank-1[%]', round(r1_lbp_sobel, 2))
        print('LBP Sobel Rank-5[%]', round(r5_lbp_sobel, 2))

        print('LBP Scharr Rank-1[%]', round(r1_lbp_scharr, 2))
        print('LBP Scharr Rank-5[%]', round(r5_lbp_scharr, 2))

        print('LBP Prewitt Rank-1[%]', round(r1_lbp_prewitt, 2))
        print('LBP Prewitt Rank-5[%]', round(r5_lbp_prewitt, 2))
        # HOG
        print('HOG unfiltered Rank-1[%]', round(r2, 2))
        print('HOG unfiltered Rank-5[%]', round(r25, 2))

        print('HOG with different parameters, unfiltered image Rank-1[%]', round(r11, 2))
        print('HOG with different parameters, unfiltered image Rank-5[%]', round(r115, 2))

        print('HOG Sobel filtered Rank-1[%]', round(r5, 2))
        print('HOG Sobel filtered Rank-5[%]', round(r55, 2))

        print('HOG Scharr filtered Rank-1[%]', round(r7, 2))
        print('HOG Scharr filtered Rank-5[%]', round(r75, 2))

        print('HOG Prewitt filtered Rank-1[%]', round(r9, 2))
        print('HOG Prewitt filtered Rank-5[%]', round(r95, 2))

        # Daisy
        print('Daisy unfiltered Rank-1[%]', round(r3, 2))
        print('Daisy unfiltered Rank-5[%]', round(r35, 2))

        print('Daisy with different parameters, unfiltered image Rank-1[%]', round(r12, 2))
        print('Daisy with different parameters, unfiltered image Rank-5[%]', round(r125, 2))

        print('Daisy Sobel filtered Rank-1[%]', round(r6, 2))
        print('Daisy Sobel filtered Rank-5[%]', round(r65, 2))

        print('Daisy Scharr filtered Rank-1[%]', round(r8, 2))
        print('Daisy Scharr filtered Rank-5[%]', round(r85, 2))

        print('Daisy Prewitt filtered Rank-1[%]', round(r10, 2))
        print('Daisy Prewitt filtered Rank-5[%]', round(r105, 2))


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()