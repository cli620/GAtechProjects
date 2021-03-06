import numpy as np
import scipy as sp
import cv2
import scipy.signal
from datetime import datetime

# Get Edges of image
def get_edges(I):

    # Median filter
    # Reduce noise --> 7x7 matrix (centroid = median of surrounding neighbors)
    # extreme specks smoothed. needs to be small enough to preserve edges
    out_I = cv2.medianBlur(src=I, ksize=7)

    # Edge detections
    # Canny edge detector (edges = single pixel edges
    # can change L2 gradient to true later compare results----------------
    # can change 1st and 2nd threshold later to compare results-----------
    sigma=0.5
    v=np.median(out_I)
    out_I = cv2.Canny(out_I, int(max(0,(1.0-sigma)*v)), int(min(255,(1.0+sigma)*v)), L2gradient=False)

    # Morphological operations
    # dilation with 2x2 structuring element --> bolden & smooth contours of edges
    # decides how contours look!
    kernel = np.ones((1,1),np.uint8)
    out_I = cv2.morphologyEx(out_I, cv2.MORPH_DILATE, kernel)

    # Edge Filter
    # Edge image separate into constituent regents --> concrete edges
    # Get rid of small contours
    # Empirically testing MINIMUM AREA THRESHOLD < 10.
    return np.subtract(255, out_I).astype(np.uint8)
    raise NotImplementedError


def get_color_map(I):
    # Bilateral Filter
    # homogenize color regions while preserving edges.
    # --> downsampled factor of 4 ( 1/2 col 1/2 row)
    ori_shape = I.shape
    downI = cv2.resize(I, (0,0), fx=0.5, fy=0.5)

    # use 9x9 filter kernel and do it 14x.
    for i in range(14):
        downI = cv2.bilateralFilter(downI, 9, 18.0, 4.5)

    # restore to original size with linear interpolation to fill in missing pixels
    out_I = cv2.resize(downI, (ori_shape[1], ori_shape[0]))

    # Median Filter
    # smooth any artifacts that occurred for upsampling
    # use 7x7 kernel (from edge median filter kernel)
    out_I = cv2.medianBlur(out_I, 7)

    # Quantize Colors:
    # p = pixel value, a = factor to reduce # of colors in each channel
    # truncate colors close to maximum
    # e.g.: if a = 24, reduce 256/24 to this how many values to be used
    a = 35
    out_I = np.divide(out_I, a)
    out_I = np.floor(out_I)
    out_I = np.multiply(out_I,a)

    return out_I.astype(np.uint8)
    raise NotImplementedError

def recombine_edge_color(I):
    # Getting edge map
    print '--- getting edge map '
    startTime = datetime.now()
    emap = get_edges(I)
    cv2.imwrite('edge_map.png', emap)
    print datetime.now() - startTime

    # Getting Color map)
    print '--- getting color map '
    startTime = datetime.now()
    cmap = get_color_map(I)
    cv2.imwrite('color_map.png', cmap)
    print datetime.now() - startTime

    # Two possible methods:
    # average the two maps
    # multiply the two maps.
    print '--- averaging edge and color maps'
    startTime = datetime.now()
    manga = cv2.bitwise_and(cmap, cmap, mask=emap)
    cv2.imwrite('manga_test.png', manga.astype(np.uint8))
    print datetime.now() - startTime

    print '------manga done | size = ', manga.shape
    return manga
    raise NotImplementedError

