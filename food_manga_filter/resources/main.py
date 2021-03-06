import errno
import os
import sys

import numpy as np
import cv2

from glob import glob

import gloss
import manga

from datetime import datetime

def readImages(image_dir):
    """This function reads in input images from a image directory

    Note: This is implemented for you since its not really relevant to
    computational photography (+ time constraints).

    Args:
    ----------
        image_dir : str
            The image directory to get images from.

    Returns:
    ----------
        images : list
            List of images in image_dir. Each image in the list is of type
            numpy.ndarray.

    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]

    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting."
            .format(image_dir))

    return images



if __name__ == "__main__":
    mainstart = datetime.now()

    use_dir = "test_dir"
    image_dir = os.path.join("pics", "source", use_dir)
    out_dir = os.path.join("pics", "out")
    try:
        _out_dir = os.path.join(out_dir, use_dir)
        not_empty = not all([os.path.isdir(x) for x in
                             glob(os.path.join(_out_dir, "*.*"))])
        os.makedirs(_out_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    print "Reading images."
    images = readImages(image_dir)

    for i in range(len(images)):
        # inputs: sign --> -1 1 0  ==> neg, pos, all
        #        freq --> -1 1 0  ==> low, high, all
        #        amp  --> -1 1 0  ==> low, high, all

        images[i] = cv2.resize(images[i], (0,0), fx=0.5, fy=0.5)
        cv2.imwrite(os.path.join(image_dir, "smaller", 'input{0:01d}.png'.format(i)),images[i])
        imagetime = datetime.now()
        #suggested 010
        freq = 0
        sign = 1
        amp = 0

        print 'final average of gloss and manga images'
        manga_img = cv2.add(manga.recombine_edge_color(images[i]).astype(np.uint8),
                            cv2.cvtColor(gloss.band_sift_operator(images[i], 1.5, sign, freq, amp).astype(np.uint8), code=cv2.COLOR_GRAY2BGR))
        cv2.imwrite(os.path.join(out_dir, use_dir, 'glossy_manga{0:01d}.png'.format(i)), manga_img.astype(np.uint8))

        print '------------ time taken by this image: ', datetime.now() - imagetime

    print '------------ TOTAL TIME: ', datetime.now() - imagetime