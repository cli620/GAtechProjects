import numpy as np
import scipy as sp
import cv2
import scipy.signal
from datetime import datetime

def band_sift_operator(image, mult_factor, sign, freq, amp):
    #inputs: sign --> -1 1 0  ==> neg, pos, all
    #        freq --> -1 1 0  ==> low, high, all
    #        amp  --> -1 1 0  ==> low, high, all

    #find luminance of image https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color

    luminance = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
    # define n = number of subbands
    #subbands = difference between two successively filtered version of the image.
    err = 0.00001
    subband = np.add(decompose(np.add(luminance, err)), err)
    n = len(subband)
    for levels in range(n):
        print '---working on gloss', levels, 'th level'
        startTime = datetime.now()
        ref = subband[levels]+err # keep copy of the subband as reference
        for coefx in range(subband[levels].shape[0]):
            for coefy in range(subband[levels].shape[1]):
                coef = subband[levels][coefx, coefy] #ref
                # print 'coef = ', coef, ' | sign = ', sign, ' | freq = ', freq, ' | n = ', n, ' | levels = ', levels
                if sign_freq_selected(coef, sign, freq, n, levels):
                    if amp == 0:
                        # straight up multiply factors
                        subband[levels][coefx, coefy] = coef * mult_factor
                        # print 'amp = all'
                    else:
                        #smoothing transition between high and low amplitudes
                        sigma = np.std(subband[levels])
                        mu = np.mean(subband[levels])

                        # print 'coef = ', coef, ' | 0.8*sigma + mu = ', 0.8*sigma+mu, ' | 1.2*sigma+mu = ', 1.2*sigma+mu
                        if abs(coef) < 0.8*sigma+mu:
                            alpha = 0
                        elif abs(coef) > 1.2*sigma+mu:
                            alpha = 1
                        else:
                            alpha = (coef - 0.8*sigma)/(1.2*sigma - 0.8*sigma)

                        # orient transition depending on amplitude selection
                        if amp == 1:
                            # print 'amp = high'
                            subband[levels][coefx,coefy] = coef*(1 + alpha*(mult_factor-1))

                        elif amp == -1:
                            # print 'amp = low'
                            subband[levels][coefx,coefy] = coef*(1+ (1-alpha)*(mult_factor-1))
                        else:
                            print 'Please insert correct values of amp [0 -1 1]... current value: ', amp
                else:
                    print
        # this part does the smoothing for applied gain map. get rid of random noises.
        gain = np.divide(subband[levels], ref)
        for i in range(n):
            gain = cv2.medianBlur(gain.astype(np.float32),ksize=5)
        for i in range(n):
            gain = cv2.blur(gain,ksize=(3,3))

        subband[levels] = np.multiply(gain, np.subtract(ref, err))
        cv2.imwrite('subband{0:01d}.png'.format(levels), subband[levels])
        print datetime.now() - startTime
    print '------gloss done | shape = ', sum(subband).shape
    out = cv2.exp(sum(subband))-err
    cv2.imwrite('gloss.png', out.astype(np.uint8))
    return out
    raise NotImplementedError

def decompose(C):
    #C = luminance --> single channel
    # number of subbands = log of minimum shape of image.
    C = np.uint8(C)
    n = round(scipy.log2(min(C.shape[0], C.shape[1])))
    n = int(n)

    # Boyadzhiev uses a Guided filter but
    # Fujieda et al. uses bilateral.
    # start off with 0

    temp1 = C
    count = 0
    subband = [None]*(int(n/2) + 1)
    for i in range(0,n+1,2):
        # double spatial extend each time. starting at 2.
        # print '---working on ', i, 'th bilateral filter '
        sigmacolor = np.std(temp1)
        sigmaspace = np.float32(i/2)

        temp2 = temp1
        temp1 = cv2.bilateralFilter(temp1, i, sigmacolor, sigmaspace)
        # try using guided filter
        subband[count] = temp2-temp1
        # print len(subband), ' | ', subband[count].shape
        cv2.imwrite('pre_subband{0:01d}.png'.format(i), subband[count])
        count = count + 1

    return subband

    raise NotImplementedError


def sign_freq_selected(c, sign, freq, n, levels):
    # print 'sign == 0? = ', (sign == 0), '  (c < 0 & sign == -1) ? = ',  (c < 0 & sign == -1) , '(c >= 0 & sign == 1) ? = ', (sign == 1) & (c >= 0 )
    # print 'sign == 1 ==> ', sign == 1, 'c >= 0 --> ', c >= 0
    # print '(sign == 1) & (c >= 0)', (sign == 1) & (c >= 0)
    signSelected = ((sign == 0) | ((sign == -1) & (c < 0)) | ((sign == 1) & (c >= 0)))

    freqSelected = ((freq == 0) | ((levels <= n/2) & (freq == -1)) | ((levels >= n/2) & (freq == 1)))
    # print 'signSelected = ', signSelected, ' | freqSelected = ', freqSelected

    return signSelected & freqSelected
    raise NotImplementedError