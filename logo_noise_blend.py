# -*- coding: utf-8 -*-
#%% Import necessary packages

import cv2
from skimage.util import random_noise
import numpy as np
#%%Read in image and preprocess

gray = cv2.imread("3m_synth_img.jpg", 0)

cv2.imshow('Grayscale: Original Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%Adding noise to the logo region

noise_img = random_noise(gray[337:508, 98:470], mode='speckle', var=np.std(gray)/255)
# random_noise returns a floating-point image on the range [0, 1]. Converting it to 'uint8' and in range [0,255]
noise_img = np.array(255*noise_img, dtype = 'uint8')

cv2.imshow('Logo with Noise', noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%% Blurring the logo region

gray[337:508, 98:470] = cv2.GaussianBlur(noise_img, (3,3), 0.5)

cv2.imshow('Gray with Blur: Final Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()