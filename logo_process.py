import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import random_noise

img_xil = plt.imread('./ICs/xilinx.png')
logo_3m = np.asarray(Image.open('./Logos/3M/0.png'))