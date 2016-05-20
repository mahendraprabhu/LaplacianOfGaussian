# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:12:48 2016
Author:     Mahendra Ramachandran
Revision:   1.0
Status:
Comments:   Trying out LoG function on a Image
----------------------------------
"""

def get_log_kernel(siz, std):
    import numpy as np
    import sys

    x = y = np.linspace(-siz, siz, 2*siz+1)
    x, y = np.meshgrid(x, y)
    arg = -(x**2 + y**2) / (2*std**2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h/h.sum() if h.sum() != 0 else h
    h1 = h*(x**2 + y**2 - 2*std**2) / (std**4)
    return h1 - h1.mean()


import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = 'Eutypa - 1.jpg'
img = cv2.imread(filename,0)

plt.close('all')

#LogKernel = get_log_kernel(5,2)
a, b = [],[]

ValueRange = np.arange(0,10,0.1)
for i in ValueRange:
    m = cv2.filter2D(img,ddepth=-1, kernel = get_log_kernel(7,i))
    a.append(i)
    b.append(m.max())

plt.plot(a,b)
#plt.subplot(1,2,1), plt.imshow(img)i,
#plt.subplot(1,2,2), plt.imshow(m)


#for i in np.arange(0,10,0.2):
#    im1 = cv2.GaussianBlur(img,(15,15),i)
#    laplacian = cv2.Laplacian(im1,cv2.CV_64F, ksize=5)
#    a.append(i)
#    b.append(np.sort(laplacian.ravel())[-5:].mean())
#
#
#b = np.array(b)
#print a[np.nonzero(b==b.max())[0]]
#plt.subplot(121),plt.plot(a,b,'ro')
#plt.subplot(122),plt.imshow(im1)
