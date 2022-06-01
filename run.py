#!/usr/bin/python
import time
import cv2
import matplotlib
import argparse
import os
import ndviLibrary as nd
import json
import convert as c
import matplotlib.pyplot as plt
import numpy as np
import imutils
import utils as u


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to image")

args = vars(ap.parse_args())


nir = args['image']

#convert the rgb image to NIR
c.convert_nir(args['image'])


#calculate the greeness of the image
crops_ndvi_a = nd.NDVI(nir, nir, False, False, args['image'])
crops_data_a = crops_ndvi_a.convert()

#calculate the greeness of the image
crops_ndvi_b = nd.NDVI_B(nir, nir, False, False)
crops_data_b = crops_ndvi_b.convert()



#responsible for saving and loading images.
#calculate the wavelength for additional of greeness.
merge_image = nd.merge(nir, nir, False, False)
resultImage = merge_image.convert()
wavelengh = merge_image.run_wavelength()

img1 = cv2.imread('image_nir.png')
img2 = cv2.imread(args['image'])

img1 = imutils.resize(img1, width = 250)
img2 = imutils.resize(img2, width = 250)

h_img = cv2.hconcat([img1, img2])

cv2.imwrite('combined_raw.png', h_img)

image = cv2.imread("NDVI.jpg")

crops_image_a = image[u.xx:(u.xx+u.xy), (u.yx):(u.yx+u.yy)]
crops_image_b = image[u.xx1:(u.xx1+u.xy1), (u.yx1):(u.yx1+u.yy1)]



dataw = {
        'leaf_color_a': crops_data_a['greeness'], 
        'leaf_color_b': crops_data_b['greeness'], 
        'leaf_count_a:': crops_data_a['leaf'], 
        'leaf_count_b:': crops_data_b['leaf'],
        'canopy_a' : crops_data_a['canopy'], 
        'canopy_b' : crops_data_b['canopy']
         }

#uncomment if you will run via terminal only
#cv2.imshow("Image Identified", crops_image_a)
#cv2.imshow("Image Identified", crops_image_b)
# cv2.imshow("Image Preprocessed", 'combined_raw.png')
# cv2.imshow("Image Greeness", 'NDVI.jpg')
# cv2.imshow("Image Wavelength", 'WavelengthColors.jpg')

crops_identified_a = cv2.imwrite('crops_image_a.png', crops_image_a)
crops_identified_b = cv2.imwrite('crops_image_b.png', crops_image_b)

# cv2.waitKey(0)
print(json.dumps(dataw))
