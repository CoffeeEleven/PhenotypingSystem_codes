import cv2
import numpy as np

def convert_nir(image):
    # read image
    img = cv2.imread(image)

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # make color channels
    red = gray.copy()
    green = gray.copy()
    blue = gray.copy()

    # set weights
    R = .642
    G = .532
    B = .44

    MWIR = 4.5

    # get sum of weights and normalize them by the sum
    R = R**4
    G = G**4
    B = B**4
    sum = R + G + B
    R = R/sum
    G = G/sum
    B = B/sum

    # combine channels with weights
    red = (R*red)
    green = (G*green)
    blue = (B*blue)
    result = cv2.merge([red,green,blue])

    # scale by ratio of 255/max to increase to fully dynamic range
    max=np.amax(result)
    result = ((255/max)*result).clip(0,255).astype(np.uint8)

    # write result to disk
    cv2.imwrite("image_nir.png", gray)

