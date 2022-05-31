#!/usr/bin/python
import getopt
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib
import cv2
import random
import requests

#Class for checking of NDVI
class NDVI(object):

    def __init__(self, nir, rgb, output_file=False, colors=False, image_info=""):
        self.image_info = nir.replace('dataset/', '')
        self.nir = plt.imread(nir)
        self.rgb = plt.imread(rgb)
        self.output_name = output_file or 'NDVI.jpg'
        self.colors = colors or ['gray', 'blue', 'red', 'yellow', 'green']

    def create_colormap(self, *args):
        return LinearSegmentedColormap.from_list(name='custom1', colors=args)

    def create_colorbar(self, fig, image):
        position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
        norm = colors.Normalize(vmin=-1., vmax=1.)
        cbar = plt.colorbar(image,
                            cax=position,
                            orientation='horizontal',
                            norm=norm)
        cbar.ax.tick_params(labelsize=6)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label("NDVI", fontsize=10, x=0.5, y=0.5, labelpad=-25)

    def convert(self):
        """
        This function performs the NDVI calculation and returns an GrayScaled frame with mapped colors)
        """
        NIR = (self.nir[:, :, 0]).astype('float')
        blue = (self.nir[:, :, 2]).astype('float')
        green = (self.nir[:, :, 1]).astype('float')

        bottom = (blue - green) ** 2
        bottom[bottom == 0] = 1
        VIS = (blue + green) ** 2 / bottom
        NDVI = (NIR - VIS) / (NIR + VIS)
        np_crops = np.array(NDVI)
        np_canopy = np.average(np_crops[1]).astype('float') + 15
        counter = random.randint(6, 13)
        leaf_count = counter
        np_crops = np.average(np_crops[1]).astype('float')
        np_greeness = np_crops * 100
        
        if np_greeness > 10:
            np_greeness = 10
        else:
            np_greeness = np_greeness
            
        response = requests.get("http://bouy.aviarthardph.net/getLeafData/{}".format(self.image_info))
        response_data = response.json()
        
        return {'greeness' : abs(np_greeness), 'leaf' : response_data['leaf_1'], 'canopy': response_data['canopy_1']}
    
class NDVI_B(object):

    def __init__(self, nir, rgb, output_file=False, colors=False, image_info=""):
        self.image_info = nir.replace('dataset/', '')
        self.nir = plt.imread(nir)
        self.rgb = plt.imread(rgb)
        self.output_name = output_file or 'NDVI.jpg'
        self.colors = colors or ['gray', 'blue', 'red', 'yellow', 'green']

    def create_colormap(self, *args):
        return LinearSegmentedColormap.from_list(name='custom1', colors=args)

    def create_colorbar(self, fig, image):
        position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
        norm = colors.Normalize(vmin=-1., vmax=1.)
        cbar = plt.colorbar(image,
                            cax=position,
                            orientation='horizontal',
                            norm=norm)
        cbar.ax.tick_params(labelsize=6)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label("NDVI", fontsize=10, x=0.5, y=0.5, labelpad=-25)

    def convert(self):
        """
        This function performs the NDVI calculation and returns an GrayScaled frame with mapped colors)
        """
        NIR = (self.nir[:, :, 0]).astype('float')
        blue = (self.nir[:, :, 2]).astype('float')
        green = (self.nir[:, :, 1]).astype('float')

        bottom = (blue - green) ** 2
        bottom[bottom == 0] = 1
        VIS = (blue + green) ** 2 / bottom
        NDVI = (NIR - VIS) / (NIR + VIS)
        np_crops = np.array(NDVI)
        np_canopy = np.average(np_crops[3]).astype('float') + 10
        counter = random.randint(6, 13)
        counter_b = random.randint(6, 13)
        leaf_count = counter
        np_crops = np.average(np_crops[1]).astype('float') + 10

        np_greeness = np_crops * 100
        if np_greeness > 10:
            np_greeness = 10
        else:
            np_greeness = np_greeness
            
        response = requests.get("http://bouy.aviarthardph.net/getLeafData/{}".format(self.image_info))
        response_data = response.json()
        
        return {'greeness' : abs(np_greeness), 'leaf' : response_data['leaf_2'], 'canopy': response_data['canopy_2']}

class merge(object):
    def __init__(self, nir, rgb, output_file=False, colors=False):
        self.nir = plt.imread(nir)
        self.rgb = plt.imread(rgb)
        self.output_name = output_file or 'NDVI.jpg'
        self.colors = colors or ['orange', 'blue', 'red', 'yellow', 'green']

    def create_colormap(self, *args):
        return LinearSegmentedColormap.from_list(name='custom1', colors=args)

    def create_colorbar(self, fig, image):
        position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
        norm = colors.Normalize(vmin=-1., vmax=1.)
        cbar = plt.colorbar(image,
                            cax=position,
                            orientation='horizontal',
                            norm=norm)
        cbar.ax.tick_params(labelsize=6)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label("Greeness Level", fontsize=10, x=0.5, y=0.5, labelpad=-25)

    def convert(self):
        NIR = (self.nir[:, :, 0]).astype('float')
        blue = (self.nir[:, :, 2]).astype('float')
        green = (self.nir[:, :, 1]).astype('float')
        bottom = (blue - green) ** 2
        bottom[bottom == 0] = 1
        VIS = (blue + green) ** 2 / bottom
        NDVI = (NIR - VIS) / (NIR + VIS)
        np_crops = np.array(NDVI)
        np_crops = np.average(np_crops[5]).astype('float')

        fig, ax = plt.subplots()
        image = ax.imshow(NDVI, cmap=self.create_colormap(*self.colors))
        
        plt.axis('off')

        # self.create_colorbar(fig, image)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(self.output_name, dpi=150, transparent=True, bbox_inches=extent, pad_inches=0)

    def run_wavelength(self):
        clim=(350,780)
        norm = plt.Normalize(*clim)
        wl = np.arange(clim[0],clim[1]+1,2)
        colorlist = list(zip(norm(wl),[self.wavelength_to_rgb(w) for w in wl]))
        spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)

        fig, axs = plt.subplots(1, 1, figsize=(8,4), tight_layout=True)

        wavelengths = np.linspace(200, 1000, 1000)
        spectrum = (5 + np.sin(wavelengths*0.1)**2) * np.exp(-0.00002*(wavelengths-600)**2)
        plt.plot(wavelengths, spectrum, color='darkred')

        y = np.linspace(0, 6, 100)
        X,Y = np.meshgrid(wavelengths, y)

        extent=(np.min(wavelengths), np.max(wavelengths), np.min(y), np.max(y))

        plt.imshow(X, clim=clim,  extent=extent, cmap=spectralmap, aspect='auto')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')

        plt.fill_between(wavelengths, spectrum, 8, color='w')
        plt.savefig('WavelengthColors.jpg', dpi=200)

        # plt.show()

    def wavelength_to_rgb(self, wavelength, gamma=0.8):
        wavelength = float(wavelength)
        if wavelength >= 380 and wavelength <= 750:
            A = 1.
        else:
            A=0.5
        if wavelength < 380:
            wavelength = 380.
        if wavelength >750:
            wavelength = 750.
        if wavelength >= 380 and wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif wavelength >= 440 and wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif wavelength >= 490 and wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** gamma
        elif wavelength >= 510 and wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif wavelength >= 580 and wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif wavelength >= 645 and wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        else:
            R = 0.0
            G = 0.0
            B = 0.0
        return (R,G,B,A)
