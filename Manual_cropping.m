%MANUAL CROPPING

clear all;close all; clc;

im=imread('DJI_0052.JPG');
imshow(im);
crop1=imcrop;
imwrite(crop1, 'Coffee_1_2.JPG');