clc
clear all
close all
warning off
RGB = imread('CoffeeSeedling_001.jpg');
subplot(1,3,1);
imshow(RGB);
title('Original Image');
%%
[BW,maskedImage] = segmentImage0001(RGB);
subplot(1,3,2);
imshow(BW);
title('Segmennted Binary Image');
subplot(1,3,3);
imshow(maskedImage);
title('Segmented Color Image');
imwrite(BW,['CoffeeSeedling_L001.jpg']);
numberOfTruePixels = sum(BW(:));
%% Measuring Length in Pixel
A=imread('CoffeeSeedling_L001.jpg');
imshow(A);
%For Measuring
h=imdistline(gca);
api=iptgetapi(h);