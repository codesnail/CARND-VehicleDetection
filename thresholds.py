# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import Camera

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = 1 if orient=='x' else 0
    y = 1 if orient=='y' else 0
    sobel = cv2.Sobel(gray, cv2.CV_64F, x, y, ksize=sobel_kernel)
    abs_sobel = np.abs(sobel)
    scaled_grad = np.uint8(255 * abs_sobel/np.max(abs_sobel))
    # Apply threshold
    grad_binary = np.zeros_like(scaled_grad)
    grad_binary[(scaled_grad>thresh[0]) & (scaled_grad<thresh[1])] = 1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_mag = np.uint8(255 * mag/np.max(mag))
    # Apply threshold
    mag_binary = np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag>mag_thresh[0]) & (scaled_mag<mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    orient = np.arctan2(sobely, sobelx)
    # Apply threshold
    dir_binary = np.zeros_like(orient)
    dir_binary[(orient>thresh[0]) & (orient<thresh[1])] = 1
    return dir_binary

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, h_thresh=(0,255), l_thresh=(0,255), s_thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]
    #print(h[600:720,250:300])
    binary_output = np.zeros_like(s)
    binary_output[((h>h_thresh[0]) & (h<=h_thresh[1])) &
                  ((l>l_thresh[0]) & (l<=l_thresh[1])) &
                  ((s>s_thresh[0]) & (s<=s_thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def thresh_pipeline(img, h_thresh=(0,255), l_thresh=(0,255), s_thresh=(190, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    #hsv= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #h_channel = hls[:,:,0]
    #s_channel = hls[:,:,1]
    #v_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    '''
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    '''
    
    # Threshold color channel
    combined_binary = np.zeros_like(s_channel)
    combined_binary[((h>h_thresh[0]) & (h<=h_thresh[1])) &
                    ((l>l_thresh[0]) & (l<=l_thresh[1])) &
                    (((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) |
                    ((scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])))] = 1


    return combined_binary # color_binary

'''
# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements
image = mpimg.imread('test_images/challenge_test02.jpg')
# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 150))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(50, 150))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(-np.pi/4, np.pi/4))
#hls_binary = thresh_pipeline(image, h_thresh=(10, 50), l_thresh=(80,100), s_thresh=(0,15), sx_thresh=(1,100))
hls_binary = hls_select(image, h_thresh=(10, 50), l_thresh=(86,130), s_thresh=(10,120))
hls_binary_white = hls_select(image, h_thresh=(160, 180), l_thresh=(130,180), s_thresh=(3,10))


binary_out = np.zeros_like(hls_binary)
binary_out[(hls_binary==1)] = 1 # | 
        #((image[:,:,0]>180) & (image[:,:,0]<=255)) |
        #((image[:,:,1]>180) & (image[:,:,1]<=255)) |
        #((image[:,:,2]>180) & (image[:,:,2]<=255))] = 1

# Plot the result
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 12))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('hls', fontsize=20)
ax3.imshow(gradx, cmap='gray')
ax3.set_title('gradx', fontsize=20)
ax4.imshow(hls_binary_white, cmap='gray')
ax4.set_title('hls white', fontsize=20)
ax5.imshow(binary_out, cmap='gray')
ax5.set_title('combined hls', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
'''