# -*- coding: utf-8 -*-
"""
Examples from the "Image Processing with Python" presentation.

@author Stephen Granade <stephen@granades.com>
"""

from __future__ import division
from __future__ import print_function

#####
# IMAGE PROCESSING WITH SCIPY

# I'm often going to show images side-by-side, so here's a helper
# function to do that
def compare_images(imgs, title=None, subtitles=None, cmaps=None):
    """Plots multiple images side by side for comparison.
    
    Args
    ----
    imgs : sequence of ndarrays
        The images to be plotted.
    title : string
        The overall plot's title, if any.
    subtitles : sequence of strings
        Titles for the sub-plots, if any.
    cmaps : sequence of color maps
        The color maps to use with the sub-plots, if any.
        If None, then all sub-plots default to grey.
    """
    fig, ax = plt.subplots(1, len(imgs))
    if title:
        plt.suptitle(title)
    for ix, img in enumerate(imgs):
        cmapstr = 'gray'
        titlestr = None
        try:
            if cmaps:
                cmapstr = cmaps[ix]
        except:
            pass
        try:
            if subtitles:
                titlestr = subtitles[ix]
        except:
            pass
        ax[ix].imshow(img, cmap=cmapstr)
        ax[ix].set_axis_off()
        if titlestr:
            ax[ix].set_title(titlestr)
    plt.tight_layout()
    return fig, ax


#####
# SIMPLE IMAGE LOADING AND DISPLAYING

import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt

img = misc.imread('Schroedinger.jpg')

plt.imshow(img)

# Why's it all rainbow like that? Because SCIENCE. We can fix that, though
plt.close('all')
plt.imshow(img, cmap='gray')

# You can also specify a color map object to use instead of a string
# For matplotlib color maps, see http://matplotlib.org/examples/color/colormaps_reference.html
plt.close('all')
plt.imshow(img, cmap=plt.cm.gray)

# Save with imsave, which lets you auto-specify the file type by extension
misc.imsave('Schroedinger-2.png', img)
# The PPT has more information about the Python Imaging Libray (PIL) that
# scipy uses for image reading and saving.
# To set image-type-specific options, convert an ndarray image to a PIL
# image object and use its save() function
pil_img = misc.toimage(img) # Gets a PIL Image
pil_img.save('Schroedinger-3.jpg', quality=30)

# You can adjust the image's luminance contrast when you show it
plt.close('all')
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img, cmap='gray', vmin=30, vmax=150)

# What if you wanted to adjust it dynamically? A library called guiqwt can help
# See https://code.google.com/p/guiqwt/
import guidata
import guiqwt
import guiqwt.image
import guiqwt.plot

_app = guidata.qapplication() # Required to start up guidata
imageitem = guiqwt.builder.make.image(img, colormap='gray')
win = guiqwt.plot.ImageDialog(edit=False, toolbar=True, wintitle="Contrast",
                              options=dict(show_contrast=True))
plot = win.get_plot()
plot.add_item(imageitem)
win.show()
win.exec_()
# Click on the plotted image to use the contrast enhancement panel


#####
# CROPPING, SCALING, ROTATING, FLIPPING

# Crop using standard array slicing notation
imgy, imgx = img.shape
cropped_img = img[0:imgy //2, :]
plt.close('all')
plt.imshow(cropped_img, cmap='gray')
# Note that, with Matplotlib's default "jet" rainbow color map, you can see things 
# like JPEG artifacts more easily
plt.close('all')
compare_images([cropped_img, cropped_img], 
               title="Check Out Those JPEG Artifacts",
               cmaps=['gray', 'jet'])

# To scale, use the imresize() function from scipy.misc
resized_img = misc.imresize(img, 0.30)
plt.close('all')
compare_images([img, resized_img], title='Float-Resized Image',
               subtitles=['Original', 'Resized'])
# imresize takes the size as a float (fraction), integer (percent), or
# tuple (final size)...or so it says. As far as I can tell, integer
# scaling is broken
resized_img = misc.imresize(img, 10)
# Tuples work, though
resized_img = misc.imresize(img, (img.shape[0] // 2, img.shape[1]))
plt.close('all')
compare_images([img, resized_img], title='Tuple-Resized Image')
# You can also define the interpolation method:
#  interp='nearest'  (Preserves hard images, so jaggy)
#  interp='bilinear'
#  interp='bicubic' (Good for smooth gradients)
#  interp='cubic'

# To rotate, use the rotate() function from scipy.ndimage
rotated_img = ndimage.rotate(img, 30)
plt.close('all')
compare_images([img, rotated_img], title='Rotated Image')
# By default rotate will make the array big enough that nothing gets cut
# off. You can change that, of course
cropped_rotated_img = ndimage.rotate(img, 30, reshape=False)
plt.close('all')
fig, ax = compare_images([rotated_img, cropped_rotated_img],
                         title='Reshaped & Non-Reshaped Rotation',
                         subtitles=['Rotation w/Reshaping', 
                         'Rotation w/o Reshaping'])
# Since the two graphs use different scales, re-scale the second, smaller
# plot to match the first
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_ylim(ax[0].get_ylim())
plt.draw()

# To flip, use the standard NumPy functions flipud() and fliprl()
flipped_img = np.flipud(img)
plt.close('all')
compare_images([img, flipped_img], title='Flipped Image')


#####
# FILTERING

# The presentation explains what filtering is, and why you might want to use a filter

# scipy.ndimage includes several common filters. For example, Gaussian 
# filters, which soften and blur images, are in scipy.ndimage
blurred_img = ndimage.gaussian_filter(img, sigma=1)
plt.close('all')
compare_images([img, blurred_img], title='Blurred Image',
               subtitles=['Original', "Gaussian Blurred $\sigma$=1"])
# The larger the Gaussian's sigma, the more it blurs
more_blurred_img = ndimage.gaussian_filter(img, sigma=3)
plt.close('all')
compare_images([img, blurred_img, more_blurred_img], title='Comparing Blurring',
               subtitles=['Original', 'Gaussian $\sigma$=1',
               'Gaussian $\sigma$=3'])

# What if you have a noisy image?
cropped_img = img[50:140, 90:180]
noisy_img = cropped_img + (cropped_img.std()*np.random.random(cropped_img.shape) - 
        (cropped_img.std()/2)*np.ones(cropped_img.shape))
plt.close('all')
compare_images([cropped_img, noisy_img], title="Noisy Image")

# You can use a Gaussian filter to de-noise the image
denoised_img = ndimage.gaussian_filter(noisy_img, sigma=1)
plt.close('all')
compare_images([cropped_img, noisy_img, denoised_img], 
               title="Gaussian Denoising",
               subtitles=['Original', 'Noisy', 'Denoised'])

# Or you can use a median filter to better preserve edges
median_denoised_img = ndimage.median_filter(noisy_img, 3)
plt.close('all')
compare_images([noisy_img, denoised_img, median_denoised_img],
               title="Gaussian vs Median Denoising",
               subtitles=['Noisy', 'Gaussian', 'Median'])


#####
# READING IMAGES INTO DIFFERENT COLOR SPACES

# You can read in color images
color_img = ndimage.imread('Commodore-Grace-Hopper.jpg')
plt.close('all')
plt.imshow(color_img)
plt.title("Color Image")
print("The color image's dimensions are %s" % str(color_img.shape))

# You can read in a color image as greyscale

grey_img = ndimage.imread('Commodore-Grace-Hopper.jpg', flatten=True)
plt.close('all')
plt.imshow(grey_img, cmap='gray')
plt.title("Color Image Read In Greyscale")
print("The dimensions of the color image read in as greyscale are %s" % 
    str(grey_img.shape))
    
# By default, color images are read in using the RGB color space
# but you can change that
ycbcr_img = ndimage.imread('Commodore-Grace-Hopper.jpg', mode='YCbCr')
plt.close('all')
plt.imshow(ycbcr_img)
plt.title("Color Image Read In in YCbCr")
print("The YCbCr image's dimensions are %s" % str(ycbcr_img.shape))

# I'm not actually using these color maps, but I'm leaving them in as
# an example of how to make a gradient color map
import matplotlib.colors
cb_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'cb_cmap', ['yellow', 'blue'])
cr_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'cr_cmap', ['yellow', 'red'])
# Create RGB representations of each individual channel
ychannel = ycbcr_img.copy();
ychannel[:,:,1] = ychannel[:,:,0]
ychannel[:,:,2] = ychannel[:,:,0]
cbchannel = ycbcr_img.copy();
cbchannel[:,:,0] = 128
cbchannel[:,:,2] = 128
crchannel = ycbcr_img.copy();
crchannel[:,:,0] = 128
crchannel[:,:,1] = 128
plt.close('all')
compare_images([color_img, ychannel, cbchannel, crchannel], 
                title="YCbCr channels",
                subtitles=['Color', 'Luminance (Y)', 'Chroma Blue (Cb)',
                           'Chroma Red (Cr)'])


#####
# IMAGE PROCESSING WITH SCIKIT-IMAGE

# The PPT lists a number of SciKits that may be of interest. Here we're going
# to work with scikit-image, or skimage.

import skimage
import skimage.color
import skimage.io


#####
# LOCAL FILTERING

# skimage has a lot of local filters available, like Sobel
import skimage.filter

img = misc.imread('Schroedinger.jpg')

schro_vsobel = skimage.filter.vsobel(img)
plt.close('all')
compare_images([img, schro_vsobel], title="Vertical Sobel Edge Detection",
               cmaps=['gray', None])

print(schro_vsobel.dtype)
# For some of the processing skimage does, it converts images to floating
# point, scaled from [-1, 1]. *NOT* [0, 1]

# There are other edge-detecting local transforms as well
schro_vprewitt = skimage.filter.vprewitt(img)
schro_vscharr = skimage.filter.vscharr(img)
plt.close('all')
compare_images([schro_vsobel, schro_vprewitt, schro_vscharr],
               title="Sobel, Prewitt, and Scharr Edge Detection",
               cmaps=[None, None, None],
               subtitles=['Sobel', 'Prewitt', 'Scharr'])

# Remember my noise reduction example earlier? skimage has better routines
cropped_img = img[50:140, 90:180]
noisy_img = cropped_img + (
        (cropped_img.std()*np.random.random(cropped_img.shape) - 
        (cropped_img.std()/2)*np.ones(cropped_img.shape)))
median_denoised_img = ndimage.median_filter(noisy_img, 3)
total_var_denoised_img = skimage.filter.denoise_tv_chambolle(noisy_img,
                                                             weight=30)
plt.close('all')
compare_images([cropped_img, noisy_img, median_denoised_img,
                total_var_denoised_img],
                title="Denoised Image",
                subtitles=['Original', 'Noisy', 'Median', 'Total Variation'])

# What if you want to create a binary image using a threshold?
sudoku = ndimage.imread('sudoku.jpg', flatten=True)
# We could do a simple global threshold using a blindly-chosen threshold
sudoku_global_thresh = sudoku >= 128
# or use a better method to find that threshold
otsu_thresh = skimage.filter.threshold_otsu(sudoku)
sudoku_otsu_thresh = sudoku >= otsu_thresh
# but skimage has an adaptive threshold function
sudoku_adaptive_thresh = skimage.filter.threshold_adaptive(sudoku, 
                                                           block_size=91,
                                                           offset=2)
plt.close('all')
compare_images([sudoku, sudoku_global_thresh, sudoku_otsu_thresh,
                sudoku_adaptive_thresh],
                title="Global, Otsu's Method, and Adaptive Thresholding",
                subtitles=['Original', 'Global Threshold', "Otsu's Method",
                           'Adaptive Threshold'])


#####
# ADJUSTING EXPOSURE

import skimage.exposure

# Using skimage, we can perform contrast enhancement automatically by
# equalizing the picture's histogram. The presentation has more information
# on histogram equalization

# Because of the flatten operation, "sudoku" is of type float
print("Sudoku is of type %s, with max value %f and min value of %f" %
    (sudoku.dtype.name, np.max(sudoku), np.min(sudoku)))
# but it's not scaled from [-1, 1] like skimage wants. Fix that!
sudoku_scaled = (sudoku - 127.5)/256
sudoku_equalized = skimage.exposure.equalize_hist(sudoku_scaled)
plt.close('all')
compare_images([sudoku_scaled, sudoku_equalized], title="Equalizing Exposure",
               subtitles=['Original', 'Equalized'])


#####
# MORPHOLOGICAL OPERATIONS

import skimage.morphology as mo

# To learn about morphological image processing like erosion, dilation,
# opening and closing, see 
# https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm

# Using mode='L' to read in greyscale prevents us from getting an array
# of floats back
squares = ndimage.imread('squares.png', mode='L')

# Erosion eats away at bright areas
squares_eroded = mo.erosion(squares, mo.square(3))
squares_diff = squares - squares_eroded
plt.close('all')
compare_images([squares, squares_eroded, squares_diff],
               title="Morphological Erosion",
               subtitles=['Original', 'Eroded', 'Difference'])

# Dilation expands bright areas
squares_dilated = mo.dilation(squares, mo.square(3))
squares_diff = squares_dilated - squares
plt.close('all')
compare_images([squares, squares_dilated, squares_diff],
               title="Morphological Dilation",
               subtitles=['Original', 'Dilated', 'Difference'])

# Opening erodes and then dilates, opening up dark gaps between features
squares_opened = mo.opening(squares, mo.square(3))
squares_diff = squares - squares_opened
plt.close('all')
compare_images([squares, squares_opened, squares_diff],
               title="Morphological Opening",
               subtitles=['Original', 'Opened', 'Difference'])

# Closing dilates and then erodes, filling in small dark gaps between features
squares_closed = mo.closing(squares, mo.square(3))
squares_diff = squares_closed - squares
plt.close('all')
compare_images([squares, squares_closed, squares_diff],
               title="Morphological Closing",
               subtitles=['Original', 'Closed', 'Difference'])


#####
# PARAMETRIC TRANSFORMATIONS

# Parametric transformations use matrices to describe translations,
# rotations, scaling, skew, and more. For more information, see
# http://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm

# Early in the class, we rotated using scipy.ndimage.rotate and scaled
# using scipy.misc.imresize (see above)
# Alternatively: you can use the skikit-image routines, which have the
# advantage of being faster and also accepting transformation matrices

import skimage.transform as tf

img = misc.imread('Schroedinger.jpg')

# skimage.transform includes rotation
rotated_img = tf.rotate(img, 30)
uncropped_rotated_img = tf.rotate(img, 30, resize=True)
plt.close('all')
compare_images([img, rotated_img, uncropped_rotated_img],
               title='Unresized & Resized scikit-image Rotation',
               subtitles=['Original', 'Rotation w/o Resizing', 
                          'Rotation w/Resizing'])
# Note that this is opposite of how scipy.ndimage.rotate() works: by
# default, the image isn't resized (or reshaped, to use ndimage.rotate()'s
# language

# skimage.transform also includes rescale()
rescaled_img = tf.rescale(img, scale=.30)
plt.close('all')
compare_images([img, rescaled_img], title='Float-Rescaled Image',
               subtitles=['Original', 'Resized'])

# skimage.transform.rescale() will let you pass a tuple to scale it
# by different percentages in each direction
rescaled_img = tf.rescale(img, scale=(.3, .5))
plt.close('all')
compare_images([img, rescaled_img], title='Tuple-Rescaled Image')

# If you want to specify the final shape of it, use
# skimage.transform.resize()
resized_img = tf.resize(img, (img.shape[0] // 2, img.shape[1]))
plt.close('all')
compare_images([img, resized_img], title='Resized Image')
# You can define the interpolation method with the "order" parameter:
#  order = 0 (nearest neighbor; preserves hard images, so jaggy)
#  order = 1 (bilinear, the default)
#  order = 2 (biquadratic)
#  order = 3 (bicubic, good for smooth gradients)
#  order = 4 (biquartic)
#  order = 5 (biquintic)

# skimage.transform includes several transformations as classes
# The SimilarityTransform is for translation, rotation, and scale
shiftright = tf.SimilarityTransform(translation=(-20, 0))
plt.close('all')
compare_images([img, tf.warp(img, shiftright)],
                title='Translation with scikit-image',
                subtitles=['Original', 'Translated'])

rotccw = tf.SimilarityTransform(rotation=np.pi / 4)
plt.close('all')
compare_images([img, tf.warp(img, rotccw)],
                title='Rotation with scikit-image',
                subtitles=['Original', 'Rotated'])

upsize = tf.SimilarityTransform(scale=0.9)
plt.close('all')
compare_images([img, tf.warp(img, upsize)],
                title='Scaling with scikit-image',
                subtitles=['Original', 'Scaled'])

# AffineTransformation adds shearing, along with translation, rotation,
# and scale
skewhoriz = tf.AffineTransform(shear=np.pi/4)
skewvert = tf.AffineTransform(matrix=skewhoriz._matrix.T)
plt.close('all')
compare_images([img,
                tf.warp(img, skewhoriz, 
                        output_shape=(img.shape[0], img.shape[1] * 2)),
                tf.warp(img, skewvert,
                        output_shape=(img.shape[0] * 2, img.shape[1]))],
                title='Affine Skew with scikit-image',
                subtitles=['Original', 'Skewed Horizontal',
                           'Skewed Vertical'])


#####
# LABELING REGIONS

# Let's generate some blobs to work with
points = np.zeros((256, 256))
num_pts = 20
point_array = (256*np.random.random((2, num_pts**2))).astype(np.int)
points[(point_array[0]), (point_array[1])] = 1
blurred_points = ndimage.gaussian_filter(points, sigma=256/(4.*num_pts))
blobs = blurred_points > np.mean(blurred_points)
plt.close('all')
compare_images([points, blurred_points, blobs], title='Generating Blobs',
               subtitles=['Points', 'Blurred Points', 'Thresholded Blobs'])

# Label the connected regions
labels = skimage.morphology.label(blobs)
plt.close('all')
compare_images([blobs, labels], title="Blobs and Their Labels",
               cmaps=['gray', 'jet'])


#####
# FEATURE MATCHING

import skimage.transform as tf

from skimage.feature import (match_descriptors, ORB, plot_matches)

schroedinger = misc.imread('Schroedinger.jpg')
# Transform the image using the skimage.transform library
# "rotate" does what you might expect
schroedinger_rotate = tf.rotate(schroedinger, 180)
# This sets up a transformation that changes the image's scale, rotates it,
# and moves it. "warp" then applies that transformation to the image
tform = tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
                           translation=(0, -200))
schroedinger_warped = tf.warp(schroedinger, tform)

# ORB is an algorithm that detects good features in an image and then
# describes them in a compact way. The descriptions can then be matched
# across multiple images.
descriptor_extractor = ORB(n_keypoints=200)

# Apply the ORB algorithm to our images
descriptor_extractor.detect_and_extract(schroedinger)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(schroedinger_rotate)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(schroedinger_warped)
keypoints3 = descriptor_extractor.keypoints
descriptors3 = descriptor_extractor.descriptors

# See which descriptors match across the images
matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=1)

plot_matches(ax[0], schroedinger, schroedinger_warped, keypoints1, keypoints2,
             matches12)
ax[0].axis('off')

plot_matches(ax[1], schroedinger, schroedinger_warped, keypoints1, keypoints3, 
             matches13)
ax[1].axis('off')

plt.show()

plt.gray()
