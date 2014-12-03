# -*- coding: utf-8 -*-
"""
Reads in the images from Fez screenshots and extracts the alphabet glyphs.

@author: Stephen Granade <stephen@granades.com>
"""

from __future__ import division
from __future__ import print_function

from collections import Counter

import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import skimage.feature

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
        if titlestr:
            ax[ix].set_title(titlestr)
        ax[ix].set_axis_off()
    plt.tight_layout()
    return fig, ax
    
def apply_cmap_to_grey_image(img, cmap_class):
    """Applies a color map to an image. Pass it a numpy image and a colormap 
    class (like matplotlib.cm.jet). The image will be scaled to maximize the
    range of the colormap.
    
    Args
    ----
    img : ndarray
        The image to apply the colormap to.
    cmap_class: Colormap object
    
    Returns
    -------
    The image with the colormap applied to it.
    """
        
    img_range = img.max() - img.min()
    scaled_img = (img - img.min()) / img_range
    return np.uint8(cmap_class(scaled_img)*255)

	
#####
# LOAD THE IMAGE

FILENAME = 'fez-image.jpg'

glyphimg = misc.imread(FILENAME)

plt.imshow(glyphimg)
plt.axis('off')
plt.tight_layout()

## Read in the image in YCbCr (luminance/chroma blue/chroma red) color space
glyphimg_ycbcr = ndimage.imread(FILENAME, mode='YCbCr')
lum = glyphimg_ycbcr[:,:,0]
chrom = glyphimg_ycbcr[:,:,1] / 2 + glyphimg_ycbcr[:,:,2] / 2
plt.close('all')
fig, ax = plt.subplots(2, 2)
ax[0,0].imshow(lum, cmap='gray')
ax[0,0].set_title('Luminance (gray)')
ax[0,1].imshow(lum)
ax[0,1].set_title('Luminance (heat map)')
ax[1,0].imshow(chrom, cmap='gray')
ax[1,0].set_title('Chrominance (gray)')
ax[1,1].imshow(chrom)
ax[1,1].set_title('Chrominance (heat map)')
ax[0,0].set_axis_off()
ax[0,1].set_axis_off()
ax[1,0].set_axis_off()
ax[1,1].set_axis_off()
plt.draw()


#####
# CREATE IMAGE MASKS

# The glyph graphics are an 11x11 square of just black and white surrounded
# by a black square. That means we're looking for areas with both high luminance
# (brightness) and low chromaticity (color).

lum_mask = lum > .94*np.max(lum)
# Low color corresponds to a chrominance of around 128 (halfway between 0 & 255)
chrom_mask = np.logical_and(chrom >= 126, chrom <= 136)

plt.close('all')
compare_images([glyphimg, lum_mask, chrom_mask], 
               title='Luminance and Chrominance Masks',
               subtitles=['Original', 'Luminance Mask', 'Chrominance Mask'])

# Let's identify the part of the image we care about
image_mask = np.logical_and(lum_mask, chrom_mask)

plt.close('all')
compare_images([glyphimg, image_mask], title='Image Mask',
               subtitles=['Original', 'Image Mask'])


#####
# FIND GLYPH CORNERS

# Find the corners of the squares in the image_mask so that we can extract
# just the squares

import skimage.feature as skf

# The corner detector needs an actual image to work with, not a boolean array
image_mask_uint = np.zeros(image_mask.shape, dtype='uint8')
image_mask_uint[image_mask] = 1

corners = skf.corner_peaks(skf.corner_harris(image_mask_uint, k=0.01, sigma=.2),
                           min_distance=8).T

plt.close('all')
plt.imshow(image_mask, cmap='gray')
plt.title('Corners')
plt.scatter(corners[1], corners[0], c='r', marker='+')
plt.tight_layout()

# That...doesn't really work that well. Crap. Okay, let's try a different
# approach.


#####
# FIND GLYPH CORNERS (TAKE 2)

# Since we know what we're looking for, let's brute force find the glyphs

# Find the border around the glyphs (low luminance, low chromaticity)
border_mask = np.logical_and(lum < 20, chrom_mask)
plt.close('all')
compare_images([glyphimg, image_mask, border_mask], title='Masks Pt 2',
               subtitles=['Original', 'Image Mask', 'Border Mask'])

# Routines to find all of the top left corners in an image
def mask_is(mask, i, j):
    """Return the true/false value of a Boolean mask image at index [i, j].
    
    If the [i, j] index is out of bounds, returns True.
    """
    try:
        return mask[i, j]
    except IndexError:
        return True
        
def is_top_left_corner(image, border, i, j):
    """Determine if image[i, j] is a top left corner that we're looking for.
    
    Args
    ----
    image : array
        A Boolean array indicating where the image pixels are.
    border : array
        A Boolean array indicating where the border around the image is.
    
    Returns
    -------
    True if image[i, j] is a top left corner, False otherwise.
    """
    return (image[i, j] & 
        mask_is(image,  i  , j+1) &
        mask_is(image,  i+1, j+1) &
        mask_is(image,  i+1, j  ) &
        mask_is(border, i+1, j-1) &
        mask_is(border, i  , j-1) &
        mask_is(border, i-1, j-1) &
        mask_is(border, i-1, j) &
        mask_is(border, i-1, j+1) &
        (not mask_is(image, i-2, j)) &
        (not mask_is(image, i, j-2)))

def find_top_left_corners(image, border):
    """Returns the top left corners in an image (i.e. surrounded by border)."""
    top_corners = []
    for x in range(image.shape[1]):
        y = 0
        while y < image.shape[0]:
            if not image[y, x]:
                y += 1
                continue
            if is_top_left_corner(image, border, y, x):
                top_corners.append([y, x])
            # Skip over contiguous image pixels
            y2 = y + 1
            while y2 < image.shape[0]:
                if not image[y2, x]:
                    break
                y2 += 1
            y = y2
    return np.array(top_corners).T

# Find the convex hull object to help find squares
# Opening/closing won't work because the gaps inside the glyphs are roughly
# the same size as the space between columns of glyphs
hull = skimage.morphology.convex_hull_object(image_mask)

plt.close('all')
plt.imshow(hull, cmap='gray')
plt.title('Convex Hull')
plt.axis('off')
plt.tight_layout();

top_left_corners = find_top_left_corners(hull, border_mask)

plt.close('all')
plt.imshow(image_mask, cmap='gray')
plt.title('Corners')
plt.scatter(top_left_corners[1], top_left_corners[0], c='g', marker='o')
plt.axis('off')
plt.tight_layout()


#####
# FIND COLUMNS

# Find the likely columns by counting occurrences and find the most-common
# spacing between adjacent columns
# (I'm using a Counter: a dict that stores elements as keys and the number of
# times that element occurs as the values)
columns = Counter(top_left_corners[1])
avg_column_count = np.mean(list(columns.values()))
likely_columns = sorted([k for k in columns
        if columns[k] > avg_column_count])
col_delta, _ = Counter([i - j for i, j 
        in zip(likely_columns[1:], likely_columns[:-1])]).most_common(1)[0]
# I happen to know that the spacing between rows is 1 more than that between
# columns, & there aren't enough columns to make the statistical analysis
# I did above work for rows.
# (I also know that the column spacing is the glyph size (11 pixels) + 2,
# but I wanted to show how I'd do this kind of analysis)
row_delta = col_delta + 1


#####
# EXTRACT GLYPHS

def extract_glyph_column(im, border, i_0, j_0, row_delta, col_delta):
    """Extract all the glyphs in a column.
    
    Args
    ----
    im : ndarray
        The Boolean image containing the glyphs.
    border : ndarray
        The Boolean image containing the border around the glyphs.
    i_0 : integer
        The row index at which the column begins.
    j_0 : integer
        The column index at which the column begins.
    row_delta : integer
        The spacing between rows in the column.
    col_delta : integer
        The spacing between columns.
    
    Returns
    -------
    A list containing all of the glyphs in the column, grouped into
    words. The returned structure a list of list of glyphs:
    [ [glyph, glyph, glyph], [glyph, glyph] ]
    """
    glyphcol = []
    i = i_0
    word = []
    while mask_is(im, i, j_0):
        word.append(im[i:i+row_delta, j_0:j_0+col_delta])
        i += row_delta
        # Sometimes you get one blank space in a column between words.
        # When that happens, make a new word
        if (mask_is(border, i, j_0) and mask_is(im, i+row_delta, j_0) and
                i + row_delta < im.shape[0]):
            glyphcol.append(word)
            word = []
            i += row_delta
    glyphcol.append(word)
    return glyphcol

# Starting from the top-left-most corner, process the glyphs as columns

# Find the index into top_left_corners where the corner in the first
# likely column is
colindex = np.where(top_left_corners[1] == likely_columns[0])[0][0]
i_start, j_start = top_left_corners[:, colindex]
glyphs = []
j = j_start # For the first trip through the loop
while mask_is(image_mask, i_start, j):
    # Process a column
    glyphs.append(extract_glyph_column(image_mask, border_mask,
                                       i_start, j,
                                       row_delta, col_delta))
    j += col_delta

    
#####
# CHECK THAT GLYPHS WERE EXTRACTED PROPERLY

def recombine_glyphs(glyphs):
    """Combines a list of words containing glyphs into one square ndarray."""
    blank_space = np.zeros(glyphs[0][0][0].shape, dtype='bool')
    # Stack all of the words in a column into a single column array
    col_list = []
    for i, col in enumerate(glyphs):
        col_stack = np.zeros((0, blank_space.shape[1]), dtype='bool')
        for j, word in enumerate(col):
            col_stack = np.append(col_stack, np.vstack(word), axis=0)
            if j < len(col) - 1:
                col_stack = np.append(col_stack, blank_space, axis=0)
        col_list.append(col_stack)
    width = sum([col.shape[1] for col in col_list])
    height = max([col.shape[0] for col in col_list])
    all_glyphs = np.zeros((height, width))
    j = 0
    for i, col in enumerate(col_list):
        all_glyphs[0:col.shape[0], j:j+col.shape[1]] = col
        j += col.shape[1]
    return all_glyphs
    
plt.close('all')
plt.imshow(recombine_glyphs(glyphs))
plt.axis('off')

