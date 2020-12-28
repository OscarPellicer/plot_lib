'''
Copyright 2020 Oscar José Pellicer Valero
Universitat de València, Spain

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or 
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This is heavily inspired by https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks

TO DOs:
    With plot_alpha & plot4, regenerate a single slice for every value (not the whole volume)
    Make code in plot more modular & make use of ipywidgets layouts & implement plot_pair
    Improve documentation
    Give the option to create a buffer of figures in another thread to display quickly
'''

#~~~~~~~~~~~~ Load required libraries ~~~~~~~~~~~~#

#Mandatory: matplotlib, numpy, ipywidgets, ipython
#Optional: scipy (if plot_label_edge==True), SimpleITK (for using SimpleITK images as input)

import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.colors as mcolors
import numpy as np
from ipywidgets import interact, IntSlider, FloatSlider
from IPython.display import display

#~~~~~~~~~~~~ Global defaults ~~~~~~~~~~~~#

#Raise Exception if trying to process automatically a mask
#containing more than MAX_UNIQUE different values
MAX_UNIQUE= 100

#If slow_mode is set to 'auto', activate only if the volume of
#the image (in voxels) is above MAX_SLOW_VOLUME
MAX_SLOW_VOLUME= 10000000
    
#~~~~~~~~~~~~ Simplified plotting functions ~~~~~~~~~~~~#
    
def plot_alpha(img1, img2, spacing=(1, 1, 1), alpha=0.5, slow_mode='auto', color='r',
                intensity_normalization=True, **kwargs):
    '''
        Plots an image mixed with another given an alpha value
    '''
    #Initial preprocessing
    imgs= []
    for img in [img1, img2]:
        img, spacing= process_initial_image(img, spacing, intensity_normalization)
        imgs.append(img)
        
    if slow_mode == 'auto': slow_mode = np.prod(imgs[0].shape) > MAX_SLOW_VOLUME
            
    #Check that shapes coincide
    shapes= [img.shape for img in imgs]
    if not np.all([np.all(s == shapes[0]) for s in shapes]):
        raise RuntimeError('Shapes of all images must be the same')

    #Alpha change callback
    color= mcolors.to_rgb(color)
    def callback(alpha):
        img1= np.stack([imgs[0]]*3, axis=-1)
        img1*= (1-alpha)
        for i in range(3): 
            img1[...,i]+= imgs[1]*color[i]*alpha
                        
            #Simple color clipping
            img1[img1 > 1.] = 1. 
            img1[img1 < 0.] = 0.
            
        plot(img1, spacing=spacing, is_color=True, slow_mode=slow_mode, 
              intensity_normalization=False, **kwargs)

    #Define slider and start interaction
    s1 = FloatSlider(min=0, max=1, step=0.05, value=alpha, 
                       description='α', continuous_update=False)
    s1.style.handle_color = 'lightblue'
    interact(callback, alpha=s1)
    
def plot_multi_mask(img, mask, spacing=(1,1,1), **kwargs):
    '''
        Plots an image along with all the channels from its mask
    '''
    mask, spacing= process_initial_image(mask, spacing, False)
    mask= mask[...,np.newaxis] if len(mask.shape) == 3 else mask
    masks= []
    for c in range(mask.shape[-1]):
        ids= np.unique(mask[...,c])[1:]
        if len(ids) > MAX_UNIQUE:
            raise RuntimeError('One of the masks has too many different values')
        masks.append([mask[...,c], ids, ['default'], ['C%d ID:%d'%(c,i) for i in ids]])
    plot(img, masks=masks, spacing=spacing, **kwargs)
    

def plot_channel_alpha(img, channels=[0,1], spacing=(1, 1, 1), **kwargs):
    '''
        Plots an overlay of any two channels of an image
    '''
    img, spacing= process_initial_image(img, spacing, intensity_normalization)
    plot_alpha(img[...,channels[0]], img[...,channels[1]], spacing=spacing, **kwargs)
    
def plot4(img, ct=None, spacing=(1, 1, 1), intensity_normalization=True, **kwargs):
    '''
        Adds a second slicer to move through time / channels
    '''
    #Initial preprocessing
    img, spacing= process_initial_image(img, spacing, intensity_normalization)
    
    #Callback for time /channel slider
    def callback(ct):
        plot(img, spacing=spacing, 
             intensity_normalization=intensity_normalization, ct=ct, **kwargs)
    
    #Define slider and start interaction
    s1 = IntSlider(min=0, max=img.shape[-1]-1, step=1, 
                     value=int(img.shape[-1]/2) if ct is None else ct, 
                     description='c/t', continuous_update=False)
    s1.style.handle_color = 'lightblue'
    interact(callback, ct=s1)
    
#~~~~~~~~~~~~ Main ploting function ~~~~~~~~~~~~#

def plot(img, title=None, dpi=80, scale='auto', spacing=(1, 1, 1),
          z=None, ct=0, is_color=False, intensity_normalization=True, slow_mode='auto',
          hide_axis=False, points=[], boxes=[], masks=[], text_kwargs= {},
          plot_label_edge=True, alpha=0.2, default_colors= mcolors.TABLEAU_COLORS,
          center_crop=[], save_as=None):
    '''
    Plots a 2D, 3D or 4D image (with an ipython slider to move through z, and optionally other 
    to move through channels / time). x and y axii are shown in mm (if spacing is provided, 
    or x is a SimpleITK image), however z axis is shown as slice index.
        
    Params:
        img: Image to show. Can either be a numpy array [(z,)y,x,(c/t,)] or a SimpleITK Image
        title: Title to show along with the plot
        dpi: DPI of the plot. It scales all plot matplotlib elements (e.g.: text sizes, ax sizes)
        scale: Float or 'auto': the scale of the plot itself
        sapcing: If img is given as an array, please provide the spacing to get faithful axis values
        z: Default slice to plot at the beginnig (None for middle slice)
        ct: If the image has multiple channels (or time steps), provide which channel (time step) to plot
        is_color: Set to True if the image is RGB(A)
        secondary_slicer: Set to True to slice through time / channels
        intensity_normalization: Set to True to preprocess image intensity for better display
        hide_axis: Do not show axis coordinates
        slow_mode: If True, image only updates after releasing mouse (useful for large images)
        points: List of points [x, y, z, (marker_style), (color), (text)]. In pixel coordinates!
        boxes: List of [xmin, xmax, ymin, ymax, zmin, zmax, (color), (text), (zorder)]. In pixel coodinates!
        masks: List of masks [mask, ([id1, id2, ...]), ([col1, col2, ...]), ([text1, text2, ...])]
        plot_label_edge: If True, shows only mask's edge, else show masks as an overlay
        alpha: If plot_label_edge == False, the alpha of the masks' overlay
        default_colors: List of matplotlib-compatible colors to be used in plotting if no colors are provided
        text_kwargs: Keyword-argument dictionary to pass to all text plotting commands (plt.text(**text_kwargs))
        center_crop: Plot only the center x*y(*z) crop of the image (in mm)
        save_as: Path to save figure. None to not save figure
    '''
    #Axii x & y respect spacing. Axis z does not
    #Get a normalized numpy image from img
    nda, spacing= process_initial_image(img, spacing, intensity_normalization)

    #By default, slicer is False (assume 2D image by default)
    slicer = False
    c = nda.shape[-1]
    if nda.ndim == 3 and not c in (3,4):
        slicer = True
    elif nda.ndim == 4:
        slicer = True
        if is_color:
            if not c in (3,4):
                raise RuntimeError('Image must have 3 or 4 channels to be considered a color image')
        else:
            nda= nda[...,ct]
            
    #Crop
    crop_mask= None
    if center_crop != []:
        if len(center_crop) not in [2,3]:
            raise ValueError('Center crop parameter must have a length of 2 or 3 [x_crop_size, y_crop_size, (z_crop_size)]')
        x_offset, y_offset, z_offset= get_crop_offsets(nda, center_crop, spacing)
        nda= crop(nda, center_crop, spacing)
        
        #Normalize again!
        if intensity_normalization:
            nda= rescale_intensity(nda)
    else:
        x_offset, y_offset, z_offset= 0, 0, 0
    
    #Get axis size
    if (slicer):
        ysize, xsize= nda.shape[1], nda.shape[2]
    else:
        ysize, xsize= nda.shape[0], nda.shape[1]
        
    #Set default colors for plotting   
    default_colors= [mcolors.to_rgb(color) for color in default_colors]*10 #Make sure not to run out
        
    #Mask info that has to be plotted for every slice [x, y, z, color, text]
    masks_info= []
       
    #Combine the image (nda) with all the masks
    if masks != []:
        OC= OverlapChecker()
        if not is_color:
            nda= np.stack([nda]*3, axis=-1)
            is_color= True
            
        curr_color= 0 #Track the current default color
        for m in masks:
            #If the first element is not a list, asume that the masks have been passed directy and use defaults
            mask=    m[0] if isinstance(m, list) else m
            mask, _= process_initial_image(mask, spacing, False)
            unique_values= np.unique(mask)[1:] #Ignore BG
            
            #Use defaults if not provided [mask, ([id1, id2, ...]), ([col1, col2, ...]), ([text1, text2, ...])]
            ids=     m[1] if isinstance(m, list) and len(m) >= 2 else unique_values
            colors=  [mcolors.to_rgb(color) for color in m[2]] if isinstance(m, list) and len(m) >= 3 and m[2][0] != 'default' \
                                                               else default_colors[curr_color: curr_color + len(ids)]
            texts=   m[3] if isinstance(m, list) and len(m) >= 4 else [''] * len(ids)
            curr_color+= len(ids)
            
            #Crop?
            if center_crop != []:
                mask= crop(mask, center_crop, spacing)

            #Perform some checks
            if len(unique_values) > MAX_UNIQUE: 
                raise RuntimeError('One of the masks has too many different values')
            if (slicer and not np.all(mask.shape[:3] == nda.shape[:3])) \
               or (not slicer and not np.all(mask.shape[:2] == nda.shape[:2])):
                raise RuntimeError('One of the masks has different shape than the image (%s vs %s)'%(mask.shape[:3], nda.shape))
            if not (len(ids) == len(colors) and len(ids) == len(texts)):
                raise RuntimeError('Lists of ids, color and text must have the same length')

            #Update the image to include the mask
            for id, color, text in zip(ids, colors, texts):
                mask_id= (mask == id)
                if plot_label_edge:
                    from scipy.ndimage.morphology import binary_erosion
                    mask_id^= binary_erosion(mask_id, np.ones((1,3,3)) if slicer else np.ones((3,3))).astype(mask_id.dtype)
                    for i in range(3): 
                        nda[mask_id,i]= color[i]
                else:
                    for i in range(3): 
                        #Two options, with slighlty different results
                        nda[mask_id,i]+= color[i] * alpha
                        #nda[mask_id,i]= color[i] * alpha + nda[mask_id,i] * (1-alpha)
                        
                #Simple color clipping
                nda[nda > 1.] = 1. 
                nda[nda < 0.] = 0.
                
                #Update masks_info for plotting additional info
                if text!= '':
                    #Get min_x & min_y for every slice in z that has some data
                    masks_info+= [[*OC.check_and_fix([np.argwhere(mask_id[z].sum(axis=0)).min(), 
                                                      np.argwhere(mask_id[z].sum(axis=1)).max()]), 
                                                      z, color, text] \
                                                      for z in range(mask_id.shape[0]) if np.sum(mask_id[z]) > 0]
            
    #If image is very big, set the slow_mode to True
    if slow_mode == 'auto': slow_mode = np.prod(nda.shape) > MAX_SLOW_VOLUME

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    if scale == 'auto': scale= 350/nda.shape[1]
    figsize = ysize / dpi*scale, xsize / dpi*scale
    extent = (0, xsize*spacing[1], 0, ysize*spacing[0])
    
    #Main callback
    def plot_slice(z=None):

        # Create fig and ax
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        if not is_color: 
            plt.set_cmap("gray")
        
        # Plot
        nda_plot= nda if z is None else nda[z,...]
        ax.imshow(nda_plot, extent=extent, interpolation=None, origin='lower')
        ax.set_ylim(ax.get_ylim()[::-1])
        
        #Show title
        if title: 
            plt.title(title)
            
        #Mask info that has to be plotted for every slice [x, y, z, color, text]
        if masks_info != []:
            for (px, py, pz, color, text) in masks_info:
                if z is None or z==round(pz):
                    x_plot, y_plot= px*spacing[0], py*spacing[1]
                    plt.text(x_plot, y_plot, text, horizontalalignment='left',
                             verticalalignment='top', color=color, **text_kwargs)
            
        #Plot points if provided
        if points != []:
            for i,p in enumerate(points):
                (px, py, pz) = p[:3]
                if z is None or z==round(pz - z_offset):
                    #Use defaults if not provided [px, py, pz, (marker_style), (color), (text)]
                    marker= p[3] if len(p) >= 4 else 'o'
                    color=  p[4] if len(p) >= 5 and p[4] != 'default' else default_colors[i]
                    text=   p[5] if len(p) >= 6 else ''
                    
                    #Plot points
                    x_plot, y_plot= (px - x_offset)*spacing[0], (py - y_offset)*spacing[1]
                    plt.plot(x_plot, y_plot, marker=marker, color=color, ms=5., mfc='none')
                    plt.text(x_plot, y_plot, text, horizontalalignment='left',
                            verticalalignment='top', color=color, **text_kwargs)
        
        #Plot boxes if provided
        if boxes != []:
            for i,b in enumerate(boxes):
                (xmin, xmax, ymin, ymax, zmin, zmax) = b[:6]
                if z is None or z >= round(zmin - z_offset) and z < round(zmax - z_offset):
                    #Use defaults if not provided [xmin, xmax, ymin, ymax, zmin, zmax, (color), (text), (zorder)]
                    color=  b[6] if len(b) >= 7 and b[6]!= 'default' else default_colors[i]
                    text=   b[7] if len(b) >= 8 else ''
                    zorder= b[8] if len(b) >= 9 else 1
                    
                    #Plot boxes
                    xmin_plot, ymin_plot= (xmin - x_offset)*spacing[0], (ymin - y_offset)*spacing[1]
                    xd_plot, yd_plot= (xmax-xmin)*spacing[0], (ymax-ymin)*spacing[1]
                    rect = patches.Rectangle((xmin_plot, ymin_plot), xd_plot, yd_plot, linewidth=1, 
                                             edgecolor=color, facecolor='none', zorder=zorder)
                    ax.add_patch(rect)
                    plt.text(xmin_plot, ymin_plot + yd_plot, text, horizontalalignment='left',
                            verticalalignment='top', color=color, **text_kwargs)
                    
        #Hide axis
        if hide_axis:
            plt.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()
        
        #Save
        if save_as is not None:
            fig.savefig(save_as, pad_inches=0)
            print('Figure saved as: %s'%save_as)

    if slicer:
        s1 = IntSlider(min=0, max=nda.shape[0]-1, step=1, value= nda.shape[0]//2 if z is None else z, 
                       description='z', continuous_update= not slow_mode)
        s1.style.handle_color = 'lightblue'
        interact(plot_slice, z=s1)
    else:
        plot_slice()
        
#~~~~~~~~~~~~ Utilities ~~~~~~~~~~~~#

def crop(img, crop, spacing):
    '''
        Crops the center of the image of size `crop` (mm) for an image `img` of spacing `spacing`
    '''
    [zs, ys, xs]= img.shape[:3]
    crop_x_start, crop_y_start, crop_z_start= get_crop_offsets(img, crop, spacing)

    return img[crop_z_start : zs - crop_z_start, 
               crop_y_start : ys - crop_y_start, 
               crop_x_start : xs - crop_x_start]

def get_crop_offsets(img, crop, spacing):
    '''
        Returns the start of the offsets in pixel coordinates
    '''
    [zs, ys, xs]= img.shape[:3]
    crop_x_start= int( (xs - crop[0]/spacing[0]) / 2 )
    crop_y_start= int( (ys - crop[1]/spacing[1]) / 2 )
    crop_z_start= int( (zs - crop[2]/spacing[2]) / 2 ) if len(crop) == 3 else 0 #len == 2
    
    return crop_x_start, crop_y_start, crop_z_start
    

class OverlapChecker():
    '''
        Checks if two positions overlap, and assigns a new position if they do
        As of now, it is disabled since it breaks sometimes
    '''
    def __init__(self, threshold=5, step_size=np.array([0, 5])):
        self.positions=[]
        self.threshold= threshold
        self.step_size= step_size
        
    def check_and_fix(self, new_pos):
#         new_pos= np.array(new_pos)
#         fixed= False
#         while not fixed:
#             fixed= True
#             for pos in self.positions:
#                 if pos[0] == new_pos[0] and np.abs(np.linalg.norm(pos - new_pos)) < self.threshold:
#                     new_pos+= self.step_size
#                     fixed= False
#         self.positions.append(new_pos)
        return new_pos
        
def process_initial_image(img, spacing, normalize):
    '''
        Processes image, either numpy or SimpleITK, and returns numpy and spacing
    '''
    if not isinstance(img, (np.ndarray, np.generic) ):
        import SimpleITK as sitk
        spacing= img.GetSpacing()
        imgp= sitk.GetArrayFromImage(img).astype(np.float32)
    else:
        imgp= img.copy()
    if normalize:
        #If there are more than MAX_UNIQUE unique values, 
        #we assume it is safe to normalize (e.g.: it is not a mask)
        if len(np.unique(imgp)) > MAX_UNIQUE:
            imgp= rescale_intensity(imgp)
        else:
            pass
            #raise RuntimeError('A mask should not be normalized')
    return imgp, spacing

def rescale_intensity(image, thres=(1.0, 99.0)):
    '''
        Clips the intensity of animage between the thresh[0]th
        and thresh[1]th percentiles, and the scales it between 0 and 1
    '''
    val_l, val_h = np.percentile(image, thres)
    image[image < val_l] = val_l
    image[image > val_h] = val_h
    return (image.astype(np.float32) - val_l) / (val_h - val_l + 1e-6)

def plot_histogram(x, no_zeros=True, bins=128, zero_threshold=0.001, 
                   intensity_normalization=False):
    '''
        Plots the historgram of an image
    '''
    x, _= process_initial_image(x, [1]*3, intensity_normalization)
    x= x.flatten()
    x= x[x>zero_threshold] if no_zeros else x
    plt.hist(x, bins=bins)
    plt.title('Histogram')
    plt.show()  
