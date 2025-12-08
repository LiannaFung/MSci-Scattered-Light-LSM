import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile
import scienceplots
plt.style.use('no-latex')  # makes them LaTeX style

def dop(bb, bd, title, blur=0):
    """heatmap for degree of polarisation for averaged images
    INPUTS:
    bb -- str, directory for bb images
    dd -- str, directory for dd images
    title -- str, save name for file
    blur -- boolean, applies gaussian blur to images before DOP calculation, default off"""

    rootdir = [bb, bd]
    averages = []
    for i in range(len(rootdir)):  # iterating through bb, bd, db, dd
        images = []
        for file in sorted(os.listdir(rootdir[i])):  # lists all files in directory
            if '.DS_Store' in file:
                continue
            full_path = os.path.join(rootdir[i], file)  # create the full path by joining the directory path and the file name
            images.append(full_path)  # append the full path to appropriate list
        image_paths = images
        images = [tifffile.imread(path) for path in image_paths]
        image_arrays = [np.array(img) for img in images]  # convert to arrays
        stacked_array = np.stack(image_arrays, axis=-1)  # stack images along a new axis
        average_array = np.mean(stacked_array, axis=-1, dtype=np.uint16)  # averaging pixel value for stacks, multiplied by 255 for the output image
        averages.append(average_array)

    bb_avrg, bd_avrg = averages
    #bb_avrg = bb_avrg_[800:1000, 1000:1200]
    #bd_avrg = bd_avrg_[800:1000, 1000:1200]

    #plt.imshow(bb_avrg)  # for guassian blur comparison
    #plt.show()
    if blur == 1:
        kernel_size = 5  # size of gaussian kernel, should be an odd number
        sigmaX = 3  # standard deviation of gaussian
        bb_avrg = cv2.GaussianBlur(bb_avrg, (kernel_size, kernel_size), sigmaX)
        bd_avrg = cv2.GaussianBlur(bd_avrg, (kernel_size, kernel_size), sigmaX)
        #plt.imshow(bb_avrg)  # for gaussian blur comparison
        #plt.show()

    DOP = cv2.subtract(bb_avrg, bd_avrg) / cv2.add(bb_avrg, bd_avrg)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.03, wspace=0.1, bottom=0.2)

    im1 = axs[0].imshow(bd_avrg, cmap='viridis')
    axs[0].set_title('HIVD', fontsize=20)
    axs[0].text(1095, 160, f'$5 \mu m$', color='w', fontsize=20)
    axs[0].add_patch(plt.Rectangle((1280 - 79 - 75, 75), 79, 2, linewidth=2, edgecolor='w', facecolor='none'))
    colorbar_ax1 = fig.add_axes([0.125, 0.18, 0.37, 0.03])  # Position for the first colorbar
    colorbar1 = fig.colorbar(im1, cax=colorbar_ax1, orientation='horizontal')
    colorbar1.set_label('Pixel Value', fontsize=20)
    colorbar1.ax.tick_params(labelsize=17)

    im2 = axs[1].imshow(DOP, cmap='viridis')
    axs[1].set_title('Degree of Polarisation', fontsize=20)
    colorbar_ax2 = fig.add_axes([0.53, 0.18, 0.37, 0.03])  # Position for the second colorbar
    colorbar2 = fig.colorbar(im2, cax=colorbar_ax2, orientation='horizontal')
    colorbar2.set_label('Degree of Polarisation', fontsize=20)
    colorbar2.ax.tick_params(labelsize=17)

    for ax in axs:
        ax.axis('off')

    plt.savefig('{} Degree of Polarisation.pdf'.format(title), bbox_inches='tight')
    plt.show()
    return

#%% Exapnder 1 0%

bb = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander, bb'
bd = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander, bd'
title = 'Expander 1'
dop(bb, bd, title)

#%% Exapnder 4 0%

bb = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 4, bb'
bd = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 4, bd'
title = 'Expander 4'
dop(bb, bd, title)

#%% Dithered TiO2 0.01%

bb = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\TiO2 0.01% 2, bb'
bd = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\TiO2 0.01% 2, bd'
title = 'Dithered TiO2 0_01'
dop(bb, bd, title)

#%% TiO2 0.0125%

bb = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\d TiO2 0.125 bb'
bd = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\d TiO2 0.125 bd'
bb = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\TiO2 2 bb'
bd = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\TiO2 2 bd'
title = 'TiO2 0_125'
title = 'TiO2 2'
dop(bb, bd, title)

#%% Latex 0.2%

bb = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\latex 0.23 bb'
bd = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\latex 0.23 bd'
bb = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\latex 3.6 bb'
bd = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\latex 3.6 bd'
title = 'latex 0_23'
title = 'latex 3_6'
dop(bb, bd, title)

#%%
def dop_fish(bb, bd, title, blur=0):
    """heatmap for degree of polarisation for averaged images
    INPUTS:
    bb -- str, directory for bb images
    dd -- str, directory for dd images
    title -- str, save name for file
    blur -- boolean, applies gaussian blur to images before DOP calculation, default off"""

    rootdir = [bb, bd]
    averages = []
    for i in range(len(rootdir)):  # iterating through bb, bd, db, dd
        images = []
        for file in sorted(os.listdir(rootdir[i])):  # lists all files in directory
            if '.DS_Store' in file:
                continue
            full_path = os.path.join(rootdir[i], file)  # create the full path by joining the directory path and the file name
            images.append(full_path)  # append the full path to appropriate list
        image_paths = images
        images = [tifffile.imread(path) for path in image_paths]
        image_arrays = [np.array(img) for img in images]  # convert to arrays
        stacked_array = np.stack(image_arrays, axis=-1)  # stack images along a new axis
        average_array = np.mean(stacked_array, axis=-1, dtype=np.uint16)  # averaging pixel value for stacks, multiplied by 255 for the output image
        averages.append(average_array)

    bb_avrg, bd_avrg = averages

    #plt.imshow(bb_avrg)  # for guassian blur comparison
    #plt.show()
    if blur == 1:
        kernel_size = 5  # size of gaussian kernel, should be an odd number
        sigmaX = 3  # standard deviation of gaussian
        bb_avrg = cv2.GaussianBlur(bb_avrg, (kernel_size, kernel_size), sigmaX)
        bd_avrg = cv2.GaussianBlur(bd_avrg, (kernel_size, kernel_size), sigmaX)
        #plt.imshow(bb_avrg)  # for gaussian blur comparison
        #plt.show()

    DOP = cv2.subtract(bb_avrg, bd_avrg) / cv2.add(bb_avrg, bd_avrg)

    fig, ax = plt.subplots(figsize=(6, 9))
    im = ax.imshow(DOP, cmap='viridis')
    ax.add_patch(plt.Rectangle((1280-98-75, 75), 98, 2, linewidth=2, edgecolor='w', facecolor='none'))
    ax.text(1060, 150, f'$25 \mu m$', color='w', fontsize=18)
    colorbar_ax = fig.add_axes([0.125, 0.12, 0.775, 0.03])  # Position for the colorbar
    colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label('Degree of Polarisation', fontsize=18)

    ax.axis('off')

    plt.savefig('{} degree of polarisation.pdf'.format(title), bbox_inches='tight')
    plt.show()
    return

#%% dome stage

title = ['dome 0_44']
bb = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\3hr fish .44, bb']
bd = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\3hr fish .44, bd']

for i in range(len(title)):
    dop_fish(bb[i], bd[i], title[i])

#%% head

title = ['head 0_1']
bb = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\fish head .1, bb']
bd = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\fish head .1, bd']

for i in range(len(title)):
    dop_fish(bb[i], bd[i], title[i])

#%% body

bb = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\fish body .2, bb']
bd = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\fish body .2, bd']
title = ['body 0_2']

for i in range(len(title)):
    dop_fish(bb[i], bd[i], title[i])

