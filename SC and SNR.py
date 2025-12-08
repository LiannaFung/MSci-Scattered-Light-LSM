#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:31:24 2023

@author: rikke.ronnow
"""

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
import matplotlib.patches as patches
from PIL import Image, ImageStat
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


fontprops = fm.FontProperties(size=18)


# def average_perceived( im_file ):
#    im = Image.open(im_file)
#    # im.show()
#    stat = ImageStat.Stat(im)
#    return stat.mean[0]

def average_perceived(im_file):
    im = tifffile.imread(im_file)
    im_arr = np.array(im)
    # im.show()

    return np.mean(im_arr)


# def RMS_perceived( im_file ):
#    im = Image.open(im_file)
#    stat = ImageStat.Stat(im)
#    rms = np.sqrt(np.mean(y**2))
#    return stat.rms[0]


def RMS_perceived(im_file):
    im = tifffile.imread(im_file)
    im_arr = np.array(im)
    # im.show()

    return np.sqrt(np.mean(im_arr ** 2))


# def std( im_file ):
#    im = Image.open(im_file)
#    stat = ImageStat.Stat(im)
#    return stat.stddev[0]

def std(im_file):
    im = tifffile.imread(im_file)
    im_arr = np.array(im)
    # im.show()

    return np.std(im_arr)

def find_min_max_brightness(rootdirs):
    """Finds the minimum and maximum pixel brightness in all the files in one
    directory"""
    min_brightness = float('inf')
    max_brightness = float('-inf')

    for rootdir in rootdirs:
        for subdir, _, files in os.walk(rootdir):
            for file in files:
                # Assuming the images are TIFF files
                image_path = os.path.join(subdir, file)

                # Read the image
                image = tifffile.imread(image_path)

                # Find minimum and maximum brightness
                current_min = np.min(image)
                current_max = np.max(image)

                # Update overall minimum and maximum brightness
                min_brightness = min(min_brightness, current_min)
                max_brightness = max(max_brightness, current_max)

    return min_brightness, max_brightness


# %% speckle contrast for averages and subtractions

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/latex pollen averages'
rootdir_sub = r'/Users/rikke.ronnow/Downloads/latex pollen subtractions'
rootdir_bb='/Users/rikke.ronnow/Downloads/latex pollen, p1 bright, p2 bright'
all_roootdir = [rootdir_avrg, rootdir_sub, rootdir_bb]

scs=[]

#all_roots = [rootdir_avrg, rootdir_sub]
#min_brightness_avrg, max_brightness_avrg = calculate_brightness(rootdir_avrg)
#min_brightness_sub, max_brightness_sub = calculate_brightness(rootdir_sub)

averages = []

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    print(all_roootdir[i])
    for file in sorted(
            os.listdir(all_roootdir[i])):  # iterates through each file in the selected bright dark combination
        print(file)
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        selected_region = image[800:1000, 1000:1200]  # width is 200
        # original_image_with_rectangle = image.copy()

        # # Convert the color values to the range 0-1
        # color = (0, 1, 0, 1)  # Green color in RGBA format

        # rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        # fig, ax = plt.subplots()  # plots with the rectangle over selected region
        # ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        # ax.add_patch(rect)

        # # Display the image with the rectangle using matplotlib
        # plt.title(file)
        # plt.show()

        # plt.imshow(selected_region, cmap='hot', vmin=0, vmax=1)
        # plt.title(file)
        # plt.show()

        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region)/mean_intensity
        
        # flat_region=np.flatten(selected_region)
        plt.title(file)
        flat_region=selected_region.flatten()
        plt.hist(flat_region, bins=40, histtype='step')
        plt.show()
        
        if i==2:
            scs.append(speckle_contrast)
        print(f'Mean Intensity: {mean_intensity}')
        print(f'Speckle Contrast: {speckle_contrast}')

#%% AVERAGES AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/latex pollen averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of latex bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('latex pollen averages stb',dpi=2000)

plt.show()

#%% SUBTRACTIONS AS SUBPLOT

rootdir_sub = r'/Users/rikke.ronnow/Downloads/latex pollen subtractions'
all_roootdir = [rootdir_sub]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of latex bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal




for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',dpi=2000)


plt.show()

# %% SNR for averages and subtractions

image = tifffile.imread(
    r'/Users/rikke.ronnow/Downloads/Mask of latex bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    print(all_roootdir[i])
    for file in sorted(
            os.listdir(all_roootdir[i])):  # iterates through each file in the selected bright dark combination
        print(file)
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        background_values = image[
            background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        print('snr', signal_average / background_average)
        
#%% AVERAGES speckle contrast AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/latex pollen averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of latex bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()


#%% SUBTRACTIONS speckle contrast AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/latex pollen subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of latex bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.savefig('testing',transparent=True, dpi=600)

plt.show()

#%% raw images AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/latex pollen'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of latex bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=255)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('latex pollen averages stb',dpi=2000)

plt.show()


#%% raw image speckle contrast AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/latex pollen'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of latex bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=255)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file)
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()

#%% all bb speckle contrast AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/latex sc'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of latex bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot')

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(wspace=0.04)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file)
        if j!=0:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
        else: 
            ax.text(10, 190, 'SC={}'.format(str(round(sc,2))), color='w', fontsize='xx-large')
plt.tight_layout()
# plt.savefig('latex pollen sc',dpi=2000)

plt.show()

# #%% 0.5 ml AS SUBPLOT

# rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0.5 ml averages'
# all_roootdir = [rootdir_avrg]

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
# axes = axes.flatten()

# image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0.5ml bb average.tiff')
# background = np.argwhere(image == 0)  # pixel values where there is background
# signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

# for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
#     for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
#         full_path = os.path.join(all_roootdir[i], file)
#         averages.append(full_path)
#         image = tifffile.imread(full_path)
        
        
        
#         background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
#         signal_values = image[signal[:, 0], signal[:, 1]]

#         signal_average = np.mean(signal_values)
#         background_average = np.mean(background_values)
#         snr = signal_average / background_average
        
        
#         selected_region = image[800:1000, 1000:1200]  # width is 200
#         original_image_with_rectangle = image.copy()

#         # Convert the color values to the range 0-1
#         color = (0, 1, 0, 1)  # Green color in RGBA format

#         rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

#         ax = axes[i*3 + j]
#         ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
#         if j==0:
#             ax.add_patch(rect)

#         # Display the image with the rectangle using matplotlib
#         ax.set_title(file[:-5])
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.set_yticks([])
#         ax.set_xticks([])
        
#         plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
#         ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# # plt.savefig('05ml averages stb',dpi=600)

# plt.show()

# #%% AVERAGES speckle contrast AS SUBPLOT


# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
# axes = axes.flatten()


# for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
#     for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
#         full_path = os.path.join(all_roootdir[i], file)
#         averages.append(full_path)
#         image = tifffile.imread(full_path)
        
        
        
#         background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
#         signal_values = image[signal[:, 0], signal[:, 1]]

#         signal_average = np.mean(signal_values)
#         background_average = np.mean(background_values)
#         snr = signal_average / background_average
        
        
#         selected_region = image[800:1000, 1000:1200]  # width is 200
#         ax = axes[i*3 + j]
#         ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
#         mean_intensity = np.mean(selected_region)
#         speckle_contrast = np.std(selected_region) / mean_intensity

#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.set_yticks([])
#         ax.set_xticks([])
        
#         plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
#         # Display the image with the rectangle using matplotlib
#         ax.set_title(file[:-5])
#         ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# # plt.tight_layout()
# # plt.savefig('latex pollen averages sc',dpi=2000)

# plt.show()


# #%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
# rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0.5 ml subtractions'
# all_roootdir = [rootdir_avrg]

# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
# axes = axes.flatten()   

# image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0.5ml bb average.tiff')
# background = np.argwhere(image == 0)  # pixel values where there is background
# signal = np.argwhere(image != 0)  # pixel values where there is signal


# for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
#     for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
#         full_path = os.path.join(all_roootdir[i], file)
#         averages.append(full_path)
#         image = tifffile.imread(full_path)
        
        
        
#         background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
#         signal_values = image[signal[:, 0], signal[:, 1]]

#         signal_average = np.mean(signal_values)
#         background_average = np.mean(background_values)
#         snr = signal_average / background_average
        
        
#         selected_region = image[800:1000, 1000:1200]  # width is 200
#         ax = axes[i*3 + j]
#         ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
#         mean_intensity = np.mean(selected_region)
#         speckle_contrast = np.std(selected_region) / mean_intensity


#         # Display the image with the rectangle using matplotlib
#         ax.set_title(file[:-5])
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.set_yticks([])
#         ax.set_xticks([])
        
#         plt.subplots_adjust(hspace=0.09,wspace=0)
#         # plt.subplots_adjust(vspace=0)

        
#         if np.isnan(speckle_contrast):    
#             ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
#         else:
#             ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# # plt.tight_layout()

# plt.show()

# #%% SUBTRACTIONS AS SUBPLOT

# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
# axes = axes.flatten()


# for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
#     for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
#         full_path = os.path.join(all_roootdir[i], file)
#         averages.append(full_path)
#         image = tifffile.imread(full_path)

        
#         background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
#         signal_values = image[signal[:, 0], signal[:, 1]]

#         signal_average = np.mean(signal_values)
#         background_average = np.mean(background_values)
#         snr = signal_average / background_average
        
#         selected_region = image[800:1000, 1000:1200]  # width is 200
#         original_image_with_rectangle = image.copy()

#         # Convert the color values to the range 0-1
#         color = (0, 1, 0, 1)  # Green color in RGBA format

#         rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

#         if i == 0:  # for averages
#             min_brightness = 0
#             max_brightness = 65025
#         else:  # for subtractions
#             min_brightness = 0
#             max_brightness = 1

#         ax = axes[i*3 + j]
#         ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
#         if j==0:
#             ax.add_patch(rect)

#         # Display the image with the rectangle using matplotlib
#         ax.set_title(file[:-5])
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.set_yticks([])
#         ax.set_xticks([])
        
#         plt.subplots_adjust(hspace=-0.4,wspace=0)
        
#         ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


# plt.tight_layout()

# # plt.savefig('latex pollen subtractions stb',dpi=2000)


# plt.show()


#%% 1 ml AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/1 ml averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 1ml bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()


#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/1 ml subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 1ml bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',dpi=2000)


plt.show()

# #%% 0 ml AS SUBPLOT

# rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0 ml averages'
# all_roootdir = [rootdir_avrg]

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
# axes = axes.flatten()

# image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0ml bb average.tiff')
# background = np.argwhere(image == 0)  # pixel values where there is background
# signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

# for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
#     for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
#         full_path = os.path.join(all_roootdir[i], file)
#         averages.append(full_path)
#         image = tifffile.imread(full_path)
        
        
        
#         background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
#         signal_values = image[signal[:, 0], signal[:, 1]]

#         signal_average = np.mean(signal_values)
#         background_average = np.mean(background_values)
#         snr = signal_average / background_average
        
        
#         selected_region = image[800:1000, 1000:1200]  # width is 200
#         original_image_with_rectangle = image.copy()

#         # Convert the color values to the range 0-1
#         color = (0, 1, 0, 1)  # Green color in RGBA format

#         rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

#         ax = axes[i*3 + j]
#         ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
#         if j==0:
#             ax.add_patch(rect)

#         # Display the image with the rectangle using matplotlib
#         ax.set_title(file[:-5])
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.set_yticks([])
#         ax.set_xticks([])
        
#         plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
#         ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# # plt.savefig('05ml averages stb',dpi=600)

# plt.show()

# #%% AVERAGES speckle contrast AS SUBPLOT


# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
# axes = axes.flatten()


# for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
#     for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
#         full_path = os.path.join(all_roootdir[i], file)
#         averages.append(full_path)
#         image = tifffile.imread(full_path)
        
        
        
#         background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
#         signal_values = image[signal[:, 0], signal[:, 1]]

#         signal_average = np.mean(signal_values)
#         background_average = np.mean(background_values)
#         snr = signal_average / background_average
        
        
#         selected_region = image[800:1000, 1000:1200]  # width is 200
#         ax = axes[i*3 + j]
#         ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
#         mean_intensity = np.mean(selected_region)
#         speckle_contrast = np.std(selected_region) / mean_intensity

#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.set_yticks([])
#         ax.set_xticks([])
        
#         plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
#         # Display the image with the rectangle using matplotlib
#         ax.set_title(file[:-5])
#         ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# # plt.tight_layout()
# # plt.savefig('latex pollen averages sc',dpi=2000)

# plt.show()




# #%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
# rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0 ml subtractions'
# all_roootdir = [rootdir_avrg]

# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
# axes = axes.flatten()   

# image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0ml bb average.tiff')
# background = np.argwhere(image == 0)  # pixel values where there is background
# signal = np.argwhere(image != 0)  # pixel values where there is signal


# for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
#     for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
#         full_path = os.path.join(all_roootdir[i], file)
#         averages.append(full_path)
#         image = tifffile.imread(full_path)
        
        
        
#         background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
#         signal_values = image[signal[:, 0], signal[:, 1]]

#         signal_average = np.mean(signal_values)
#         background_average = np.mean(background_values)
#         snr = signal_average / background_average
        
        
#         selected_region = image[800:1000, 1000:1200]  # width is 200
#         ax = axes[i*3 + j]
#         ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
#         mean_intensity = np.mean(selected_region)
#         speckle_contrast = np.std(selected_region) / mean_intensity


#         # Display the image with the rectangle using matplotlib
#         ax.set_title(file[:-5])
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.set_yticks([])
#         ax.set_xticks([])
        
#         plt.subplots_adjust(hspace=0.09,wspace=0)
#         # plt.subplots_adjust(vspace=0)

        
#         if np.isnan(speckle_contrast):    
#             ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
#         else:
#             ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# # plt.tight_layout()

# plt.show()

# #%% SUBTRACTIONS AS SUBPLOT


# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
# axes = axes.flatten()


# for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
#     for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
#         full_path = os.path.join(all_roootdir[i], file)
#         averages.append(full_path)
#         image = tifffile.imread(full_path)

        
#         background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
#         signal_values = image[signal[:, 0], signal[:, 1]]

#         signal_average = np.mean(signal_values)
#         background_average = np.mean(background_values)
#         snr = signal_average / background_average
        
#         selected_region = image[800:1000, 1000:1200]  # width is 200
#         original_image_with_rectangle = image.copy()

#         # Convert the color values to the range 0-1
#         color = (0, 1, 0, 1)  # Green color in RGBA format

#         rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

#         if i == 0:  # for averages
#             min_brightness = 0
#             max_brightness = 65025
#         else:  # for subtractions
#             min_brightness = 0
#             max_brightness = 1

#         ax = axes[i*3 + j]
#         ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
#         if j==0:
#             ax.add_patch(rect)

#         # Display the image with the rectangle using matplotlib
#         ax.set_title(file[:-5])
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.set_yticks([])
#         ax.set_xticks([])
        
#         plt.subplots_adjust(hspace=-0.4,wspace=0)
        
#         ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


# plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


# plt.show()

# #%% 0% AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0% averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[500:700, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 500), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[500:700, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0% subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[500:700, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[500:700, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 500), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% 2% AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/2% averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 2% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/2% subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 2% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% 8% AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/8% averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 8% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j-1]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==1:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j-1]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/8% subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 8% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% 4% AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/4% averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 4% bb average.tiff')
background = np.argwhere(image != 0)  # pixel values where there is background
signal = np.argwhere(image == 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/4% subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 4% bb average.tiff')
background = np.argwhere(image != 0)  # pixel values where there is background
signal = np.argwhere(image == 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()
#%% 0.2% AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0.2% averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0.2% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0.2% subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0.2% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% 0.4% AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0.4% averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0.4% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0.4% subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0.4% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% 0.8% AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0.8% averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0.8% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0.8% subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0.8% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% 0% AS SUBPLOT attempt 2

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0% averages attempt 2'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0% bb average attempt 2.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/0% subtractions attempt 2'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 0% bb average attempt 2.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% 2% AS SUBPLOT attempt 2

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/2% averages attempt 2'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 2% bb average attempt 2.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/2% subtractions attempt 2'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 2% bb average attempt 2.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% 4% AS SUBPLOT attempt 2

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/4% averages attempt 2'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 4% bb average attempt 2.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/4% subtractions attempt 2'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 4% bb average attempt 2.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% 8% AS SUBPLOT attempt 2

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/8% averages attempt 2'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of 8% bb average attempt 2.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/8% subtractions attempt 2'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Mask of % bb average attempt 2.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()

#%% TiO2 0.2% AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/TiO2 0.2% averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Masks new/Mask of TiO2 0.2% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        scalebar = AnchoredSizeBar(ax.transData,
                           404, '100 μm', 'upper left', 
                           pad=1,
                           color='white',
                           frameon=False,
                           size_vertical=3,
                           fontproperties=fontprops)
        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)
            ax.add_artist(scalebar)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=65025)
        
        scalebar = AnchoredSizeBar(ax.transData,
                           42, '10 μm', 'upper left', 
                           pad=0.5,
                           color='white',
                           frameon=False,
                           size_vertical=0.8,
                           fontproperties=fontprops)

        if j==0:
            ax.add_artist(scalebar)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/TiO2 0.2% subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Masks new/Mask of TiO2 0.2% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='hot', vmin=0, vmax=1)

        scalebar = AnchoredSizeBar(ax.transData,
                           42, '10 μm', 'upper left', 
                           pad=0.5,
                           color='white',
                           frameon=False,
                           size_vertical=0.8,
                           fontproperties=fontprops)

        if j==0:
            ax.add_artist(scalebar)        
        
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT

resolution=1000
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        scalebar = AnchoredSizeBar(ax.transData,
                           404 * resolution/fig.dpi , '100 μm', 'upper left', 
                           pad=1,
                           color='white',
                           frameon=False,
                           size_vertical=3,
                           fontproperties=fontprops)
        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='hot', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)
            ax.add_artist(scalebar)



        # Display the image with the rectangle using matplotlib
        # ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.1,wspace=0.04)
        
        # plt.subplots_adjust(hspace=-0.4,wspace=0)

        
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity
        
        ax.text(50, 910, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')
        ax.text(50, 1000, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')


# plt.tight_layout()

plt.savefig('TiO2 02% sub no titles',transparent=True, dpi=resolution)

plt.show()

 #%% TiO2 0.4% AS SUBPLOT

rootdir_avrg = r'/Users/rikke.ronnow/Downloads/TiO2 0.4% averages'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Masks new/Mask of TiO2 0.4% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal



        
        
        

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')
        
        scalebar = AnchoredSizeBar(ax.transData,
                           404, '100 μm', 'upper left', 
                           pad=1,
                           color='white',
                           frameon=False,
                           size_vertical=3,
                           fontproperties=fontprops)
        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='viridis', vmin=0, vmax=65025)
        if j==0:
            ax.add_patch(rect)
            ax.add_artist(scalebar)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.4)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')

plt.tight_layout()
# plt.savefig('05ml averages stb',dpi=600)

plt.show()

#%% AVERAGES speckle contrast AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if '.DS_Store' in full_path:
            continue
        image = plt.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='viridis', vmin=0, vmax=65025)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity
        
        scalebar = AnchoredSizeBar(ax.transData,
                           42, '10 μm', 'upper left', 
                           pad=0.5,
                           color='white',
                           frameon=False,
                           size_vertical=0.8,
                           fontproperties=fontprops)
        ax = axes[i*3 + j]

        if j==0:
            ax.add_artist(scalebar)


        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=-0.5)
        
        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')

# plt.tight_layout()
# plt.savefig('latex pollen averages sc',dpi=2000)

plt.show()




#%% SUBTRACTIONS speckle contrast AS SUBPLOT
        
rootdir_avrg = r'/Users/rikke.ronnow/Downloads/TiO2 0.4% subtractions'
all_roootdir = [rootdir_avrg]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()   

image = tifffile.imread(r'/Users/rikke.ronnow/Downloads/Masks new/Mask of TiO2 0.4% bb average.tiff')
background = np.argwhere(image == 0)  # pixel values where there is background
signal = np.argwhere(image != 0)  # pixel values where there is signal


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)
        
        
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        ax = axes[i*3 + j]
        ax.imshow(selected_region, cmap='viridis', vmin=0, vmax=1)

            
        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region) / mean_intensity


        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=0.09,wspace=0)
        # plt.subplots_adjust(vspace=0)

        
        if np.isnan(speckle_contrast):    
            ax.text(10, 190, 'SC=1', color='w', fontsize='xx-large')
        
        else:
            ax.text(10, 190, 'SC={}'.format(str(round(speckle_contrast,2))), color='w', fontsize='xx-large')
            
# plt.tight_layout()

plt.show()

#%% SUBTRACTIONS AS SUBPLOT


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()


for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for j, file in enumerate(sorted(os.listdir(all_roootdir[i]))):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        image = tifffile.imread(full_path)

        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        snr = signal_average / background_average
        
        selected_region = image[800:1000, 1000:1200]  # width is 200
        original_image_with_rectangle = image.copy()

        # Convert the color values to the range 0-1
        color = (0, 1, 0, 1)  # Green color in RGBA format

        rect = patches.Rectangle((1000, 800), 200, 200, linewidth=2, edgecolor=color, facecolor='none')

        if i == 0:  # for averages
            min_brightness = 0
            max_brightness = 65025
        else:  # for subtractions
            min_brightness = 0
            max_brightness = 1

        ax = axes[i*3 + j]
        ax.imshow(original_image_with_rectangle, cmap='viridis', vmin=0, vmax=1)
        if j==0:
            ax.add_patch(rect)

        # Display the image with the rectangle using matplotlib
        ax.set_title(file[:-5])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        
        plt.subplots_adjust(hspace=-0.4,wspace=0)
        
        ax.text(50, 1000, 'S/B={}'.format(str(round(snr,2))), color='w', fontsize='xx-large')


plt.tight_layout()

# plt.savefig('latex pollen subtractions stb',transparent=True, dpi=300)


plt.show()
