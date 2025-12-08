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


bb_stb=[]
bd_stb=[]
db_stb=[]
dd_stb=[]

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

def split(list_a, chunk_size):
    full_list=[]
    for i in range(0, len(list_a), chunk_size):
        full_list.append(list_a[i:i + chunk_size])
    return full_list
        
    

# %% s/b for averages

conc0=r'/Users/rikke.ronnow/Downloads/0% averages'
conc0a2=r'/Users/rikke.ronnow/Downloads/0% averages attempt 2'

conc02=r'/Users/rikke.ronnow/Downloads/0.2% averages'

conc04=r'/Users/rikke.ronnow/Downloads/0.4% averages'

conc08=r'/Users/rikke.ronnow/Downloads/1 ml averages'
conc082=r'/Users/rikke.ronnow/Downloads/0.8% averages'

conc2=r'/Users/rikke.ronnow/Downloads/2% averages'
conc22=r'/Users/rikke.ronnow/Downloads/2% averages attempt 2' #odd

conc4=r'/Users/rikke.ronnow/Downloads/4% averages' #weird
conc42=r'/Users/rikke.ronnow/Downloads/4% averages attempt 2'

conc8=r'/Users/rikke.ronnow/Downloads/8% averages'
conc82=r'/Users/rikke.ronnow/Downloads/8% averages attempt 2'

all_roootdir = [conc0, conc0a2, conc02, conc04, conc08, conc082, conc2, conc42, conc8, conc82]
masks=r'/Users/rikke.ronnow/Downloads/Masks-kopi 2'

maskpath=[]


for j, file in enumerate(sorted(os.listdir(masks))):  # iterates through each file in the selected bright dark combination
    full_path = os.path.join(masks, file)
    maskpath.append(full_path)

stb=[]

signals = []

averages = []

backgrounds = []

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    for file in sorted(
            os.listdir(all_roootdir[i])):  # iterates through each file in the selected bright dark combination
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if i==8:
            if '.DS_Store' in full_path:
                continue
            image=plt.imread(full_path)
        else:
            image = plt.imread(full_path)
        
        if '.DS_Store' in maskpath:
            continue
        mask=plt.imread(maskpath[i+1])
        
        if i==300:
            background = np.argwhere(mask == 0)  # pixel values where there is background
            signal = np.argwhere(mask != 0)
        else:
            background = np.argwhere(mask != 0)  # pixel values where there is background
            signal = np.argwhere(mask == 0)
            
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        
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

        # plt.imshow(image, cmap='hot', vmin=0, vmax=65025)
        # plt.title(file)
        # plt.show()
        
        # plt.imshow(mask, cmap='hot', vmin=0, vmax=1)
        # plt.title(file)
        # plt.show()

        mean_intensity = np.mean(selected_region)
        speckle_contrast = np.std(selected_region)/mean_intensity
        
        stb.append(signal_average/background_average)
        signals.append(signal_average)
        backgrounds.append(background_average)

        
        # if i==2:
        #     scs.append(speckle_contrast)
        # print(f'Mean Intensity: {mean_intensity}')
        # print(f'Speckle Contrast: {speckle_contrast}')

signals=split(signals,4)
backgrounds=split(backgrounds,4)

#%%

sts=[]
btb=[]

for i in signals:
    i=i/i[0]
    sts.append(i)
    
for i in backgrounds:
    i=i/i[0]
    btb.append(i)
    
background_polarization=[]
signal_polarization=[]

for i in sts:
    signal_polarization1=(i[1]-i[0])/(i[1]+i[0])
    signal_polarization2=(i[3]-i[2])/(i[3]+i[2])
    
    signal_polarization.append(signal_polarization1)
    

for i in btb:
    background_polarization1=(i[1]-i[0])/(i[1]+i[0])
    background_polarization2=(i[3]-i[2])/(i[3]+i[2])
    
    background_polarization.append(background_polarization1)
    
ratio=np.array(signal_polarization)/np.array(background_polarization)


    
#%%
n=4
x=[0,0,0.2,0.4,0.8,0.8,2,4,8,8]
# plt.scatter(x,stb[::n],label='HIHO')

plt.xlabel('Concentration %')
plt.ylabel('S/B')

# plt.scatter(x,stb[1::n],label='HIVO')

plt.scatter(x,stb[2::n],label='VIHO')

plt.scatter(x,stb[3::n],label='VIVO')

plt.legend()
plt.show()

#%%
n=4
x=[0,0,0.2,0.4,0.8,0.8,2,4,8,8]
# plt.scatter(x,stb[::n],label='HIHO')

plt.xlabel('Concentration %')
plt.ylabel('Polarization')

# plt.scatter(x,stb[1::n],label='HIVO')

plt.scatter(x,signal_polarization,label='signal polarization')

plt.scatter(x,background_polarization,label='background polarization')

plt.legend()
plt.show()

#%%
plt.scatter(x,ratio)

plt.show()

# %% s/b for subtractions

conc0=r'/Users/rikke.ronnow/Downloads/0% subtractions'
conc0a2=r'/Users/rikke.ronnow/Downloads/0% subtractions attempt 2'

conc02=r'/Users/rikke.ronnow/Downloads/0.2% subtractions'

conc04=r'/Users/rikke.ronnow/Downloads/0.4% subtractions'

conc08=r'/Users/rikke.ronnow/Downloads/1 ml subtractions'
conc082=r'/Users/rikke.ronnow/Downloads/0.8% subtractions'

conc2=r'/Users/rikke.ronnow/Downloads/2% subtractions'
conc22=r'/Users/rikke.ronnow/Downloads/2% subtractions attempt 2'

conc4=r'/Users/rikke.ronnow/Downloads/4% subtractions'
conc42=r'/Users/rikke.ronnow/Downloads/4% subtractions attempt 2'

conc8=r'/Users/rikke.ronnow/Downloads/8% subtractions'
conc82=r'/Users/rikke.ronnow/Downloads/8% subtractions attempt 2'

all_roootdir = [conc0, conc0a2, conc02, conc04, conc08, conc082, conc2, conc22, conc4, conc42, conc8, conc82]
masks=r'/Users/rikke.ronnow/Downloads/Masks-kopi'

maskpath=[]


for j, file in enumerate(sorted(os.listdir(masks))):  # iterates through each file in the selected bright dark combination
    full_path = os.path.join(masks, file)
    maskpath.append(full_path)

stb=[]

averages = []

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    print(all_roootdir[i])
    for file in sorted(
            os.listdir(all_roootdir[i])):  # iterates through each file in the selected bright dark combination
        print(file)
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if i==4:
            if '.DS_Store' in full_path:
                continue
            image=tifffile.imread(full_path)
        else:
            image = tifffile.imread(full_path)
         
        if '.DS_Store' in maskpath:
            continue
        mask=tifffile.imread(maskpath[i+1])
        
        if i==8:
            background = np.argwhere(mask != 0)  # pixel values where there is background
            signal = np.argwhere(mask == 0)
        else:
            background = np.argwhere(mask == 0)  # pixel values where there is background
            signal = np.argwhere(mask != 0)

        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        
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
        
        stb.append(signal_average/background_average)
        
        # if i==2:
        #     scs.append(speckle_contrast)
        # print(f'Mean Intensity: {mean_intensity}')
        # print(f'Speckle Contrast: {speckle_contrast}')

#%%
n=6
x=[0,0,0.2,0.4,0.8,0.8,2,2,4,4,8,8]
plt.plot(x,stb[::n],'x',label='HIHO-HIVO')

plt.xlabel('Concentration %')
plt.ylabel('S/B')

plt.plot(x,stb[1::n],'x',label='HIHO-VIHO')
plt.plot(x,stb[2::n],'x',label='HIHO-VIVO')
plt.plot(x,stb[3::n],'x',label='HIVO-VIHO')
plt.plot(x,stb[4::n],'x',label='HIVO-VIVO')
plt.plot(x,stb[5::n],'x',label='VIHO-VIVO')


plt.legend()
plt.show()

# %% s/b for averages

conc01=r'/Users/rikke.ronnow/Downloads/TiO2 0.1% averages'

conc02=r'/Users/rikke.ronnow/Downloads/TiO2 0.2% averages'

conc04=r'/Users/rikke.ronnow/Downloads/TiO2 0.4% averages'

all_roootdir = [conc01, conc02, conc04]
masks=r'/Users/rikke.ronnow/Downloads/Masks new'

maskpath=[]


for j, file in enumerate(sorted(os.listdir(masks))):  # iterates through each file in the selected bright dark combination
    full_path = os.path.join(masks, file)
    maskpath.append(full_path)

stb=[]

averages = []

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    print(all_roootdir[i])
    for file in sorted(
            os.listdir(all_roootdir[i])):  # iterates through each file in the selected bright dark combination
        print(file)
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if i==10:
            if '.DS_Store' in full_path:
                continue
            image=plt.imread(full_path)
        else:
            image = plt.imread(full_path)
        
        if '.DS_Store' in maskpath:
            continue
        mask=plt.imread(maskpath[i])
        
        if i==300:
            background = np.argwhere(mask == 0)  # pixel values where there is background
            signal = np.argwhere(mask != 0)
        else:
            background = np.argwhere(mask != 0)  # pixel values where there is background
            signal = np.argwhere(mask == 0)
            
        plt.imshow(image)
        plt.show()
        plt.imshow(mask)
        plt.show()
        
        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        
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
        
        stb.append(signal_average/background_average)
        
        # if i==2:
        #     scs.append(speckle_contrast)
        # print(f'Mean Intensity: {mean_intensity}')
        # print(f'Speckle Contrast: {speckle_contrast}')
#%%
n=4
x=[0.1,0.2,0.4]
plt.scatter(x,stb[::n],label='HIHO')

plt.xlabel('Concentration %')
plt.ylabel('S/B')

plt.scatter(x,stb[1::n],label='HIVO')

plt.scatter(x,stb[2::n],label='VIHO')

plt.scatter(x,stb[3::n],label='VIVO')

plt.legend()
plt.show()

# %% s/b for subtractions

conc01=r'/Users/rikke.ronnow/Downloads/TiO2 0.1% subtractions'

conc02=r'/Users/rikke.ronnow/Downloads/TiO2 0.2% subtractions'

conc04=r'/Users/rikke.ronnow/Downloads/TiO2 0.4% subtractions'

all_roootdir = [conc01, conc02, conc04]
masks=r'/Users/rikke.ronnow/Downloads/Masks new'

maskpath=[]


for j, file in enumerate(sorted(os.listdir(masks))):  # iterates through each file in the selected bright dark combination
    full_path = os.path.join(masks, file)
    maskpath.append(full_path)

stb=[]

averages = []

for i in range(len(all_roootdir)):  # iterates through the bright dark combinations
    print(all_roootdir[i])
    for file in sorted(
            os.listdir(all_roootdir[i])):  # iterates through each file in the selected bright dark combination
        print(file)
        full_path = os.path.join(all_roootdir[i], file)
        averages.append(full_path)
        if i==4:
            if '.DS_Store' in full_path:
                continue
            image=tifffile.imread(full_path)
        else:
            image = tifffile.imread(full_path)
         
        if '.DS_Store' in maskpath:
            continue
        mask=tifffile.imread(maskpath[i])
        
        if i==8:
            background = np.argwhere(mask != 0)  # pixel values where there is background
            signal = np.argwhere(mask == 0)
        else:
            background = np.argwhere(mask == 0)  # pixel values where there is background
            signal = np.argwhere(mask != 0)

        background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
        signal_values = image[signal[:, 0], signal[:, 1]]

        signal_average = np.mean(signal_values)
        background_average = np.mean(background_values)
        
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
        
        stb.append(signal_average/background_average)
        
        # if i==2:
        #     scs.append(speckle_contrast)
        # print(f'Mean Intensity: {mean_intensity}')
        # print(f'Speckle Contrast: {speckle_contrast}')

#%%
n=6
plt.figure(figsize=(7,8))
plt.plot(x,stb[::n],'o',label='HIHD-HIVD')

plt.xlabel('Concentration %',fontsize=20)
plt.ylabel('S/B',fontsize=20)

plt.plot(x,stb[1::n],'o',label='HIHD-VIHD')
plt.plot(x,stb[2::n],'o',label='HIHD-VIVD')
plt.plot(x,stb[3::n],'o',label='HIVD-VIHD')
plt.plot(x,stb[4::n],'o',label='HIVD-VIVD')
plt.plot(x,stb[5::n],'o',label='VIHD-VIVD')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig('concentration',transparent=True, dpi=2000)
plt.show()

