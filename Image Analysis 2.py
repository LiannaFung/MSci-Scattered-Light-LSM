#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:25:40 2023

@author: rikke.ronnow
"""

from skimage import io
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageStat

image = io.imread('/Users/rikke.ronnow/Downloads/Pictures/picture20231123_162159')
plt.hist(image.ravel(),bins=200)
plt.xlabel('Intensity Value')
plt.ylabel('Count')
plt.show()

image=Image.open('/Users/rikke.ronnow/Downloads/Pictures/picture20231123_162159')
image.show()

cropped=image.crop((0,0,1280,1024))
cropped.show()


#%%

def average_perceived( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   return stat.mean[0]

def RMS_perceived( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   return stat.rms[0]

def std( im_file ):
   im = Image.open(im_file)
   stat = ImageStat.Stat(im)
   return stat.stddev[0]

# def perceived_average( im_file ):
#    im = Image.open(im_file)
#    stat = ImageStat.Stat(im)
#    gs = (np.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)) 
#          for r,g,b in im.getdata())
#    return sum(gs)/stat.count[0]


#%%
import glob, os

rootdir = '/Users/rikke.ronnow/Downloads/P1 0-90, P2 fixed 140'

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    print("Current File Being Processed is: ", file)
    

#%%
rootdir = '/Users/rikke.ronnow/Downloads/P1 0-90, P2 fixed 140'

start=0
perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    # plt.title(file[-6:])
    # plt.hist(image.ravel(),bins=100)
    # plt.xlabel('Intensity Value')
    # plt.ylabel('Count')
    # plt.xlim(35,150)
    # plt.ylim(0,20000)
    # plt.show()
    # print(file)
    # print(average_perceived(os.path.join(subdir, file)))
    # print(RMS_perceived(os.path.join(subdir, file)))
    # print(perceived_average(os.path.join(subdir, file)))
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    print(std(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmax(perceived_brightness)
rms_max=np.argmax(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))*5+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))*5+start,perceived_brightness)
plt.show()

print('max brightness for polarizer 1 at:',start+5*p_max)
print('max rms brightness for polarizer 1 at:',start+5*rms_max)


#%%

rootdir = '/Users/rikke.ronnow/Downloads/P1 165-260, P2 fixed 140'
start=165

perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmax(perceived_brightness)
rms_max=np.argmax(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))*5+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))*5+start,perceived_brightness)
plt.show()
print('max brightness for polarizer 1 at:',start+5*p_max)
print('max rms brightness for polarizer 1 at:',start+5*rms_max)

#%%

rootdir = '/Users/rikke.ronnow/Downloads/P2 280-360, P1 fixed 210'
start=280

perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmax(perceived_brightness)
rms_max=np.argmax(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))*5+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))*5+start,perceived_brightness)
plt.show()

print('max brightness for polarizer 2 at:',start+5*p_max)
print('max rms brightness for polarizer 2 at:',start+5*rms_max)

#%%

rootdir = '/Users/rikke.ronnow/Downloads/P2 100-200, P1 fixed 210'
start=100

perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmax(perceived_brightness)
rms_max=np.argmax(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))*5+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))*5+start,perceived_brightness)
plt.show()

print('max brightness for polarizer 2 at:',start+5*p_max)
print('max rms brightness for polarizer 2 at:',start+5*rms_max)

#%%

rootdir = '/Users/rikke.ronnow/Downloads/P1 120-160, P2 fixed 210'
start=120

perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmin(perceived_brightness)
rms_max=np.argmin(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))*5+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))*5+start,perceived_brightness)
plt.show()

print('min brightness for polarizer 1 at:',start+5*p_max)
print('min rms brightness for polarizer 1 at:',start+5*rms_max)

#%%

rootdir = '/Users/rikke.ronnow/Downloads/P1 280-360, P2 fixed 140'
start=280

perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmin(perceived_brightness)
rms_max=np.argmin(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))*5+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))*5+start,perceived_brightness)
plt.show()

print('min brightness for polarizer 1 at:',start+5*p_max)
print('min rms brightness for polarizer 1 at:',start+5*rms_max)

#%%

rootdir = '/Users/rikke.ronnow/Downloads/P2 0-100, P1 fixed 210'
start=0

perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmin(perceived_brightness)
rms_max=np.argmin(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))*5+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))*5+start,perceived_brightness)
plt.show()

print('min brightness for polarizer 2 at:',start+5*p_max)
print('min rms brightness for polarizer 2 at:',start+5*rms_max)

#%%

rootdir = '/Users/rikke.ronnow/Downloads/P2 190-270, P1 fixed 210'
start=190

perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmin(perceived_brightness)
rms_max=np.argmin(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))*5+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))*5+start,perceived_brightness)
plt.show()

print('min brightness for polarizer 2 at:',start+5*p_max)
print('min rms brightness for polarizer 2 at:',start+5*rms_max)

#%%
rootdir = '/Users/rikke.ronnow/Downloads/P1 58-68, P2 fixed 140'

start=58
perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmax(perceived_brightness)
rms_max=np.argmax(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))+start,perceived_brightness)
plt.show()

print('max brightness for polarizer 1 at:',start+1*p_max)
print('max rms brightness for polarizer 1 at:',start+1*rms_max)


#%%
rootdir = '/Users/rikke.ronnow/Downloads/P1 207-217, P2 fixed 140'

start=207
perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmax(perceived_brightness)
rms_max=np.argmax(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))+start,perceived_brightness)
plt.show()

print('max brightness for polarizer 1 at:',start+1*p_max)
print('max rms brightness for polarizer 1 at:',start+1*rms_max)

#%%
rootdir = '/Users/rikke.ronnow/Downloads/P2 147-157, P1 fixed 210'

start=147
perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmax(perceived_brightness)
rms_max=np.argmax(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(RMS_brightness))+start,RMS_brightness)
plt.show()


plt.ylabel('Perceived brightness')
plt.xlabel('Polarizer angle')

plt.plot(np.arange(len(perceived_brightness))+start,perceived_brightness)
plt.show()

print('max brightness for polarizer 2 at:',start+1*p_max)
print('max rms brightness for polarizer 2 at:',start+1*rms_max)

#%%
rootdir = '/Users/rikke.ronnow/Downloads/P2 327-337, P1 fixed 210'

start=207
perceived_brightness=[]
RMS_brightness=[]
picture=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    image = io.imread(file)
    perceived_brightness.append(average_perceived(os.path.join(file)))
    RMS_brightness.append(RMS_perceived(os.path.join(file)))
    picture.append(file)
        
#%%

perceived_brightness=np.array(perceived_brightness)
RMS_brightness=np.array(RMS_brightness)
picture=np.array(picture)

p_max=np.argmax(perceived_brightness)
rms_max=np.argmax(RMS_brightness)

print(p_max,picture[p_max])
print(rms_max,picture[rms_max])

plt.plot(np.arange(len(RMS_brightness)),RMS_brightness)
plt.show()
plt.plot(np.arange(len(perceived_brightness)),perceived_brightness)
plt.show()

print('max brightness for polarizer 2 at:',start+1*p_max)
print('max rms brightness for polarizer 2 at:',start+1*rms_max)

#%%

rootdir = '/Users/rikke.ronnow/Downloads/9 bright'

brightimages=[]
darkimages=[]

for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    brightimages.append(file)

rootdir = '/Users/rikke.ronnow/Downloads/9 dark'


for file in sorted(glob.glob(os.path.join(rootdir,'*'))):
    darkimages.append(file)
    
import tifffile 

 
def average(images,title):
# Load 9 TIFF images 

    image_paths = images
    # title=str(images)
    
    images = [tifffile.imread(path) for path in image_paths] 
    # Convert images to numpy arrays 
    
    image_arrays = [np.array(img) for img in images] 
    # Stack images along a new axis 
    
    stacked_array = np.stack(image_arrays, axis=-1) 
    # Calculate the average pixel value across the stacked images 
    
    average_array = np.mean(stacked_array, axis=-1, dtype=np.uint16)*255
    # Create a new Image object from the average array 
    
    average_image = Image.fromarray(average_array) 
    
    # print(title)
    
    # Save or display the result 
        
    average_image.save(title+".tiff") 
    
    return average_array

    
average(brightimages,"bright")
average(darkimages,"dark")
    
def sub_image(image1,image2):
    
    subtract=image1-image2
    subtract_image=Image.fromarray(subtract)
    subtract_image.show()
    
    
sub_image(average(brightimages,"bright"),average(darkimages,"dark"))