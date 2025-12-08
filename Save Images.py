from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile

from PIL import Image, ImageStat
# %% averaging images

def average(images, title):
    """Averages pixels over a stack of images. It default returns averaged image array * 255 for image viewing, if you want to do
        further computations, need to divide by 255 again.
    images -- path to a list of images
    title -- file title of the final averaged image, don't include the .tiff extension"""
    image_paths = images
    images = [tifffile.imread(path) for path in image_paths]
    image_arrays = [np.array(img) for img in images]  # convert to arrays
    stacked_array = np.stack(image_arrays, axis=-1)  # stack images along a new axis
    average_array = np.mean(stacked_array, axis=-1,
                            dtype=np.uint16) * 255  # averaging pixel value for stacks, multiplied by 255 for the output image
    average_image = Image.fromarray(average_array.astype(np.uint16))  # creating a new object from the averaged array

    # Save the image using a proper file path
    output_path = r'Downloads\{}.tiff'.format(title)
    average_image.save(title+".tiff") 
    
    plt.imshow(average_image, cmap='gray')  # using 'gray' colormap for grayscale images
    plt.title('Averaging Result')
    plt.colorbar()  # NOT SURE WHY THERE ARE TWO COLORBARS
    plt.show()

    return average_array / 255, output_path  # IDK IF I HAVE TO DIVIDE BY 255, DID FOR IMAGE PROCESSING REASONS?


# %% subtracting images

def subtract_arrays(array1, array2,title):
    """Subtracts images. If the pixel value is negative, it is left as negative.
    array1 -- first image array
    array2 -- second image array"""
    result = np.subtract(array2, array1)  # subtracting two arrays
    result = result.clip(min=0)
    plt.imshow(result, cmap='gray')  # using 'gray' colormap for grayscale images
    plt.title('Subtraction Result')
    plt.colorbar()
    plt.show()
    result_image = Image.fromarray(result/255) 
    result_image.save(title+".tiff") 
    
    return result
#%%
def save_images(full_paths, title):
    '''order: bb, bd, db, dd'''
    rootdir = full_paths[0]
    brightbright = []

    for file in sorted(os.listdir(rootdir)):  # lists all files in directory
        full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
        brightbright.append(full_path)

    rootdir = full_paths[2]
    darkbright = []

    for file in sorted(os.listdir(rootdir)):  # similarly for dark images
        full_path = os.path.join(rootdir, file)
        darkbright.append(full_path)

    rootdir = full_paths[1]
    brightdark = []

    for file in sorted(os.listdir(rootdir)):  # lists all files in directory
        full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
        brightdark.append(full_path)  # append the full path to appropriate list

    rootdir = full_paths[3]
    darkdark = []

    for file in sorted(os.listdir(rootdir)):  # similarly for dark images
        full_path = os.path.join(rootdir, file)
        darkdark.append(full_path)

    bb_array, bb_output_path = average(brightbright, title+' bb average')
    db_array, dark_output_path = average(darkbright, title+' db average')
    bd_array, bd_output_path = average(brightdark, title+' bd average')
    dd_array, dd_output_path = average(darkdark, title+' dd average')

    subtract_arrays(dd_array, bd_array, title+' bd-dd')
    subtract_arrays(db_array, bb_array, title+' bb-db')
    subtract_arrays(dd_array, bb_array, title+' bb-dd')
    subtract_arrays(bd_array, bb_array, title+' bb-bd')
    subtract_arrays(db_array, bd_array, title+' bd-db')
    subtract_arrays(dd_array, db_array, title+' db-dd')

#%%
bb = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\0.2%, p1 bright, p2 bright'
bd = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\0.2%, p1 bright, p2 dark'
db = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\0.2%, p1 dark, p2 bright'
dd = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\0.2%, p1 dark, p2 dark'


paths=[bb,bd,db,dd]

save_images(paths,'hello world')
