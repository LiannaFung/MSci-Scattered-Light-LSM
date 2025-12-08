from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile

from PIL import Image, ImageStat

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/p1 dark, p2 dark'
brightimages = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightimages.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/p1 dark, p2 bright'
darkimages = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkimages.append(full_path)


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

bright_array, bright_output_path = average(brightimages, 'avrg_bright')
dark_array, dark_output_path = average(darkimages, 'avrg_dark')

print(np.amax(bright_array))
print(np.amax(dark_array))

subtract_arrays(dark_array, bright_array,'dark subtraction')


# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/p1 bright, p2 bright, attempt 2'
brightimages = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightimages.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/p1 bright, p2 dark, attempt 2'
darkimages = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkimages.append(full_path)


bright_array, bright_output_path = average(brightimages, 'avrg_bright')
dark_array, dark_output_path = average(darkimages, 'avrg_dark')

print(np.amax(bright_array))
print(np.amax(dark_array))

subtract_arrays(dark_array, bright_array,'bright subtraction')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/pollen p2 bright, p1 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/pollen p2 bright, p1 dark'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'bb average')
db_array, dark_output_path = average(darkbright, 'db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'bright subtraction')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/pollen p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/pollen p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'bd average')
dd_array, dd_output_path = average(darkdark, 'dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'bd-dd')
subtract_arrays(db_array, bb_array,'bb-db')
subtract_arrays(dd_array, bb_array,'bb-dd')
subtract_arrays(bd_array, bb_array,'bb-bd')
subtract_arrays(db_array, bd_array,'bd-db')
subtract_arrays(bd_array, db_array,'db-bd')


# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/Pollen 2, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/Pollen 2, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'bb average')
db_array, dark_output_path = average(darkbright, 'db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'bright subtraction')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/Pollen 2, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/Pollen 2, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'bd average')
dd_array, dd_output_path = average(darkdark, 'dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

bd_sub_dd=subtract_arrays(dd_array, bd_array,'bd-dd')
bb_sub_db=subtract_arrays(db_array, bb_array,'bb-db')
bb_sub_dd=subtract_arrays(dd_array, bb_array,'bb-dd')
bb_sub_bd=subtract_arrays(bd_array, bb_array,'bb-bd')
bd_sub_db=subtract_arrays(db_array, bd_array,'bd-db')
db_sub_bd=subtract_arrays(bd_array, db_array,'db-bd')

#%%

# bb_sub_bd_sub_db=subtract_arrays(db_array*0.1, bb_sub_bd,'bb-bd-db')

bb_sub_bd_sub_dd=subtract_arrays(dd_array, bb_sub_bd,'bb-bd-dd')



# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/Our pollen, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/Our pollen, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'new bb average')
db_array, dark_output_path = average(darkbright, 'new db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'new bright subtraction')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/Our pollen, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/Our pollen, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'new bd average')
dd_array, dd_output_path = average(darkdark, 'new dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'new bd-dd')
subtract_arrays(db_array, bb_array,'new bb-db')
subtract_arrays(dd_array, bb_array,'new bb-dd')
subtract_arrays(bd_array, bb_array,'new bb-bd')
subtract_arrays(db_array, bd_array,'new bd-db')
subtract_arrays(bd_array, db_array,'new db-bd')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/Updated setup, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/Updated setup, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'updated bb average')
db_array, dark_output_path = average(darkbright, 'updated db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'updated bright subtraction')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/Updated setup, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/Updated setup, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'updated bd average')
dd_array, dd_output_path = average(darkdark, 'updated dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'updated bd-dd')
subtract_arrays(db_array, bb_array,'updated bb-db')
subtract_arrays(dd_array, bb_array,'updated bb-dd')
subtract_arrays(bd_array, bb_array,'updated bb-bd')
subtract_arrays(db_array, bd_array,'updated bd-db')
subtract_arrays(bd_array, db_array,'updated db-bd')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/Updated setup 2, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/Updated setup 2, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'updated 2 bb average')
db_array, dark_output_path = average(darkbright, 'updated 2 db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'updated 2 bright subtraction')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/Updated setup 2, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/Updated setup 2, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'updated 2 bd average')
dd_array, dd_output_path = average(darkdark, 'updated 2 dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'updated 2 bd-dd')
subtract_arrays(db_array, bb_array,'updated 2 bb-db')
subtract_arrays(dd_array, bb_array,'updated 2 bb-dd')
subtract_arrays(bd_array, bb_array,'updated 2 bb-bd')
subtract_arrays(db_array, bd_array,'updated 2 bd-db')
subtract_arrays(bd_array, db_array,'updated 2 db-bd')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/flipped, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/flipped, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'flipped bb average')
db_array, dark_output_path = average(darkbright, 'flipped db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'flipped bright subtraction')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/flipped, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/flipped, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'flipped bd average')
dd_array, dd_output_path = average(darkdark, 'flipped dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'flipped bd-dd')
subtract_arrays(db_array, bb_array,'flipped bb-db')
subtract_arrays(dd_array, bb_array,'flipped bb-dd')
subtract_arrays(bd_array, bb_array,'flipped bb-bd')
subtract_arrays(db_array, bd_array,'flipped bd-db')
subtract_arrays(bd_array, db_array,'flipped db-bd')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/latex pollen, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/latex pollen, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'latex bb average')
db_array, dark_output_path = average(darkbright, 'latex db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'latex bright subtraction')

# %% loading files

rootdir = '/Users/rikke.ronnow/Downloads/latex pollen, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/latex pollen, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'latex bd average')
dd_array, dd_output_path = average(darkdark, 'latex dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'latex bd-dd')
subtract_arrays(db_array, bb_array,'latex bb-db')
subtract_arrays(dd_array, bb_array,'latex bb-dd')
subtract_arrays(bd_array, bb_array,'latex bb-bd')
subtract_arrays(db_array, bd_array,'latex bd-db')
subtract_arrays(bd_array, db_array,'latex db-bd')

# %% loading 0.25 ml

rootdir = '/Users/rikke.ronnow/Downloads/0.25ml, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/0.25ml, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, '0.25ml bb average')
db_array, dark_output_path = average(darkbright, '0.25ml db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'0.25ml bright subtraction')

# %% subtracting 0.25 ml

rootdir = '/Users/rikke.ronnow/Downloads/0.25ml, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/0.25ml, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, '0.25ml bd average')
dd_array, dd_output_path = average(darkdark, '0.25ml dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'0.25ml bd-dd')
subtract_arrays(db_array, bb_array,'0.25ml bb-db')
subtract_arrays(dd_array, bb_array,'0.25ml bb-dd')
subtract_arrays(bd_array, bb_array,'0.25ml bb-bd')
subtract_arrays(db_array, bd_array,'0.25ml bd-db')
subtract_arrays(bd_array, db_array,'0.25ml db-bd')

# %% loading 0.5 ml

rootdir = '/Users/rikke.ronnow/Downloads/0.5ml, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/0.5ml, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, '0.5ml bb average')
db_array, dark_output_path = average(darkbright, '0.5ml db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'0.5ml bright subtraction')

# %% subtracting 0.5 ml

rootdir = '/Users/rikke.ronnow/Downloads/0.5ml, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/0.5ml, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, '0.5ml bd average')
dd_array, dd_output_path = average(darkdark, '0.5ml dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'0.5ml bd-dd')
subtract_arrays(db_array, bb_array,'0.5ml bb-db')
subtract_arrays(dd_array, bb_array,'0.5ml bb-dd')
subtract_arrays(bd_array, bb_array,'0.5ml bb-bd')
subtract_arrays(db_array, bd_array,'0.5ml bd-db')
subtract_arrays(dd_array, db_array,'0.5ml db-dd')

# %% loading 1 ml

rootdir = '/Users/rikke.ronnow/Downloads/1ml, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/1ml, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, '1ml bb average')
db_array, dark_output_path = average(darkbright, '1ml db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'1ml bright subtraction')

# %% subtracting 0.5 ml

rootdir = '/Users/rikke.ronnow/Downloads/1ml, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/1ml, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, '1ml bd average')
dd_array, dd_output_path = average(darkdark, '1ml dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'1ml bd-dd')
subtract_arrays(db_array, bb_array,'1ml bb-db')
subtract_arrays(dd_array, bb_array,'1ml bb-dd')
subtract_arrays(bd_array, bb_array,'1ml bb-bd')
subtract_arrays(db_array, bd_array,'1ml bd-db')
subtract_arrays(dd_array, db_array,'1ml db-dd')

#%% loading 0 ml

rootdir = '/Users/rikke.ronnow/Downloads/0ml, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/0ml, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, '0ml bb average')
db_array, dark_output_path = average(darkbright, '0ml db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'0ml bright subtraction')

#%% subtracting 0.5 ml

rootdir = '/Users/rikke.ronnow/Downloads/0ml, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/0ml, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, '0ml bd average')
dd_array, dd_output_path = average(darkdark, '0ml dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'0ml bd-dd')
subtract_arrays(db_array, bb_array,'0ml bb-db')
subtract_arrays(dd_array, bb_array,'0ml bb-dd')
subtract_arrays(bd_array, bb_array,'0ml bb-bd')
subtract_arrays(db_array, bd_array,'0ml bd-db')
subtract_arrays(dd_array, db_array,'0ml db-dd')

#%% loading 0 %

rootdir = '/Users/rikke.ronnow/Downloads/0%, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/0%, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, '0% bb average')
db_array, dark_output_path = average(darkbright, '0% db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'0% bright subtraction')

#%% subtracting 0 %

rootdir = '/Users/rikke.ronnow/Downloads/0%, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/0%, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, '0% bd average')
dd_array, dd_output_path = average(darkdark, '0% dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'0% bd-dd')
subtract_arrays(db_array, bb_array,'0% bb-db')
subtract_arrays(dd_array, bb_array,'0% bb-dd')
subtract_arrays(bd_array, bb_array,'0% bb-bd')
subtract_arrays(db_array, bd_array,'0% bd-db')
subtract_arrays(dd_array, db_array,'0% db-dd')

#%% loading 2%

rootdir = '/Users/rikke.ronnow/Downloads/2%, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/2%, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, '2% bb average')
db_array, dark_output_path = average(darkbright, '2% db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'2% bright subtraction')

#%% subtracting 2%

rootdir = '/Users/rikke.ronnow/Downloads/2%, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/2%, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, '2% bd average')
dd_array, dd_output_path = average(darkdark, '2% dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'2% bd-dd')
subtract_arrays(db_array, bb_array,'2% bb-db')
subtract_arrays(dd_array, bb_array,'2% bb-dd')
subtract_arrays(bd_array, bb_array,'2% bb-bd')
subtract_arrays(db_array, bd_array,'2% bd-db')
subtract_arrays(dd_array, db_array,'2% db-dd')

#%% loading 4%

rootdir = '/Users/rikke.ronnow/Downloads/4%, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/4%, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, '4% bb average')
db_array, dark_output_path = average(darkbright, '4% db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'4% bright subtraction')

#%% subtracting 4%

rootdir = '/Users/rikke.ronnow/Downloads/4%, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/4%, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, '4% bd average')
dd_array, dd_output_path = average(darkdark, '4% dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'4% bd-dd')
subtract_arrays(db_array, bb_array,'4% bb-db')
subtract_arrays(dd_array, bb_array,'4% bb-dd')
subtract_arrays(bd_array, bb_array,'4% bb-bd')
subtract_arrays(db_array, bd_array,'4% bd-db')
subtract_arrays(dd_array, db_array,'4% db-dd')

#%% loading 8%

rootdir = '/Users/rikke.ronnow/Downloads/8%, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/8%, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, '8% bb average')
db_array, dark_output_path = average(darkbright, '8% db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'8% bright subtraction')

#%% subtracting 8%

rootdir = '/Users/rikke.ronnow/Downloads/8%, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/8%, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, '8% bd average')
dd_array, dd_output_path = average(darkdark, '8% dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'8% bd-dd')
subtract_arrays(db_array, bb_array,'8% bb-db')
subtract_arrays(dd_array, bb_array,'8% bb-dd')
subtract_arrays(bd_array, bb_array,'8% bb-bd')
subtract_arrays(db_array, bd_array,'8% bd-db')
subtract_arrays(dd_array, db_array,'8% db-dd')

#%% loading 8%

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.2%, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.2%, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'TiO2 0.2% bb average')
db_array, dark_output_path = average(darkbright, 'TiO2 0.2% db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'TiO2 0.2% bright subtraction')

#%% subtracting 8%

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.2%, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.2%, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'TiO2 0.2% bd average')
dd_array, dd_output_path = average(darkdark, 'TiO2 0.2% dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'TiO2 0.2% bd-dd')
subtract_arrays(db_array, bb_array,'TiO2 0.2% bb-db')
subtract_arrays(dd_array, bb_array,'TiO2 0.2% bb-dd')
subtract_arrays(bd_array, bb_array,'TiO2 0.2% bb-bd')
subtract_arrays(db_array, bd_array,'TiO2 0.2% bd-db')
subtract_arrays(dd_array, db_array,'TiO2 0.2% db-dd')

#%% loading 8%

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.4%, p1 bright, p2 bright'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.4%, p1 dark, p2 bright'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'TiO2 0.4% bb average')
db_array, dark_output_path = average(darkbright, 'TiO2 0.4% db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'TiO2 0.4% bright subtraction')

#%% subtracting 8%

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.4%, p1 bright, p2 dark'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.4%, p1 dark, p2 dark'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'TiO2 0.4% bd average')
dd_array, dd_output_path = average(darkdark, 'TiO2 0.4% dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'TiO2 0.4% bd-dd')
subtract_arrays(db_array, bb_array,'TiO2 0.4% bb-db')
subtract_arrays(dd_array, bb_array,'TiO2 0.4% bb-dd')
subtract_arrays(bd_array, bb_array,'TiO2 0.4% bb-bd')
subtract_arrays(db_array, bd_array,'TiO2 0.4% bd-db')
subtract_arrays(dd_array, db_array,'TiO2 0.4% db-dd')

#%% loading 0.1%

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.1%, bb'
brightbright = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightbright.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.1%, db'
darkbright = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkbright.append(full_path)


bb_array, bb_output_path = average(brightbright, 'TiO2 0.1% bb average')
db_array, dark_output_path = average(darkbright, 'TiO2 0.1% db average')

print(np.amax(bb_array))
print(np.amax(db_array))

subtract_arrays(db_array, bb_array,'TiO2 0.1% bright subtraction')

#%% subtracting 8%

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.1%, bd'
brightdark = []

for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    brightdark.append(full_path)  # append the full path to appropriate list

rootdir = '/Users/rikke.ronnow/Downloads/TiO2 0.1%, dd'
darkdark = []

for file in sorted(os.listdir(rootdir)):  # similarly for dark images
    full_path = os.path.join(rootdir, file)
    darkdark.append(full_path)


bd_array, bd_output_path = average(brightdark, 'TiO2 0.1% bd average')
dd_array, dd_output_path = average(darkdark, 'TiO2 0.1% dd average')

print(np.amax(bd_array))
print(np.amax(dd_array))

subtract_arrays(dd_array, bd_array,'TiO2 0.1% bd-dd')
subtract_arrays(db_array, bb_array,'TiO2 0.1% bb-db')
subtract_arrays(dd_array, bb_array,'TiO2 0.1% bb-dd')
subtract_arrays(bd_array, bb_array,'TiO2 0.1% bb-bd')
subtract_arrays(db_array, bd_array,'TiO2 0.1% bd-db')
subtract_arrays(dd_array, db_array,'TiO2 0.1% db-dd')
