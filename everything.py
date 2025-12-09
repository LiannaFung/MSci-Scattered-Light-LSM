import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
from PIL import Image
from itertools import combinations
import cv2
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredSizeBar)
import matplotlib.font_manager as fm
import scienceplots
plt.style.use('science')  # makes them LaTeX style

def everything(rootdir, mask_rootdir, title, ymin=800, ymax=1000, xmin=1000, xmax=1200, individual_sub_plot=0):
    """INPUTS
    rootdir -- list of str, [bb, bd, db, dd] list of root directories
    mask_rootdir -- str or boolean, root directory of mask, if 0 = saves the averages as tiff files and returns
    title -- str, plot title name
    x and y, min max -- int, bounds of selected background region
    individual_sub_plot -- boolean, 1 = saves individual subtraction tiff files

    OUTPUTS
    sb_avrg, sc_avrg, sb_sub, sc_sub, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background as nested list
    if mask_rootdir == 0, then it only does the averages as nested list bb, bd, db, dd"""

    if mask_rootdir != 0:
        mask = tifffile.imread(mask_rootdir)
        background = np.argwhere(mask == 0)  # pixel values where there is background
        signal = np.argwhere(mask != 0)  # pixel values where there is signal

    # COMPUTING AVERAGES
    averages = []
    for i in range(len(rootdir)):  # iterating through bb, bd, db, dd
        images = []

        for file in sorted(os.listdir(rootdir[i])):  # lists all files in directory
            full_path = os.path.join(rootdir[i], file)  # create the full path by joining the directory path and the file name
            images.append(full_path)  # append the full path to appropriate list

        image_paths = images
        images = [tifffile.imread(path) for path in image_paths]
        image_arrays = [np.array(img) for img in images]  # convert to arrays
        stacked_array = np.stack(image_arrays, axis=-1)  # stack images along a new axis
        average_array = np.mean(stacked_array, axis=-1, dtype=np.uint16)  # averaging pixel value for stacks, multiplied by 255 for the output image
        averages.append(average_array)

    if mask_rootdir == 0:
        bb = Image.fromarray(np.uint8(averages[0] * 255))
        bd = Image.fromarray(np.uint8(averages[1] * 255))
        db = Image.fromarray(np.uint8(averages[2] * 255))
        dd = Image.fromarray(np.uint8(averages[3] * 255))
        bb.save(f'{title} bb average.tiff')
        bd.save(f'{title} bd average.tiff')
        db.save(f'{title} db average.tiff')
        dd.save(f'{title} dd average.tiff')
        return averages
    else:
        # COMPUTING S/B AND SC FOR AVERAGES
        sb_avrg = []
        sc_avrg = []
        I_signal = []
        I_background = []
        error_sb = []  # I used standard deviation error propagation eqn 2b https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
        for i in range(len(averages)):
            # S/B
            image = averages[i]
            background_values = image[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
            signal_values = image[signal[:, 0], signal[:, 1]]

            signal_average = np.mean(signal_values)
            background_average = np.mean(background_values)
            sb = signal_average / background_average
            sb_avrg.append(sb)

            error_sb.append(sb * np.sqrt( (np.std(signal_values)/np.mean(signal_values))**2 + (np.std(background_values)/np.mean(background_values))**2 ))

            # SC
            selected_region = image[ymin:ymax, xmin:xmax]  # width is 200
            sc_avrg.append(np.std(selected_region) / np.mean(selected_region))

            # POLARISATION
            I_signal.append(signal_average)
            I_background.append(background_average)

        I_bb_signal, I_bd_signal, I_db_signal, I_dd_signal = I_signal
        I_bb_background, I_bd_background, I_db_background, I_dd_background = I_background

        DOP_b_signal = (I_bb_signal - I_bd_signal) / (I_bb_signal + I_bd_signal)
        DOP_b_background = (I_bb_background - I_bd_background) / (I_bb_background + I_bd_background)
        DOP_v_signal = (I_db_signal - I_dd_signal) / (I_db_signal + I_dd_signal)
        DOP_v_background = (I_db_background - I_dd_background) / (I_db_background + I_dd_background)

        # USING AVERAGES TO DO SUBTRACTION COMBINATIONS, CALCULATE S/B AND SC
        array_names = ['HIHD', 'HIVD', 'VIHD', 'VIVD']  # for graph titles
        err_array_names = ['err_hihd', 'err_hivd', 'err_vihd', 'err_vivd']
        combinations_list = list(combinations(array_names, 2))  # all possible combinations of array names
        err_combinations_list = list(combinations(err_array_names, 2))
        for i, name in enumerate(array_names):  # assigning the names of averages ie bb_avrg = averages[0]
            globals()[name] = np.array(averages[i])
        for i, name in enumerate(err_array_names):  # assigning names to errors
            globals()[name] = np.array(error_sb[i])
        print('averages: number of rows in an image', len(averages[0]))
        print('averages: number of columns in an image', len(averages[0][0]))  # for scalebar and FOV calculation

        # FINAL AVERAGE SUBPLOT code is in a weird order bc I need array_names for subplot
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.12, wspace=0)
        #fig.suptitle('{} averages'.format(title), fontsize=16)
        selected_region = []
        for i in range(len(averages)):
            selected_region.append(image[ymin:ymax, xmin:xmax])
            ax = axs.flatten()[i]
            ax.imshow(averages[i], cmap='hot')
            ax.set_title(f'{array_names[i]}')
            ax.text(50, 215, f'S/B: {sb_avrg[i]:.2f}\nSC: {sc_avrg[i]:.2f}', color='w', fontsize='xx-large')
            fontprops = fm.FontProperties(size=14)  # changes scalebar font
            scalebar = AnchoredSizeBar(ax.transData, 103, r'25 $\mu$m', 'upper right', pad=1, color='white', frameon=False, size_vertical=3, fontproperties=fontprops)  # scale bar definitions
            if i == 0:  # adding rectangle on first plot
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none'))
                ax.add_artist(scalebar)
            ax.axis('off')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{} averages'.format(title), transparent=True, dpi=1000, bbox_inches='tight')
        plt.show()

        print('subtractions: number of rows in an image', len(selected_region[0]))
        print('subtractions: number of columns in an image', len(selected_region[0][0]))
        # FINAL AVERAGE BACKGROUND SUBPLOT
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.12, wspace=0)
        #fig.suptitle('{} averages background'.format(title), fontsize=16)
        scalebar = AnchoredSizeBar(ax.transData, 265, r'10 $\mu$m', 'upper right', pad=0.5, color='white', frameon=False, size_vertical=0.8, fontproperties=fontprops)
        for i in range(len(averages)):
            ax = axs.flatten()[i]
            ax.imshow(selected_region[i], cmap='hot')
            ax.set_title(f'{array_names[i]}')
            ax.text(9, 42, f'S/B: {sb_avrg[i]:.2f}\nSC: {sc_avrg[i]:.2f}', color='w', fontsize='xx-large')
            ax.axis('off')
            if i == 0:
                ax.add_artist(scalebar)

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{} averages background'.format(title), transparent=True, dpi=1000, bbox_inches='tight')
        plt.show()

        err_sub_sb = []  # propagating the error on sb from the averages
        for i, names_comb in enumerate(err_combinations_list):
            err1, err2 = globals()[names_comb[0]], globals()[names_comb[1]]
            err_sub_sb.append(np.sqrt(err1**2 + err2**2))  # using standard deviation error propagation eqn 1b https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html

        # SUBTRACTIONS SUBPLOT
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        fig.subplots_adjust(hspace=0.12, wspace=0)
        #fig.suptitle('{} subtractions'.format(title), fontsize=16)
        selected_region = []
        subtractions = []
        sb_sub_all = []
        sc_sub_all = []
        for i, names_comb in enumerate(combinations_list):
            ax = axs.flatten()[i]
            array1, array2 = globals()[names_comb[0]], globals()[names_comb[1]]
            subtraction = np.maximum(cv2.subtract(np.array(array1), np.array(array2)), 0)
            subtractions.append(subtraction)

            selected_region.append(subtraction[ymin:ymax, xmin:xmax])

            # S/B
            background_values = subtraction[background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual image
            signal_values = subtraction[signal[:, 0], signal[:, 1]]

            signal_average = np.mean(signal_values)
            sb_sub = signal_average / np.mean(background_values)
            sb_sub_all.append(sb_sub)

            # SC
            sc_sub = np.std(selected_region) / np.mean(selected_region)
            sc_sub_all.append(sc_sub)

            ax.imshow(subtractions[i], cmap='hot')
            ax.set_title(f'{names_comb[0]} - {names_comb[1]}')
            ax.text(50, 215, f'S/B: {sb_sub:.2f}$\pm${err_sub_sb[i]:.2f}\nSC: {sc_sub:.2f}', color='w', fontsize='xx-large')
            scalebar = AnchoredSizeBar(ax.transData, 103, r'25 $\mu$m', 'upper right', pad=1, color='white', frameon=False,                                       size_vertical=3, fontproperties=fontprops)  # scale bar definitions

            if i == 0:  # adding rectangle on first plot
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none'))
                ax.add_artist(scalebar)
            ax.axis('off')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{} subtractions'.format(title), transparent=True, dpi=1000, bbox_inches='tight')
        plt.show()

        if individual_sub_plot == 1:
            bb_bd = Image.fromarray(np.uint8(subtractions[0] * 255))
            bb_db = Image.fromarray(np.uint8(subtractions[1] * 255))
            bb_dd = Image.fromarray(np.uint8(subtractions[2] * 255))
            bd_db = Image.fromarray(np.uint8(subtractions[3] * 255))
            bd_dd = Image.fromarray(np.uint8(subtractions[4] * 255))
            db_dd = Image.fromarray(np.uint8(subtractions[5] * 255))
            bb_bd.save(f'{title} bb_bd subtraction.tiff')
            bb_db.save(f'{title} bb_db subtraction.tiff')
            bb_dd.save(f'{title} bb_dd subtraction.tiff')
            bd_db.save(f'{title} bd_db subtraction.tiff')
            bd_dd.save(f'{title} bd_dd subtraction.tiff')
            db_dd.save(f'{title} db_dd subtraction.tiff')

        # FINAL SUBTRACTION BACKGROUND SUBPLOT
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        fig.subplots_adjust(hspace=0.12, wspace=0)
        #fig.suptitle('{} subtractions background'.format(title), fontsize=16)
        scalebar = AnchoredSizeBar(ax.transData, 265, r'10 $\mu$m', 'upper right', pad=0.5, color='white', frameon=False, size_vertical=0.8, fontproperties=fontprops)
        for i in range(len(selected_region)):
            ax = axs.flatten()[i]
            ax.imshow(selected_region[i], cmap='hot')
            ax.set_title(f'{names_comb[0]} - {names_comb[1]}')
            ax.text(9, 42, f'S/B: {sb_sub:.2f}$\pm${err_sub_sb[i]:.2f}\nSC: {sc_sub_all[i]:.2f}', color='w', fontsize='xx-large')
            ax.axis('off')
            if i == 0:
                ax.add_artist(scalebar)

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{} subtractions background'.format(title), transparent=True, dpi=1000, bbox_inches='tight')
        plt.show()

        return sb_avrg, sc_avrg, sb_sub_all, sc_sub_all, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background

#%% FOR SINGLE PLOTS

title = 'Beam Expander 4'
bb = r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 4, bb'
bd = r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 4, bd'
db = r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 4, db'
dd = r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 4, dd'
mask = r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 0_0125% TiO2 bb average.tif'

averages = everything([bb, bd, db, dd], 0, title)  # does save averages individually
#sb_avrg, sc_avrg, sb_sub, sc_sub, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background = everything([bb, bd, db, dd], mask, title, individual_sub_plot=1)  # does save subtractions individually

#%% TIO2 0.0125, 0.025, 0.05

title = ['Beam Expander 1', 'Beam Expander 2', 'Beam Expander 3',  'Beam Expander 4']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander, bb', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 2, bb', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 3, bb', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 4, bb']
bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander, bd', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 2, bd', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 3, bd', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 4, bd']
db = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander, db', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 2, db', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 3, db', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 4, db']
dd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander, dd', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 2, dd', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 3, dd', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\Expander 4, dd']
mask = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of expander 1 bb average.tiff', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of expander 2 bb average.tiff', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of expander 2 bb average.tiff', r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of expander 4 bb average.tiff']

sb_avrgs = []
sc_avrgs = []
sb_subs = []
sc_subs = []
DOP_b_signals = []
DOP_b_backgrounds = []
DOP_v_signals = []
DOP_v_backgrounds = []
for i in range(len(title)):
    sb_avrg, sc_avrg, sb_sub, sc_sub, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background = everything([bb[i], bd[i], db[i], dd[i]], mask[i], title[i])  # does save subtractions individually
    sb_avrgs.append(sb_avrg)
    sc_avrgs.append(sc_avrg)
    sb_subs.append(sb_sub)
    sc_subs.append(sc_sub)
    DOP_b_signals.append(DOP_b_signal)
    DOP_b_backgrounds.append(DOP_b_background)
    DOP_v_signals.append(DOP_v_signal)
    DOP_v_backgrounds.append(DOP_v_background)

x = [0.0125, 0.025, 0.05]  # these are the concentrations

sb_avrgs_ = np.array(sb_avrgs).T.tolist()  # bb, bd, db, dd
plt.plot(x, sb_avrgs_[0], 'x', label='HIHO')
plt.plot(x, sb_avrgs_[1], 'x', label='HIVO')
plt.plot(x, sb_avrgs_[2], 'x', label='VIHO')
plt.plot(x, sb_avrgs_[3], 'x', label='VIVO')
plt.xlabel('Concentration (% mass)')
plt.ylabel('S/B')
plt.title('TiO2 Averages S/B')
plt.legend()
plt.show()

sc_avrgs_ = np.array(sc_avrgs).T.tolist()  # bb, bd, db, dd
plt.plot(x, sc_avrgs_[0], 'x', label='HIHO')
plt.plot(x, sc_avrgs_[1], 'x', label='HIVO')
plt.plot(x, sc_avrgs_[2], 'x', label='VIHO')
plt.plot(x, sc_avrgs_[3], 'x', label='VIVO')
plt.xlabel('Concentration (% mass)')
plt.ylabel('SC')
plt.title('TiO2 Averages SC')
plt.legend()
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
plt.plot(x, sb_subs_[0], 'x', label='HIHO - HIVO')
plt.plot(x, sb_subs_[1], 'x', label='HIHO - VIHO')
plt.plot(x, sb_subs_[2], 'x', label='HIHO - VIVO')
plt.plot(x, sb_subs_[3], 'x', label='HIVO - VIHO')
plt.plot(x, sb_subs_[4], 'x', label='HIVO - VIVO')
plt.plot(x, sb_subs_[5], 'x', label='VIHO - VIVO')
plt.xlabel('Concentration (% mass)')
plt.ylabel('S/B')
plt.title('TiO2 Subtractions S/B')
plt.legend()
plt.show()

sc_subs_ = np.array(sc_subs).T.tolist()  # bb, bd, db, dd
plt.plot(x, sc_subs_[0], 'x', label='HIHO - HIVO')
plt.plot(x, sc_subs_[1], 'x', label='HIHO - VIHO')
plt.plot(x, sc_subs_[2], 'x', label='HIHO - VIVO')
plt.plot(x, sc_subs_[3], 'x', label='HIVO - VIHO')
plt.plot(x, sc_subs_[4], 'x', label='HIVO - VIVO')
plt.plot(x, sc_subs_[5], 'x', label='VIHO - VIVO')
plt.xlabel('Concentration (% mass)')
plt.ylabel('S/B')
plt.title('TiO2 Subtractions S/B')
plt.legend()
plt.show()

plt.plot(x, DOP_b_signals, 'x', label='Signal')
plt.plot(x, DOP_b_backgrounds, 'x', label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
plt.title('TiO2 Horizontal Illumination Degree of Polarisation')
plt.legend()
plt.show()

plt.plot(x, DOP_v_signals, 'x', label='Signal')
plt.plot(x, DOP_v_backgrounds, 'x', label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
plt.title('TiO2 Vertical Illumination Degree of Polarisation')
plt.legend()
plt.show()