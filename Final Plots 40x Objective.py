import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(plt.rcParamsDefault)

import os
import tifffile
from PIL import Image
from itertools import combinations
import cv2
import matplotlib.font_manager as fm
import scienceplots
plt.style.use('no-latex')  # makes them LaTeX style
import collections

def flatten(l):  # flattens a 3d nested list
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


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
    all_selected_regions = []
    err_std = []
    err_sig = []
    err_back = []
    
    for i in range(len(rootdir)):  # iterating through bb, bd, db, dd
        images = []

        for file in sorted(os.listdir(rootdir[i])):  # lists all files in directory
            full_path = os.path.join(rootdir[i],
                                     file)  # create the full path by joining the directory path and the file name
            images.append(full_path)  # append the full path to appropriate list

        image_paths = images
        images = [tifffile.imread(path) for path in image_paths]
        selected_regions = [image[ymin:ymax, xmin:xmax] for image in images]
        stds = [np.std(region) for region in selected_regions]
        err_std.append(np.std(stds))# / np.sqrt(len(stds)))
        image_arrays = [np.array(img) for img in images]  # convert to arrays
        stacked_array = np.stack(image_arrays, axis=-1)  # stack images along a new axis
        average_array = np.mean(stacked_array, axis=-1, dtype=np.uint16)  # averaging pixel value for stacks, multiplied by 255 for the output image
        averages.append(average_array)
        all_selected_regions.append(selected_regions)
                
        if mask_rootdir != 0:
            
            signals = [image[signal[:, 0], signal[:, 1]] for image in images]
            backgrounds = [image[background[:, 0], background[:, 1]]  for image in images]

            signal_averages = [np.mean(sig) for sig in signals]
            background_averages = [np.mean(back) for back in backgrounds]

            err = np.std(signal_averages) #/ np.sqrt(len(signal_averages))
            err_sig.append(err)
            
            err = np.std(background_averages) #/ np.sqrt(len(background_averages))
            err_back.append(err)
            
        
    if mask_rootdir == 0:
        bb = Image.fromarray(np.uint8(averages[0]))
        bd = Image.fromarray(np.uint8(averages[1]))
        db = Image.fromarray(np.uint8(averages[2]))
        dd = Image.fromarray(np.uint8(averages[3]))
        bb.save(f'{title} bb average.tiff')
        bd.save(f'{title} bd average.tiff')
        db.save(f'{title} db average.tiff')
        dd.save(f'{title} dd average.tiff')
        return averages
    else:
        # COMPUTING S/B AND SC FOR AVERAGES
        avrg_vmax = max(list(flatten(averages)))
        avrg_vmin = min(list(flatten(averages)))
        avrg_background_vmax = max(list(flatten(all_selected_regions)))
        avrg_background_vmin = min(list(flatten(all_selected_regions)))

        sb_avrg = []
        sc_avrg = []
        I_signal = []
        I_background = []
        error_sb = []  # I used standard deviation error propagation eqn 2b https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
        error_sc = []
        for i in range(len(averages)):
            # S/B
            image = averages[i]
            background_values = image[
                background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual images
            signal_values = image[signal[:, 0], signal[:, 1]]

            signal_average = np.mean(signal_values)
            background_average = np.mean(background_values)
            sb = signal_average / background_average
            sb_avrg.append(sb)

            # error_sb.append(sb * np.sqrt((np.std(signal_values) / np.mean(signal_values)) ** 2 + (
            #             np.std(background_values) / np.mean(background_values)) ** 2))

            error_sb.append(sb * np.sqrt((err_sig[i]/signal_average ) ** 2 + (err_back [i]/background_average) ** 2))


            # SC
            selected_region = image[ymin:ymax, xmin:xmax]  # width is 200
            sc = np.std(selected_region) / np.mean(selected_region)
            sc_avrg.append(sc)

            error_sc.append(sc * np.sqrt((err_std[i] / np.std(selected_region)) ** 2 + (
                        np.std(selected_region) / np.mean(selected_region)) ** 2))

            # POLARISATION
            I_signal.append(signal_average)
            I_background.append(background_average)

        I_bb_signal, I_bd_signal, I_db_signal, I_dd_signal = I_signal
        I_bb_background, I_bd_background, I_db_background, I_dd_background = I_background

        err_bb_signal, err_bd_signal, err_db_signal, err_dd_signal = err_sig
        err_bb_background, err_bd_background, err_db_background, err_dd_background = err_back


        DOP_b_signal = (I_bb_signal - I_bd_signal) / (I_bb_signal + I_bd_signal)
        DOP_b_background = (I_bb_background - I_bd_background) / (I_bb_background + I_bd_background)
        DOP_v_signal = (I_db_signal - I_dd_signal) / (I_db_signal + I_dd_signal)
        DOP_v_background = (I_db_background - I_dd_background) / (I_db_background + I_dd_background)

        DOP_b_signal_err = DOP_b_signal * ((err_bb_signal + err_bd_signal) / (I_bb_signal + I_bd_signal) + (err_bb_signal + err_bd_signal) / (I_bb_signal - I_bd_signal))
        DOP_b_background_err = DOP_b_background * ((err_bb_background + err_bd_background) / (I_bb_background + I_bd_background) + (err_bb_background + err_bd_background) / (I_bb_background - I_bd_background))
        DOP_v_signal_err = DOP_v_signal * ((err_db_signal + err_dd_signal) / (I_db_signal + I_dd_signal) + (err_db_signal + err_dd_signal) / (I_db_signal - I_dd_signal))
        DOP_v_background_err = DOP_v_background * ((err_db_background + err_dd_background) / (I_db_background + I_dd_background) + (err_db_background + err_dd_background) / (I_db_background - I_dd_background))


        # USING AVERAGES TO DO SUBTRACTION COMBINATIONS, CALCULATE S/B AND SC
        array_names = ['HIHD', 'HIVD', 'VIHD', 'VIVD']  # for graph titles
        err_array_names_sb = ['err_hihd_sb', 'err_hivd_sb', 'err_vihd_sb', 'err_vivd_sb']
        err_array_names_sc = ['err_hihd_sc', 'err_hivd_sc', 'err_vihd_sc', 'err_vivd_sc']
        combinations_list = list(combinations(array_names, 2))  # all possible combinations of array names
        err_combinations_list_sb = list(combinations(err_array_names_sb, 2))
        err_combinations_list_sc = list(combinations(err_array_names_sc, 2))
        for i, name in enumerate(array_names):  # assigning the names of averages ie bb_avrg = averages[0]
            globals()[name] = np.array(averages[i])
        for i, name in enumerate(err_array_names_sb):  # assigning names to errors
            globals()[name] = np.array(error_sb[i])
        for i, name in enumerate(err_array_names_sc):  # assigning names to errors
            globals()[name] = np.array(error_sc[i])

        # FINAL AVERAGE SUBPLOT code is in a weird order bc I need array_names for subplot
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.03, wspace=0.1, bottom=0.2)
        # fig.suptitle('{} averages'.format(title), fontsize=16)
        selected_region = []
        for i in range(len(averages)):
            # selected_region.append(image[ymin:ymax, xmin:xmax])
            selected_region.append(averages[i][ymin:ymax, xmin:xmax])
            ax = axs.flatten()[i]
            im = ax.imshow(averages[i], cmap='viridis', vmin=avrg_vmin, vmax=avrg_vmax)
            ax.set_title(f'{array_names[i]}')
            ax.text(50, 220, f'S/B: {sb_avrg[i]:.2f}$\pm${error_sb[i]:.1g}\nSC: {sc_avrg[i]:.2f}$\pm${error_sc[i]:.1g}',
                    color='w', fontsize='x-large')
            if i == 0:  # adding rectangle on first plot
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none'))
                ax.add_patch(plt.Rectangle((1280-79-75, 75), 79, 2, linewidth=2, edgecolor='w', facecolor='none'))
                ax.text(1070, 160, f'$5 \mu m$', color='w', fontsize='x-large')
            ax.axis('off')

        colorbar_ax = fig.add_axes([0.125, 0.16, 0.775, 0.03])  # adjust positions as needed [0.2, 0.02, 0.6, 0.02]
        colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label('Pixel Value', fontsize='x-large')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{} averages.pdf'.format(title), bbox_inches='tight')
        plt.show()

        # FINAL AVERAGE BACKGROUND SUBPLOT
        fig, axs = plt.subplots(2, 2, figsize=(7, 8))
        fig.subplots_adjust(hspace=0.12, wspace=0.1, bottom=0.2)
        # fig.suptitle('{} averages background'.format(title), fontsize=16)
        for i in range(len(averages)):
            ax = axs.flatten()[i]
            im = ax.imshow(selected_region[i], cmap='viridis', vmin=avrg_background_vmin, vmax=avrg_background_vmax)
            ax.set_title(f'{array_names[i]}')
            ax.text(9, 42, f'S/B: {sb_avrg[i]:.2f}$\pm${error_sb[i]:.1g}\nSC: {sc_avrg[i]:.2f}$\pm${error_sc[i]:.1g}',
                    color='w', fontsize='x-large')
            ax.axis('off')
            if i == 0:
                ax.add_patch(plt.Rectangle((200 - 32 - 15, 15), 32, 2, linewidth=2, edgecolor='w', facecolor='none'))
                ax.text(154, 32, f'$2 \mu m$', color='w', fontsize='x-large')

        colorbar_ax = fig.add_axes([0.125, 0.14, 0.775, 0.03])  # adjust positions as needed [0.2, 0.02, 0.6, 0.02]
        colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label('Pixel Value', fontsize='x-large')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{} averages background.pdf'.format(title), bbox_inches='tight')
        plt.show()

        err_sub_sb = []  # propagating the error on sb from the averages
        err_sub_sc = []
        for i, names_comb in enumerate(err_combinations_list_sb):
            err1, err2 = globals()[names_comb[0]], globals()[names_comb[1]]
            err_sub_sb.append(np.sqrt(
                err1 ** 2 + err2 ** 2))  # using standard deviation error propagation eqn 1b https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
            err_sub_sb

        for i, names_comb in enumerate(err_combinations_list_sc):
            err1, err2 = globals()[names_comb[0]], globals()[names_comb[1]]
            err_sub_sc.append(np.sqrt(
                err1 ** 2 + err2 ** 2))  # using standard deviation error propagation eqn 1b https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
            err_sub_sc

        # SUBTRACTIONS SUBPLOT
        selected_region = []
        subtractions = []
        sb_sub_all = []
        sc_sub_all = []
        for i, names_comb in enumerate(combinations_list):
            array1, array2 = globals()[names_comb[0]], globals()[names_comb[1]]
            subtraction = cv2.subtract(np.array(array1), np.array(array2))
            subtractions.append(subtraction)

            selected_region.append(subtraction[ymin:ymax, xmin:xmax])

            # S/B
            background_values = subtraction[
                background[:, 0], background[:, 1]]  # finding intensity corresponding signal in the actual image
            signal_values = subtraction[signal[:, 0], signal[:, 1]]

            signal_average = np.mean(signal_values)
            sb_sub = signal_average / np.mean(background_values)
            sb_sub_all.append(sb_sub)

            # SC
            sc_sub = np.std(selected_region) / np.mean(selected_region)
            sc_sub_all.append(sc_sub)

        sub_vmin = min(list(flatten(subtractions)))
        sub_vmax = max(list(flatten(subtractions)))
        sub_background_vmin = min(list(flatten(selected_region)))
        sub_background_vmax = max(list(flatten(selected_region)))

        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        fig.subplots_adjust(hspace=0.14, wspace=0, bottom=0.2)
        # fig.suptitle('{} subtractions'.format(title), fontsize=16)
        for i, names_comb in enumerate(combinations_list):
            ax = axs.flatten()[i]
            im = ax.imshow(subtractions[i], cmap='viridis', vmin=sub_vmin, vmax=sub_vmax)
            ax.set_title(f'{names_comb[0]} - {names_comb[1]}')
            ax.text(50, 210, f'S/B: {sb_sub_all[i]:.2f}$\pm${err_sub_sb[i]:.1g}\nSC: {sc_sub_all[i]:.2f}$\pm${err_sub_sc[i]:.1g}', color='w', fontsize='x-large')
            if i == 0:  # adding rectangle on first plot
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none'))
                ax.add_patch(plt.Rectangle((1280 - 79 - 75, 75), 79, 2, linewidth=2, edgecolor='w', facecolor='none'))
                ax.text(1070, 160, f'$5 \mu m$', color='w', fontsize='x-large')
            ax.axis('off')

        colorbar_ax = fig.add_axes([0.14, 0.14, 0.745, 0.03])  # adjust positions as needed [0.2, 0.02, 0.6, 0.02]
        colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label('Pixel Value', fontsize='x-large')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{} subtractions.pdf'.format(title), bbox_inches='tight')
        plt.show()

        if individual_sub_plot == 1:
            bb_bd = Image.fromarray(np.uint8(subtractions[0]))
            bb_db = Image.fromarray(np.uint8(subtractions[1]))
            bb_dd = Image.fromarray(np.uint8(subtractions[2]))
            bd_db = Image.fromarray(np.uint8(subtractions[3]))
            bd_dd = Image.fromarray(np.uint8(subtractions[4]))
            db_dd = Image.fromarray(np.uint8(subtractions[5]))

            bb_bd.save(f'{title} bb_bd subtraction.tiff')
            bb_db.save(f'{title} bb_db subtraction.tiff')
            bb_dd.save(f'{title} bb_dd subtraction.tiff')
            bd_db.save(f'{title} bd_db subtraction.tiff')
            bd_dd.save(f'{title} bd_dd subtraction.tiff')
            db_dd.save(f'{title} db_dd subtraction.tiff')

        # FINAL SUBTRACTION BACKGROUND SUBPLOT
        fig, axs = plt.subplots(2, 3, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.2)
        # fig.suptitle('{} subtractions background'.format(title), fontsize=16)
        for i, names_comb in enumerate(combinations_list):
            ax = axs.flatten()[i]
            im = ax.imshow(selected_region[i], cmap='viridis', vmin=sub_background_vmin, vmax=sub_background_vmax)
            ax.set_title(f'{names_comb[0]} - {names_comb[1]}')
            ax.text(9, 42, f'S/B: {sb_sub_all[i]:.2f}$\pm${err_sub_sb[i]:.1g}\nSC: {sc_sub_all[i]:.2f}$\pm${err_sub_sc[i]:.1}', color='w', fontsize='x-large')
            ax.axis('off')
            if i == 0:
                ax.add_patch(plt.Rectangle((200 - 32 - 15, 15), 32, 2, linewidth=2, edgecolor='w', facecolor='none'))
                ax.text(154, 32, f'$2 \mu m$', color='w', fontsize='x-large')

        colorbar_ax = fig.add_axes([0.126, 0.14, 0.776, 0.03])  # adjust positions as needed [0.2, 0.02, 0.6, 0.02]
        colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label('Pixel Value', fontsize='x-large')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{} subtractions background.pdf'.format(title), bbox_inches='tight')
        plt.show()

        return sb_avrg, error_sb, sc_avrg, error_sc, sb_sub_all, err_sub_sb, sc_sub_all, err_sub_sc, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background, DOP_b_signal_err, DOP_b_background_err, DOP_v_signal_err, DOP_v_background_err

#%% DEM

bb=('/Users/rikke.ronnow/Downloads/DEM4, bb')
bd=('/Users/rikke.ronnow/Downloads/DEM4, bd')
db=('/Users/rikke.ronnow/Downloads/DEM4, db')
dd=('/Users/rikke.ronnow/Downloads/DEM4, dd')

paths = [bb, bd, db, dd]

mask =  '/Users/rikke.ronnow/Downloads/Mask of DEM4 bb average.tiff'

sb_avrg, error_sb, sc_avrg, error_sc, sb_sub_all, err_sub_sb, sc_sub_all, err_sub_sc, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background, DOP_b_signal_err, DOP_b_background_err, DOP_v_signal_err, DOP_v_background_err = everything(paths, mask,'DEM4')


# %% 0%

title = ['0 1', '0 2', '0 3', '0 4', '0 5', '0 expander 1', '0 expander 2', '0 expander 3', '0 expander 4', '0 D', '0 DM', '0 DEM1', '0 DEM2', '0 DEM3', '0 DEM4']
bb = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0%, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 2, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 3, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 4, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 5, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 2, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 3, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 4, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither more, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither even more, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM2, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM3, bb',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM4, bb']
bd = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0%, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 2, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 3, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 4, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 5, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 2, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 3, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 4, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither more, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither even more, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM2, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM3, bd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM4, bd']
db = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0%, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 2, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 3, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 4, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 5, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 2, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 3, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 4, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither more, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither even more, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM2, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM3, db',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM4, db']
dd = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0%, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 2, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 3, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 4, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Pollen 0% 5, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 2, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 3, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Expander 4, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither more, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Dither even more, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM2, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM3, dd',
      r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\DEM4, dd']
mask = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of Pollen Only bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of pollen only 2 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of pollen only 3 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of pollen only 4 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of pollen only 5 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of expander 1 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of expander 2 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of expander 3 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of expander 4 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of Dither bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of Dither more bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of DEM bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of DEM2 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of DEM3 bb average.tiff',
r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\#Masks\Mask of DEM4 bb average.tiff']

sb_avrgs = []
sb_avrgs_errs = []
sc_avrgs = []
sc_avrgs_errs = []
sb_subs = []
sb_subs_errs = []
sc_subs = []
sc_subs_errs = []
DOP_b_signals = []
DOP_b_backgrounds = []
DOP_v_signals = []
DOP_v_backgrounds = []
DOP_b_signal_errs = []
DOP_b_background_errs = []
DOP_v_signal_errs = []
DOP_v_background_errs = []

for i in range(len(title)):
    sb_avrg, error_sb, sc_avrg, error_sc, sb_sub_all, err_sub_sb, sc_sub_all, err_sub_sc, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background , DOP_b_signal_err, DOP_b_background_err, DOP_v_signal_err, DOP_v_background_err = everything(
        [bb[i], bd[i], db[i], dd[i]], mask[i], title[i])  # does save subtractions individually
    sb_avrgs.append(sb_avrg)
    sb_avrgs_errs.append(error_sb)
    sc_avrgs.append(sc_avrg)
    sc_avrgs_errs.append(error_sc)
    sb_subs.append(sb_sub_all)
    sb_subs_errs.append(err_sub_sb)
    sc_subs.append(sc_sub_all)
    sc_subs_errs.append(err_sub_sc)
    DOP_b_signals.append(DOP_b_signal)
    DOP_b_backgrounds.append(DOP_b_background)
    DOP_v_signals.append(DOP_v_signal)
    DOP_v_backgrounds.append(DOP_v_background)
    DOP_b_signal_errs.append(DOP_b_signal_err)
    DOP_b_background_errs.append(DOP_b_background_err)
    DOP_v_signal_errs.append(DOP_v_signal_err)
    DOP_v_background_errs.append(DOP_v_background_err)
