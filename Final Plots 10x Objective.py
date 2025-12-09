import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(plt.rcParamsDefault)

import os
import tifffile
from PIL import Image
from itertools import combinations
import cv2
import scienceplots
plt.style.use('no-latex')  # makes them LaTeX style
import collections
import pickle

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
    err_std2 = []  #TEST
    err_sig = []
    err_back = []

    for i in range(len(rootdir)):  # iterating through bb, bd, db, dd
        images = []

        for file in sorted(os.listdir(rootdir[i])):  # lists all files in directory
            if '.DS_Store' in file:
                continue
            full_path = os.path.join(rootdir[i], file)  # create the full path by joining the directory path and the file name
            images.append(full_path)  # append the full path to appropriate list
        image_paths = images
        images = [tifffile.imread(path) for path in image_paths]
        selected_regions = [image[ymin:ymax, xmin:xmax] for image in images]
        stds = [np.std(region) for region in selected_regions]
        err_std.append(np.std(stds))# / np.sqrt(len(stds)))
        err_std2.append(np.std(stds) / np.sqrt(len(stds)))
        image_arrays = [np.array(img) for img in images]  # convert to arrays
        stacked_array = np.stack(image_arrays, axis=-1)  # stack images along a new axis
        average_array = np.mean(stacked_array, axis=-1,
                                dtype=np.uint16)  # averaging pixel value for stacks, multiplied by 255 for the output image
        averages.append(average_array)
        all_selected_regions.append(selected_regions)

        if mask_rootdir != 0:
            signals = [image[signal[:, 0], signal[:, 1]] for image in images]
            backgrounds = [image[background[:, 0], background[:, 1]] for image in images]

            signal_averages = [np.mean(sig) for sig in signals]
            background_averages = [np.mean(back) for back in backgrounds]

            err = np.std(signal_averages)# / np.sqrt(len(signal_averages))
            err_sig.append(err)

            err = np.std(background_averages)# / np.sqrt(len(background_averages))
            err_back.append(err)

    if mask_rootdir == 0:
        bb = Image.fromarray(np.uint8(averages[0]))
        #bd = Image.fromarray(np.uint8(averages[1]))
        #db = Image.fromarray(np.uint8(averages[2]))
        #dd = Image.fromarray(np.uint8(averages[3]))
        bb.save(f'{title} bb average.tiff')
        #bd.save(f'{title} bd average.tiff')
        #db.save(f'{title} db average.tiff')
        #dd.save(f'{title} dd average.tiff')
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
            print('sig', signal_average)
            print('back', background_average)
            print('sb', sb)

            error_sb.append(sb * np.sqrt((err_sig[i]/signal_average) ** 2 + (err_back[i]/background_average) ** 2))
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
                ax.add_patch(plt.Rectangle((1280-98-75, 75), 98, 2, linewidth=2, edgecolor='w', facecolor='none'))
                ax.text(1050, 160, f'$25 \mu m$', color='w', fontsize='x-large')
            ax.axis('off')

        colorbar_ax = fig.add_axes([0.125, 0.16, 0.775, 0.03])  # adjust positions as needed [0.2, 0.02, 0.6, 0.02]
        colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label('Pixel Value', fontsize='x-large')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.savefig('{} averages.pdf'.format(title), bbox_inches='tight')
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
                ax.add_patch(plt.Rectangle((200 - 39 - 15, 15), 39, 2, linewidth=2, edgecolor='w', facecolor='none'))
                ax.text(145, 32, f'$10 \mu m$', color='w', fontsize='x-large')

        colorbar_ax = fig.add_axes([0.125, 0.14, 0.775, 0.03])  # adjust positions as needed [0.2, 0.02, 0.6, 0.02]
        colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label('Pixel Value', fontsize='x-large')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.savefig('{} averages background.pdf'.format(title), bbox_inches='tight')
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
                ax.add_patch(plt.Rectangle((1280 - 98 - 75, 75), 98, 2, linewidth=2, edgecolor='w', facecolor='none'))
                ax.text(1050, 160, f'$25 \mu m$', color='w', fontsize='x-large')
            ax.axis('off')

        colorbar_ax = fig.add_axes([0.14, 0.14, 0.745, 0.03])  # adjust positions as needed [0.2, 0.02, 0.6, 0.02]
        colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label('Pixel Value', fontsize='x-large')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.savefig('{} subtractions.pdf'.format(title), bbox_inches='tight')
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
            ax.text(9, 42, f'S/B: {sb_sub_all[i]:.2f}$\pm${err_sub_sb[i]:.1g}\nSC: {sc_sub_all[i]:.2f}$\pm${err_sub_sc[i]:.1g}', color='w', fontsize='x-large')
            ax.axis('off')
            if i == 0:
                ax.add_patch(plt.Rectangle((200 - 39 - 15, 15), 39, 2, linewidth=2, edgecolor='w', facecolor='none'))
                ax.text(145, 32, f'$10 \mu m$', color='w', fontsize='x-large')

        colorbar_ax = fig.add_axes([0.126, 0.14, 0.776, 0.03])  # adjust positions as needed [0.2, 0.02, 0.6, 0.02]
        colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label('Pixel Value', fontsize='x-large')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.savefig('{} subtractions background.pdf'.format(title), bbox_inches='tight')
    plt.show()

    return sb_avrg, error_sb, sc_avrg, error_sc, sb_sub_all, err_sub_sb, sc_sub_all, err_sub_sc, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background, DOP_b_signal_err, DOP_b_background_err, DOP_v_signal_err, DOP_v_background_err

#%%

def fish(rootdir, title):
    """INPUTS
    rootdir -- list of str, [bb, bd] list of root directories
    title -- str, plot title name"""

    # COMPUTING AVERAGES
    averages = []
    plot_images = []
    all_images = []
    for i in range(len(rootdir)):  # iterating through bb, bd, db, dd
        images = []
        for file in sorted(os.listdir(rootdir[i])):  # lists all files in directory
            if '.DS_Store' in file:
                continue
            full_path = os.path.join(rootdir[i],
                                     file)  # create the full path by joining the directory path and the file name
            images.append(full_path)  # append the full path to appropriate list
        image_paths = images
        images = [tifffile.imread(path) for path in image_paths]
        all_images.append(images)
        image_arrays = [np.array(img) for img in images]  # convert to arrays
        stacked_array = np.stack(image_arrays, axis=-1)  # stack images along a new axis
        average_array = np.mean(stacked_array, axis=-1,
                                dtype=np.uint16)  # averaging pixel value for stacks, multiplied by 255 for the output image
        averages.append(average_array)

    else:
        avrg_vmax = max(list(flatten(averages)))
        avrg_vmin = min(list(flatten(averages)))
        print('avrg', avrg_vmax, avrg_vmin)

        bb = averages[0]
        bd = averages[1]

        plot_images.append(bb)
        plot_images.append(bd)
        plot_images.append(all_images[0][0])

        # SUBTRACTIONS SUBPLOT
        subtraction = cv2.subtract(np.array(bb), np.array(bd))
        plot_images.append(subtraction)
        sub_vmax = max(list(flatten(subtraction)))
        sub_vmin = min(list(flatten(subtraction)))
        print('sub', sub_vmax, sub_vmin)

        fig, axs =  plt.subplots(2, 2, figsize=(8, 8))
        fig.subplots_adjust(hspace=-0.03, wspace=0.1, bottom=0.2)
        # fig.suptitle('{} subtractions'.format(title), fontsize=16)
        for i, names_comb in enumerate(plot_images):
            ax = axs.flatten()[i]
            ax.imshow(plot_images[i], cmap='viridis', vmin=avrg_vmin, vmax=avrg_vmax)
            if i == 0:
                im = ax.imshow(plot_images[i], cmap='viridis', vmin=avrg_vmin, vmax=avrg_vmax)  # to set the colorbar
            axs.flatten()[0].set_title('HIHD')
            axs.flatten()[1].set_title('HIVD')
            axs.flatten()[2].set_title('HIHD Raw')
            axs.flatten()[3].set_title('HIHD - HIVD')
            if i == 0:  # adding rectangle on first plot
                ax.add_patch(plt.Rectangle((1280 - 98 - 75, 75), 98, 2, linewidth=2, edgecolor='w', facecolor='none'))
                ax.text(1050, 160, '$25 \mu m$', color='w', fontsize='x-large')
            ax.axis('off')

        colorbar_ax = fig.add_axes([0.125, 0.16, 0.775, 0.03])  # adjust positions as needed [0.2, 0.02, 0.6, 0.02]
        colorbar = fig.colorbar(im, cax=colorbar_ax, orientation='horizontal')
        colorbar.set_label('Pixel Value', fontsize='x-large')

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('{} comparison.pdf'.format(title), bbox_inches='tight')
        plt.show()

        return

#%% mask combination

rootdir=r'C:\Users\liann\Documents\GitHub\MSci-Project\Masks\d tio2 2 redone'
images=[]
for file in sorted(os.listdir(rootdir)):  # lists all files in directory
    if '.DS_Store' in file:
        continue
    full_path = os.path.join(rootdir, file)  # create the full path by joining the directory path and the file name
    images.append(full_path)  # append the full path to appropriate list
image_paths = images
images = [tifffile.imread(path) for path in image_paths]
mask=np.zeros([1024,1280])
for image in images:
    mask+=image

mask[(mask>0)]=1
mask[(mask==0)]=255
mask[(mask==1)]=0

mask_im=Image.fromarray(mask)
mask_im.save(r'C:\Users\liann\Documents\GitHub\MSci-Project\Masks\Mask of d TiO2 2 redone bb average.tiff')

# %% rikke fish

bb=('/Users/rikke.ronnow/Downloads/3hr fish .41, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .41, bd')
paths = [bb, bd]
fish(paths,'3hr fish 41')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .42, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .42, bd')
paths = [bb, bd]
fish(paths,'3hr fish 42')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .43, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .43, bd')
paths = [bb, bd]
fish(paths,'3hr fish 43')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .44, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .44, bd')
paths = [bb, bd]
fish(paths,'3hr fish 44')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .45, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .45, bd')
paths = [bb, bd]
fish(paths,'3hr fish 45')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .46, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .46, bd')
paths = [bb, bd]
fish(paths,'3hr fish 46')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .47, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .47, bd')
paths = [bb, bd]
fish(paths,'3hr fish 47')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .48, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .48, bd')
paths = [bb, bd]
fish(paths,'3hr fish 48')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .49, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .49, bd')
paths = [bb, bd]
fish(paths,'3hr fish 49')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .50, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .50, bd')
paths = [bb, bd]
fish(paths,'3hr fish 50')

bb=('/Users/rikke.ronnow/Downloads/3hr fish .51, bb')
bd=('/Users/rikke.ronnow/Downloads/3hr fish .51, bd')
paths = [bb, bd]
fish(paths,'3hr fish 51')

bb=('/Users/rikke.ronnow/Downloads/fish head .6, bb')
bd=('/Users/rikke.ronnow/Downloads/fish head .6, bd')
paths = [bb, bd]
fish(paths,'fish head 6')

bb=('/Users/rikke.ronnow/Downloads/fish head .5, bb')
bd=('/Users/rikke.ronnow/Downloads/fish head .5, bd')
paths = [bb, bd]
fish(paths,'fish head 5')

bb=('/Users/rikke.ronnow/Downloads/fish head .4, bb')
bd=('/Users/rikke.ronnow/Downloads/fish head .4, bd')
paths = [bb, bd]
fish(paths,'fish head 4')

bb=('/Users/rikke.ronnow/Downloads/fish head .3, bb')
bd=('/Users/rikke.ronnow/Downloads/fish head .3, bd')
paths = [bb, bd]
fish(paths,'fish head 3')

bb=('/Users/rikke.ronnow/Downloads/fish head .2, bb')
bd=('/Users/rikke.ronnow/Downloads/fish head .2, bd')
paths = [bb, bd]
fish(paths,'fish head 2')

bb=('/Users/rikke.ronnow/Downloads/fish head .1, bb')
bd=('/Users/rikke.ronnow/Downloads/fish head .1, bd')
paths = [bb, bd]
fish(paths,'fish head 1')

bb=('/Users/rikke.ronnow/Downloads/fish head .0, bb')
bd=('/Users/rikke.ronnow/Downloads/fish head .0, bd')
paths = [bb, bd]
fish(paths,'fish head ')

bb=('/Users/rikke.ronnow/Downloads/fish head -.1, bb')
bd=('/Users/rikke.ronnow/Downloads/fish head -.1, bd')
paths = [bb, bd]
fish(paths,'fish head -1')

bb=('/Users/rikke.ronnow/Downloads/fish body -.1, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body -.1, bd')
paths = [bb, bd]
fish(paths,'fish body -1')

bb=('/Users/rikke.ronnow/Downloads/fish body .0, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body .0, bd')
paths = [bb, bd]
fish(paths,'fish body 0')

bb=('/Users/rikke.ronnow/Downloads/fish body .1, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body .1, bd')
paths = [bb, bd]
fish(paths,'fish body 1')

bb=('/Users/rikke.ronnow/Downloads/fish body .2, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body .2, bd')
paths = [bb, bd]
fish(paths,'fish body 2')

bb=('/Users/rikke.ronnow/Downloads/fish body .3, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body .3, bd')
paths = [bb, bd]
fish(paths,'fish body 3')

bb=('/Users/rikke.ronnow/Downloads/fish body .4, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body .4, bd')
paths = [bb, bd]
fish(paths,'fish body 4')

bb=('/Users/rikke.ronnow/Downloads/fish body .5, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body .5, bd')
paths = [bb, bd]
fish(paths,'fish body 5')

bb=('/Users/rikke.ronnow/Downloads/fish body .6, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body .6, bd')
paths = [bb, bd]
fish(paths,'fish body 6')

bb=('/Users/rikke.ronnow/Downloads/fish body .7, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body .7, bd')
paths = [bb, bd]
fish(paths,'fish body 7')

bb=('/Users/rikke.ronnow/Downloads/fish body .8, bb')
bd=('/Users/rikke.ronnow/Downloads/fish body .8, bd')
paths = [bb, bd]
fish(paths,'fish body 8')


#%% lianna TiO2

title = ['TiO2 0_0125', 'TiO2 0_0250', 'TiO2 0_0500', 'TiO2 0_1000', 'TiO2 0_2000', 'TiO2 0_4000', 'TiO2 0_01 Dithered', 'TiO2 0_01 Dithered 2']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.0125%, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.025%, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.05%, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.1%, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.2%, p1 bright, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.4%, p1 bright, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01%, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01% 2, bb']
bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.0125%, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.025%, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.05%, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.1%, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.2%, p1 bright, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.4%, p1 bright, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01%, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01% 2, bd']
db = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.0125%, db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.025%, db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.05%, db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.1%, db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.2%, p1 dark, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.4%, p1 dark, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01%, db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01% 2, db']
dd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.0125%, dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.025%, dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.05%, dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.1%, dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.2%, p1 dark, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.4%, p1 dark, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01%, dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01% 2, dd']
mask = [
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 0_0125% TiO2 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 0_025% TiO2 bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 0_05% TiO2 bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of TiO2 0.1% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of TiO2 0.2% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of TiO2 0.4% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of TiO2 0.01% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of TiO2 0.01% 2 bb average.tiff']

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

x = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.01, 0.01]  # these are the concentrations, the last two are dithered

nested_lists = [sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs]
title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
# Save to a file
for i in range(len(title)):
    with open('tio2_{}.pkl'.format(title[i]), 'wb') as f:
        pickle.dump(np.array(nested_lists[i]).T.tolist(), f)  # formatted as nested list at concentrations

sb_avrgs_ = np.array(sb_avrgs).T.tolist()  # bb, bd, db, dd
sb_avrgs_errs_ = np.array(sb_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_avrgs_[1], yerr=sb_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sb_avrgs_[2], yerr=sb_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sb_avrgs_[3], yerr=sb_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('S/B')
plt.title(f'TiO$_2$ Averages S/B')
plt.legend()
plt.savefig('TiO2 Averages SB.pdf', bbox_inches='tight')
plt.show()

sc_avrgs_ = np.array(sc_avrgs).T.tolist()  # bb, bd, db, dd
sc_avrgs_errs_ = np.array(sc_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sc_avrgs_[1], yerr=sc_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sc_avrgs_[2], yerr=sc_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sc_avrgs_[3], yerr=sc_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('SC')
plt.title(f'TiO$_2$ Averages SC')
plt.legend()
plt.savefig('TiO2 Averages SC.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
#plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('S/B')
plt.title(f'TiO$_2$ Subtractions S/B')
plt.legend()
plt.savefig('TiO2 Subtractions SB, HIHD vs HIHD-HIVD.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('S/B')
plt.title(f'TiO$_2$ Subtractions S/B')
plt.legend()
plt.savefig('TiO2 Subtractions SB, HIHD vs HIHD-VIHD vs HIHD-VIHD.pdf', bbox_inches='tight')
plt.show()

sc_subs_ = np.array(sc_subs).T.tolist()  # bb, bd, db, dd
sc_subs_errs_ = np.array(sc_subs_errs).T.tolist()
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
#plt.errorbar(x, sc_subs_[0], yerr=sc_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sc_subs_[1], yerr=sc_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sc_subs_[2], yerr=sc_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sc_subs_[3], yerr=sc_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sc_subs_[4], yerr=sc_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sc_subs_[5], yerr=sc_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('SC')
plt.title(f'TiO$_2$ Subtractions SC')
plt.legend()
plt.savefig('TiO2 Subtractions SC, HIHD vs HIHD-VIHD.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_b_signals, yerr=DOP_b_signal_errs,fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_b_backgrounds, yerr=DOP_b_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
plt.title(f'TiO$_2$ Horizontal Illumination Degree of Polarisation')
plt.legend()
plt.savefig('TiO2 Horizontal Illumination Degree of Polarisation.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_v_signals, yerr=DOP_v_signal_errs, fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_v_backgrounds, yerr=DOP_v_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
plt.title(f'TiO$_2$ Vertical Illumination Degree of Polarisation')
plt.legend()
plt.savefig('TiO2 Vertical Illumination Degree of Polarisation.pdf', bbox_inches='tight')
plt.show()

# %% lianna latex

title = ['Latex 0_2', 'latex 0_4', 'latex 0_8', 'latex 2', 'latex 4', 'latex 8']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.2%, p1 bright, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.4%, p1 bright, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.8%, p1 bright, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\2%, p1 bright, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\4%, p1 bright, p2 bright, attempt 2',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\8%, p1 bright, p2 bright, attempt 2']
bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.2%, p1 bright, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.4%, p1 bright, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.8%, p1 bright, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\2%, p1 bright, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\4%, p1 bright, p2 dark, attempt 2',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\8%, p1 bright, p2 dark, attempt 2']
db = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.2%, p1 dark, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.4%, p1 dark, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.8%, p1 dark, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\2%, p1 dark, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\4%, p1 dark, p2 bright, attempt 2',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\8%, p1 dark, p2 bright, attempt 2']
dd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.2%, p1 dark, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.4%, p1 dark, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0.8%, p1 dark, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\2%, p1 dark, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\4%, p1 dark, p2 dark, attempt 2',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\8%, p1 dark, p2 dark, attempt 2']
mask = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 0.2% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 0.4% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 0.8% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 2% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 4% bb average attempt 2.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 8% bb average attempt 2.tiff']

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
    sb_avrg, error_sb, sc_avrg, error_sc, sb_sub_all, err_sub_sb, sc_sub_all, err_sub_sc, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background, DOP_b_signal_err, DOP_b_background_err, DOP_v_signal_err, DOP_v_background_err = everything(
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

x = [0.2, 0.4, 0.8, 2, 4, 8]  # these are the concentrations

nested_lists = [sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs]
title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
# Save to a file
for i in range(len(title)):
    with open('latex_{}.pkl'.format(title[i]), 'wb') as f:
        pickle.dump(np.array(nested_lists[i]).T.tolist(), f)  # formatted as nested list at concentrations

sb_avrgs_ = np.array(sb_avrgs).T.tolist()  # bb, bd, db, dd
sb_avrgs_errs_ = np.array(sb_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_avrgs_[1], yerr=sb_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sb_avrgs_[2], yerr=sb_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sb_avrgs_[3], yerr=sb_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('S/B')
plt.title('Latex Averages S/B')
plt.legend()
plt.savefig('Latex Averages SB.pdf', bbox_inches='tight')
plt.show()

sc_avrgs_ = np.array(sc_avrgs).T.tolist()  # bb, bd, db, dd
sc_avrgs_errs_ = np.array(sc_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sc_avrgs_[1], yerr=sc_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sc_avrgs_[2], yerr=sc_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sc_avrgs_[3], yerr=sc_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('SC')
plt.title('Latex Averages SC')
plt.legend()
plt.savefig('Latex Averages SC.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('S/B')
plt.title('Latex Subtractions S/B')
plt.legend()
plt.savefig('Latex Subtractions SB, HIHD vs HIHD-HIVD vs HIHD-VIHD.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
#plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
#plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('S/B')
plt.title('Latex Subtractions S/B')
plt.legend()
plt.savefig('Latex Subtractions SB, HIHD vs HIVD-VIHD vs HIVD-VIVD.pdf', bbox_inches='tight')
plt.show()

sc_subs_ = np.array(sc_subs).T.tolist()  # bb, bd, db, dd
sc_subs_errs_ = np.array(sc_subs_errs).T.tolist()
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sc_subs_[0], yerr=sc_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
#plt.errorbar(x, sc_subs_[1], yerr=sc_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sc_subs_[2], yerr=sc_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sc_subs_[3], yerr=sc_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sc_subs_[4], yerr=sc_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sc_subs_[5], yerr=sc_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('SC')
plt.title('Latex Subtractions SC')
plt.legend()
plt.savefig('Latex Subtractions SC, HIHD vs HIHD-HIVD.pdf', bbox_inches='tight')
plt.show()

sc_subs_ = np.array(sc_subs).T.tolist()  # bb, bd, db, dd
sc_subs_errs_ = np.array(sc_subs_errs).T.tolist()
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
#plt.errorbar(x, sc_subs_[0], yerr=sc_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sc_subs_[1], yerr=sc_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sc_subs_[2], yerr=sc_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sc_subs_[3], yerr=sc_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sc_subs_[4], yerr=sc_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sc_subs_[5], yerr=sc_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (% mass)')
plt.ylabel('SC')
plt.title('Latex Subtractions SC')
plt.legend()
plt.savefig('Latex Subtractions SC, HIHD vs HIHD-VIHD.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_b_signals, yerr=DOP_b_signal_errs,fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_b_backgrounds, yerr=DOP_b_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
plt.title('Latex Horizontal Illumination Degree of Polarisation')
plt.legend()
plt.savefig('Latex Horizontal Illumination Degree of Polarisation.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_v_signals, yerr=DOP_v_signal_errs, fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_v_backgrounds, yerr=DOP_v_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
plt.title('Latex Vertical Illumination Degree of Polarisation')
plt.legend()
plt.savefig('Latex Vertical Illumination Degree of Polarisation.pdf', bbox_inches='tight')
plt.show()

#%% lianna 0%

title = ['0', '0 attempt 2']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0%, p1 bright, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0%, p1 bright, p2 bright, attempt 2']
bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0%, p1 bright, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0%, p1 bright, p2 dark, attempt 2']
db = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0%, p1 dark, p2 bright',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0%, p1 dark, p2 bright, attempt 2']
dd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0%, p1 dark, p2 dark',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\0%, p1 dark, p2 dark, attempt 2']
mask = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 0% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of 0% bb average attempt 2.tiff']

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
    sb_avrg, error_sb, sc_avrg, error_sc, sb_sub_all, err_sub_sb, sc_sub_all, err_sub_sc, DOP_b_signal, DOP_b_background, DOP_v_signal, DOP_v_background, DOP_b_signal_err, DOP_b_background_err, DOP_v_signal_err, DOP_v_background_err = everything(
        [bb[i], bd[i], db[i], dd[i]], mask[i], title[i], ymin=400, ymax=600)  # does save subtractions individually
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
    
#%% rikke fish 2

rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .41, bb','/Users/rikke.ronnow/Downloads/3hr fish .41, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .42, bb','/Users/rikke.ronnow/Downloads/3hr fish .42, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .43, bb','/Users/rikke.ronnow/Downloads/3hr fish .43, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .44, bb','/Users/rikke.ronnow/Downloads/3hr fish .44, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .45, bb','/Users/rikke.ronnow/Downloads/3hr fish .45, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .46, bb','/Users/rikke.ronnow/Downloads/3hr fish .46, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .47, bb','/Users/rikke.ronnow/Downloads/3hr fish .47, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .48, bb','/Users/rikke.ronnow/Downloads/3hr fish .48, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .49, bb','/Users/rikke.ronnow/Downloads/3hr fish .49, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .50, bb','/Users/rikke.ronnow/Downloads/3hr fish .50, bd']
rootdirs=['/Users/rikke.ronnow/Downloads/3hr fish .51, bb','/Users/rikke.ronnow/Downloads/3hr fish .51, bd']

#%% lianna TiO2 dithered

title = ['TiO2 0_01 Dithered', 'TiO2 0_01 Dithered 2']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01%, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01% 2, bb']
bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01%, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01% 2, bd']
db = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01%, db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01% 2, db']
dd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01%, dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\TiO2 0.01% 2, dd']
mask = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of TiO2 0.01% bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\#Masks\Mask of TiO2 0.01% 2 bb average.tiff']

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

#%% lianna 3hr fish

title = ['3hr _41', '3hr _42', '3hr _43', '3hr _44', '3hr_45', '3hr_46', '3hr _47', '3hr_48', '3hr _49', '3hr _50', '3hr _51']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .41, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .42, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .43, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .44, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .45, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .46, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .47, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .48, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .49, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .50, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .51, bb']
bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .41, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .42, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .43, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .44, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .45, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .46, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .47, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .48, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .49, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .50, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\3hr fish .51, bd']

for i in range(len(title)):
     fish([bb[i], bd[i]], title[i])

#%% fish head

title = ['head -_1', 'head _0', 'head _1', 'head _2', 'head _3', 'head _4', 'head _5', 'head _6']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head -.1, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .0, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .1, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .2, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .3, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .4, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .5, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .6, bb']
bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head -.1, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .0, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .1, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .2, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .3, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .4, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .5, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish head .6, bd']

for i in range(len(title)):
    fish([bb[i], bd[i]], title[i])

#%% fish body

title = ['body -_1', 'body _0', 'body _1', 'body _2', 'body _3', 'body _4', 'body _5', 'body _6', 'body _7', 'body _8']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body -.1, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .0, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .1, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .2, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .3, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .4, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .5, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .6, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .7, bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .8, bb']
bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body -.1, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .0, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .1, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .2, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .3, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .4, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .5, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .6, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .7, bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\fish body .8, bd']

for i in range(len(title)):
    fish([bb[i], bd[i]], title[i])

#%% REDONE TiO2

title = ['TiO2 0_125', 'TiO2 0_25', 'TiO2 0_5', 'TiO2 1', 'TiO2 2']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.125 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.25 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.5 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 1 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 2 bb']

bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.125 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.25 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.5 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 1 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 2 bd']

db = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.125 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.25 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.5 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 1 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 2 db']

dd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.125 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.25 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 0.5 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 1 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\TiO2 2 dd']
mask = [
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 0.125 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 0.25 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 0.5 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 1 bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 2 bb average AMENDED.tif']

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

x = [0.125, 0.25, 0.5, 1, 2]  # these are the concentrations, the last two are dithered

nested_lists = [sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs]
title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
# Save to a file
for i in range(len(title)):
    with open('tio2_{}.pkl'.format(title[i]), 'wb') as f:
        pickle.dump(np.array(nested_lists[i]).T.tolist(), f)  # formatted as nested list at concentrations


sb_avrgs_ = np.array(sb_avrgs).T.tolist()  # bb, bd, db, dd
sb_avrgs_errs_ = np.array(sb_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_avrgs_[1], yerr=sb_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sb_avrgs_[2], yerr=sb_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sb_avrgs_[3], yerr=sb_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (ml/mg)')
plt.ylabel('S/B')
plt.title(f'TiO$_2$ Averages S/B')
plt.legend()
plt.savefig('TiO2 Averages SB REDONE.pdf', bbox_inches='tight')
plt.show()

sc_avrgs_ = np.array(sc_avrgs).T.tolist()  # bb, bd, db, dd
sc_avrgs_errs_ = np.array(sc_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sc_avrgs_[1], yerr=sc_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sc_avrgs_[2], yerr=sc_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sc_avrgs_[3], yerr=sc_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (ml/mg)')
plt.ylabel('SC')
plt.title(f'TiO$_2$ Averages SC')
plt.legend()
plt.savefig('TiO2 Averages SC REDONE.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
#plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (ml/mg)')
plt.ylabel('S/B')
plt.title(f'TiO$_2$ Subtractions S/B')
plt.legend()
plt.savefig('TiO2 Subtractions SB, HIHD vs HIHD-HIVD REDONE.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (ml/mg)')
plt.ylabel('S/B')
plt.title(f'TiO$_2$ Subtractions S/B')
plt.legend()
plt.savefig('TiO2 Subtractions SB, HIHD vs HIHD-VIHD vs HIHD-VIHD REDONE.pdf', bbox_inches='tight')
plt.show()

sc_subs_ = np.array(sc_subs).T.tolist()  # bb, bd, db, dd
sc_subs_errs_ = np.array(sc_subs_errs).T.tolist()
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
#plt.errorbar(x, sc_subs_[0], yerr=sc_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sc_subs_[1], yerr=sc_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sc_subs_[2], yerr=sc_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sc_subs_[3], yerr=sc_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sc_subs_[4], yerr=sc_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sc_subs_[5], yerr=sc_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (ml/mg)')
plt.ylabel('SC')
plt.title(f'TiO$_2$ Subtractions SC')
plt.legend()
plt.savefig('TiO2 Subtractions SC, HIHD vs HIHD-VIHD REDONE.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_b_signals, yerr=DOP_b_signal_errs,fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_b_backgrounds, yerr=DOP_b_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (ml/mg)')
plt.ylabel('Degree of Polarisation')
plt.title(f'TiO$_2$ Horizontal Illumination Degree of Polarisation')
plt.legend()
plt.savefig('TiO2 Horizontal Illumination Degree of Polarisation REDONE.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_v_signals, yerr=DOP_v_signal_errs, fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_v_backgrounds, yerr=DOP_v_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (ml/mg)')
plt.ylabel('Degree of Polarisation')
plt.title(f'TiO$_2$ Vertical Illumination Degree of Polarisation')
plt.legend()
plt.savefig('TiO2 Vertical Illumination Degree of Polarisation REDONE.pdf', bbox_inches='tight')
plt.show()

#%% REDONE TiO2 dithered

title = ['d_TiO2 0_125', 'd_TiO2 0_25', 'd_TiO2 0_5', 'd_TiO2 1', 'd_TiO2 2']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.125 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.25 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.5 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 1 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 2 bb']

bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.125 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.25 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.5 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 1 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 2 bd']

db = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.125 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.25 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.5 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 1 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 2 db']

dd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.125 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.25 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 0.5 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 1 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\d TiO2 2 dd']
mask = [
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 0.125 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 0.25 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 0.5 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 1 bb average.tiff',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d TiO2 2 bb average AMENDED.tif']


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

x = [0.125, 0.25, 0.5, 1, 2]  # these are the concentrations, the last two are dithered

nested_lists = [sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs]
title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
# Save to a file
for i in range(len(title)):
    with open('d_tio2_{}.pkl'.format(title[i]), 'wb') as f:
        pickle.dump(np.array(nested_lists[i]).T.tolist(), f)  # formatted as nested list at concentrations


sb_avrgs_ = np.array(sb_avrgs).T.tolist()  # bb, bd, db, dd
sb_avrgs_errs_ = np.array(sb_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_avrgs_[1], yerr=sb_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sb_avrgs_[2], yerr=sb_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sb_avrgs_[3], yerr=sb_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('S/B')
plt.title(f'Dithered TiO$_2$ Averages S/B')
plt.legend()
plt.savefig('Dithered TiO2 Averages SB REDONE.pdf', bbox_inches='tight')
plt.show()

sc_avrgs_ = np.array(sc_avrgs).T.tolist()  # bb, bd, db, dd
sc_avrgs_errs_ = np.array(sc_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sc_avrgs_[1], yerr=sc_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sc_avrgs_[2], yerr=sc_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sc_avrgs_[3], yerr=sc_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('SC')
plt.title(f'Dithered TiO$_2$ Averages SC')
plt.legend()
plt.savefig('Dithered TiO2 Averages SC REDONE.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
#plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('S/B')
plt.title(f'TiO$_2$ Subtractions S/B')
plt.legend()
plt.savefig('Dithered TiO2 Subtractions SB, HIHD vs HIHD-HIVD REDONE.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('S/B')
plt.title(f'Dithered TiO$_2$ Subtractions S/B')
plt.legend()
plt.savefig('Dithered TiO2 Subtractions SB, HIHD vs HIHD-VIHD vs HIHD-VIHD REDONE.pdf', bbox_inches='tight')
plt.show()

sc_subs_ = np.array(sc_subs).T.tolist()  # bb, bd, db, dd
sc_subs_errs_ = np.array(sc_subs_errs).T.tolist()
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
#plt.errorbar(x, sc_subs_[0], yerr=sc_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sc_subs_[1], yerr=sc_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sc_subs_[2], yerr=sc_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sc_subs_[3], yerr=sc_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sc_subs_[4], yerr=sc_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sc_subs_[5], yerr=sc_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('SC')
plt.title(f'Dithered TiO$_2$ Subtractions SC')
plt.legend()
plt.savefig('Dithered TiO2 Subtractions SC, HIHD vs HIHD-VIHD REDONE.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_b_signals, yerr=DOP_b_signal_errs,fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_b_backgrounds, yerr=DOP_b_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('Degree of Polarisation')
plt.title(f'Dithered TiO$_2$ Horizontal Illumination Degree of Polarisation')
plt.legend()
plt.savefig('Dithered TiO2 Horizontal Illumination Degree of Polarisation REDONE.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_v_signals, yerr=DOP_v_signal_errs, fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_v_backgrounds, yerr=DOP_v_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('Degree of Polarisation')
plt.title(f'Dithered TiO$_2$ Vertical Illumination Degree of Polarisation')
plt.legend()
plt.savefig('Dithered TiO2 Vertical Illumination Degree of Polarisation REDONE.pdf', bbox_inches='tight')
plt.show()

#%% REDONE latex

title = ['latex 0_23', 'latex 0_46', 'latex 0_92', 'latex 0_18', 'latex 0_36']
bb = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.23 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.46 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.92 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 1.8 bb',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 3.6 bb']

bd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.23 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.46 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.92 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 1.8 bd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 3.6 bd']

db = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.23 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.46 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.92 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 1.8 db',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 3.6 db']

dd = [r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.23 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.46 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 0.92 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 1.8 dd',
      r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\latex 3.6 dd']
mask = [
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d latex 0.23 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d latex 0.46 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d latex 0.92 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d latex 1.8 bb average.tif',
    r'C:\Users\liann\Documents\GitHub\MSci-Scattered-Light-LSM\Pictures\REDONE\Masks\masks\Mask of d latex 3.6 bb average.tif']

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

x = [0.23, 0.46, 0.92, 1.8, 3.6]  # these are the concentrations, the last two are dithered

nested_lists = [sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs]
title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
# Save to a file
for i in range(len(title)):
    with open('latex_{}.pkl'.format(title[i]), 'wb') as f:
        pickle.dump(np.array(nested_lists[i]).T.tolist(), f)  # formatted as nested list at concentrations

sb_avrgs_ = np.array(sb_avrgs).T.tolist()  # bb, bd, db, dd
sb_avrgs_errs_ = np.array(sb_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_avrgs_[1], yerr=sb_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sb_avrgs_[2], yerr=sb_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sb_avrgs_[3], yerr=sb_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('S/B')
plt.title('Latex Averages S/B')
plt.legend()
plt.savefig('Latex Averages SB REDONE.pdf', bbox_inches='tight')
plt.show()

sc_avrgs_ = np.array(sc_avrgs).T.tolist()  # bb, bd, db, dd
sc_avrgs_errs_ = np.array(sc_avrgs_errs).T.tolist()  # bb, bd, db, dd
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sc_avrgs_[1], yerr=sc_avrgs_errs_[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sc_avrgs_[2], yerr=sc_avrgs_errs_[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sc_avrgs_[3], yerr=sc_avrgs_errs_[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('SC')
plt.title('Averages SC')
plt.legend()
plt.savefig('Latex Averages SC REDONE.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
#plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('S/B')
plt.title('Subtractions S/B')
plt.legend()
plt.savefig('Latex Subtractions SB, HIHD vs HIHD-HIVD REDONE.pdf', bbox_inches='tight')
plt.show()

sb_subs_ = np.array(sb_subs).T.tolist()
sb_subs_errs_ = np.array(sb_subs_errs).T.tolist()
plt.errorbar(x, sb_avrgs_[0], yerr=sb_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs_[0], yerr=sb_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sb_subs_[1], yerr=sb_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs_[2], yerr=sb_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs_[3], yerr=sb_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs_[4], yerr=sb_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs_[5], yerr=sb_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('S/B')
plt.title('Subtractions S/B')
plt.legend()
plt.savefig('Latex Subtractions SB, HIHD vs HIHD-VIHD vs HIHD-VIHD REDONE.pdf', bbox_inches='tight')
plt.show()

sc_subs_ = np.array(sc_subs).T.tolist()  # bb, bd, db, dd
sc_subs_errs_ = np.array(sc_subs_errs).T.tolist()
plt.errorbar(x, sc_avrgs_[0], yerr=sc_avrgs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD')
#plt.errorbar(x, sc_subs_[0], yerr=sc_subs_errs_[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sc_subs_[1], yerr=sc_subs_errs_[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sc_subs_[2], yerr=sc_subs_errs_[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sc_subs_[3], yerr=sc_subs_errs_[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sc_subs_[4], yerr=sc_subs_errs_[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sc_subs_[5], yerr=sc_subs_errs_[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('SC')
plt.title('Subtractions SC')
plt.legend()
plt.savefig('Latex Subtractions SC, HIHD vs HIHD-VIHD REDONE.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_b_signals, yerr=DOP_b_signal_errs,fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_b_backgrounds, yerr=DOP_b_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('Degree of Polarisation')
plt.title('Horizontal Illumination Degree of Polarisation')
plt.legend()
plt.savefig('Latex Horizontal Illumination Degree of Polarisation REDONE.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_v_signals, yerr=DOP_v_signal_errs, fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_v_backgrounds, yerr=DOP_v_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (mg/ml)')
plt.ylabel('Degree of Polarisation')
plt.title('Vertical Illumination Degree of Polarisation')
plt.legend()
plt.savefig('Latex Vertical Illumination Degree of Polarisation REDONE.pdf', bbox_inches='tight')
plt.show()

