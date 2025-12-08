import os
import matplotlib.pyplot as plt
import pickle
import scienceplots
#import matplotlib as mpl
#mpl.use('pgf')
plt.style.use('science')
directory = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\TiO2 REDONE'
#save_dir = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\TiO2 REDONE\Scatter Plots'

# TIO2
title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
loaded_lists = []
# reading pickle
for i in range(len(title)):
    with open(os.path.join(directory, 'tio2_{}.pkl'.format(title[i])), 'rb') as f:
        loaded_lists.append(pickle.load(f))

sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs = loaded_lists
x = [0.125, 0.25, 0.5, 1, 2]  # these are the concentrations, the last two are dithered

plt.figure(figsize=(8, 5))
plt.errorbar(x, sb_avrgs[0], yerr=sb_avrgs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_avrgs[1], yerr=sb_avrgs_errs[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sb_avrgs[2], yerr=sb_avrgs_errs[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sb_avrgs[3], yerr=sb_avrgs_errs[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel(r'Concentration (mgml$^{-1}$)', fontsize=12)
plt.xscale('log', base=2)
plt.ylabel('S/B', fontsize=12)
plt.legend()
plt.savefig('TiO2 Averages SB.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 7.2))  # Set the width to 8 inches and the height as specified
plt.errorbar(x, sc_avrgs[0], yerr=sc_avrgs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD')  # ordering bb, bd, db, dd
plt.errorbar(x, sc_avrgs[1], yerr=sc_avrgs_errs[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sc_avrgs[2], yerr=sc_avrgs_errs[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sc_avrgs[3], yerr=sc_avrgs_errs[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel(r'Concentration (mgml$^{-1}$)', fontsize=14)
plt.xscale('log', base=2)
plt.ylabel('SC' , fontsize=14)
#plt.xscale('log', base=2)
#plt.title(f'TiO$_2$ Averages S/B')
plt.legend()
plt.savefig('TiO2 Averages SC.pdf', bbox_inches='tight')
plt.show()

import pandas as pd
plotdata = pd.DataFrame({
    '0.0125%': [sublist[0] for sublist in sc_avrgs],
    '0.01%': [sublist[-1] for sublist in sc_avrgs]},
    index=['HIHD', 'HIVD', 'VIHD', 'VIVD'])
errors = pd.DataFrame({
    '0.0125%': [sublist[0] for sublist in sc_avrgs_errs],
    '0.01%': [sublist[-1] for sublist in sc_avrgs_errs]},
    index=['HIHD', 'HIVD', 'VIHD', 'VIVD'])
ax = plotdata.plot(kind='bar', figsize=(6, 6), rot=0, yerr=errors, error_kw=dict(capsize=3, alpha=0.8))
plt.xlabel('Polarisation Combination')
plt.ylabel('SC')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'TiO2_Averages_SC_dithering_avrg.pdf'), bbox_inches='tight')
plt.show()

plotdata = pd.DataFrame({
    '0.0125%': [sublist[0] for sublist in sc_subs],
    '0.01%': [sublist[-1] for sublist in sc_subs]},
    index=['HIHD-HIVD', 'HIHD-VIHD', 'HIHD-VIVD', 'HIVD-VIHD', 'HIVD-VIVD', 'VIHD-VIVD'])
errors = pd.DataFrame({
    '0.0125%': [sublist[0] for sublist in sc_subs_errs],
    '0.01%': [sublist[-1] for sublist in sc_subs_errs]},
    index=['HIHD-HIVD', 'HIHD-VIHD', 'HIHD-VIVD', 'HIVD-VIHD', 'HIVD-VIVD', 'VIHD-VIVD'])
ax = plotdata.plot(kind='bar', figsize=(8, 6), rot=0, yerr=errors, error_kw=dict(capsize=3, alpha=0.8))
plt.xlabel('Polarisation Combination')
plt.ylabel('SC')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'TiO2_Averages_SC_dithering_sub.pdf'), bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 5))
plt.errorbar(x, sb_avrgs[0], yerr=sb_avrgs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs[0], yerr=sb_subs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sb_subs[1], yerr=sb_subs_errs[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs[2], yerr=sb_subs_errs[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs[3], yerr=sb_subs_errs[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs[4], yerr=sb_subs_errs[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs[5], yerr=sb_subs_errs[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel(r'Concentration (mgml$^{-1}$)', fontsize=12)
plt.xscale('log', base=2)
plt.ylabel('S/B', fontsize=12)
#plt.title(f'TiO$_2$ Subtractions S/B')
plt.legend()
plt.savefig('TiO2 Subtractions SB, improvement.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x[:-2], DOP_b_signals[:-2], yerr=DOP_b_signal_errs[:-2], fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x[:-2], DOP_b_backgrounds[:-2], yerr=DOP_b_background_errs[:-2], fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
#plt.title(f'TiO$_2$ Horizontal Illumination Degree of Polarisation')
plt.legend()
plt.savefig('TiO2 Horizontal Illumination Degree of Polarisation.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x[:-2], DOP_v_signals[:-2], yerr=DOP_v_signal_errs[:-2], fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x[:-2], DOP_v_backgrounds[:-2], yerr=DOP_v_background_errs[:-2], fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
#plt.title(f'TiO$_2$ Vertical Illumination Degree of Polarisation')
plt.legend()
plt.savefig('TiO2 Vertical Illumination Degree of Polarisation.pdf', bbox_inches='tight')
plt.show()

#%% LATEX

directory = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\latex REDONE'
#save_dir = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\latex REDONE\Scatter Plots'

title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
loaded_lists = []
# reading pickle
for i in range(len(title)):
    with open(os.path.join(directory, 'latex_{}.pkl'.format(title[i])), 'rb') as f:
        loaded_lists.append(pickle.load(f))
sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs = loaded_lists
x = [2.3*2**i for i in range(5)]  # these are the concentrations

plt.figure(figsize=(6, 6))  # Set the width to 8 inches and the height as specified
plt.errorbar(x, sb_avrgs[0], yerr=sb_avrgs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_avrgs[1], yerr=sb_avrgs_errs[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sb_avrgs[2], yerr=sb_avrgs_errs[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sb_avrgs[3], yerr=sb_avrgs_errs[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel(r'Concentration (mgml$^{-1}$)', fontsize=14)
plt.xscale('log', base=2)
plt.ylabel('S/B', fontsize=14)
#plt.title('Latex Averages S/B')
plt.legend()
plt.savefig('Latex Averages SB.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 7.4))  # Set the width to 8 inches and the height as specified
plt.errorbar(x, sc_avrgs[0], yerr=sc_avrgs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sc_avrgs[1], yerr=sc_avrgs_errs[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, sc_avrgs[2], yerr=sc_avrgs_errs[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, sc_avrgs[3], yerr=sc_avrgs_errs[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel(r'Concentration (mgml$^{-1}$)', fontsize=15)
plt.xscale('log', base=2)
plt.ylabel('SC', fontsize=15)
#plt.title('Latex Averages SC')
plt.legend()
plt.savefig('Latex Averages SC.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 5))
plt.errorbar(x, sb_avrgs[0], yerr=sb_avrgs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, sb_subs[0], yerr=sb_subs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, sb_subs[1], yerr=sb_subs_errs[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, sb_subs[2], yerr=sb_subs_errs[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, sb_subs[3], yerr=sb_subs_errs[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, sb_subs[4], yerr=sb_subs_errs[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, sb_subs[5], yerr=sb_subs_errs[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel(r'Concentration (mgml$^{-1}$)', fontsize=12)
plt.xscale('log', base=2)
plt.ylabel('S/B', fontsize=12)
#plt.title('Latex Subtractions S/B')
plt.legend()
plt.savefig('Latex Subtractions SB, Improvements.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_b_signals, yerr=DOP_b_signal_errs, fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_b_backgrounds, yerr=DOP_b_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
#plt.title('Latex Horizontal Illumination Degree of Polarisation')
plt.legend()
plt.savefig('Latex Horizontal Illumination Degree of Polarisation.pdf', bbox_inches='tight')
plt.show()

plt.errorbar(x, DOP_v_signals, yerr=DOP_v_signal_errs, fmt='x', ls='none', capsize=3, label='Signal')
plt.errorbar(x, DOP_v_backgrounds, yerr=DOP_v_background_errs, fmt='x', ls='none', capsize=3, label='Background')
plt.xlabel('Concentration (% mass)')
plt.ylabel('Degree of Polarisation')
#plt.title('Latex Vertical Illumination Degree of Polarisation')
plt.legend()
plt.savefig('Latex Vertical Illumination Degree of Polarisation.pdf', bbox_inches='tight')
plt.show()

#%% background poisson

import matplotlib.pyplot as plt
import tifffile
import collections
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import poisson
import scienceplots
plt.style.use('no-latex')  # makes them LaTeX style

def background_region(rootdir, ymin=800, ymax=1000, xmin=1000, xmax=1200):
    averages = []
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
        image_arrays = [np.array(img) for img in images]  # convert to arrays
        stacked_array = np.stack(image_arrays, axis=-1)  # stack images along a new axis
        average_array = np.mean(stacked_array, axis=-1,
                                dtype=np.uint16)  # averaging pixel value for stacks, multiplied by 255 for the output image
        averages.append(average_array)

    selected_regions = [image[ymin:ymax, xmin:xmax] for image in averages]
    return selected_regions, averages

def flatten(l):  # flattens a 3d nested list
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def poisson_distribution(x, A, mu):
    return A*poisson.pmf(x, mu)

rootdir = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\TiO2 0.125 bb',
           r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\TiO2 0.125 bd',
           r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\TiO2 0.125 db',
           r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\TiO2 0.125 dd']
bb_background, bd_background, db_background, dd_background,  = background_region(rootdir)[0]
bb, bd, db, dd = background_region(rootdir)[1]
bb_flat = np.array(bb_background).flatten()
hist, bins, _ = plt.hist(bb_flat, bins=80, label='Pixel Values')

bin_centers = 0.5 * (bins[:-1] + bins[1:])  # bin centre
params, _ = curve_fit(poisson_distribution, bin_centers, hist)  # curve fit
plt.plot(np.arange(np.max(bin_centers)), poisson_distribution(np.arange(np.max(bin_centers)), *params), linewidth=2, label='Fitted Poisson')

plt.xlabel('Pixel Value')
plt.ylabel('Number')
plt.legend()
plt.savefig('TiO2 Poisson Noise.pdf', bbox_inches='tight')
plt.show()

#%% tring poisson for latex

rootdir = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\latex 0.23 bb',
           r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\latex 0.23 bd',
           r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\latex 0.23 db',
           r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\latex 0.23 dd']
bb_background, bd_background, db_background, dd_background,  = background_region(rootdir)[0]
bb, bd, db, dd = background_region(rootdir)[1]
bb_flat = np.array(bb_background).flatten()
hist, bins, _ = plt.hist(bb_flat, bins=50, label='Pixel Values')

bin_centers = 0.5 * (bins[:-1] + bins[1:])  # bin centre
params, _ = curve_fit(poisson_distribution, bin_centers, hist)  # curve fit
plt.plot(np.arange(np.max(bin_centers)), poisson_distribution(np.arange(np.max(bin_centers)), *params), linewidth=2, label='Fitted Poisson')

plt.xlabel('Pixel Value')
plt.ylabel('Number')
plt.legend()
plt.show()

#%% tring poisson for dithered

rootdir = [r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\d TiO2 0.125 bb',
           r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\d TiO2 0.125 bd',
           r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\d TiO2 0.125 db',
           r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\REDONE\d TiO2 0.125 dd']
bb_background, bd_background, db_background, dd_background,  = background_region(rootdir)[0]
bb, bd, db, dd = background_region(rootdir)[1]
bb_flat = np.array(bb_background).flatten()
hist, bins, _ = plt.hist(bb_flat, bins=50, label='Pixel Values')

bin_centers = 0.5 * (bins[:-1] + bins[1:])  # bin centre
#params, _ = curve_fit(poisson_distribution, bin_centers, hist)  # curve fit
#plt.plot(np.arange(np.max(bin_centers)), poisson_distribution(np.arange(np.max(bin_centers)), *params), linewidth=2, label='Fitted Poisson')

plt.xlabel('Pixel Value')
plt.ylabel('Number')
plt.legend()
plt.show()

#%% finding percentage improvements for sb TIO2

import os
import matplotlib.pyplot as plt
import pickle
import scienceplots
plt.style.use('no-latex')  # makes them LaTeX style
directory = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\TiO2 REDONE'

# TIO2
title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
loaded_lists = []
# reading pickle
for i in range(len(title)):
    with open(os.path.join(directory, 'tio2_{}.pkl'.format(title[i])), 'rb') as f:
        loaded_lists.append(pickle.load(f))

sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs = loaded_lists

sb_sub = sb_subs[:3]  # all hihd - ? subtractions
sb_sub_err = sb_subs_errs[:3]
sb_avrg = sb_avrgs[0]  # hihd
sb_avrg_err = sb_avrgs_errs[0]
result = []
result_err = []
for i in range(len(sb_sub[0])):
    x = []
    y = []
    for j in range(len(sb_sub)):
        improvement = (sb_sub[j][i] - sb_avrg[i]) * 100 / sb_avrg[i]
        err = improvement * np.sqrt((np.sqrt(sb_avrg_err[i]**2 + sb_sub_err[j][i]**2)/(sb_sub[j][i]+sb_avrg[i]))**2 + (sb_avrg_err[i]/sb_avrg[i])**2)
        x.append(improvement)
        y.append(err)
    result.append(x)
    result_err.append(y)
hihd_ = np.array(result).T.tolist()
hihd_err = np.array(result_err).T.tolist()

sb_sub = sb_subs[3:-1]  # all hivd - ? subtractions
sb_sub_err = sb_subs_errs[3:-1]
sb_avrg = sb_avrgs[1]  # hivd
sb_avrg_err = sb_avrgs_errs[1]
result = []
result_err = []
for i in range(len(sb_sub[0])):
    x = []
    y = []
    for j in range(len(sb_sub)):
        improvement = (sb_sub[j][i] - sb_avrg[i]) * 100 / sb_avrg[i]
        err = improvement * np.sqrt((np.sqrt(sb_avrg_err[i]**2 + sb_sub_err[j][i]**2)/(sb_sub[j][i]+sb_avrg[i]))**2 + (sb_avrg_err[i]/sb_avrg[i])**2)
        x.append(improvement)
        y.append(err)
    result.append(x)
    result_err.append(y)
hivd_ = np.array(result).T.tolist()
hivd_err = np.array(result_err).T.tolist()

sb_sub = [sb_subs[-1]]  # all vihd - ? subtractions
sb_sub_err = [sb_subs_errs[-1]]
sb_avrg = sb_avrgs[2]  # vihd
sb_avrg_err = sb_avrgs_errs[2]
result = []
result_err = []
for i in range(len(sb_sub[0])):
    x = []
    y = []
    for j in range(len(sb_sub)):
        improvement = (sb_sub[j][i] - sb_avrg[i]) * 100 / sb_avrg[i]
        err = improvement * np.sqrt((np.sqrt(sb_avrg_err[i]**2 + sb_sub_err[j][i]**2)/(sb_sub[j][i]+sb_avrg[i]))**2 + (sb_avrg_err[i]/sb_avrg[i])**2)
        x.append(improvement)
        y.append(err)
    result.append(x)
    result_err.append(y)
vihd_ = np.array(result).T.tolist()
vihd_err = np.array(result_err).T.tolist()

x = [0.125, 0.25, 0.5, 1, 2]  # these are the concentrations, the last two are dithered
final_result = hihd_ + hivd_ + vihd_
final_err = hihd_err + hivd_err + vihd_err
labels = ['HIHD - HIVD', 'HIVD - VIHD', 'HIHD - VIVD', 'HIVD - VIHD', 'HIVD - VIVD', 'VIHD - VIVD']
for i in range(len(final_result)):
    print(labels[i])
    print('percentage change', final_result[i])
    print('percentage change error', final_err[i])
    print('sb', np.mean(sb_subs[i]))
    plt.errorbar(x[:-2], final_result[i], yerr=final_err[i], fmt='x', ls='none', capsize=3, label=labels[i])
plt.xlabel('Concentration (% mass)')
plt.ylabel('Percentage Improvement')
plt.legend()
plt.show()

#%% finding percentage improvements for sb latex

directory = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\latex REDONE'

title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
loaded_lists = []
# reading pickle
for i in range(len(title)):
    with open(os.path.join(directory, 'latex_{}.pkl'.format(title[i])), 'rb') as f:
        loaded_lists.append(pickle.load(f))
sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs = loaded_lists

sb_sub = sb_subs[:3]  # all hihd - ? subtractions
sb_sub_err = sb_subs_errs[:3]
sb_avrg = sb_avrgs[0]  # hihd
sb_avrg_err = sb_avrgs_errs[0]
result = []
result_err = []
for i in range(len(sb_sub[0])):
    x = []
    y = []
    for j in range(len(sb_sub)):
        improvement = (sb_sub[j][i] - sb_avrg[i]) * 100 / sb_avrg[i]
        err = improvement * np.sqrt((np.sqrt(sb_avrg_err[i]**2 + sb_sub_err[j][i]**2)/(sb_sub[j][i]+sb_avrg[i]))**2 + (sb_avrg_err[i]/sb_avrg[i])**2)
        x.append(improvement)
        y.append(err)
    result.append(x)
    result_err.append(y)
hihd_ = np.array(result).T.tolist()
hihd_err = np.array(result_err).T.tolist()

sb_sub = sb_subs[3:-1]  # all hivd - ? subtractions
sb_sub_err = sb_subs_errs[3:-1]
sb_avrg = sb_avrgs[1]  # hivd
sb_avrg_err = sb_avrgs_errs[1]
result = []
result_err = []
for i in range(len(sb_sub[0])):
    x = []
    y = []
    for j in range(len(sb_sub)):
        improvement = (sb_sub[j][i] - sb_avrg[i]) * 100 / sb_avrg[i]
        err = improvement * np.sqrt((np.sqrt(sb_avrg_err[i]**2 + sb_sub_err[j][i]**2)/(sb_sub[j][i]+sb_avrg[i]))**2 + (sb_avrg_err[i]/sb_avrg[i])**2)
        x.append(improvement)
        y.append(err)
    result.append(x)
    result_err.append(y)
hivd_ = np.array(result).T.tolist()
hivd_err = np.array(result_err).T.tolist()

sb_sub = [sb_subs[-1]]  # all vihd - ? subtractions
sb_sub_err = [sb_subs_errs[-1]]
sb_avrg = sb_avrgs[2]  # vihd
sb_avrg_err = sb_avrgs_errs[2]
result = []
result_err = []
for i in range(len(sb_sub[0])):
    x = []
    y = []
    for j in range(len(sb_sub)):
        improvement = (sb_sub[j][i] - sb_avrg[i]) * 100 / sb_avrg[i]
        err = improvement * np.sqrt((np.sqrt(sb_avrg_err[i]**2 + sb_sub_err[j][i]**2)/(sb_sub[j][i]+sb_avrg[i]))**2 + (sb_avrg_err[i]/sb_avrg[i])**2)
        x.append(improvement)
        y.append(err)
    result.append(x)
    result_err.append(y)
vihd_ = np.array(result).T.tolist()
vihd_err = np.array(result_err).T.tolist()

x = [0.23*2**i for i in range(5)]  # these are the concentrations
final_result = hihd_ + hivd_ + vihd_
final_err = hihd_err + hivd_err + vihd_err
labels = ['HIHD - HIVD', 'HIVD - VIHD', 'HIHD - VIVD', 'HIVD - VIHD', 'HIVD - VIVD', 'VIHD - VIVD']
for i in range(len(final_result)):
    print(labels[i])
    print('percentage change', final_result[i])
    print('percentage change error', final_err[i])
    print('sb', np.mean(sb_subs[i]))
    plt.errorbar(x, final_result[i], yerr=final_err[i], fmt='x', ls='none', capsize=3, label=labels[i])
plt.xlabel('Concentration (% mass)')
plt.ylabel('Percentage Improvement')
plt.legend()
plt.show()

#%% percentage improvements for sc dithered subtractions

dithered = sc_avrgs[0][-2:][-1]  # lowest dithered in hihd
dithered_err = sc_avrgs_errs[0][-2:][-1]
undithered = np.max(sc_avrgs[0][0])  # 0.0125 in hihd
undithered_err = sc_avrgs_errs[0][0]
per_improvement = (dithered - undithered)/undithered * 100
err = per_improvement * np.sqrt(
    (np.sqrt(undithered_err ** 2 + dithered_err ** 2) / (undithered + dithered)) ** 2 + (
                undithered_err / undithered) ** 2)
print('tio2 average', per_improvement, 'err', err)

dithered = sc_subs[0][-2:][-1]  # lowest dithered in hihd
dithered_err = sc_subs_errs[0][-2:][-1]
undithered = np.max(sc_subs[0][0])  # 0.0125 in hihd
undithered_err = sc_subs_errs[0][0]
per_improvement = (dithered - undithered)/undithered * 100
err = per_improvement * np.sqrt(
    (np.sqrt(undithered_err ** 2 + dithered_err ** 2) / (undithered + dithered)) ** 2 + (
                undithered_err / undithered) ** 2)
print('tio2 subtraction', per_improvement, 'err', err)

#%% percentage imrpovements for sc dithered averages

import matplotlib.cm as cm

directory = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\d TiO2 REDONE'
title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
loaded_lists = []
for i in range(len(title)):
    with open(os.path.join(directory, 'd_tio2_{}.pkl'.format(title[i])), 'rb') as f:
        loaded_lists.append(pickle.load(f))

d_sb_avrgs, d_sb_avrgs_errs, d_sc_avrgs, d_sc_avrgs_errs, d_sb_subs, d_sb_subs_errs, d_sc_subs, d_sc_subs_errs, d_DOP_b_signals, d_DOP_b_backgrounds, d_DOP_v_signals, d_DOP_v_backgrounds, d_DOP_b_signal_errs, d_DOP_b_background_errs, d_DOP_v_signal_errs, d_DOP_v_background_errs = loaded_lists

directory = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\TiO2 REDONE'

title = ['sb_avrgs', 'sb_avrgs_errs', 'sc_avrgs', 'sc_avrgs_errs', 'sb_subs', 'sb_subs_errs', 'sc_subs', 'sc_subs_errs', 'DOP_b_signals', 'DOP_b_backgrounds', 'DOP_v_signals', 'DOP_v_backgrounds', 'DOP_b_signal_errs', 'DOP_b_background_errs', 'DOP_v_signal_errs', 'DOP_v_background_errs']
loaded_lists = []
for i in range(len(title)):
    with open(os.path.join(directory, 'tio2_{}.pkl'.format(title[i])), 'rb') as f:
        loaded_lists.append(pickle.load(f))

sb_avrgs, sb_avrgs_errs, sc_avrgs, sc_avrgs_errs, sb_subs, sb_subs_errs, sc_subs, sc_subs_errs, DOP_b_signals, DOP_b_backgrounds, DOP_v_signals, DOP_v_backgrounds, DOP_b_signal_errs, DOP_b_background_errs, DOP_v_signal_errs, DOP_v_background_errs = loaded_lists
x = [0.125, 0.25, 0.5, 1, 2]  # these are the concentrations, the last two are dithered

# SC avrg
values = []
values_err = []
for i in range(len(sc_avrgs)):  # iterate through polariser combination
    d_pol_com = d_sc_avrgs[i]
    d_pol_com_err = d_sc_avrgs_errs[i]
    pol_com = sc_avrgs[i]
    pol_com_err = sc_avrgs_errs[i]
    pol_com_values = []  # e.g HIHD, all concentrations as list
    pol_com_values_err = []
    for b in range(len(sc_avrgs[i])):  # for all concentrations in a given polariser orientation
        percentage_improvement = 100*(d_pol_com[b] - pol_com[b])/pol_com[b]
        percentage_improvement_err = percentage_improvement * np.sqrt(
            (np.sqrt(pol_com_err[b] ** 2 + d_pol_com_err[b] ** 2) / (pol_com[b] + d_pol_com[b])) ** 2 + (
                    pol_com_err[b] / pol_com[b]) ** 2)
        pol_com_values.append(percentage_improvement)
        pol_com_values_err.append(percentage_improvement_err)
    values.append(pol_com_values)
    values_err.append(pol_com_values_err)
    print('avrg', pol_com_values, pol_com_values_err)

categories_y = ['VIVD', 'VIHD', 'HIVD', 'HIHD']
categories_x = ['0.125', '0.25', '0.5', '1', '2']
values_forplot = np.array(values)  # Random values for demonstration
x_indices = np.arange(len(categories_x))
y_indices = np.arange(len(categories_y))

fig = plt.figure(figsize=(6, 7))
plt.imshow(values_forplot, cmap=cm.viridis.reversed(), extent=[x_indices[0]-0.5, x_indices[-1]+0.5, y_indices[0]-0.5, y_indices[-1]+0.5], aspect='auto')
cbar = plt.colorbar(label=r'Percentage Improvement from Undithered ($\%$)', orientation='horizontal', pad=0.1)  # Adjust pad to move colorbar closer
cbar.ax.xaxis.label.set_fontsize(14)  # Increase font size of colorbar label

plt.xticks(x_indices, categories_x)
plt.yticks(y_indices, categories_y)
plt.xlabel(r'Concentrations (mgml$^{-1}$)', fontsize=14)
plt.ylabel('Polariser Combinations', fontsize=14)
plt.grid(False)  # Turn off grid lines

plt.subplots_adjust(bottom=0.1, left=0.2)
plt.savefig('SC improvement avrg.pdf', bbox_inches='tight')
plt.show()

# SC sub
values = []
values_err = []
for i in range(len(sc_subs)):  # iterate through polariser combination
    d_pol_com = d_sc_subs[i]
    d_pol_com_err = d_sc_subs_errs[i]
    pol_com = sc_subs[i]
    pol_com_err = sc_subs_errs[i]
    pol_com_values = []  # e.g HIHD, all concentrations as list
    pol_com_values_err = []
    for b in range(len(sc_subs[i])):  # for all concentrations in a given polariser orientation
        percentage_improvement = 100*(d_pol_com[b] - pol_com[b])/pol_com[b]
        percentage_improvement_err = percentage_improvement * np.sqrt(
            (np.sqrt(pol_com_err[b] ** 2 + d_pol_com_err[b] ** 2) / (pol_com[b] + d_pol_com[b])) ** 2 + (
                    pol_com_err[b] / pol_com[b]) ** 2)
        pol_com_values.append(percentage_improvement)
        pol_com_values_err.append(percentage_improvement_err)
    values.append(pol_com_values)
    values_err.append(pol_com_values_err)
    print('sub', pol_com_values, pol_com_values_err)

categories_y = ['VIHD - VIVD', 'HIVD - VIVD', 'HIVD - VIHD', 'HIHD - VIVD', 'HIHD - VIHD', 'HIHD - HIVD']
categories_x = ['0.125', '0.25', '0.5', '1', '2']
values_forplot = np.array(values)  # Random values for demonstration
x_indices = np.arange(len(categories_x))
y_indices = np.arange(len(categories_y))

fig = plt.figure(figsize=(6, 7))
plt.imshow(values_forplot, cmap=cm.viridis.reversed(), extent=[x_indices[0]-0.5, x_indices[-1]+0.5, y_indices[0]-0.5, y_indices[-1]+0.5], aspect='auto')
cbar = plt.colorbar(label=r'Percentage Improvement from Undithered ($\%$)', orientation='horizontal', pad=0.1)  # Adjust pad to move colorbar closer
cbar.ax.xaxis.label.set_fontsize(14)  # Increase font size of colorbar label

plt.xticks(x_indices, categories_x)
plt.yticks(y_indices, categories_y)
plt.xlabel(r'Concentrations (mgml$^{-1}$)', fontsize=14)
plt.ylabel('Polariser Combinations', fontsize=14)
plt.grid(False)  # Turn off grid lines

plt.subplots_adjust(bottom=0.1, left=0.2)
plt.savefig('SC improvement sub.pdf', bbox_inches='tight')
plt.show()

#%% sb improvement from dithering

# Sb avrg
values = []
values_err = []
for i in range(len(sb_avrgs)):  # iterate through polariser combination
    d_pol_com = d_sb_avrgs[i]
    d_pol_com_err = d_sb_avrgs_errs[i]
    pol_com = sb_avrgs[i]
    pol_com_err = sb_avrgs_errs[i]
    pol_com_values = []  # e.g HIHD, all concentrations as list
    pol_com_values_err = []
    for b in range(len(sb_avrgs[i])):  # for all concentrations in a given polariser orientation
        percentage_improvement = 100*(d_pol_com[b] - pol_com[b])/pol_com[b]
        percentage_improvement_err = percentage_improvement * np.sqrt(
            (np.sqrt(pol_com_err[b] ** 2 + d_pol_com_err[b] ** 2) / (pol_com[b] + d_pol_com[b])) ** 2 + (
                    pol_com_err[b] / pol_com[b]) ** 2)
        if percentage_improvement > 0:
            print(i, b)
        pol_com_values.append(percentage_improvement)
        pol_com_values_err.append(percentage_improvement_err)
    values.append(pol_com_values)
    values_err.append(pol_com_values_err)

categories_y = ['VIVD', 'VIHD', 'HIVD', 'HIHD']
categories_x = ['0.125', '0.25', '0.5', '1', '2']
values_forplot = np.array(values)  # Random values for demonstration
x_indices = np.arange(len(categories_x))
y_indices = np.arange(len(categories_y))

fig = plt.figure(figsize=(8, 6))
plt.imshow(values_forplot, cmap='viridis', extent=[x_indices[0]-0.5, x_indices[-1]+0.5, y_indices[0]-0.5, y_indices[-1]+0.5], aspect='auto')
cbar = plt.colorbar(label='Percentage Improvement from Undithered (%)', orientation='horizontal')  # Add horizontal colorbar
cbar.ax.xaxis.label.set_fontsize('large')
plt.xticks(x_indices, categories_x)
plt.yticks(y_indices, categories_y)
plt.xlabel(r'Concentrations (mgml$^{-1}$)', fontsize='x-large')
plt.ylabel('Polariser Combinations', fontsize='x-large')
plt.grid(False)  # Turn off grid lines

plt.subplots_adjust(bottom=0.1)
plt.savefig('SB improvement avrg.pdf', bbox_inches='tight')
plt.show()

# SB sub
values = []
values_err = []
for i in range(len(sb_subs)):  # iterate through polariser combination
    d_pol_com = d_sb_subs[i]
    d_pol_com_err = d_sb_subs_errs[i]
    pol_com = sb_subs[i]
    pol_com_err = sb_subs_errs[i]
    pol_com_values = []  # e.g HIHD, all concentrations as list
    pol_com_values_err = []
    for b in range(len(sb_subs[i])):  # for all concentrations in a given polariser orientation
        percentage_improvement = 100*(d_pol_com[b] - pol_com[b])/pol_com[b]
        percentage_improvement_err = percentage_improvement * np.sqrt(
            (np.sqrt(pol_com_err[b] ** 2 + d_pol_com_err[b] ** 2) / (pol_com[b] + d_pol_com[b])) ** 2 + (
                    pol_com_err[b] / pol_com[b]) ** 2)
        #if percentage_improvement > 500:
        #    percentage_improvement = 0  # gets rid of hihd - vivd 0.5 is 1000% improvement
        pol_com_values.append(percentage_improvement)
        pol_com_values_err.append(percentage_improvement_err)
    values.append(pol_com_values)
    values_err.append(pol_com_values_err)
    print('sub', pol_com_values, pol_com_values_err)

categories_y = ['VIHD - VIVD', 'HIVD - VIVD', 'HIVD - VIHD', 'HIHD - VIVD', 'HIHD - VIHD', 'HIHD - HIVD']
categories_x = ['0.125', '0.25', '0.5', '1', '2']
values_forplot = np.array(values)  # Random values for demonstration
x_indices = np.arange(len(categories_x))
y_indices = np.arange(len(categories_y))

fig = plt.figure(figsize=(8, 6))
plt.imshow(values_forplot, cmap='viridis', extent=[x_indices[0]-0.5, x_indices[-1]+0.5, y_indices[0]-0.5, y_indices[-1]+0.5], aspect='auto')
cbar = plt.colorbar(label='Percentage Improvement from Undithered (%)', orientation='horizontal')  # Add horizontal colorbar
cbar.ax.xaxis.label.set_fontsize('large')
plt.xticks(x_indices, categories_x)
plt.yticks(y_indices, categories_y)
plt.xlabel(r'Concentrations (mgml$^{-1}$)', fontsize='x-large')
plt.ylabel('Polariser Combinations', fontsize='x-large')
plt.grid(False)  # Turn off grid lines

plt.subplots_adjust(bottom=0.1, left=0.2)
plt.savefig('SB improvement sub.pdf', bbox_inches='tight')
plt.show()

import os
import matplotlib.pyplot as plt
import pickle
import scienceplots
#import matplotlib as mpl
#mpl.use('pgf')
plt.style.use('no-latex')  # makes them LaTeX style
directory = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\TiO2 REDONE'
save_dir = r'C:\Users\liann\OneDrive - Imperial College London\Documents\Project\Pictures\Final plots\TiO2 REDONE\Scatter Plots'


plt.figure(figsize=(6, 6.8))
plt.errorbar(x, d_sb_avrgs[0], yerr=d_sb_avrgs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, d_sb_avrgs[1], yerr=d_sb_avrgs_errs[1], fmt='x', ls='none', capsize=3, label='HIVD')
plt.errorbar(x, d_sb_avrgs[2], yerr=d_sb_avrgs_errs[2], fmt='x', ls='none', capsize=3, label='VIHD')
plt.errorbar(x, d_sb_avrgs[3], yerr=d_sb_avrgs_errs[3], fmt='x', ls='none', capsize=3, label='VIVD')
plt.xlabel(r'Concentration (mgml$^{-1}$)', fontsize=14)
plt.xscale('log', base=2)
plt.ylabel('S/B', fontsize=14)
plt.legend()
plt.savefig('d TiO2 Averages SB.pdf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 6))
plt.errorbar(x, d_sb_avrgs[0], yerr=d_sb_avrgs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD')
plt.errorbar(x, d_sb_subs[0], yerr=d_sb_subs_errs[0], fmt='x', ls='none', capsize=3, label='HIHD - HIVD')
plt.errorbar(x, d_sb_subs[1], yerr=d_sb_subs_errs[1], fmt='x', ls='none', capsize=3, label='HIHD - VIHD')
#plt.errorbar(x, d_sb_subs[2], yerr=d_sb_subs_errs[2], fmt='x', ls='none', capsize=3, label='HIHD - VIVD')
#plt.errorbar(x, d_sb_subs[3], yerr=d_sb_subs_errs[3], fmt='x', ls='none', capsize=3, label='HIVD - VIHD')
#plt.errorbar(x, d_sb_subs[4], yerr=d_sb_subs_errs[4], fmt='x', ls='none', capsize=3, label='HIVD - VIVD')
#plt.errorbar(x, d_sb_subs[5], yerr=d_sb_subs_errs[5], fmt='x', ls='none', capsize=3, label='VIHD - VIVD')
plt.xlabel(r'Concentration (mgml$^{-1}$)', fontsize=14)
plt.xscale('log', base=2)
plt.ylabel('S/B', fontsize=14)
#plt.title(f'TiO$_2$ Subtractions S/B')
plt.legend()
plt.savefig('d TiO2 Subtractions SB.pdf', bbox_inches='tight')
plt.show()