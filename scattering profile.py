#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:10:04 2024

@author: rikke.ronnow
"""

  
import matplotlib.pyplot as plt
import numpy as np

# X=4.94739
# Qsca=3.8793

# natural      = (S1**2+S2**2)/2*1/(np.pi *X**2 *Qsca)
# perpen       = S1**2/(np.pi *X**2 *Qsca)
# parallel     = S2**2/(np.pi *X**2 *Qsca)

# r = np.arange(0, 2, 0.01)
# theta = 2 * np.pi * r

data=np.loadtxt('/Users/rikke.ronnow/Downloads/Uden navn2.txt',skiprows=1)
theta=data[:,0]
theta=theta/350*(2*np.pi)
natural=data[:,4]
perpen=data[:,5]
parallel=data[:,6]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.set_rscale('symlog')
ax.plot(theta, natural, label='natural', color='#E07A5F')
ax.plot(theta, perpen, label='perpendicular', color='#3D405B')
ax.plot(theta, parallel, label='parallel', color='#81B29A')
ax.legend(bbox_to_anchor=(1,0), loc="lower right",  bbox_transform=fig.transFigure)

ax.set_rmax(1)
ax.set_rticks([0.25,0.5,0.75, 1])  # Less radial ticks
ax.set_yticklabels([])
# ax.set_xticklabels([])
ax.set_rlabel_position(0)  # Move radial labels away from plotted line
ax.grid(True)


plt.savefig('scattering profile',transparent=True, dpi=2000)
plt.show()

#%% polar plot

# updated function with symlog
def scatter_logpolar(ArithmeticErrortheta, r_, bullseye=0.3, **kwargs):
    min10 = np.log10(np.min(r_))
    max10 = np.log10(np.max(r_))
    r = np.log10(r_) - min10 + bullseye
    ax.plot(theta, r, **kwargs)
    l = np.arange(np.floor(min10), max10)
    ax.set_rticks(l - min10 + bullseye) 
    ax.set_yticklabels(["1e%d" % x for x in l])
    ax.set_rlim(0, max10 - min10 + bullseye)
    ax.set_title('log-polar manual')
    return ax

# setup the plot
r = natural

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# scatter_polar_mpl(ax[0], theta, r)
# scatter_logpolar_mpl(ax[1], theta, r)
scatter_logpolar(theta, r, color='#E07A5F')
scatter_logpolar(theta, perpen, color='#3D405B')
scatter_logpolar(theta, parallel, color='#81B29A')
ax.set_rlabel_position(-90)  # Move radial labels away from plotted line


plt.tight_layout()
plt.show()