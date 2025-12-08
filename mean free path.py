import numpy as np
import miepython

radius_ball = 21e-9/2  # in m
vol_ball = 4/3 * np.pi * radius_ball**3  # in m^3
density_ball = 4.26 * 1e3  # in kg/m^3
mass_ball = density_ball*vol_ball  # in kg

index_ball = 2.4892
m = index_ball #- 1j*1.8899e-9

l = 0.5e-3  # mean path is 0.5 mm

wavelength = 635e-9  # m
k = 2*np.pi/wavelength  # m^-1
F_m = abs((m**2 - 1)/(m**2 +2))**2

# TESTING OUT DIFFERENT SIGMAS
rayleigh_solidangle = 8*np.pi/3 * k**4 * radius_ball**6 * F_m  # from book, this is over the full solid angle, not illumination from one side?
rayleigh_sigma = 2 * np.pi**5 * (2*radius_ball)**6 / (3 * wavelength**4) * ((index_ball**2-1)/(index_ball**2+2))**2  # from wikipedia
sigma = 7.0093e-08 / 1e12  # from mie scattering website, scattering cross section in m^2

# USING MIEPYTHON
x = 2 * np.pi/wavelength * radius_ball  # wave number * radius_ball
geometric_cross_section = np.pi * radius_ball**2  # m^2, area
qext, qsca, qback, g = miepython.mie(m, x)  # scattering efficiency qsca unitless
sigma_miepython = qsca * geometric_cross_section
# https://miepython.readthedocs.io/en/latest/02_efficiencies.html

n = 1/(l * rayleigh_solidangle)  # in m^-3

print('number of target particles per m^3 =', n)

fifty_ml = 50e-6  # in m^3
mass_needed = mass_ball*n*fifty_ml  # in kg

print('mass needed =', mass_needed*1000, 'g for 50ml')
print('mass needed =', mass_needed*1000/50, 'g for 1ml eppendorf')
for i in range(4):
    print(mass_needed*1000/(50*2**i))

#%% potting scattering cross section as graph

import matplotlib.pyplot as plt

wavelength_range = np.linspace(400e-9, 700e-9, num=1000)
x = 2 * np.pi * radius_ball / wavelength_range

qext, qsca, qback, g = miepython.mie(m, x)
scatt = qsca * geometric_cross_section

plt.plot(wavelength_range, scatt, color='red')

plt.xlabel("Wavelength (m)")
plt.ylabel("Cross Section (m²)")
plt.title("Cross Sections for 21 nm TiO2")
plt.show()

#%% IGNORE THIS AND ONWARDS, IS THE SOLID ANGLE STUFF

import miepython
import matplotlib.pyplot as plt

#diff_sca = np.pi * r**2 * miepython.i_unpolarized(m,x,mu,norm='qsca')

m = index_ball - 1j*1.8899e-9
lambda0 = 635e-9  # m
theta = 90#np.linspace(0,180,1000)  # for a range of angles, we only want 90 degrees tho?
mu = np.cos(theta * np.pi/180)

d = 21e-9  # I THINK THIS IS THE DIAMETER?
x = 2 * np.pi/lambda0 * d/2  # wave number * radius_ball
geometric_cross_section = np.pi * d**2/4 * 1e4  # is pi * r^2 in cm**2
qext, qsca, qback, g = miepython.mie(m, x)
sigma_sca = geometric_cross_section * miepython.i_unpolarized(m, x, mu, 'qsca')

print('differential scattering cross section', sigma_sca*1e-3, 'cm^2/steradian')
print('differential scattering cross section', sigma_sca*1e-3/10000, 'm^2/steradian')  # I THINK STERADIANS ARE UNITLESS? M^2/M^2 IN SI UNITS?
# I THINK NEED TO INTEGRATE OVER FULL SOLIDANGLE? BUT WE WANT JUST A CIRCLE? IDK?

#  https://miepython.readthedocs.io/en/latest/03a_normalization.html#Differential-Scattering-Cross-Section

#%%

wavelength = 635e-9
radius_ball = 21e-9
x = 2 * np.pi/wavelength * radius_ball  # wave number * radius_ball
geometric_cross_section = np.pi * radius_ball**2  # m^2, area
qext, qsca, qback, g = miepython.mie(m, x)
tio2 = qsca*geometric_cross_section

radius_ball = 1e-6
x = 2 * np.pi/wavelength * radius_ball  # wave number * radius_ball
geometric_cross_section = np.pi * radius_ball**2  # m^2, area
qext, qsca, qback, g = miepython.mie(m, x)
latex = qsca*geometric_cross_section

radius_ball = 25e-6
x = 2 * np.pi/wavelength * radius_ball  # wave number * radius_ball
geometric_cross_section = np.pi * radius_ball**2  # m^2, area
qext, qsca, qback, g = miepython.mie(m, x)
pollen = qsca*geometric_cross_section

