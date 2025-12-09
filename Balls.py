### Calculates the mean free path of background

import numpy as np

radius_ball = 6e-9  #m

density_ball = 2.65 * 1000 #kg/m^3

index = 1.457
wavelength = 635e-9

vol_ball = (4/3) * np.pi * (radius_ball**3)

number_ball = density_ball * vol_ball

vol_agarose = 50*(0.01**3) #ml = cm^3 = 0.01^3 m^3

vol_required_1ball = (10e-6)**3  # 1 ball per 10 micron cubed SHOULD THE 10 ALSO BE CUBED? ASK CHRIS

number_ball = vol_agarose / vol_required_1ball

print(density_ball * number_ball * vol_ball, 'kg')
print(density_ball * number_ball * vol_ball * 1000, 'g')

l = 0.06e-3 #  m mean free path
sigma = np.pi*radius_ball**2  # flux area presented m^2

n=80000000 * 1e18
#n = 1/(l*sigma)  # balls per volume
m_balls = n * density_ball * vol_ball  # mass of all balls in the volume

vol_agarose = 5e-5  # in m^3
abs_m_balls = m_balls * vol_agarose
print(abs_m_balls*1000,'g in 50ml')

# using the website chris gave us for scattering area
sigma_real=2.7794e-10 # micron^2
sigma_rayleigh = 1e12 * (2 * np.pi**5 *(2*radius_ball)**6 * (index**2 - 1)**2) / (3 * (wavelength)**4 * (index**2 + 2)**2)  # micron^2

#n_real=n/1e+18
n_real=1.5e13  # CHANGE THIS

l_real=(1/(sigma_rayleigh*n_real))*1e-3
print('mean free path', l_real, 'mm')

#%%

suspension_and_ball = abs_m_balls*1000/0.4  # in g
water_for_agarose = 50-suspension_and_ball

print(suspension_and_ball/5, water_for_agarose/5)


