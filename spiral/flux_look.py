import numpy as np
# Assuming that flux_fraction, load_profile, sim_obs_string are correctly defined in 'utils'
from utils import flux_fraction, flux_total, load_profile, sim_obs_string, load_geo
import matplotlib.pyplot as plt

# load frank profile
config = 'band6_C6'
uJy = 50
a_or_i = 'i'
Mth = '3'
dustnum = ['0', '1', '2', '3']
disk_models = []

for dust in dustnum:
    disk_models.append(sim_obs_string(config, uJy, a_or_i, Mth, dust))

all_disks = ['BP_Tau', 'DO_Tau', 'DR_Tau', 'FT_Tau', 'RY_Tau', 'UZ_Tau'] + disk_models

for disk in all_disks:
    r, intensity = load_profile(disk)

    # Uncomment and fix the plotting code if needed
    # ax = plt.subplot()
    # ax.plot(r*180*60*60/np.pi, intensity/1e10, color='red')
    # ax.set(xlabel="r [arcsec]", ylabel='I [1e10 Jy/sr]')
    # plt.show()


    if disk in disk_models:


        truncated_edges = np.where((r > (10/140)*np.pi/(180*3600)) & (r < (110/140)*np.pi/(180*3600)))
        r = r[truncated_edges]
        intensity = intensity[truncated_edges]

    inc, _, _, _ = load_geo(disk)
    inc *= np.pi/180 # radians


    flux = flux_total(r, intensity, inc) 
    print(f'{disk[:101]}: flux = {flux:.4g} Jy')

    # flux, f = flux_fraction(0.67, r, intensity) #0.67'' ~= 90 au and gives 95% of flux
    # print(f'{disk[:101]}: flux = {flux:.2g} Jy, fraction of total flux = {f:.2f}')




