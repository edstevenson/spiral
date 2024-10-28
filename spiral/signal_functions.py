import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from astropy.io import fits
from gofish import imagecube
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, brute
import pandas as pd
from utils import sim_obs_image_string, sim_obs_string

def custom_cmap(red=False):
    
    if red:
        colors = ['white', plt.cm.Reds(0.4), plt.cm.Reds(0.95)]
    else:
        colors = ['white', plt.cm.Blues(0.4), plt.cm.Blues(0.95)]

    # Create a new colormap with these colors
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_red", colors)

    return cmap


def clean_profile_plotter(config, uJy, Mth, dustnum, a_or_i_list=['a','i'], save_fig=False, label=True, size=20):
    """
    Plot radial profile
    
    - save_fig (bool): Whether to save the plotted figure.
    """

    # Plotting

    sns.set_theme(style='whitegrid', palette='deep')
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    fig, ax = plt.subplots(figsize=(7, 6))

    y = [0,0]
    line_color = ['r','b'] # red for adiabatic, blue for isothermal
    for i, a_or_i in enumerate(a_or_i_list):

        image = sim_obs_image_string(config, uJy, a_or_i, Mth, dustnum)

        # Load the FITS file and extract header information
        with fits.open(image) as hdul:
            header = hdul[0].header

            bmaj = header['BMAJ']
            bmin = header['BMIN']

            if 'CDELT1' in header and 'CDELT2' in header:
                pixel_scale_y = header['CDELT2']  # degrees/pixel along y
                pixel_scale_y *= 60**2 # units: arcsec

                pixel_scale_x = header['CDELT1']  # degrees/pixel along x
                pixel_scale_x *= 60**2 # units: arcsec
                # print(f'pixel_scale_x = {pixel_scale_x:.4g} arcsec/pixel \npixel_scale_y = {pixel_scale_y:.4g} arcsec/pixel')
            else:
                print("Resolution keywords not found in header.")

            # Calculate the beam area in steradians
            beam_area = np.pi * bmaj * bmin / (4 * np.log(2)) # units: deg^2
            beam_area *= (np.pi/180)**2 # units: sr


        # Compute the radial profile using the gofish library
        cube = imagecube(image, FOV=2*0.8, verbose=False)
        x, y[i], _ = cube.radial_profile(dr=pixel_scale_y, unit='Jy/beam', r_cavity=0.1)
        y[i] /= beam_area*1e10

        # Add to plot 
        ax.plot(x, y[i], color=line_color[i], lw=1.0)
        

    ax.axvline(x=50/140, color='lightblue', linestyle='--')
    ax.set_xlim(x.min(), x.max())
    ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(10/140, 110/140+1e-6, 10/140)))
    ax.set_ylim(0, 10)
    ax.set_xlim(10/140, 110/140)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_xticklabels([f'{x*140:.0f}' for x in ax.get_xticks()])
    ax.tick_params(axis='both', labelsize=size)
    if label:
        ax.set_xlabel('$r \;/\, \mathrm{au}$', fontsize=size)
        ax.set_ylabel('$I \;(10^{10}\mathrm{Jy/sr})$', fontsize=size)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xticklabels([f'{x*140:.0f}' for x in ax.get_xticks()])
        ax.tick_params(axis='both', labelsize=size)
    else:
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xticklabels([f'{x*140:.0f}' for x in ax.get_xticks()])
        ax.tick_params(axis='both', labelsize=size, colors='white')       

    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')  

    plt.subplots_adjust(bottom=0.135, left=0.14, top=0.96, right=0.955)

    if save_fig:
        # add pad to bottom of figure to fit xlabel
        fig.savefig(f'/home/es833/Proj42_results/gofish_profiles/profile_{config}_{uJy}uJy_both-{Mth}Mth-dust{dustnum}.png', dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close()
        



def gap_finder(config, uJy, a_or_i, Mth, dustnum, plot=True, save_fig=False, label=True, size=20):
    """
    Process a given image to compute and optionally plot its radial profile.
    
    Parameters:
    - plot (bool): Whether to plot the radial profile.
    - save_fig (bool): Whether to save the plotted figure.
    
    Returns:
    - float: Difference between the minimum and maximum of the radial profile.
    """

    if a_or_i == 'a':
        eos = 'adi'
        beta = '-beta10'
    elif a_or_i == 'i':
        eos = 'iso'
        beta = ''
    else:
        raise KeyError("a_or_i = 'a' or 'i'")

    image = sim_obs_image_string(config, uJy, a_or_i, Mth, dustnum)

    # Load the FITS file and extract header information
    with fits.open(image) as hdul:
        header = hdul[0].header

        bmaj = header['BMAJ']
        bmin = header['BMIN']

        if 'CDELT1' in header and 'CDELT2' in header:
            pixel_scale_y = header['CDELT2']  # degrees/pixel along y
            pixel_scale_y *= 60**2 # units: arcsec

            pixel_scale_x = header['CDELT1']  # degrees/pixel along x
            pixel_scale_x *= 60**2 # units: arcsec
            # print(f'pixel_scale_x = {pixel_scale_x:.4g} arcsec/pixel \npixel_scale_y = {pixel_scale_y:.4g} arcsec/pixel')
        else:
            print("Resolution keywords not found in header.")

        # Calculate the beam area in steradians
        beam_area = np.pi * bmaj * bmin / (4 * np.log(2)) # units: deg^2
        beam_area *= (np.pi/180)**2 # units: sr



    # Function to plot the radial profile
    def plot_radial_profile(x, y, save_fig, label=label, size=size):

        # Set the theme for plotting
        sns.set_theme(palette='deep')
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(x, y, color='r', lw=1.0)
        ax.axvline(x=50/140, color='lightblue', linestyle='--')
        ax.set_xlim(x.min(), x.max())
        ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(10/140, 110/140+1e-6, 10/140)))
        ax.set_ylim(0, 10)
        ax.set_xlim(10/140, 110/140)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xticklabels([f'{x*140:.0f}' for x in ax.get_xticks()])
        ax.tick_params(axis='both', labelsize=size)
        if label:
            ax.set_xlabel('$r \;/\, \mathrm{au}$', fontsize=size)
            ax.set_ylabel('$I \;(10^{10}\mathrm{Jy/sr})$', fontsize=size)
            ax.yaxis.set_major_locator(plt.MultipleLocator(1))
            ax.set_xticklabels([f'{x*140:.0f}' for x in ax.get_xticks()])
            ax.tick_params(axis='both', labelsize=size)
        else:
            ax.yaxis.set_major_locator(plt.MultipleLocator(1))
            ax.set_xticklabels([f'{x*140:.0f}' for x in ax.get_xticks()])
            ax.tick_params(axis='both', labelsize=size, colors='white')
            
     

        plt.subplots_adjust(bottom=0.135, left=0.14, top=0.96, right=0.955)
        if save_fig:
            # add pad to bottom of figure to fit xlabel
            fig.savefig(f'/home/es833/Proj42_results/gofish_profiles/profile_{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}.png', dpi=330)
            plt.close(fig)
        else:
            plt.show()
           

    # Compute the radial profile using the gofish library
    cube = imagecube(image, FOV=2*0.8, verbose=False)
    x, y, _ = cube.radial_profile(dr=pixel_scale_y, unit='Jy/beam', r_cavity=0.1)
    y /= beam_area*1e10

    # global_max_y = np.max(y)

    bounds = (0.2, 0.4)
    indices_within_bounds = np.where((x >= bounds[0]) & (x <= bounds[1]))[0]

    min_y = np.min(y[indices_within_bounds])
    min_x = x[np.argmin(y[indices_within_bounds])+indices_within_bounds[0]]

    bounds = (min_x, 0.8)
    indices_within_bounds = np.where((x >= bounds[0]) & (x <= bounds[1]))[0]

    max_y = np.max(y[indices_within_bounds])
    max_x = x[np.argmax(y[indices_within_bounds])+indices_within_bounds[0]]


    DI = max_y/min_y if max_y > min_y else 0
    # print(f'dI/I = {DI:.4g} ({a_or_i}-{Mth}Mth-dust{dustnum})')

    Dx = max_x - min_x if DI != 0 else 0
    # print(f'Dx = {Dx:.4g} arcsec ({a_or_i}-{Mth}Mth-dust{dustnum})')
    # Plot the radial profile if plot is True
    if plot:
        plot_radial_profile(x, y, save_fig)

    # print(f'gap width = {Dx:.3g} arcsec, gap depth = {DI:.3g} ({a_or_i}-{Mth}Mth-dust{dustnum})')

    marginal = 1.5
    depth_level = 2 if DI > marginal else 1 if DI > 0 else 0
    print(f'gap depth = {DI} ({a_or_i}-{Mth}Mth-dust{dustnum})')

    return depth_level

def spiral_strength(config, uJy, a_or_i, Mth, dustnum):
    '''
    returns the spiral strength of a given spiral model

    Note: this function relies on having models of both hands
    '''
    diskname = f'{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}'
    Ds_values = {}
    r_values = {}
    phi_values = {}
    Nphi = {}
    Nr = {}

    for hand in ['R', 'L']:
        diskname_hand = f'{diskname}-{hand}'
        txtfile = f'/data/es833/chi2_txtfiles/{diskname_hand}_chi2.txt'
        df = pd.read_csv(txtfile, sep='\t')
        Dchi2 = df['dchi2/chi2 (%)'].values
        Ds_values[hand] = Dchi2[1:]
        r_values[hand] = df['r_planet [arcsec]'].values[1:]
        phi_values[hand] = df['spiral angle [deg]'].values[1:]
        Nphi[hand] = np.argmax(phi_values[hand])+1
        Nr[hand] = len(r_values[hand]/Nphi[hand])

        # this is to correct for unusual thing where chi2txtfile contains a few repeats at the end
        r_values[hand], phi_values[hand], Ds_values[hand] = r_values[hand][:Nphi[hand]*Nr[hand]], phi_values[hand][:Nphi[hand]*Nr[hand]], Ds_values[hand][:Nphi[hand]*Nr[hand]]

    print(f'\033[1m{diskname}\033[0m') 

    Ds_peak_R = Ds_values['R'].max()
    Ds_peak_L = Ds_values['L'].max()

    Ds_avg_R = Ds_values['R'].mean()
    Ds_avg_L = Ds_values['L'].mean()

    Ds_avg = (Ds_avg_R + Ds_avg_L) / 2

    if Ds_avg >= 0:
        raise ValueError("Ds_avg must be >= 0, this cannot be used for a spiral strength calculation")
     
    ss = math.log(1 + np.abs(Ds_peak_R - Ds_peak_L)) - 0.1
    print(f'non-truncated spiral strength = {ss:.4g}\n')
    ss = ss if ss > 0 else 0
    
    hand_str = 'right' if Ds_peak_R - Ds_peak_L > 0 else 'left'
    print(f'spiral strength = {ss:.4g}\t for {hand_str}-handed spiral\n')

    return ss

def spiral_strength_close(config, uJy, a_or_i, Mth, dustnum, dR=0.05, r_max=0.5, dPhi=30, phi_max=360):
    '''
    returns the spiral strength of a given spiral model

    dR: radial half-width from peak in better spiral hand to look in other hand for, [''] 

    r_max: max disk radius considered, ['']

    Dphi: half-width of phi range to consider, [deg]

    phi_max: phi range considered, [deg]
    
    Note: this function relies on having models of both hands
    '''
    diskname = f'{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}'
    Ds_values = {}
    r_values = {}
    phi_values = {}
    Nphi = {}
    Nr = {}

    for i, hand in enumerate(['R', 'L']):
        diskname_hand = f'{diskname}-{hand}'
        txtfile = f'/data/es833/chi2_txtfiles/{diskname_hand}_chi2.txt'
        df = pd.read_csv(txtfile, sep='\t')
        Dchi2 = df['dchi2/chi2 (%)'].values
        Ds_values[hand] = Dchi2[1:]
        r_values[hand] = df['r_planet [arcsec]'].values[1:]
        phi_values[hand] = df['spiral angle [deg]'].values[1:]
        if i == 0:
            Nphi = np.argmax(phi_values[hand])+1
            Nr = len(r_values[hand])//Nphi
            # should be the same for both hands

        # # this is to correct for unusual thing where chi2txtfile contains a few repeats at the end
        # r_values[hand], phi_values[hand], Ds_values[hand] = r_values[hand][:Nphi[hand]*Nr[hand]], phi_values[hand][:Nphi[hand]*Nr[hand]], Ds_values[hand][:Nphi[hand]*Nr[hand]]

    print(f'\033[1m{diskname}\033[0m') 

    Ds_avg_R = Ds_values['R'].mean()
    Ds_avg_L = Ds_values['L'].mean()
    Ds_avg = (Ds_avg_R + Ds_avg_L) / 2

    if Ds_avg >= 0:
        raise ValueError("Ds_avg must be < 0, this cannot be used for a spiral strength calculation")
    
    # Zone of comparison 
     
    shape = Nr, Nphi 

    peak = {}
    peak['R'] = Ds_values['R'].max()
    peak['L'] = Ds_values['L'].max()
    
    if peak['R'] > peak['L']:
        hand_str = 'R'
        F0 = Ds_values['R'].reshape(shape)  # strong hand matrix
        idx = np.argmax(F0)
        F = Ds_values['L'].reshape(shape)  # weak hand matrix
    else:
        hand_str = 'L'
        F0 = Ds_values['L'].reshape(shape)  
        idx = np.argmax(F0)
        F = Ds_values['R'].reshape(shape)

    r_idx, phi_idx = np.unravel_index(idx, F0.shape)
    dR_idx = int((dR*Nr) // r_max)
    dPhi_idx = int((dPhi*Nphi) // phi_max)

    r_min_idx = max(0, r_idx - dR_idx) 
    r_max_idx = min(F.shape[0], r_idx + dR_idx + 1)
    phi_min_idx = max(0, phi_idx - dPhi_idx)
    phi_max_idx = min(F.shape[1], phi_idx + dPhi_idx + 1)

    Fzone = F[r_min_idx:r_max_idx, phi_min_idx:phi_max_idx] # zone of comparison for spiral of opposite hand
    weak_peak = Fzone.max() # weak hand, zone-of-comparison max
     
    ss = math.log(1 + np.abs(peak[hand_str] - weak_peak)) - 0.1
    print(f'non-truncated spiral strength = {ss:.4g}\n')
    ss = ss if ss > 0 else 0
    print(f'spiral strength = {ss:.4g}\t ({hand_str}-handed)\n')

    return ss




def detection_space(za, zi, spirals=False, save_fig=False, figname='', size=27):
    """spirals: True if plotting for spirals, False if plotting for gaps (just changes the cmap label)"""

    if spirals:
        label = r'$S$'
        rotation = 'horizontal'
        cmap = 'Reds'
    else:
        label = r'$\Delta$'
        rotation = 'horizontal'
        cmap = 'Blues'


    x = [0, 1, 2]
    y = [0, 1, 2, 3]

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # Mask zeros. 
    za = np.ma.masked_where(za == 0, za)
    zi = np.ma.masked_where(zi == 0, zi)

    fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    # Plot za
    c1 = axs[0].pcolormesh(x, y, za, shading='auto', cmap=cmap)
    axs[0].set_xlabel('$M_\mathrm{p}/M_\mathrm{th}$', fontsize=size)
    axs[0].set_ylabel(r'$\tau_{0}$', fontsize=size)
    axs[0].set_title(r'adiabatic ($\beta=10$)', fontsize=size)

    # Plot zi
    c2 = axs[1].pcolormesh(x, y, zi, shading='auto', cmap=cmap)
    axs[1].set_xlabel('$M_\mathrm{p}/M_\mathrm{th}$', fontsize=size)
    axs[1].set_title(r'isothermal ($\beta=0$)', fontsize=size)

    xlabels = [0.3, 1, 3]
    ylabels = [0.1, 0.3, 1, 3]
    xticks = [0, 1, 2]
    yticks = [0, 1, 2, 3]

    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xlabels, fontsize=size)
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels(ylabels, fontsize=size)

    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xlabels, fontsize=size) 



    # Set the same color scale for both plots
    vmin = min(za.min(), zi.min())
    vmax = max(za.max(), zi.max())
    c1.set_clim(vmin, vmax)
    c2.set_clim(vmin, vmax)

    cbar = fig.colorbar(c1, ax=axs, orientation='vertical', pad=0.06)

    # cbar.ax.tick_params()  # Adjust '14' to your desired fontsize
    # cbar.set_ticks([]) # empty for now as magnitudes not comparable

    cbar.set_label(label,rotation=rotation,fontsize=size+2, labelpad=20)
    if save_fig:
        plt.savefig(f'/home/es833/Proj42_results/detection_space_{figname}.pdf', dpi=330)
    plt.show()

def detection_space_qual(za, zi, spirals=False, save_fig=False, figname='', size=32):
    """spirals: True if plotting for spirals, False if plotting for gaps (just changes the cmap label)"""

    if spirals:
        cmap = custom_cmap(red=True)
    else:
        cmap = custom_cmap(red=False)

    x = [0, 1, 2]
    y = [0, 1, 2, 3]

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    fig, axs = plt.subplots(1, 2, figsize=(14, 7.7), sharey=True)

    # Plot za
    c1 = axs[0].pcolormesh(x, y, za, shading='auto', cmap=cmap)
    axs[0].set_xlabel('$M_\mathrm{p}/M_\mathrm{th}$', fontsize=size)
    axs[0].set_ylabel(r'$\tau_{0}$', fontsize=size)
    axs[0].set_title(r'$\beta=10$', fontsize=size)

    # Plot zi
    c2 = axs[1].pcolormesh(x, y, zi, shading='auto', cmap=cmap)
    axs[1].set_xlabel('$M_\mathrm{p}/M_\mathrm{th}$', fontsize=size)
    axs[1].set_title(r'$\beta=0$', fontsize=size)

    xlabels = [0.3, 1, 3]
    ylabels = [0.1, 0.3, 1, 3]
    xticks = [0, 1, 2]
    yticks = [0, 1, 2, 3]

    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xlabels, fontsize=size)
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels(ylabels, fontsize=size)

    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xlabels, fontsize=size) 

    # Set the same color scale for both plots
    c1.set_clim(0, 2)
    c2.set_clim(0, 2)
    plt.subplots_adjust(bottom=0.17)
   
    if save_fig:
        plt.savefig(f'/home/es833/Proj42_results/detection_space_qual_{figname}.pdf', dpi=330)
    plt.show()
    plt.close()