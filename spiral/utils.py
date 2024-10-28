import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
import seaborn as sns
from astropy.io import fits
import pandas as pd
import json
import cv2
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import rotate
from frank.utilities import UVDataBinner
from frank.plot import plot_vis_quantity
from scipy.integrate import trapz, simps



def round_to_1sf(x):
    """round list elements each to 1sf"""
    
    for i in range(len(x)):
        x[i] = np.round(x[i], (-np.floor(np.log10(np.abs(x[i])))).astype(int))
    
    return x

def ensure_list(input_value):
    # Check if the input is a string
    if isinstance(input_value, str):
        # Return a list containing the string
        return [input_value]
    else:
        # Return the input as it is (assuming it's already a list or other type)
        return input_value

def str1(num, rep=None):
    """
    Converts to a string, and removes the . if its there
    optionally replaces it with rep
    """

    num_str = str(num)
    
    if rep is None:
        rep = ''

    if '.' in num_str:
        num_str = num_str.replace('.', rep)
    
    return num_str

def add_dot(s):
    if len(s) > 0 and s[0] == '0':
        return '0.' + s[1:]
    else:
        return s
    
def obs_wle(disk):
    """Extract obs. wavelength"""

    if disk.startswith(('/data/es833/modeldata','/data/es833/C4C7', '/data/es833/C5C8', '/data/es833/C6C9', '/data/es833/band')):

        wavelength = None
        with open(f'{disk}.txt', 'r') as file:
            for _ in range(2):
                line = file.readline()
                if line.startswith("# wavelength"):
                    # Extract the value after the '=' sign and convert it to a float
                    wavelength = float(line.split('=')[1].strip()) # [wavelength] = m
                    break
        
    else:
        raise TypeError('no obs. wavelength to extract. Are u,v in wavelength units already?')

    return wavelength


def load_uvtable(disk, frank=False):
    """read in uvtable data ; frank uses frank uv fit instead of data"""

    if disk.startswith(('/data/es833/modeldata','/data/es833/C4C7', '/data/es833/C5C8', '/data/es833/C6C9', '/data/es833/band')):

        filename_f = f'{disk}_frank_uv_fit_nn.txt'
        d_f = pd.read_csv(filename_f, header = None, skiprows = 1, delim_whitespace=True, names = ['u','v','Re_V', 'Im_V', 'weights'])
        
        if frank:

            u, v, Re_V, Im_V, w = d_f['u'].to_numpy(), d_f['v'].to_numpy(), d_f['Re_V'].to_numpy(), d_f['Im_V'].to_numpy(), d_f['weights'].to_numpy()
            V = Re_V + 1j * Im_V
            
        else:

            filename = f'{disk}.txt'
            d = pd.read_csv(filename, header = None, skiprows = 3, delim_whitespace=True, names = ['u','v','Re_V', 'Im_V', 'weights'])
            w = d_f['weights'].to_numpy()
            u, v, Re_V, Im_V = d['u'].to_numpy(), d['v'].to_numpy(), d['Re_V'].to_numpy(), d['Im_V'].to_numpy()
            V = Re_V + 1j * Im_V

            wle = obs_wle(disk)
            u, v = u/wle, v/wle

    else:

        if frank:
            filename = "/data/es833/dataforfrank/uv_tables/" + disk + "_selfcal_cont_bin60s_1chan_nopnt_nofl_frank_uv_fit.npz" 
        else:
            filename = "/data/es833/dataforfrank/uv_tables/" + disk + "_selfcal_cont_bin60s_1chan_nopnt_nofl.npz"
        
        d = np.load(filename, allow_pickle=True)
        u, v, V, w = d['u'], d['v'], d['V'], d['weights']
    
    return u, v, V, w


def load_profile(disk):
    """read radial profile from frank ; converts [r] from arcsec to radians [intensity] is Jy/sr"""

    if disk.startswith(('/data/es833/modeldata','/data/es833/C4C7', '/data/es833/C5C8', '/data/es833/C6C9', '/data/es833/band')):
        # takes non-negative profile by default
        filename = f'{disk}_frank_profile_fit_nn.txt'
    
    else:
        filename = '/data/es833/dataforfrank/uv_tables/' + disk + '_selfcal_cont_bin60s_1chan_nopnt_nofl_frank_profile_fit.txt'
    
    profile = pd.read_csv(filename, header = None, skiprows = 1, delim_whitespace=True, names = ['r','I','I_unc'])
    r,intensity = profile['r'].to_numpy(), profile['I'].to_numpy()
    r = r*np.pi/(180*60*60) # convert from arsec to rad

    return r, intensity

def load_geo(disk):
    """outputs inc, PA, dRA, dDec in degrees and arcsec"""

    if disk.startswith(('/data/es833/modeldata','/data/es833/C4C7', '/data/es833/C5C8', '/data/es833/C6C9', '/data/es833/band')):
        inc, PA, dRA, dDec = 0., 0., 0., 0.
    else:
        filename = '/data/es833/dataforfrank/uv_tables/' + disk + '_geo.json'
        with open(filename, 'r') as f:
            geometry = json.load(f)
        inc = geometry.get('inc')
        PA = geometry.get('PA')
        dRA = geometry.get('dRA')
        dDec = geometry.get('dDec')

    return inc, PA, dRA, dDec

def load_disk(disk):    
    """loads uvdata, profile and disk parameters ; [r] = radians, [I] = Jy/sr"""
    
    u, v, V, w = load_uvtable(disk)
    r, intensity = load_profile(disk)

    return u, v, V, w, r, intensity


def write_txtfile(filename, header, values):
    """make txt file to store data"""
    
    df = pd.DataFrame(values, columns=header)
    df.to_csv(filename, sep='\t', index=True) 


def recentre_V(u, v, V, dRA, dDec):

    """recentre visibilites (NOT deprojection) dRA and dDec in radians"""

    phi = 2*np.pi*(dRA*u + dDec*v)

    shifted_V = V * (np.cos(phi) - 1j * np.sin(phi)) #e^-2pijphi

    return shifted_V


def chi2(observed, expected, weights):
    
    wgtd_sqr_residuals = np.square(observed - expected)*weights

    return np.sum(wgtd_sqr_residuals)


def flux_fraction(r_planet, r, I):
    """
    Calculates the fraction of total flux of an axisymmetric disk within r_planet
    using a simple rectangle rule.
    [r_planet] = arcsec
    [flux] = Jy
    """
    r_p = r_planet * np.pi / (180 * 60 * 60)  # Convert to radians
    n = len(r)
    dr = np.zeros(n - 1)
    flux_total = 0
    flux = 0  # Initialize flux within r_planet

    if r_p < r[0]:
        raise ValueError('r_p < r[0]')

    for i in range(n - 1):
        dr[i] = r[i + 1] - r[i]
        flux_total += 2 * np.pi * r[i] * dr[i] * I[i]

        # Accumulate flux only if r[i] is less than r_p
        if r[i] < r_p:
            flux += 2 * np.pi * r[i] * dr[i] * I[i]

    if flux_total == 0:
        raise ValueError('Total flux is zero, division by zero encountered')

    f = flux / flux_total
    return flux, f


def flux_total(r, I, inc=0.):
    """
    Calculates the total flux
    using numerical integration (trapezoidal rule or Simpson's rule).
    [flux] = Jy
    corrects for inclination
    """

    # Calculate differential area
    circumference = 2 * np.pi * r

    # Integrate using the trapezoidal rule
    flux_total = trapz(I * circumference, r) * np.cos(inc)
    # Or, for Simpson's rule, use:
    # flux_total = simps(I * circumference, r)

    if flux_total == 0:
        raise ValueError('Total flux is zero, division by zero encountered')

    return flux_total


def polar_to_cart(polar_data, r, x, y, order=3):

    """
    Auxiliary function to map polar data to a cartesian plane ; 
    default order = 3 => cubic interpolation ;  
    a 2D array of data polar_data is assumed, with shape Nphi x Nr ; 
    r is the *evenly spaced* radial coordinate array ; 
    phi goes 0-~360 ['~' because step size might not perfectly divide 360] [vs. in polar_to_cart_log it goes -pi,pi]
    this is accounted for with phi argument (which is in degrees) (can be left out for later versions where I make sure 
    step size divides 360)
    """

    # "x" and "y" are numpy arrays with the desired cartesian coordinates
    # we make a meshgrid with them
    X, Y = np.meshgrid(x, y)

    # Now that we have the X and Y coordinates of each point in the output plane
    # we can calculate their corresponding phi and r
    Phi = (np.arctan2(Y, X)).ravel()
    R = (np.sqrt(X**2 + Y**2)).ravel()
    R_shift = R - r.min()

    # convert angles into (0,2pi) range
    Phi %= (2*np.pi)

    # get polar_data shape
    Nphi, Nr = polar_data.shape
    
    # Using the known phi and r, the coordinates are mapped to
    # those of the data grid

    Phi *= Nphi / (2*np.pi) 
    R_shift *= Nr / (r.max() - r.min())

    # An array of polar coordinates is created stacking the previous arrays
    coords = np.vstack((Phi, R_shift))

    # To avoid holes in the 0,2pi boundary, the first row of the data
    # copied in the end
    polar_data = np.vstack((polar_data, polar_data[0,:]))

    # The data is mapped to the new coordinates
    # Values outside range are substituted with 0.
    cart_data = map_coordinates(polar_data, coords, order=order, mode='constant', cval=0.)

    # The data is reshaped and returned
    return cart_data.reshape(len(x), len(y))



def polar_to_cart_log_CV(polar_data, r, x, y, SA=0., linear=False):

    """
    auxiliary function to map polar data to a cartesian plane with optional initial azimuthal angle offset (anticlockwise) SA, uses OpenCV's remap function;

    defaults to cubic interpolation; linear=True => linear interpolation;

    SA in degrees;
    """
    if linear:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_CUBIC
    

    # Create a meshgrid with the desired cartesian coordinates
    X, Y = np.meshgrid(x, y)
    rmin, rmax = r.min(), r.max()
    ln_rmin, ln_rmax = np.log(rmin), np.log(rmax)

    Phi = np.arctan2(Y, X)
    Phi -= SA # add optional spiral angle (minus because we are effectively shifting coordinates here whilst keeping spiral fixed)
 
    # convert angles into (0,2pi) range
    Phi %= (2*np.pi)
    R = np.sqrt(X**2 + Y**2)
    
    # Mask values at r<rmin 
    ln_R = np.empty_like(R)
    ln_R[R < rmin] = -1e12 #-1e12 < ln_rmin therefore gets set to zero as out of bounds.
    ln_R[R >= rmin] = np.log(R[R >= rmin])
    ln_R_shift = ln_R - ln_rmin

    Nphi, Nr = polar_data.shape
    
    # Mapping: Adjust the known phi and ln_r values to map to the data grid
    Phi_idx = Phi * (Nphi-1) / (2*np.pi)
    ln_R_idx = ln_R_shift * (Nr-1) / (ln_rmax - ln_rmin)

    # To avoid holes in the 0,2pi boundary, the last row of the data
    # copied in the begining 
    polar_data = np.vstack((polar_data, polar_data[-1,:]))

    # Use OpenCV's remap function. Values outside range are substituted with 0.
    cart_data = cv2.remap(polar_data.astype(np.float32), ln_R_idx.astype(np.float32), Phi_idx.astype(np.float32), interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # flip along x-axis (to match convention of RA increasing right to left) 
    cart_data = np.flip(cart_data, axis=0)

    return cart_data

def polar_to_cart_log_CV_fast(polar_data, r, Phi, R, Nphi, Nr, SA=0., linear=False):

    """
    version of polar_cart_log_CV that takes in Phi and R arrays, and Nphi and Nr, to avoid having to calculate them each time;

    defaults to cubic interpolation; linear=True => linear interpolation;
    """
    # copies to avoid overwriting original arrays
    Phi_copy = Phi.copy()

    if linear:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_CUBIC
    

    rmin, rmax = r.min(), r.max()
    ln_rmin, ln_rmax = np.log(rmin), np.log(rmax)

    Phi_copy -= SA # add optional spiral angle (minus because we are effectively shifting coordinates here whilst keeping spiral fixed)
 
    # convert angles into (0,2pi) range
    Phi_copy %= (2*np.pi)
    
    # Mask values at r<rmin 
    ln_R = np.empty_like(R)
    ln_R[R < rmin] = -1e12 #-1e12 < ln_rmin therefore gets set to zero as out of bounds.
    ln_R[R >= rmin] = np.log(R[R >= rmin])
    ln_R_shift = ln_R - ln_rmin

    
    # Mapping: Adjust the known Phi_copy and ln_r values to map to the data grid
    Phi_idx = Phi_copy * (Nphi-1) / (2*np.pi)
    ln_R_idx = ln_R_shift * (Nr-1) / (ln_rmax - ln_rmin)
    # Use OpenCV's remap function. Values outside range are substituted with 0.
    cart_data = cv2.remap(polar_data.astype(np.float32), ln_R_idx.astype(np.float32), Phi_idx.astype(np.float32), interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    # flip along x-axis (to match convention of RA increasing right to left) 
    cart_data = np.flip(cart_data, axis=0)

    return cart_data

def polar_to_cart_log(polar_data, r, x, y, SA=0., order=3):

    """
    like polar_to_cart except with evenly log-spaced r;

    auxiliary function to map polar data to a cartesian plane with optional initial azimuthal angle offset (anticlockwise) SA,
    [SA] = rad; 

    default order = 3 => cubic interpolation; 

    a 2D array of data polar_data is assumed, with shape Nphi x Nr;

    r is the evenly log-spaced radial coordinate array;

    r_planet = 1
    
    assumes polar_data.shape = (Nphi,Nr)
    """

    # "x" and "y" are numpy arrays with the desired cartesian coordinates
    # we make a meshgrid with them
    X, Y = np.meshgrid(x, y)
    rmin, rmax = r.min(), r.max()
    ln_rmin, ln_rmax = np.log(rmin), np.log(rmax)

    # Now that we have the X and Y coordinates of each point in the output plane
    # we can calculate their corresponding phi and ln_r
    Phi = (np.arctan2(Y, X)).ravel()

    R = np.sqrt(X**2 + Y**2).ravel()
    # mask values at r<rmin 
    ln_R = np.empty_like(R)
    ln_R[R < rmin] = -1e12 #-1e12 < ln_rmin therefore gets set to zero as out of bounds.
    ln_R[R >= rmin] = np.log(R[R >= rmin])

    # shift coordinate so ln_R_shift[~mask] starts at zero (needed for Mapping below)
    ln_R_shift = ln_R - ln_rmin 

    # add optional spiral angle (- because we are effectively shifting coordinates here whilst keeping spiral fixed)
    Phi -= SA

    # convert angles into (0,2pi) range
    Phi %= (2*np.pi) 
    
    # get polar_data shape
    Nphi, Nr = polar_data.shape
    
    # Mapping: using the known phi and ln_r, the coordinates are mapped to
    # those of the data grid
    Phi_idx = Phi * (Nphi-1) / (2*np.pi)  
    ln_R_idx = ln_R_shift * (Nr-1) / (ln_rmax - ln_rmin)

    # An array of polar coordinates is created stacking the previous arrays
    coords = np.vstack((Phi_idx, ln_R_idx))

    # To avoid holes in the 0,2pi boundary, the last row of the data
    # copied in the begining 
    polar_data = np.vstack((polar_data, polar_data[-1,:]))

    # The data is mapped to the new coordinates
    # Values outside range are substituted with 0.
    cart_data = map_coordinates(polar_data, coords, order=order, mode='constant', cval=0.)

    # The data is reshaped, flipped along x-axis (to fit RA increasing right to left) and returned
    # this ensures that image_s fits with origin='upper' default in galario
    cart_data = cart_data.reshape(len(x), len(y))
    cart_data = np.flip(cart_data, axis=0)


    return cart_data


def p2i(r, intensity, nxy, dxy, inc=0.):
    """
    profile to image: 
    returns list containing x, y and square matrix 'image' of shape (nxy,nxy) from intensity radial profile 'intensity' ; 
    [dxy] = rad/pixel ; [r] = rad ; [intensity] = Jy
    returns x, y in rad and image in Jy/pixel

    """
    
    # create meshgrids
    x = (np.linspace(0.5, -0.5 + 1/nxy, nxy)) * dxy * nxy
    x /= np.cos(inc)# increases rad per pixel (of axisymmetric model) along x-axis, therefore reduces no. pixels that sample intensity along x-direction, therefore reduces total flux of axisymmetric disk by cos(inc)
    y = (np.linspace(0.5, -0.5 + 1/nxy, nxy)) * dxy * nxy
    # galario chi2Image and sampleImage require x-axis increases from right to left
    # and the y-axis increases from bottom to top 
    # note this odd axes set-up will necessitate a flip along x-axis for the spiral simulations later

    x_mesh, y_mesh = np.meshgrid(x, y) 
    r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)

    # linear interpolation
    intensity = intensity*dxy**2 # convert from Jy/sr (sr=rad**2) to Jy/pixel
    I = interp1d(r, intensity, kind='linear', bounds_error=False, fill_value=0.,  assume_sorted=True) 

    image = I(r_mesh) 
    image[nxy//2, nxy//2] = intensity[0] # approximates intensity at central pixel as that of min radius in frank profile 

    return image, x, y



def vis_baseline_plot(u, v, vis, vis_fit, weights, bin_widths, vis_frank=None, figsize=(8,6)):

    r"""
    Re(V) against baseline plot
    
    bin_widths : list, unit = \lambda
        Bin widths in which to bin the observed visibilities

    """
    # Set the fontsize variable
    size = 15

    # global settings for plots
    cs = ['#a4a4a4', 'k', '#f781bf', '#dede00']
    
    # baseline
    uv = (u**2 + v**2)**0.5 

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    for i in range(len(bin_widths)):
        binned_vis = UVDataBinner(uv, vis, weights, bin_widths[i])
        vis_re_kl = binned_vis.V.real * 1e3
        if bin_widths[i] > 50e3:
            vis_err_re_kl = binned_vis.error.real * 1e3 
            plot_vis_quantity(binned_vis.uv / 1e6, vis_re_kl, ax,
                     vis_err_re_kl, c=cs[i],
                     marker='x', ls='None', 
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))
        else: 
            plot_vis_quantity(binned_vis.uv / 1e6, vis_re_kl, ax,
                     c=cs[i],
                     marker='x', ls='None', 
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))
    
    # Add a custom legend with the desired fontsize
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, fontsize=size)

    binned_vis_fit = UVDataBinner(uv, vis_fit, weights, 20e3)
    vis_fit_kl = binned_vis_fit.V.real * 1e3
    plot_vis_quantity(binned_vis_fit.uv / 1e6, vis_fit_kl, ax, c='blue', linewidth=1, label= r'a.m. fit with {:.0f} k$\lambda$ bins'.format(20), zorder=11)

    zoom_ylim_guess = abs(vis_fit_kl[int(.5 * len(vis_fit_kl)):]).max()
    zoom_bounds = [-5 * zoom_ylim_guess, 20 * zoom_ylim_guess]
    ax.set_ylim(zoom_bounds)
    ax.set_xlim(right=max(uv) / 1e6 * 1.01)
    xlims = ax.get_xlim()
    ax.set_xlim(xlims)
    plt.setp(ax.get_xticklabels(), visible=True, fontsize=size)
    plt.setp(ax.get_yticklabels(), visible=True, fontsize=size)
    ax.set_xlabel(r'Baseline [M$\lambda$]', fontsize=size)
    ax.set_ylabel('Re(V) [mJy]', fontsize=size)
    
    if vis_frank is not None:
        binned_vis_frank = UVDataBinner(uv, vis_frank, weights, 20e3)
        vis_frank_kl = binned_vis_frank.V.real * 1e3
        plot_vis_quantity(binned_vis_frank.uv / 1e6, vis_frank_kl, ax, c='r', linestyle=':', label= r'frank fit with {:.0f} k$\lambda$ bins'.format(20), zorder=12)


    return fig

def test_fit_with_chi2(u, v, vis, vis_fit, weights, bin_width, reduced=False):

    """
    returns binned chi2 or reduced chi2 for fit of real part of 
    model visibilities
    """

    uv = (u**2 + v**2)**0.5 

    binned_vis = UVDataBinner(uv, vis, weights, bin_width)
    vis_re = binned_vis.V.real
    w = binned_vis._w

    binned_vis_fit = UVDataBinner(uv, vis_fit, weights, bin_width)
    vis_fit_re = binned_vis_fit.V.real
   
    chi2_ = chi2(vis_fit_re, vis_re, w)

    if reduced:
        nbins = np.ceil(uv.max() / bin_width).astype('int')
        dof = nbins - 1
        chi2_ = chi2_/dof
    
    return chi2_



def get_spiral(a_or_i, Mth, dustnum, band=7, plot=False, save_fig=False):
    """
    gets dI_I, r and dphi ; [dphi] = rad
    arguments:
    a_or_i: ('a') or isothermal ('i')
    Mth: '01','03','1','3'(01=>0.1)
    dustnum: '0', '1', '2' and '3' correspond to tau_(time=0, radius=Rp) = 0.1, 0.3, 1.0, and 3.0 respectively
    plot: optionally plots dI_I map in xy coords
    note: spirals are R-handed by default
    """

    if a_or_i == 'a':
        beta = '-beta10'
        eos = 'adi'
    else:  
        beta = ''
        eos = 'iso'

    if band == 7:
        band_str1, band_str2 = '', ''
    else:
        band_str1, band_str2 = f'band{band}_', f'band{band}/'

    base_path = f'/data/es833/{band_str1}dI_I_models/ds{a_or_i}-{Mth}qth-h007{beta}/{band_str2}f3d2c_dust{dustnum}_orbit1500_EOS{eos}_EPSTEIN_2D_CPD0'
    dI = f'{base_path}_intens_frac_rt.fits'
    r = f'{base_path}_rcenters.fits'
    tcenters = f'{base_path}_tcenters.fits'
    tcenters_notshifted = f'{base_path}_tcenters_notshifted.fits'

    
    dI_I_fits = fits.open(dI)
    dI_I = dI_I_fits[0].data

    r_spiral_fits = fits.open(r)
    r_spiral = r_spiral_fits[0].data * 1/50  # assuming r_p = 50 au this scales so that r_p = 1 

    tcenters = fits.open(tcenters)
    shift = tcenters[0].data[0]
    tcenters_notshifted = fits.open(tcenters_notshifted)
    noshift = tcenters_notshifted[0].data[0]
    dphi = shift - noshift

    # get rid of nans and transpose to fit polar_to_cart_log
    dI_I = np.nan_to_num(dI_I, nan=0.)
    dI_I = dI_I.T

    if plot:
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral' 
        r_max = r_spiral.max()
        x = np.linspace(-r_max, r_max, num=4096)
        y = np.linspace(-r_max, r_max, num=4096)

        dI_I_xy = polar_to_cart_log_CV(dI_I, r_spiral, x, y, SA=-dphi, linear=False)
        dI_I_xy = np.flip(dI_I_xy, axis=0)

        plt.Figure()
        divnorm=colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1)
        plt.pcolormesh(x, y, dI_I_xy, cmap='bwr', norm=divnorm)
        cbar = plt.colorbar()#label=r'$\frac{\Delta I}{<I>_{\phi}}$')
        cbar.set_label(r'$\frac{\Delta I}{\left<I\right>_{\!\phi}}$', rotation='horizontal', labelpad=20)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.show()

        if save_fig:
            filename = f'/home/es833/Proj42_results/pretties/simulation_{a_or_i}-{Mth}Mth-dust{dustnum}.png'
            plt.savefig(filename, bbox_inches='tight', dpi=330)
        

    return dI_I, r_spiral, dphi


def add_spiral_fast(dI_I, r_spiral, dphi, hand, r_planet, SA, image, Phi, R, Nphi, Nr):
    """
    generates 'image_s' (disk with spiral added) ; 
    includes angular correction dphi
    dI_I.shape = (Nphi, Nr) ; 
    r_spiral = the evenly log-spaced radial coordinate array for dI_I polar grid ; 
    [x] = [y] = rad ; [r_planet] = arcsec ;
    this function assumes dI_I = 0 outside r_spiral range ;
    it also assumes r_spiral = 1 corresponds to planet location ;
    SA is spiral offset (anticlockwise)
    [SA] = rad; 

    """

    # convert fargos dI_I polar mesh to cartesian mesh

    r_planet *= np.pi/(180*60*60) # convert to rad

    # correct for azimuthal shift - so SA=0 corresponds to planet on disk x-axis (semi-minor axis of projected disk)
    SA += dphi

    ## convert to cartesian mesh

    # Using scipy map_coordinates
    # dI_I_xy = polar_to_cart_log(dI_I, r_spiral*r_planet, x, y, SA=SA, order=1) # order=1 for speed

    # Using OpenCV remap to be faster
    dI_I_xy = polar_to_cart_log_CV_fast(dI_I, r_spiral*r_planet, Phi, R, Nphi, Nr, SA=SA, linear=True) # linear for speed

    if hand == 'L':
        dI_I_xy = np.flip(dI_I_xy, axis=0)

    image_s = image + dI_I_xy * image # image with spiral added

    return image_s

def add_spiral(a_or_i, Mth, dustnum, hand, r_planet, SA, image, x, y, band=7):
    """
    generates 'image_s' (disk with spiral added) ; 
    includes angular correction dphi
    dI_I.shape = (Nphi, Nr) ; 
    r_spiral = the evenly log-spaced radial coordinate array for dI_I polar grid ; 
    [x] = [y] = rad ; [r_planet] = arcsec ;
    this function assumes dI_I = 0 outside r_spiral range ;
    it also assumes r_spiral = 1 corresponds to planet location ;
    SA is spiral offset (anticlockwise)
    [SA] = rad; 

    """

    dI_I, r_spiral, dphi = get_spiral(a_or_i, Mth, dustnum, band=band) # r_spiral = 1 corresponds to planet position

    # convert fargos dI_I polar mesh to cartesian mesh

    r_planet *= np.pi/(180*60*60) # convert to rad

    # correct for azimuthal shift - so SA=0 corresponds to planet on disk x-axis (semi-minor axis of projected disk)
    SA += dphi

    ## convert to cartesian mesh

    # Using scipy map_coordinates
    # dI_I_xy = polar_to_cart_log(dI_I, r_spiral*r_planet, x, y, SA=SA, order=1) # order=1 for speed

    # Using OpenCV remap to be faster 

    dI_I_xy = polar_to_cart_log_CV(dI_I, r_spiral*r_planet, x, y, SA=SA, linear=True) # linear for speed
    

    if hand == 'L':
        dI_I_xy = np.flip(dI_I_xy, axis=0)

    image_s = image + dI_I_xy * image # image with spiral added

    return image_s

def pretty_image(disk, r_planet, SA, a_or_i, Mth, dustnum, hand, nxy=1024, rmax=None, axisym=False, project=False, log=False, png=False, size=13, band=7):
    """
    makes a pretty image of disk+spiral: 
    -weak spirals look best with linear colormap, 
    -strong spirals wash out the image so look better with log colormap

    rmax = max radius to plot to ; nxy = no. pixels
    [r_planet] = [rmax] = arcsec, [SA] = deg
    """

    # profile
    r, intens = load_profile(disk)
    rad2arcsec = 180*60*60/np.pi
    r *= rad2arcsec
    intens /= rad2arcsec**2 # now [I] = Jy / (arcsec)**2

    if project:
        # disk geometry
        inc, PA, _, _ = load_geo(disk)
        inc *= np.pi/180
        # convert inc to rad, leave PA in degrees as scipy.ndimage.rotate wants that
    else:
        inc = PA = 0.

    # truncate at appropriate rmax
    if rmax is not None:
        idx = np.where(r > rmax)[0][0]
    else:
        idx = np.where(intens < 0)[0][0]
        rmax = r[idx-1]
    
    r = r[:idx]
    intens = intens[:idx]
    
    # to enable interpolation at small radii
    r = np.insert(r, 0, 0)
    intens = np.insert(intens, 0, intens[0]) 
    
    if log: 
        intens[intens<1e-2] = 1e-2
        I = interp1d(r, intens, kind='linear', bounds_error=False, fill_value=1e-2,  assume_sorted=True) 
        # gets rid of unplottably small values in log plot
    else:
        I = interp1d(r, intens, kind='linear', bounds_error=False, fill_value=0,  assume_sorted=True) 

    # Mesh 
    x = np.linspace(-1, 1, nxy) * rmax
    b = np.cos(inc)
    x /= b # inclinination is measured parallel to x-axis, reduces flux appropriately (see p2i)
    y = np.linspace(-1, 1, nxy) * rmax
    X, Y = np.meshgrid(x, y)

    # map intensity onto mesh
    R = np.sqrt(X**2 + Y**2)
    image = I(R)

    if axisym:
        image_s = image

    else:
        # # Add SPIRAL
        image_s = add_spiral(a_or_i, Mth, dustnum, hand, r_planet, SA*np.pi/180, image, x*1/rad2arcsec, y*1/rad2arcsec, band=band)

        # # Add SPIRAL fast (used as a check)
        # # for add_spiral
        # dI_I, r_spiral, dphi = get_spiral(a_or_i, Mth, dustnum, band=band) # r_spiral = 1 corresponds to planet position
        # X, Y = np.meshgrid(x*1/rad2arcsec, y*1/rad2arcsec)
        # R = np.sqrt(X**2 + Y**2)
        # Phi = np.arctan2(Y, X)
        # Nphi, Nr = dI_I.shape
        # dI_I = np.vstack((dI_I, dI_I[-1,:])) # To avoid holes in the 0,2pi boundary, the last row of the data copied in the begining 
        # image_s = add_spiral_fast(dI_I, r_spiral, dphi, hand, r_planet, SA*np.pi/180, image, Phi, R, Nphi, Nr)

        # Rotate the image by PA (anticlockwise)
        image_s = rotate(image_s, PA, reshape=False, order=3)

    if log:
        image_s[image_s < 1e-2] = 1e-2
    else:
        image_s[image_s < 0] = 0

    ### Plotting

    # theme
    sns.set_theme(palette='deep')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    cmap='inferno'

    fig, ax = plt.subplots()
    # ax.set_title(f'{disk} Spiral-Including Model Disk', pad=20, fontsize=size+1, fontweight='bold')

    if rmax > 0.5:
        step = 0.2
    else:
        step = 0.1

    if log:
        plt.pcolormesh(x*b, y, image_s, cmap=cmap, norm=colors.LogNorm(vmin=1e-2))
        # we use x*b instead of x as we just want to represent spatial scales,
        # so we have to reverse the scaling needed to generate correct images earlier (note x*b = y so really its just a square background grid)
    else: 
        plt.pcolormesh(x*b, y, image_s, cmap=cmap) 

    cbar = plt.colorbar()
    cbar.set_label(r'I [Jy arcsec$^{-2}$]', fontsize=size)
    cbar.ax.tick_params(labelsize=size)
    ax.set_xlabel("RA offset ['']", fontsize=size)
    ax.set_ylabel("Dec offset ['']", fontsize=size)
    ax.set_xlim(rmax,-rmax) # forces RA to run right to left 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=size)
    plt.subplots_adjust(bottom=0.15)  # Adjust the bottom space to a suitable value


    # save file
    if png and log and project:
        filename = '/home/es833/Proj42_results/pretties/' + disk + '_' + str1(r_planet) + 'rp-' + str(round(SA)) + 'deg_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_pretty_log_projected.png'
        fig.savefig(filename, bbox_inches='tight', dpi=330)
    
    elif png and project:
        filename = '/home/es833/Proj42_results/pretties/' + disk + '_' + str1(r_planet) + 'rp-' + str(round(SA)) + 'deg_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_pretty_projected.png'
        fig.savefig(filename, bbox_inches='tight', dpi=330)
    
    elif png and log:
        filename = '/home/es833/Proj42_results/pretties/' + disk + '_' + str1(r_planet) + 'rp-' + str(round(SA)) + 'deg_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_pretty_log.png'
        fig.savefig(filename, bbox_inches='tight', dpi=330)
    
    elif png:
        filename = '/home/es833/Proj42_results/pretties/' + disk + '_' + str1(r_planet) + 'rp-' + str(round(SA)) + 'deg_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '_pretty.png'
        fig.savefig(filename, bbox_inches='tight', dpi=330)
    
    else:
        plt.show()
    plt.close()


def sim_obs_string(config, uJy, a_or_i, Mth, dustnum):
    '''
    [uJy] = uJy/beam
    config = 'C4C7' or 'C6C9'
    '''

    # Check and convert list inputs to strings
    if isinstance(a_or_i, list) and len(a_or_i) == 1:
        a_or_i = a_or_i[0]
    if isinstance(Mth, list) and len(Mth) == 1:
        Mth = Mth[0]
    if isinstance(dustnum, list) and len(dustnum) == 1:
        dustnum = dustnum[0]

    if config.startswith('C4C7'):
        Ca, Cb = '8.4', '8.7'
    elif config == 'C5C8':
        Ca, Cb = '8.5', '8.8'
    elif config == 'C6C9':
        Ca, Cb = '7.6', '7.9'
    elif config == 'band6_C3C6':
        Ca, Cb = '4.3', '4.6'
    elif config == 'band6_C6':
        Ca, Cb = '4.6', None
    else:
        raise KeyError("need to add this config to sim_obs_string function")
    
    freq = '230' if 'band6' in config else '345'
    cycles = f'concat.cycle{Ca}.cycle{Cb}' if Cb is not None else f'cycle{Ca}'
    h = 'h010' if 'h010' in config else 'h005' if 'h005' in config else 'h007'
    config = config.replace('_h005', '').replace('_h010', '').replace('band6_C3', 'band6_')
    
    if a_or_i == 'a':
        eos = 'adi'
        beta = '-beta10'
    elif a_or_i == 'i':
        eos = 'iso'
        beta = ''
    else:
        raise KeyError("a_or_i = 'a' or 'i'")

    return f"/data/es833/{config}_{uJy}uJy_simulated_observations/dusty_spirals_{eos}/ds{a_or_i}-{Mth}qth-{h}{beta}/f3d2c_dust{dustnum}_orbit1500_EOS{eos}_EPSTEIN_2D_CPD0/{freq}GHz.{uJy}uJy/{freq}GHz.{uJy}uJy.alma.{cycles}.noisy.ms.uvtable"

def sim_obs_image_string(config, uJy, a_or_i, Mth, dustnum):
    '''
    [uJy] = uJy/beam
    config = 'C4C7' or 'C6C9'
    '''

    # Check and convert list inputs to strings
    if isinstance(a_or_i, list) and len(a_or_i) == 1:
        a_or_i = a_or_i[0]
    if isinstance(Mth, list) and len(Mth) == 1:
        Mth = Mth[0]
    if isinstance(dustnum, list) and len(dustnum) == 1:
        dustnum = dustnum[0]

    if config.startswith('C4C7'):
        Ca, Cb = '8.4', '8.7'
    elif config == 'C5C8':
        Ca, Cb = '8.5', '8.8'
    elif config == 'C6C9':
        Ca, Cb = '7.6', '7.9'
    elif config in ['band6_C3C6', 'band6_C6']:
        Ca, Cb = '4.3', '4.6'
    else:
        raise KeyError("need to add this config to sim_obs_string function")
    
    freq = '230' if 'band6' in config else '345'
    cycles = f'concat.cycle{Ca}.cycle{Cb}' if Cb is not None else f'cycle{Ca}'
    h = 'h010' if 'h010' in config else 'h005' if 'h005' in config else 'h007'
    config = config.replace('_h005', '').replace('_h010', '').replace('band6_C3', 'band6_')
    
    if a_or_i == 'a':
        eos = 'adi'
        beta = '-beta10'
    elif a_or_i == 'i':
        eos = 'iso'
        beta = ''
    else:
        raise KeyError("a_or_i = 'a' or 'i'")

    return f"/data/es833/{config}_{uJy}uJy_simulated_observations/dusty_spirals_{eos}/ds{a_or_i}-{Mth}qth-{h}{beta}/f3d2c_dust{dustnum}_orbit1500_EOS{eos}_EPSTEIN_2D_CPD0/{freq}GHz.{uJy}uJy/{freq}GHz.{uJy}uJy.alma.{cycles}.noisy.image.fits"    