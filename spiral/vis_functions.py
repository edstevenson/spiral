import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm 
from galario.double import get_image_size, sampleImage, chi2Image, threads
from utils import load_uvtable, load_disk, load_geo, p2i, add_spiral, add_spiral_fast, get_spiral, write_txtfile, chi2, recentre_V, vis_baseline_plot, test_fit_with_chi2, sim_obs_string
import seaborn as sns


def no_spiral_chi2(disk):
    """finds chi2_AO for given disk"""

    # load uvdata, profile
    u, v, V, w, r, intensity = load_disk(disk)

    # load disk geometry
    inc, PA, dRA, dDec = load_geo(disk)
    inc *= np.pi/180
    PA *= np.pi/180
    dRA *= np.pi/(180*60*60) 
    dDec *= np.pi/(180*60*60) 

    # get required pixel number and size from galario, [dxy] = rad
    nxy, dxy = get_image_size(u, v, verbose=False) 
    
    # calculate chi2 for spiral-less disk
    image, _, _ = p2i(r, intensity, nxy, dxy, inc=inc)
    Re_V = np.ascontiguousarray(np.real(V))
    Im_V = np.ascontiguousarray(np.imag(V))
    chi2_ = chi2Image(image, dxy, u, v, Re_V, Im_V, w, dRA=dRA, dDec=dDec, PA=PA, origin='upper')
    red_chi2_ = chi2_ / len(V)

    return chi2_, red_chi2_


def set_iterator(r_planet, SA, config, uJy, a_or_i, Mth, dustnum, hand, stuff=[None, 1]):
    """
    Iterates over spiral position space to find (projected, non-centered) chi2 for each r, phi. 
    Takes a given simulated observation / spiral model pair as input

    Arguments:  
    r_planet: planet radius (arcsec)
    SA: azimuthal spiral angle anticlockwise from x=0+ (deg)
    config: alma configuration e.g. 'C4C7'
    a_or_i: ('a') or isothermal ('i')
    Mth: '01','03','1','3'(01=>0.1)
    dustnum: '0', '1', '2' and '3' correspond to tau_(time=0, radius=Rp) = 0.1, 0.3, 1.0, and 3.0 respectively 
    stuff: list containing tag and Nthreads
    """

    disk = sim_obs_string(config, uJy, a_or_i, Mth, dustnum)
    diskname = f'{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}-{hand}'

    tag, Nthreads = stuff

    if tag is not None:
        headline = f'{tag}_{diskname}'
    else:
        headline = diskname
    print(f'\033[1m{headline}\033[0m')

    # load uvdata, profile
    u, v, V, w, r, intensity = load_disk(disk)
    Re_V = np.ascontiguousarray(np.real(V))
    Im_V = np.ascontiguousarray(np.imag(V))

    # load disk geometry
    inc, PA, dRA, dDec = load_geo(disk)
    inc *= np.pi/180
    PA *= np.pi/180
    dRA *= np.pi/(180*60*60) 
    dDec *= np.pi/(180*60*60) 

    # get required pixel number and size from galario, [dxy] = rad
    nxy, dxy = get_image_size(u, v, verbose=False) 
    print(nxy,dxy)

    threads(Nthreads) # set number of threads used by galario
    
    # calculate chi2 for spiral-less disk
    image, x, y = p2i(r, intensity, nxy, dxy, inc=inc)
    chi2_ = chi2Image(image, dxy, u, v, Re_V, Im_V, w, dRA=dRA, dDec=dDec, PA=PA, origin='upper')
    print(f'spiral-less chi2 = {chi2_:.1f}')

    M = len(r_planet)*len(SA)
    results = np.zeros((M, 4))
    no_spiral = np.array([np.nan, np.nan, chi2_, 0])

    # for add_spiral
    band = 6 if 'band6' in config else 7
    dI_I, r_spiral, dphi = get_spiral(a_or_i, Mth, dustnum, band=band) # r_spiral = 1 corresponds to planet position
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)
    Nphi, Nr = dI_I.shape
    dI_I = np.vstack((dI_I, dI_I[-1,:])) # To avoid holes in the 0,2pi boundary, the last row of the data copied in the begining 


    with tqdm(total=M) as pbar:
        # iterate over position parameter space
        for i, (r_p, sa) in enumerate(itertools.product(r_planet, SA)):
            # generate image with spiral added
            image_s = add_spiral_fast(dI_I, r_spiral, dphi, hand, r_p, sa*np.pi/180, image, Phi, R, Nphi, Nr)
            # image_s = add_spiral(a_or_i, Mth, dustnum, hand, r_p, sa*np.pi/180, image, x, y, band=band)
 
            # calculate chi2 
            chi2_spiral = chi2Image(image_s, dxy, u, v, Re_V, Im_V, w, dRA=dRA, dDec=dDec, PA=PA, origin='upper')
         
            # store values in results array
            results[i, 0:3] = [r_p, sa, chi2_spiral]
            
            # Update the progress bar
            pbar.update(1)

    # calculates fractional improvement in chi2 column
    results[:,3] = (chi2_ - results[:,2]) / chi2_ * 100


    # write to txt file to store data for each spiral model
    results = np.vstack((no_spiral, results))
    header = ['r_planet [arcsec]', 'spiral angle [deg]', 'chi2', 'dchi2/chi2 (%)']
    if tag is not None:
        txtfile = f'/data/es833/chi2_txtfiles/{tag}_{diskname}_chi2.txt' 
    else:
        txtfile = f'/data/es833/chi2_txtfiles/{diskname}_chi2.txt'            

    write_txtfile(txtfile, header, results)
    
    return 

def iterator(r_planet, SA, disk, a_or_i, Mth, dustnum, hand, stuff4iterator):
    """
    Iterates over spiral position space to find (projected, non-centered) chi2 for each r, phi. 
    Takes a given spiral model as input (cf. super_iterator iterates over 
    spiral models)
    Prints txt file if txt=True in stuff4iterator list
    band=6 default (1.3mm)

    Arguments:
    r_planet: planet radius
    SA: azimuthal spiral angle anticlockwise from x=0+
    disk: name of disk e.g. 'FT_Tau'
    [spiral arguments]
    a_or_i: ('a') or isothermal ('i')
    Mth: '01','03','1','3'(01=>0.1)
    dustnum: '0', '1', '2' and '3' correspond to tau_(time=0, radius=Rp) = 0.1, 0.3, 1.0, and 3.0 respectively 
    image, x, y, dxy, u, v, dRA, dDEc, PA from parallel_iterator
    """

    tag, Nthreads = stuff4iterator

    if disk.startswith('model'):
        diskpath = f'/data/es833/modeldata/{disk}'
        band = 7
    else:
        diskpath = disk
        band = 6

    # load uvdata, profile
    u, v, V, w, r, intensity = load_disk(diskpath)
    Re_V = np.ascontiguousarray(np.real(V))
    Im_V = np.ascontiguousarray(np.imag(V))

    # load disk geometry
    inc, PA, dRA, dDec = load_geo(diskpath) #deg, deg, arcsec, arcsec
    # inc, PA, dRA, dDec = 30, 90, 0.0013, -0.0018

    print('inc, PA, dRA, dDec:', inc, PA, dRA, dDec)
    inc *= np.pi/180
    PA *= np.pi/180
    dRA *= np.pi/(180*60*60) 
    dDec *= np.pi/(180*60*60) 

    # get required pixel number and size from galario, [dxy] = rad
    nxy, dxy = get_image_size(u, v, verbose=False) 
    print(nxy, dxy)

    # calculate chi2 for spiral-less disk
    image, x, y = p2i(r, intensity, nxy, dxy, inc=inc)
    chi2_ = chi2Image(image, dxy, u, v, Re_V, Im_V, w, dRA=dRA, dDec=dDec, PA=PA, origin='upper')

    M = len(r_planet)*len(SA)
    results = np.zeros((M, 4))
    no_spiral = np.array([np.nan, np.nan, chi2_, 0])

    threads(Nthreads) # set number of threads used by galario
    # for add_spiral
    dI_I, r_spiral, dphi = get_spiral(a_or_i, Mth, dustnum, band=band) # r_spiral = 1 corresponds to planet position
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)
    Nphi, Nr = dI_I.shape
    dI_I = np.vstack((dI_I, dI_I[-1,:])) # To avoid holes in the 0,2pi boundary, the last row of the data copied in the begining

    with tqdm(total=M) as pbar:
        # iterate over position parameter space
        for i, (r_p, sa) in enumerate(itertools.product(r_planet, SA)):

            # generate image with spiral added
            # image_s = add_spiral(a_or_i, Mth, dustnum, hand, r_p, sa*np.pi/180, image, x, y) 
            image_s = add_spiral_fast(dI_I, r_spiral, dphi, hand, r_p, sa*np.pi/180, image, Phi, R, Nphi, Nr)
            # calculate chi2 
            chi2_spiral = chi2Image(image_s, dxy, u, v, Re_V, Im_V, w, dRA=dRA, dDec=dDec, PA=PA, origin='upper')

            # store values in results array
            results[i, 0:3] = [r_p, sa, chi2_spiral]
            
            # Update the progress bar
            pbar.update(1)

    # calculates fractional improvement in chi2 column
    results[:,3] = (chi2_ - results[:,2]) / chi2_ * 100


    # write to txt file to store data for each spiral model
    results = np.vstack((no_spiral, results))
    header = ['r_planet [arcsec]', 'spiral angle [deg]', 'chi2', 'dchi2/chi2 (%)']
    if tag is not None:
        txtfile = '/data/es833/chi2_txtfiles/' + disk + '_' + tag + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt' 
    else:
        txtfile = '/data/es833/chi2_txtfiles/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'            

    write_txtfile(txtfile, header, results)
    
    return 


def spliterator(r_planet, SA, disk, a_or_i, Mth, dustnum, hand, stuff4iterator):
    """
    Iterates over spiral position space to find (projected, non-centered) chi2 for each r, phi. 
    Uses Im(V) from spiral model and Re(V) from frank axisym fit.
    Takes a given spiral model as input (cf. super_iterator iterates over 
    spiral models)

    Arguments:
    r_planet: planet radius
    SA: azimuthal spiral angle anticlockwise from x=0+
    disk: name of disk e.g. 'FT_Tau'
    [spiral arguments]
    a_or_i: ('a') or isothermal ('i')
    Mth: '01','03','1','3'(01=>0.1)
    dustnum: '0', '1', '2' and '3' correspond to tau_(time=0, radius=Rp) = 0.1, 0.3, 1.0, and 3.0 respectively 
    image, x, y, dxy, u, v, dRA, dDEc, PA from parallel_iterator
    """

    Re_V, Im_V, image, x, y, dxy, u, v, w, dRA, dDec, PA, tag, chi2_, Nthreads = stuff4iterator

    M = len(r_planet)*len(SA)
    results = np.zeros((M, 4))
    no_spiral = np.array([np.nan, np.nan, chi2_, 0])

    threads(Nthreads) # set number of threads used by galario

    with tqdm(total=M) as pbar:
        # iterate over position parameter space
        for i, (r_p, sa) in enumerate(itertools.product(r_planet, SA)):

            # generate images
            image_s = add_spiral(a_or_i, Mth, dustnum, hand, r_p, sa*np.pi/180, image, x, y) 

            # calculate chi2 
            V_S = sampleImage(image_s, dxy, u, v, dRA=dRA, dDec=dDec, PA=PA, origin='upper') 
            V_A = sampleImage(image, dxy, u, v, dRA=dRA, dDec=dDec, PA=PA, origin='upper') 
            chi2_spiral_real = chi2(np.real(V_A), Re_V, weights=w)
            chi2_spiral_imag = chi2(np.imag(V_S), Im_V, weights=w)
            chi2_spiral = chi2_spiral_real + chi2_spiral_imag

            # store values in results array
            results[i, 0:3] = [r_p, sa, chi2_spiral]
            
            # Update the progress bar
            pbar.update(1)
        

    # calculates fractional improvement in chi2 column
    results[:,3] = (chi2_ - results[:,2]) / chi2_ * 100

    
    # write to txt file to store data for each spiral model
    results = np.vstack((no_spiral, results))
    header = ['r_planet [arcsec]', 'spiral angle [deg]', 'chi2', 'dchi2/chi2 (%)']
    if tag is not None:
        txtfile = '/data/es833/chi2_txtfiles/' + disk + '_' + tag + '_split_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt' 
    else:
        txtfile = '/data/es833/chi2_txtfiles/' + disk + '_split_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'            
    write_txtfile(txtfile, header, results)

    return 




def SA_iterator(r_planet, SA, disk, a_or_i, Mth, dustnum, hand, stuff4SAiterator):
    """
    Like iterator but for chi2_SA
    prints to txt file in 'chi2_SA' folder
    
    Parameters:
    r_planet (array): Radial positions of the planet in arcsec.
    SA (array): Spiral angles in degrees.
    disk (str): Disk name.
    a_or_i (str): Specifies the type of spiral model.
    Mth (str): Mass of planet.
    dustnum (str): Optical thickness.
    stuff4SAiterator (tuple): A tuple containing required data for calculations.
    
    Returns:
    None. Writes results to a text file.
    """

    Re_vis, Im_vis, image, x, y, dxy, u, v, w, dRA, dDec, PA, chi2_0 = stuff4SAiterator

    M = len(r_planet)*len(SA)
    results = np.zeros((M, 4))
    no_spiral = np.array([np.nan, np.nan, chi2_0, 0])

    threads(1) # set number of threads used by galario to 1

    with tqdm(total=M) as pbar:
        # iterate over position parameter space
        for i, (r_p, sa) in enumerate(itertools.product(r_planet, SA)):

            # generate image with spiral added
            image_s = add_spiral(a_or_i, Mth, dustnum, hand, r_p, sa*np.pi/180, image, x, y) 

            # calculate chi2 
            chi2_spiral = chi2Image(image_s, dxy, u, v, Re_vis, Im_vis, w, dRA=dRA, dDec=dDec, PA=PA, origin='upper')

            # store values in results array
            results[i, 0:3] = [r_p, sa, chi2_spiral]
            
            # Update the progress bar
            pbar.update(1)

    results[:,3] = results[:,2] / chi2_0 * 100 # normalize wrt chi2_AO (%)

    # write to txt file to store data for each spiral model
    results = np.vstack((no_spiral, results))
    header = ['r_planet [arcsec]', 'spiral angle [deg]', 'chi2', 'chi2/chi2_0 (%)']
    txtfile = '/data/es833/chi2_txtfiles/chi2_SA/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'   
    write_txtfile(txtfile, header, results)

    return 


def radial_iterator(r_planet, disk, a_or_i, Mth, dustnum, hand):
    """
    Like iterator but only radial position and 
    different data stored (used by chi2_scale multiprocessing file)
    prints to txt file in 'scales' folder
    """
    # load uvdata, profile
    u, v, V, w, r, intensity = load_disk(disk)
    nxy, dxy = get_image_size(u, v, verbose=False)

    # load disk geometry
    inc, PA, dRA, dDec = load_geo(disk)
    inc *= np.pi/180
    PA *= np.pi/180
    dRA *= np.pi/(180*60*60) 
    dDec *= np.pi/(180*60*60) 

    # vis and chi2 for spiral-less disk
    image, x, y = p2i(r, intensity, nxy, dxy, inc=inc)
    Re_V = np.ascontiguousarray(np.real(V))
    Im_V = np.ascontiguousarray(np.imag(V))
    vis = sampleImage(image, dxy, u, v, dRA=dRA, dDec=dDec, PA=PA, origin='upper')
    Re_vis = np.ascontiguousarray(np.real(vis))
    Im_vis = np.ascontiguousarray(np.imag(vis))

    chi2_real = chi2(Re_vis, Re_V, weights=w)
    chi2_imag = chi2(Im_vis, Im_V, weights=w)
    chi2_0 = chi2_real + chi2_imag  


    Nr = len(r_planet)
    results = np.zeros((Nr, 3))
    no_spiral = np.array([np.nan, chi2_0, 0])

    threads(1) # set number of threads used by galario to 1

    with tqdm(total=Nr) as pbar:
        # iterate over radial position
        for i, r_p in enumerate(r_planet):

            # generate image with spiral added
            image_s = add_spiral(a_or_i, Mth, dustnum, hand, r_p, 0, image, x, y) # SA=0

            # calculate chi2 
            chi2_spiral = chi2Image(image_s, dxy, u, v, Re_vis, Im_vis, w, dRA=dRA, dDec=dDec, PA=PA, origin='upper')

            # store values in results array
            results[i, 0:2] = [r_p, chi2_spiral]
            
            # Update the progress bar
            pbar.update(1)

    results[:,2] = results[:,1] / chi2_0 * 100 # normalize wrt axisym-model chi2 (%)

    # write to txt file to store data for each spiral model
    results = np.vstack((no_spiral, results))
    header = ['r_planet [arcsec]', 'chi2', 'chi2/chi2_0 (%)']
    txtfile = '/data/es833/chi2_txtfiles/scales/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'   
    write_txtfile(txtfile, header, results)

    return 


def diagnostics(disk, recentre=True, png=False):
    """
    Prints diagnostic graph of recentred Re(V) against baseline,
    with optional save to png ; 
    Also shows chi2 for:
    binned real visibilities
    unbinned complex visibilities

    """

    # load uvdata, profile, disk parameters 
    u, v, V, w, r, intensity = load_disk(disk)

    # load disk geometry
    inc, PA, dRA, dDec = load_geo(disk)
    inc *= np.pi/180
    PA *= np.pi/180
    dRA *= np.pi/(180*60*60) 
    dDec *= np.pi/(180*60*60) 

    # get required pixel number and size from galario, [dxy] = rad
    nxy, dxy = get_image_size(u, v, verbose=False) 
    
    # calculate vis
    image, _, _ = p2i(r, intensity, nxy, dxy, inc=inc)
    vis = sampleImage(image, dxy, u, v, dRA=dRA, dDec=dDec, PA=PA, origin='upper')  

    # load frank uv fit baselines, visibilities, weights 
    _, _, vis_frank, _, = load_uvtable(disk, frank=True)


    if recentre:
        # recentre visibilities
        V = recentre_V(u, v, V, dRA=dRA, dDec=dDec)
        vis = recentre_V(u, v, vis, dRA=dRA, dDec=dDec)
        vis_frank = recentre_V(u, v, vis_frank, dRA=dRA, dDec=dDec)


    ### plot
    size = 15

    # theme
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    fig = vis_baseline_plot(u, v, vis=V, vis_fit=vis, weights=w, bin_widths=[20e3,100e3], vis_frank=vis_frank)
    plt.title(disk, pad=40)
    
    # chi2 values
    rchi2 = test_fit_with_chi2(u, v, vis=V, vis_fit=vis, weights=w, bin_width=20e3, reduced=True)
    chi2_ = test_fit_with_chi2(u, v, vis=V, vis_fit=vis, weights=w, bin_width=20e3, reduced=False)
    chi2_unbinned_real = chi2(np.real(vis), np.real(V), weights=w)
    chi2_unbinned_imag = chi2(np.imag(vis), np.imag(V), weights=w)
    chi2_unbinned = chi2_unbinned_real + chi2_unbinned_imag
    chi2_unbinned_frank_real = chi2(np.real(vis_frank), np.real(V), weights=w)
    chi2_unbinned_frank_imag = chi2(np.imag(vis_frank), np.imag(V), weights=w)
    chi2_unbinned_frank = (chi2_unbinned_frank_real + chi2_unbinned_frank_imag) / len(u)
    print(f'chi2_unbinned_frank = {chi2_unbinned_frank}')
    dchi2_chi2 = 100*(chi2_unbinned - chi2_unbinned_frank) / chi2_unbinned

    # add chi2 to plot
    plt.text(0.97, 0.58, '$\\chi^{{2}}$ = {:.1f}\n$\Delta\chi^{{2}} / \chi^{{2}}$ = {:.2g}%'.format(chi2_unbinned, dchi2_chi2),
         transform=plt.gca().transAxes,
         fontsize=size, verticalalignment='bottom', horizontalalignment='right', 
         bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7, boxstyle='round'))

    # correct legend ordering
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,0,3]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=size)

    plt.show()

    if png:
        fig.savefig('/home/es833/Proj42/diagnostic_plots/' + disk + '_diagnostic.png', dpi=300)

    return