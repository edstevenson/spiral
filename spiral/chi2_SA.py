import numpy as np
from vis_functions import SA_iterator
import itertools
import multiprocessing
from galario.double import get_image_size, sampleImage
from utils import load_disk, load_geo, p2i, chi2


disk = 'FT_Tau'

### simulation parameter space
a_or_i = ['a','i']
Mth = ['01','03','1','3']
dustnum = ['0','1', '2','3']
hand = ['R']

### position parameter space

r_planet = np.arange(0.02, 0.5+1e-8, step=0.02) # arcsec
SA = np.arange(0, 360, 4) # deg                                             

#################################################

"""
- finds chi2_SA/chi2_OA /%
- parallel iterations over planet radial position for a given disk
- chi2 taken between axisym. disk + spiral model, and axisym. disk model (rather than real disk)
- saves txt files with data: 
r, chi2, and chi2/chi2_0 (the chi2 taken between axisym model and observed disk) to give a sense of the fractional increase in chi2 expected for an 'incorrect' spiral model
"""

print(f'\033[1m{disk}\033[0m (chi2_SA)\n')

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

stuff4SAiterator = Re_vis, Im_vis, image, x, y, dxy, u, v, w, dRA, dDec, PA, chi2_0

spiral_models = list(itertools.product(a_or_i, Mth, dustnum, hand))

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        pool.starmap(SA_iterator, [(r_planet, SA, disk, a_or_i, Mth, dustnum, hand, stuff4SAiterator) for a_or_i, Mth, dustnum, hand in spiral_models])
        pool.close()
        pool.join()

print(f'\033[1m{disk}\033[0m (chi2_SA) done!')
