import numpy as np
from vis_functions import radial_iterator
import itertools
import multiprocessing
from galario.double import get_image_size, sampleImage
from utils import load_disk, load_geo, p2i, chi2


disk = 'FT_Tau'

### simulation parameter space
a_or_i = ['i', 'a']
Mth = ['01','03','1','3']
dustnum = ['0','1', '2','3']
hand = ['R']

### position parameter space

r_planet = np.arange(0.02, 0.5+1e-8, step=0.001) # arcsec


#################################################

"""
- finds chi2_SA/chi2_OA /%
- parallel iterations over planet radial position for a given disk
- chi2 taken between axisym. disk + spiral model, and axisym. disk model (rather than real disk)
- saves txt files with data: 
r, chi2, and chi2/chi2_0 (the chi2 taken between axisym model and observed disk) to give a sense of the fractional increase in chi2 expected for an 'incorrect' spiral model
"""

print(f'\033[1m{disk}\033[0m (chi2 scale)\n')

spiral_models = list(itertools.product(a_or_i, Mth, dustnum, hand))

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        pool.starmap(radial_iterator, [(r_planet, disk, a_or_i, Mth, dustnum, hand) for a_or_i, Mth, dustnum, hand in spiral_models])
        pool.close()
        pool.join()

print(f'\033[1m{disk}\033[0m (chi2 scale) done!')
