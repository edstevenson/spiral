import numpy as np
from vis_functions import iterator, spliterator
import itertools
import multiprocessing
from galario.double import get_image_size, chi2Image
from utils import load_disk, load_geo, p2i

disk = 'model_fixphase_inc25_rot90_35uJy_C4C7_a-03Mth-dust2'

### simulation parameter space
a_or_i = ['a']
Mth = ['03']
dustnum = ['2']
hand = ['R'] 

### position parameter space
r_planet = np.arange(0.02, 0.5 + 1e-8, step=0.02) # arcsec
SA = np.arange(0, 360, 4) # deg

### txt files
tag = None

Nthreads = 32 # no. galario threads


#################################################

### parallel iterations over spiral model parameter space for a given disk

if tag is not None:
    headline = disk + '_' + tag
else:
    headline = disk

print(f'\033[1m{headline}\033[0m')

spiral_models = list(itertools.product(a_or_i, Mth, dustnum, hand))

# setting number of threads used by galario (12,24 core,thread machine)

stuff4iterator = tag, Nthreads 

    
if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        if Nthreads in range(1,17):
            pool.starmap(iterator, [(r_planet, SA, disk, a_or_i, Mth, dustnum, hand, stuff4iterator) for a_or_i, Mth, dustnum, hand in spiral_models])
            pool.close()
            pool.join()
        else:
            iterator(r_planet, SA, disk, a_or_i[0], Mth[0], dustnum[0], hand[0], stuff4iterator)

print(f'\033[1m{headline}\033[0m done!')
