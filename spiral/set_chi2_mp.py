import numpy as np
from vis_functions import set_iterator
import itertools
import multiprocessing

config = 'C5C8'
uJy = 35


## observation parameters
a_or_i = ['i'] 
Mth = ['03']
dustnum = ['3'] #,'2']
hand = ['R']

# 'C5C8' next to do:


spiral_models = list(itertools.product(a_or_i, Mth, dustnum, hand))

### position parameter space
r_planet = np.arange(0.35, 0.39 + 1e-8, step=0.02) # arcsec
SA = np.arange(-10, 10, 2) # deg

### txt files
tag = None

### setting number of threads used by galario 
Nthreads = 64

#####################################################################

### parallel iterations over parameter space

stuff = tag, Nthreads
    
if __name__ == '__main__':
    if Nthreads in range(1,17):
        with multiprocessing.Pool() as pool:
            pool.starmap(set_iterator, [(r_planet, SA, config, uJy, a_or_i, Mth, dustnum, hand, stuff) for a_or_i, Mth, dustnum, hand in spiral_models])
            pool.close()
            pool.join()
    else:
        set_iterator(r_planet, SA, config, uJy, a_or_i[0], Mth[0], dustnum[0], hand[0], stuff)

# Done!
if tag is not None:
    print(f'\033[1m{tag}_{config}_{uJy}uJy\033[0m done!')
else:
    print(f'\033[1m{config}_{uJy}uJy\033[0m done!')