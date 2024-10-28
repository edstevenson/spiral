from utils import pretty_image
import numpy as np
import itertools

disk = 'RY_Tau' 

### Parameters

r_planet = 0.34; SA = 30 # arcsec, deg

a_or_i = 'a'#,'i']
Mth = '1'     #['01','03','1','3']
dustnum = '1' #['0','1','2','3']
hand = 'R'

# for _, (a_or_i, Mth, dustnum) in enumerate(itertools.product(a_or_i, Mth, dustnum)):

pretty_image(disk, r_planet, SA, a_or_i, Mth, dustnum, hand, nxy=2**11, rmax=0.8,  project=True, log=True, png=True, size=15)


print(disk + ' done!')

