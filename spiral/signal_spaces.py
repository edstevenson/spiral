from signal_functions import gap_finder, detection_space, detection_space_qual, spiral_strength, spiral_strength_close
import numpy as np  
import itertools

config = 'C4C7'
uJy = 50

spirals = True
signal = 'spiral' if spirals else 'gap'

a_or_i = ['a','i'] 
Mth_list = ['03','1','3']   
dustnum_list = ['0','1','2','3']

z = {key: np.zeros((len(dustnum_list), len(Mth_list))) for key in a_or_i}

# for eos in a_or_i:
#     for (i, dustnum), (j, Mth) in itertools.product(enumerate(dustnum_list), enumerate(Mth_list)):      
#         if spirals:
#             # z[eos][i,j] = spiral_strength(config, uJy, eos, Mth, dustnum)
#             z[eos][i,j] = spiral_strength_close(config, uJy, eos, Mth, dustnum, dR=0.1, dPhi=90)
#         else:
#             z[eos][i,j] = gap_finder(config, uJy, eos, Mth, dustnum, plot=False, save_fig=False)

# detection_space(z['a'], z['i'], spirals=spirals, save_fig=False, figname=f'{config}_{uJy}uJy_{signal}')


# ---------------------------------------------------------------

# ### QUALITATIVE COMPARISON OF PLANET STRENGTHS; 0,1,2 --> no, marginal, clear
z['a'] = [[2,2,2],
          [2,2,2],
          [2,2,2],
          [2,2,2]]

z['i'] = [[0,2,2],
          [0,2,2],
          [0,2,2],
          [1,2,2]]

# # for C5C8 35: the i-1-0, a-03-0 cases clearly recoverd, i-03-3 is marginal, i-03-2 not quite recoverable 

detection_space_qual(z['a'], z['i'], spirals=spirals, save_fig=True, figname='Tau_comparison', size=44)


