from signal_functions import gap_finder, clean_profile_plotter
import numpy as np  
import itertools

config = 'C5C8'
uJy = 35

a_or_i = ['a','i']
Mth_list = ['03','1','3'] 
dustnum_list = ['0','1','2','3']

for eos in a_or_i:
    for (i, dustnum), (j, Mth) in itertools.product(enumerate(dustnum_list), enumerate(Mth_list)):      
        _ = gap_finder(config, uJy, eos, Mth, dustnum, plot=True, save_fig=True, label=True, size=24)

# for (i, dustnum), (j, Mth) in itertools.product(enumerate(dustnum_list), enumerate(Mth_list)):  
#     label = True # if dustnum == '0' and Mth == '03' else False
#     clean_profile_plotter(config, uJy, Mth, dustnum, save_fig=True, label=label, size=24)