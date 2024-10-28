import itertools
from chi2_functions import (dchi2_info, LRT, LRT_radial_plot, LRT_radial_multi, 
                            chi2_scale_plot, chi2_scale_multi, chi2_heatmap, yscale_setter, set_chi2_heatmap)


# chi2_heatmap(disk, a_or_i, Mth, dustnum, hand, LRT=False, png=False, size=14)

# chi2_SA_heatmap(disk, a_or_i, Mth, dustnum, hand, polarplot=True, png=True)


## PIMDOs
config = 'C4C7'
uJy = 35


## observation parameters
a_or_i = ['a'] 
Mth = ['1']
dustnum = ['3'] #,'2']
hand = ['R', 'L']

set_chi2_heatmap(config, uJy, a_or_i, Mth, dustnum, hand, norm=False, LRT=False, save_fig=False, size=17, truncate=0.18, label=False, order_of_mag_offset=True) # 14 is good for one column, one map; 13 is good for two columns, two maps, one cbar, 15 is good for the 4by3 grid

# for a_or_i, Mth, dustnum in itertools.product(a_or_i, Mth, dustnum):

#     yscale = yscale_setter(config, uJy, a_or_i, Mth, dustnum)
#     print(yscale)

#     set_chi2_heatmap(config, uJy, a_or_i, Mth, dustnum, 'L', norm=False, LRT=False, save_fig=True, size=15, truncate=0.08, label=False, yscale=yscale, order_of_mag_offset=False, tag='_rh-axis') 



print('done!')




