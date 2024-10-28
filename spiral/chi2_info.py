from chi2_functions import (dchi2_info, LRT, LRT_radial_plot, LRT_radial_multi, 
                            chi2_scale_plot, chi2_scale_multi, chi2_heatmap, chi2_SA_heatmap,
                            set_chi2_heatmap)

disk = 'model_inc50_rot90_35uJy_C4C7_a-03Mth-dust2'

### simulation parameter space
a_or_i = ['a']
Mth = ['03']
dustnum = ['2']
hand = ['R'] 

chi2_heatmap(disk, a_or_i, Mth, dustnum, hand, norm=False, size=19, truncate=0.02, cbar_scale=[], cbar_ticks=[], label=False, order_of_mag_offset=True, save_fig=False) # 14 is good for one column, one map; 13 is good for two columns, two maps, one cbar, 15 is good for two columns and three maps
# # disk = 'model_35uJy_C4C7_a-1Mth-dust3' 

# ### simulation parameter space

# a_or_i = ['a']
# Mth = ['03']
# dustnum = ['2']
# hand = ['R'] 

# cbar_ticks = [100*e for e in [-9, -6, -3, 0, 0.8, 1.6, 2.4]]

# for thing in ('_inc35', '_inc25'):
#     for string in ('03Mth',):
#         disk = f'model{thing}_inc30_rot90_35uJy_C4C7_a-{string}-dust2'

#         chi2_heatmap(disk, a_or_i, Mth, dustnum, hand, norm=False, size=19, truncate=0.02, cbar_scale=[-9.63e2,2.7e2], cbar_ticks=cbar_ticks, label=False, order_of_mag_offset=True, save_fig=False, folder='/home/es833/Proj42_results/incorrect_geom') # 14 is good for one column, one map; 13 is good for two columns, two maps, one cbar, 15 is good for two columns and three maps



# chi2_scale_plot(disk, a_or_i, Mth, dustnum, hand, png=False)

# chi2_SA_heatmap(disk, a_or_i, Mth, dustnum, hand, png=True)



    



print('done!')




