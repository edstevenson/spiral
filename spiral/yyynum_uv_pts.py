from utils import load_uvtable, sim_obs_string
from vis_functions import no_spiral_chi2
import itertools
# config = 'C5C8'
# uJy = 35
# 
# disk = sim_obs_string(config, uJy, 'a', '03', '3')

# taurus_disks = ['FT_Tau', 'DR_Tau', 'BP_Tau', 'DO_Tau', 'RY_Tau']
# 
# syn_disks = [sim_obs_string('band6_C6', 50, a_or_i, Mth, dustnum) for a_or_i, Mth, dustnum in itertools.product(['a', 'i'], ['03', '1', '3'], ['0', '1', '2', '3'])]
# syn_params = [f'{a_or_i}_{Mth}_{dustnum}' for a_or_i, Mth, dustnum in itertools.product(['a', 'i'], ['03', '1', '3'], ['0', '1', '2', '3'])]
# 
# disks = zip(taurus_disks+syn_disks, taurus_disks+syn_params)

disks = [(sim_obs_string('C4C7', 35, 'a', '3', '3'), 'disks')]


for disk, name in disks:

    u,_,_,_ = load_uvtable(disk)
    chi2_, red_chi2_ = no_spiral_chi2(disk)
    print(f'{name} has {len(u)} data points; chi2_AO = {chi2_:.0f}, reduced chi2_AO = {red_chi2_:.2f}')
    