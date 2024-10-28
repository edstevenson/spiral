from vis_functions import diagnostics
from utils import sim_obs_string

disk = sim_obs_string('C4C7','10', 'a', '03', '3')

diagnostics(disk, recentre=True, png=False)