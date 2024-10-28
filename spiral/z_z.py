import numpy as np
from utils import load_profile, sim_obs_string
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# theme
sns.set_theme(palette='deep')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

disk = sim_obs_string('band6_C6', 50, 'i', '3', '3')

# load frank profile
r, intensity = load_profile(disk)

fig, ax = plt.subplots()
ax.plot(r*180*60*60/np.pi, intensity/1e10, color='red') #
ax.set(xlabel="$r$ / arcsec", ylabel='I [$10^{10}$ Jy/sr]')
ax.set_xlim(0,0.8)
fig.set_size_inches(6, 2.8)
plt.show()
