### enter /data/es833/spirals before running as uses cwd ./ to find spirals files 
### run $ python /home/es833/Proj42/fargo/spirals.py . 1 to plot deltaI/I for the first snapshot.
 

import sys, os
try:
    FARGO_DIR = os.environ["FARGO_PATH"]
    sys.path.append(os.path.join(FARGO_DIR, "python"))
except KeyError:
    pass

from fargo_lib import FARGO_Sim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    Sim = FARGO_Sim(sys.argv[1])
except IndexError:
    Sim = FARGO_Sim('./') 
    
try:
    snap_num = int(sys.argv[2])
except IndexError:
    snap_num = 0

    
r   = Sim.Rmed
phi = Sim.phimed 
print(phi[0:2], phi[-3:])

rho_g  = Sim.load_field('gasdens', snap_num)
vr_g   = Sim.load_field('gasvy',  snap_num)
vphi_g = Sim.load_field('gasvx',  snap_num)

vphi_g =  (vphi_g + Sim.get_parameter("OMEGAFRAME") * r) - r**-0.5


# Get the temperature
# Does not have correct units, but doesn't matter either
T = Sim.load_field('gasenergy', snap_num)
T0 = T.mean(0)

# Get the optical depth
tau_planet = 0.1 # Specify the optical depth scale
kappa = tau_planet / (Sim.get_parameter('SIGMA0')*Sim.get_parameter('DUST_TO_GAS'))


# Get the optical depth and the azimuthally averaged optical depth.
tau0 = Sim.load_field('dustdens1_', snap_num).mean(0) * kappa
tau = Sim.load_field('dustdens1_', snap_num) * kappa

# Compute the intensity to the azimuthal average
I0 = T0 * np.exp(-tau0)
dI_I = T*np.exp(-tau)/I0 - 1

plt.pcolormesh(r, phi/np.pi, dI_I)
plt.colorbar(label=r'$\delta I/ I$')
plt.xlabel(r"$r$")
plt.ylabel(r"$\phi$")

plt.show()
