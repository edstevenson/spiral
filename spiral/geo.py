### use to output fitted disk geometry values to json file 
### can also use to check that gaussian fit is accurate 

import json
from utils import load_disk, load_parameters
from frank.geometry import FitGeometryGaussian, FitGeometryFourierBessel

disk = 'DR_Tau'
fixed_inc_pa = True # True => using Long's values
Fourier_Bessel = False

# load uvdata, profile, disk parameters 
u, v, V, w, _, _ = load_disk(disk)
params = load_parameters(disk)
Rmax = params['hyperparameters']['rout']
N = params['hyperparameters']['n']
inc = params['geometry']['inc']
PA = params['geometry']['pa']
dRA = params['geometry']['dra']
dDec = params['geometry']['ddec']
guess = [inc, PA, dRA, dDec]

print('\nFixed geometry:\n\t\t inc  = {:.2f} deg,\n\t\t PA   = {:.2f} deg,\n\t\t'
' dRA  = {:.2e} mas,\n\t\t dDec = {:.2e} mas\n'.format(inc, PA, dRA*1e3, dDec*1e3))


if Fourier_Bessel and fixed_inc_pa:

    fit_type = 'Fourier-Bessel'
    geom = FitGeometryFourierBessel(Rmax=Rmax, N=N, inc_pa=[inc,PA], guess=guess,verbose=True)

elif Fourier_Bessel:

    fit_type = 'Fourier-Bessel'
    geom = FitGeometryFourierBessel(Rmax=Rmax, N=N, guess=guess,verbose=True)

elif fixed_inc_pa:

    fit_type = 'Gaussian'
    geom = FitGeometryGaussian(inc_pa=[inc,PA])

else:

    fit_type = 'Gaussian'
    geom = FitGeometryGaussian(guess=guess)

    
geom.fit(u, v, V, w)


print('\n{} Fitted geometry:\n\t\t inc  = {:.2f} deg,\n\t\t PA   = {:.2f} deg,\n\t\t'
' dRA  = {:.2e} mas,\n\t\t dDec = {:.2e} mas\n'.format(fit_type, geom.inc, geom.PA, geom.dRA*1e3, geom.dDec*1e3))

inc, PA, dRA, dDec = geom.inc, geom.PA, geom.dRA, geom.dDec 

geometry = {'inc': inc, 'PA': PA, 'dRA': dRA, 'dDec': dDec}
param_filename = '/data/es833/dataforfrank/uv_tables/' + disk + '_geo.json'
with open(param_filename, 'w') as f:
    json.dump(geometry, f, indent=4)