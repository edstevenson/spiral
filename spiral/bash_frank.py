import json
import os
import subprocess
from galario.double import get_image_size
from utils import obs_wle, load_uvtable, sim_obs_string
import itertools

def run_frank_set(config, uJy, a_or_i, Mth, dustnum, params_file, radial_profile=False):
    for a_or_i, Mth, dustnum in itertools.product(a_or_i, Mth, dustnum):

        disk = sim_obs_string(config, uJy, a_or_i, Mth, dustnum)

        # Info. to extract
        # fit is still saved as txt file if radial_profile is False
        fit = True ; save_profile = False ;  image_specs = False ; chi2_AO = False

        if fit:
            ### FRANK FIT
            # uses frank fit from terminal with edited frank package 
            # (and edits params_file to account for different uv-table format)

            ### Edit params_file

            # Load file
            with open(f'/home/es833/Proj42/params/{params_file}.json', 'r') as json_file:
                params = json.load(json_file)

            # Edit the "norm_wle" parameter
            wavelength = obs_wle(disk) # Extract obs. wavelength
            params["modify_data"]["norm_wle"] = wavelength

            # Save modified file
            with open(f'/home/es833/Proj42/params/{params_file}.json', 'w') as json_file:
                json.dump(params, json_file, indent=2)


            ### Run frank fit in terminal

            # uses edited (Jess's edits + np.atleast_1d fix in utilities.py)
            # by setting the PYTHONPATH environment variable to include the edited frank package directory
            os.environ['PYTHONPATH'] = '/home/es833/edited_frank'

            cmd = f'python -m frank.fit -uv {disk}.txt -p /home/es833/Proj42/params/{params_file}.json'

            subprocess.run(cmd, shell=True, check=True)

        ### Info.

        if radial_profile:

            import numpy as np
            from utils import load_profile
            import matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns

            # theme
            sns.set_theme(palette='deep')
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'

            # load frank profile
            r, intensity = load_profile(disk)

            fig, ax = plt.subplots()
            ax.plot(r*180*60*60/np.pi, intensity/1e10, color='red') #
            ax.set(xlabel="$r$ / arcsec", ylabel='I [$10^{10}$ Jy/sr]')
            ax.set_xlim(0,0.8)
            fig.set_size_inches(6, 2.8)
            plt.show()

            if save_profile:
                    
                filename = f'/home/es833/Proj42_results/{disk.split("es833/", 1)[1]}_profile.png'
        
                # Create the directory if it doesn't exist
                directory = os.path.dirname(filename)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                fig.savefig(filename, bbox_inches='tight', dpi=300)

        if image_specs:

            u, v, _, _ = load_uvtable(disk)
            nxy, dxy = get_image_size(u, v, verbose=False) 
            print(nxy, dxy)

        if chi2_AO:

            from vis_functions import no_spiral_chi2
            no_spiral_chi2(disk)

def run_frank_single(single_disk, params_file, radial_profile=False):

    disk = f'/data/es833/modeldata/{single_disk}'

    with open(f'/home/es833/Proj42/params/{params_file}.json', 'r') as json_file:
        params = json.load(json_file)

    # Edit the "norm_wle" parameter
    wavelength = obs_wle(disk) # Extract obs. wavelength
    params["modify_data"]["norm_wle"] = wavelength

    # Save modified file
    with open(f'/home/es833/Proj42/params/{params_file}.json', 'w') as json_file:
        json.dump(params, json_file, indent=2)


    ### Run frank fit in terminal

    # uses edited (Jess's edits + np.atleast_1d fix in utilities.py)
    # by setting the PYTHONPATH environment variable to include the edited frank package directory
    os.environ['PYTHONPATH'] = '/home/es833/edited_frank'

    cmd = f'python -m frank.fit -uv {disk}.txt -p /home/es833/Proj42/params/{params_file}.json'

    subprocess.run(cmd, shell=True, check=True)
    if radial_profile:

            import numpy as np
            from utils import load_profile
            import matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns

            # theme
            sns.set_theme(palette='deep')
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'

            # load frank profile
            r, intensity = load_profile(disk)

            fig, ax = plt.subplots()
            ax.plot(r*180*60*60/np.pi, intensity/1e10, color='red') #
            ax.set(xlabel="$r$ / arcsec", ylabel='I [$10^{10}$ Jy/sr]')
            ax.set_xlim(0,0.8)
            fig.set_size_inches(6, 2.8)
            plt.show()


# correction due to some wierd thing that happens with Jess's disks (not sure why)
# have tried changing p2i which does change things but no clear improvement
# for now, use dra=0.0013, ddec=-0.0018 as phase centre. 

# config = 'C5C8'
# uJy = 35

# a_or_i = ['a', 'i'] 
# Mth = ['03', '1', '3']
# dustnum = ['0', '1', '2', '3']

# params_file = 'model_parameters'
# run_frank_set(config, uJy, a_or_i, Mth, dustnum, params_file, radial_profile=False)

###################

params_file = 'off_model_parameters' 

for mass in ('03Mth',):
    single_disk = f'model_fixphase_inc35_rot90_35uJy_C4C7_a-{mass}-dust2'
    run_frank_single(single_disk, params_file, radial_profile=True)


###################

# /data/es833/C4C7_35uJy_simulated_observations/dusty_spirals_adi/dsa-1qth-h007-beta10/f3d2c_dust2_orbit1500_EOSadi_EPSTEIN_2D_CPD0/345GHz.35uJy/345GHz.35uJy.alma.concat.cycle8.4.cycle8.7.noisy.ms.uvtable.txt

