import pandas as pd
import numpy as np
from tqdm import tqdm 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter, ScalarFormatter, FuncFormatter
import scipy.stats as stats
import itertools
from utils import round_to_1sf, add_dot, load_profile, ensure_list
import seaborn as sns
from cmcrameri import cm


distance = {
    'FT_Tau': 127,
    'DR_Tau': 195,
    'BP_Tau': 129,
    'DO_Tau': 139,
    'RY_Tau': 128,
    'UZ_Tau': 131,
}


def dchi2_info(disk, a_or_i, Mth, dustnum, hand, avg_dchi2=False, best_dchi2=False):
    """
    prints best and/or avg spiral dchi2/chi2 (chi2 is for axisym model) for each spiral model
    prints r (arcsec), SA (deg) for best spiral if you set best_dchi2=True only
    """

    for _, (a_or_i, Mth, dustnum, hand) in enumerate(itertools.product(a_or_i, Mth, dustnum, hand)):

        txtfile = '/data/es833/chi2_txtfiles/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'
        df = pd.read_csv(txtfile, sep='\t')
        Dchi2 = df['dchi2/chi2 (%)'].values
        Ds = Dchi2[1:]
        r = df['r_planet [arcsec]'].values
        r = r[1:]
        phi = df['spiral angle [deg]'].values
        phi = phi[1:]
        # correct for azimuthal shift - so every map has phi=0 corresponding to NEED FOR OLD WAY 
        # planet at phi = 0
        # dphi = get_dphi(a_or_i, Mth, dustnum)
        # phi += dphi*180/np.pi
        # phi %= 360 

        if avg_dchi2 and best_dchi2:
            
            Ds_avg = np.sum(Ds) / len(Ds)
            Ds_best = Ds.max()
            print(f'{Ds_avg:.4f} %  \t{Ds_best:.4f} %  \t', '(' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + ')')

        elif avg_dchi2:
            ### average Ds 
            Ds_avg = np.sum(Ds) / len(Ds)
            print(f'{Ds_avg:.4f} %  \t', '(' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + ')')

        elif best_dchi2:
            ### minimum chi2 
            idx = np.argmax(Ds)  
            print(f'{Ds[idx]:.4f} %\t{round(r[idx],3)}\t{round(phi[idx],1)}  \t', '(' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + ')')
        
        else:
            pass
    
    if avg_dchi2 == False and best_dchi2 == False:
        print('choose avg_dchi2 or best_dchi2 for me to show something\n')
    

def LRT(disk, a_or_i, Mth, dustnum, hand):
    """Likelihood Ratio Test"""

    print('probability that best spiral is a better model than no spiral:\n')
    for _, (a_or_i, Mth, dustnum, hand) in enumerate(itertools.product(a_or_i, Mth, dustnum, hand)):

        txtfile = '/data/es833/chi2_txtfiles/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'
        df = pd.read_csv(txtfile, sep='\t')
        chi2 = df['chi2']
        chi2_0 = chi2[0]
        chi2_s = chi2[1:]
        chi2_best = chi2_s.min()

        ### LRT
        dof = 6
        lambda_LR = chi2_0 - min(chi2_best, chi2_0) 

        chi2_dist = stats.chi2(dof)
        cdf = chi2_dist.cdf(lambda_LR)

        # p = 1 - cdf 
        # p = statistical significance = prob. of obtaining at least as extreme a result given that null hypo. is true
        # null hypo. = no spiral is best model
        # therefore, p = prob. the spiral-including model is a worse fit, cdf = prob. spiral-including model is a better fit

        print(f'{cdf*100} %  \t', '(' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + ')') 


def LRT_radial_plot(disk, a_or_i, Mth, dustnum, hand, png=False):
    """Statistical significance from Likelihood Ratio Test for each radius"""

    N = len(list(itertools.product(a_or_i, Mth, dustnum, hand)))
    with tqdm(total=N) as pbar:
        for _, (a_or_i, Mth, dustnum, hand) in enumerate(itertools.product(a_or_i, Mth, dustnum, hand)):

            txtfile = '/data/es833/chi2_txtfiles/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'
            df = pd.read_csv(txtfile, sep='\t')
            chi2 = df['chi2'].values
            chi2_0 = chi2[0]
            chi2_s = chi2[1:]
            r = df['r_planet [arcsec]'].values
            r = r[1:]
            phi = df['spiral angle [deg]'].values
            phi = phi[1:]

            Nphi = np.argmax(phi)+1
            Nr = int(len(r)/Nphi)

            # this is to correct for unusual thing where chi2txtfile sometimes contains a few repeats at the end
            r, chi2_s = r[:Nphi*Nr], chi2_s[:Nphi*Nr]

            shape = Nr, Nphi # each row is a particular radius
            R = r.reshape(shape) 
            rad = R[:,0]
            CHI2 = chi2_s.reshape(shape)
            chi2_best = np.min(CHI2, axis=1)

            # LRT
            dof = 6
            lambda_LR = chi2_0 - chi2_best
            chi2_dist = stats.chi2(dof)
            cdf = chi2_dist.cdf(lambda_LR) 
            # cdf translates as prob. of correctly rejecting null hyp. i.e. prob. that spiral model is a better fit

            ### plotting

            # theme
            sns.set_theme(palette='deep')
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'

            fig, ax = plt.subplots()
            ax.plot(rad, cdf, zorder=2, linewidth=1.5)
            ax.set_xlabel('$r$')
            ax.set_ylabel('$1-p$') # p=p-value.
            ax.set_xlim(0, rad[-1]+0.002)  
            tick_locs = np.arange(rad[0], rad[-1]+1e-8, 0.02)
            if rad[-1] >= 0.8:
                tick_labels = [str(round(rad[0],2))]+['']*3+[0.1]+['']*4+[0.2]+['']*4+[0.3]+['']*4+[0.4]+['']*4+[0.5]+['']*4+[0.6]+['']*4+[0.7]+['']*4+[str(round(rad[-1],2))]
            else:
                tick_labels = [str(round(rad[0],2))]+['']*3+[0.1]+['']*4+[0.2]+['']*4+[0.3]+['']*4+[0.4]+['']*4+[str(round(rad[-1],2))]
            ax.set_xticks(tick_locs, tick_labels)
            ax.set_ylim(-0.005,1.005)
            ax.set_yticks([0.1*i for i in range(11)], [0]+['']*9+[1])    
            ax.set_title(disk + ', ' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand, pad=20)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            

            if png:
                filename = '/home/es833/Proj42_results/' + disk + '/LRT_radial_plots/' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_LRT.png'
                fig.savefig(filename, bbox_inches='tight', dpi=300)
            else:
                plt.show()
            plt.close()

            # Update the progress bar
            pbar.update(1)


def LRT_radial_multi(disk, a_or_i, Mth, dustnum, hand, png=False, tag='multi'):
    """Statistical significance from Likelihood Ratio Test for each radius"""

    # theme
    sns.set_theme(palette='deep')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    N = len(list(itertools.product(a_or_i, Mth, dustnum, hand)))
    with tqdm(total=N) as pbar:
        for j, (a_or_i, Mth, dustnum, hand) in enumerate(itertools.product(a_or_i, Mth, dustnum, hand)):

            txtfile = '/data/es833/chi2_txtfiles/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'
            df = pd.read_csv(txtfile, sep='\t')

            if j == 0:    
                r = df['r_planet [arcsec]'].values
                r = r[1:]
                phi = df['spiral angle [deg]'].values
                phi = phi[1:]
                Nphi = np.argmax(phi)+1
                Nr = int(len(r)/Nphi)
                
                ### plotting
                fig, ax = plt.subplots()
                ax.set_xlabel('$r$ [arcsec]')
                ax.set_ylabel('$1-p$', rotation='horizontal') # p=p-value.
                ax.set_xlim(0, r[-1]+0.002)  
                tick_locs = np.arange(r[0], r[-1]+1e-8, 0.02)
                if r[-1] >= 0.8:
                    tick_labels = [str(round(r[0],2))]+['']*3+[0.1]+['']*4+[0.2]+['']*4+[0.3]+['']*4+[0.4]+['']*4+[0.5]+['']*4+[0.6]+['']*4+[0.7]+['']*4+[str(round(r[-1],2))]
                else:
                    tick_labels = [str(round(r[0],2))]+['']*3+[0.1]+['']*4+[0.2]+['']*4+[0.3]+['']*4+[0.4]+['']*4+[str(round(r[-1],2))]
                ax.set_xticks(tick_locs, tick_labels)
                ax.set_ylim(-0.005,1.005)
                ax.set_yticks([0.1*i for i in range(11)], [0]+['']*9+[1])    
                ax.set_title(disk, pad=20)
                r = r[:Nphi*Nr]
                shape = Nr, Nphi
                R = r.reshape(shape) 
                r = R[:,0]
            
            chi2 = df['chi2'].values
            chi2_0 = chi2[0]
            chi2_s = chi2[1:]

            # this is to correct for unusual thing where chi2txtfile sometimes contains a few repeats at the end
            chi2_s =  chi2_s[:Nphi*Nr]

            # each row is a particular radius
            CHI2 = chi2_s.reshape(shape)
            chi2_best = np.min(CHI2, axis=1)

            # LRT
            dof = 6
            lambda_LR = chi2_0 - chi2_best
            chi2_dist = stats.chi2(dof)
            cdf = chi2_dist.cdf(lambda_LR) 
            # cdf translates as prob. of correctly rejecting null hyp. i.e. prob. that spiral model is a better fit

            label = a_or_i + '-' + Mth + 'Mth-dust' + dustnum
            ax.plot(r, cdf, zorder=2, linewidth=1.5, label=label)

            # Update the progress bar
            pbar.update(1)

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if png:
        filename = '/home/es833/Proj42_results/' + disk + '/LRT_radial_plots/multis/' + tag + '_LRT.png'
        fig.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()


def chi2_scale_plot(disk, a_or_i, Mth, dustnum, hand, png=False):
    """
    - chi2 taken between axisym. disk + spiral model, and axisym. disk model
    - chi2/chi2_0 plotted against radius
    - gives a sense of the fractional increase in chi2 expected 
    for an 'incorrect' disk model
    """
    N = len(list(itertools.product(a_or_i, Mth, dustnum, hand)))
    with tqdm(total=N) as pbar:
        for _, (a_or_i, Mth, dustnum, hand) in enumerate(itertools.product(a_or_i, Mth, dustnum, hand)):

            txtfile = '/data/es833/chi2_txtfiles/scales/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'
            df = pd.read_csv(txtfile, sep='\t')
            r = df['r_planet [arcsec]'].values
            rad = r[1:]
            Ds = df['chi2/chi2_0 (%)'].values
            Ds = Ds[1:]

            ### plotting

            # theme
            sns.set_theme(palette='deep')
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'

            fig, ax = plt.subplots()
            ax.set_xlabel('$r$') 
            ax.set_xlim(rad[0]-0.005, rad[-1]+0.005)        
            ax.set_ylabel(r'$\dfrac{\chi_{SA}^{2}}{\chi_{AO}^{2}}$ /%', rotation='horizontal', labelpad=20)
            ax.set_title('Effect of Spiral on Axisymmetric Disk Model\n' + disk + ', ' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand, pad=20)
            tick_locs = np.arange(rad[0], rad[-1]+1e-8, 0.02)
            if rad[-1] >= 0.8:
                tick_labels = [str(round(rad[0],2))]+['']*3+[0.1]+['']*4+[0.2]+['']*4+[0.3]+['']*4+[0.4]+['']*4+[0.5]+['']*4+[0.6]+['']*4+[0.7]+['']*4+[str(round(rad[-1],2))]
            else:
                tick_labels = [str(round(rad[0],2))]+['']*3+[0.1]+['']*4+[0.2]+['']*4+[0.3]+['']*4+[0.4]+['']*4+[str(round(rad[-1],2))]
            ax.set_xticks(tick_locs, tick_labels)

            ax.plot(rad, Ds, zorder=2, linewidth=1.5)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(6))  

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            if png:
                filename = '/home/es833/Proj42_results/' + disk + '/chi2_scale_plots/' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_scale.png'
                fig.savefig(filename, bbox_inches='tight', dpi=300)
            else:
                plt.show()
            plt.close()

            # Update the progress bar
            pbar.update(1)


def chi2_scale_multi(disk, a_or_i, Mth, dustnum, hand, png=False, log=False, tag='multi', size=18, I_profile=False):
    """
    like chi2_scale_plot except all on same figure
    """

    # theme
    sns.set_theme(style='whitegrid', palette='deep')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    N = len(list(itertools.product(a_or_i, Mth, dustnum, hand)))
    color_dict = {}  # Dictionary to store colors

    with tqdm(total=N) as pbar:
        for j, (dustnum, Mth, a_or_i, hand) in enumerate(itertools.product(dustnum, Mth, a_or_i, hand)):

            txtfile = '/data/es833/chi2_txtfiles/scales/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'
            # txtfile = '/data/es833/chi2_txtfiles/scales/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '_chi2.txt'
            df = pd.read_csv(txtfile, sep='\t')

            if j == 0:
                r = df['r_planet [arcsec]'].values
                rad = r[1:]
                Nr = len(rad)
                Ds = np.zeros((N,Nr))

                ### plotting

                fig, ax = plt.subplots()
                ax.set_xlabel('$r$ / arcsec', fontsize=size) 
                ax.set_xlim(rad[0]-0.005, rad[-1]+0.005)        
                ax.set_ylabel(r'$\dfrac{\chi_{SA}^{2}}{\chi_{AO}^{2}}$', rotation='horizontal', labelpad=25, fontsize=size, )
                ax.yaxis.set_major_locator(ticker.MaxNLocator(6)) 
                tick_locs = np.arange(rad[0], rad[-1]+1e-8, 0.02)
                if rad[-1] >= 0.8:
                    tick_labels = [str(round(rad[0],2))]+['']*3+[0.1]+['']*4+[0.2]+['']*4+[0.3]+['']*4+[0.4]+['']*4+[0.5]+['']*4+[0.6]+['']*4+[0.7]+['']*4+[str(round(rad[-1],2))]
                else:
                    tick_labels = [str(round(rad[0],2))]+['']*3+[0.1]+['']*4+[0.2]+['']*4+[0.3]+['']*4+[0.4]+['']*4+[str(round(rad[-1],2))]
                ax.set_xticks(tick_locs, tick_labels)
                ax.tick_params(axis='both', which='major', labelsize=size)
                ax.yaxis.set_label_coords(-0.21, 0.4)


                if I_profile:
                    # Create a second y-axis
                    ax2 = ax.twinx()
                    r, intensity = load_profile(disk)
                    ax2.plot(r*180*60*60/np.pi, intensity/1e10, color='black') 

                    # Set label for the right y-axis
                    ax2.set_ylabel('I [$10^{10}$ Jy/sr]', fontsize=size)  # Modify as needed

                    # ax2.set_ylim(min_value, max_value)

                    # Additional formatting for right y-axis
                    ax2.tick_params(axis='y', labelsize=size)

                    ax2.grid(False)

                    
            Ds = df['chi2/chi2_0 (%)'].values
            Ds = Ds[1:] / 100 # Convert to fraction

            if 'Mth' in tag:

                if a_or_i == 'a':
                    cooling_time = 10
                
                else:
                    cooling_time = 0
            
                if dustnum == '0':
                    d = 0.1
                    space = '\,'
                elif dustnum == '1':
                    d = 0.3
                    space = '\,'
                elif dustnum == '2':
                    d = 1
                    space = r'\quad\,'
                else:
                    d = 3
                    space = r'\quad\,'

                label = fr'$\tau_0 = {d},{space} \beta = {cooling_time}$'


                # Generate a unique key for each combination of Mth and dustnum
                color_key = (Mth, dustnum)

                if a_or_i == 'a':
                    # Plot normally for 'a' and store the line color
                    line, = ax.plot(rad, Ds, zorder=2, linewidth=1.5, label=label)  # Keep reference to the line
                    color_dict[color_key] = line.get_color()  # Store the color
                elif a_or_i == 'i':
                    # For 'i', use dashed lines and retrieve color from corresponding 'a'
                    if color_key in color_dict:
                        color = color_dict[color_key]
                        ax.plot(rad, Ds, zorder=2, linewidth=1.5, label=label, linestyle='--', color=color)
                    else:
                        # Fallback color if 'a' case not plotted yet
                        ax.plot(rad, Ds, zorder=2, linewidth=1.5, label=label, linestyle='--', color='grey')

            else:
                label = fr'${add_dot(Mth)}\, M_{{\mathrm{{th}}}}$'
                ax.plot(rad, Ds, zorder=2, linewidth=1.5, label=label)

            ax.legend(fontsize=size) 
            if log:
                ax.set_yscale('log')

            # Update the progress bar
            pbar.update(1)

    if I_profile:
        ax.legend(bbox_to_anchor=(1.135, 1.04), loc='upper left', fontsize=size) 
    else:
        ax.legend(bbox_to_anchor=(1.02, 1.05), loc='upper left', fontsize=size) 
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    

    if png:
        filename = '/home/es833/Proj42_results/'+ disk + '/chi2_scale_plots/multis/' + tag + '_scale.pdf'
        fig.savefig(filename, bbox_inches='tight', dpi=300)

    else:
        plt.show()
    plt.close()


def chi2_SA_heatmap(disk, a_or_i, Mth, dustnum, hand, png=False):

    N = len(list(itertools.product(a_or_i, Mth, dustnum, hand)))
    with tqdm(total=N) as pbar:
        for _, (a_or_i, Mth, dustnum, hand) in enumerate(itertools.product(a_or_i, Mth, dustnum, hand)):

            txtfile = '/data/es833/chi2_txtfiles/chi2_SA/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'
            df = pd.read_csv(txtfile, sep='\t')
            r = df['r_planet [arcsec]'].values
            r = r[1:]
            phi = df['spiral angle [deg]'].values
            phi = phi[1:]
            Ds = df['chi2/chi2_0 (%)'].values
            Ds = Ds[1:] / 100 # convert to fraction

            Nphi = np.argmax(phi)+1
            Nr = int(len(r)/Nphi)

            # this is to correct for unusual thing where chi2txtfile contaings a few repeats at the end
            r, phi, Ds = r[:Nphi*Nr], phi[:Nphi*Nr], Ds[:Nphi*Nr]
            
            # this is to correct for gap between 360-step and 0 
            for i in range(Nr-1):
            # inserts values at 360deg points (just before 0deg points)
                phi = np.insert(phi, (i+1)*Nphi+i, 360)
                r = np.insert(r, (i+1)*Nphi+i, r[i*Nphi+i])
                Ds = np.insert(Ds, (i+1)*Nphi+i, Ds[i*Nphi+i])
            phi = np.append(phi, 360)
            r = np.append(r, r[-Nphi])
            Ds = np.append(Ds, Ds[-Nphi])

            # # correct for azimuthal shift - so every map has phi=0 corresponding to NEED FOR OLD WAY 
            # # planet at phi = 0
            # dphi = get_dphi(a_or_i, Mth, dustnum)
            # phi += dphi*180/np.pi

            shape = Nr, Nphi +1
            R = r.reshape(shape)
            Phi = phi.reshape(shape)
            F = Ds.reshape(shape)

            # theme
            sns.set_theme(palette='deep')
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'

            divnorm = colors.LogNorm() # vmin=Ds.min(), vmax=Ds.max())
            cmap = cm.devon

            ### Plotting
            size = 16

   
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            c = ax.pcolormesh(Phi*np.pi/180, R, F, cmap=cmap, norm=divnorm, zorder=1)
            
            # worth altering for high res versions or other disks
            if r[-1] >= 0.8:
                ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            else:                        
                ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])

            ax.tick_params(axis='both', labelsize=size)
            ax.yaxis.set_zorder(10)
            ax.set_xticks([])
            ax.grid(True, axis='y', color='black', linewidth=0.1)


            # Create a colorbar
            cbar = fig.colorbar(c, ax=ax)
            
            cbar.ax.set_ylabel(ylabel=r'$\dfrac{\chi_{SA}^{2}}{\chi_{AO}^{2}}$', rotation='horizontal', labelpad=30, fontsize=size+1)
            cbar.ax.yaxis.set_label_coords(5.9, 0.59)

            cbar.ax.tick_params(labelsize=size)
            fmt = ticker.FormatStrFormatter('%.2g')
            cbar.ax.yaxis.set_major_formatter(fmt)
            cbar.formatter = ticker.LogFormatterSciNotation()
            cbar.update_ticks()

            if png:
                filename = '/home/es833/Proj42_results/' + disk + '/chi2_SA/' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2_' +'.png'
                fig.savefig(filename, bbox_inches='tight', dpi=300)
            else:
                plt.show()

            plt.close()

            # Update the progress bar
            pbar.update(1)            
    

def chi2_heatmap(disk, a_or_i, Mth, dustnum, hand, norm=False, size=14, truncate=0, cbar_scale=[], cbar_ticks=[], label=True, order_of_mag_offset=True, save_fig=False, folder=None):

    N = len(list(itertools.product(a_or_i, Mth, dustnum, hand)))
    with tqdm(total=N) as pbar:
        for _, (a_or_i, Mth, dustnum, hand) in enumerate(itertools.product(a_or_i, Mth, dustnum, hand)):

            txtfile = '/data/es833/chi2_txtfiles/' + disk + '_' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2.txt'
            df = pd.read_csv(txtfile, sep='\t')
            r = df['r_planet [arcsec]'].values
            r = r[1:]
            phi = df['spiral angle [deg]'].values
            phi = phi[1:]
            chi2 = df['chi2'].values
            chi2_0 = chi2[0]
            Dchi2 = chi2_0 - chi2
            Dchi2 = Dchi2[1:]
            print(f'peak Dchi2 = {Dchi2.max()}\t min Dchi2 = {Dchi2.min()}')
            Dchi2_norm = df['dchi2/chi2 (%)'].values
            Dchi2_norm = Dchi2_norm[1:]

            if norm:
                Ds = Dchi2_norm
            else:
                Ds = Dchi2

            if truncate:
                trunc_index = np.where(np.isclose(r, truncate, atol=1e-6))[0][0] if np.any(np.isclose(r, truncate, atol=1e-6)) else print('this radius not in data')
                r = r[trunc_index:]
                phi = phi[trunc_index:]
                Ds = Ds[trunc_index:]
                
            Nphi = np.argmax(phi)+1
            Nr = int(len(r)/Nphi)

            # this is to correct for unusual thing where chi2txtfile contains a few repeats at the end
            r, phi, Ds = r[:Nphi*Nr], phi[:Nphi*Nr], Ds[:Nphi*Nr]

            shape = Nr, Nphi #+1
            R = r.reshape(shape)
            Phi = phi.reshape(shape)
            F = Ds.reshape(shape)

            # theme
            sns.set_theme(palette='deep')
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'


            if cbar_scale:
                Dsmin = Ds.min() if cbar_scale[0] is None else cbar_scale[0]
                Dsmax = Ds.max() if cbar_scale[1] is None else cbar_scale[1]
            else:
                Dsmin = Ds.min()
                Dsmax = Ds.max()
            divnorm = colors.TwoSlopeNorm(vmin=min(Dsmin,-1e-8), vcenter=0., vmax=max(Dsmax,1e-8))
            cmap = 'coolwarm'

            ### Plotting

            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

            plt.pcolormesh(Phi*np.pi/180, R, F, cmap=cmap, norm=divnorm, zorder=1)
        
            # r ticks

            d=distance.get(disk, 140)
            # ax.set_yticks(np.arange(20/d, r[-1], step=20/d))
            # ax.set_yticklabels(np.arange(20, round(r[-1]*d), step=20))
            ax.set_yticks(np.arange(10/140, 0.51, step=20/140))
            ax.set_yticklabels(np.arange(10, 71, step=20), fontsize=size)
            ax.yaxis.set_zorder(10) 
            ax.set_xticks([])
            ax.grid(True, axis='y', color='black', linewidth=0.1)
            ax.set_rlabel_position(22.5)
            # r_label_position = (22.5*np.pi/180, 0.9)  # Set the label position (angle, radius)
            # ax.text(*r_label_position, r"$r ['']$", rotation=0, ha='center', va='center', fontsize=9)
            if save_fig == False:
                ax.set_title(f"{disk}_{a_or_i}-{Mth}Mth-dust{dustnum}-{hand}", pad=20)
            style = 'polar'

            if cbar_ticks:
                ticks = cbar_ticks
            else:
                if Dsmin > 0:
                    t1 = round_to_1sf(np.array([Dsmax/4])).item()
                    ticks = np.linspace(0,4*t1,5)
                elif Dsmax < 0:
                    t2 = round_to_1sf(np.array([Dsmin/4])).item()
                    ticks = np.linspace(4*t2,0,5)
                else:
                    t1 = round_to_1sf(np.array([Dsmax/4])).item()
                    t2 = round_to_1sf(np.array([Dsmin/4])).item()
                    # t1 = 0.02
                    ticks1 = np.linspace(0,5*t1,6)
                    # t2 = -0.05
                    ticks2 = np.linspace(4*t2,t2,4)
                    ticks = np.hstack([ticks2, ticks1])

            cbar = plt.colorbar(ticks=ticks)
            if label:
                if norm:
                    cbar.ax.set_ylabel(ylabel=r'$\dfrac{\Delta\chi^{2}}{\chi_{AO}^{2}}$ /%', rotation='horizontal', labelpad=10, fontsize=size+1) 
                else:
                    cbar.ax.set_ylabel(ylabel=r'$\Delta\chi^{2}$', rotation='horizontal', labelpad=10, fontsize=size+1)

            cbar.ax.yaxis.set_label_coords(5, 0.5)
            cbar.ax.tick_params(labelsize=size)
            if order_of_mag_offset:
                fmt = ScalarFormatter(useMathText=True, useOffset=True)
                fmt.set_powerlimits((-1, 1))
                cbar.ax.yaxis.set_major_formatter(fmt)
                offset_text = cbar.ax.yaxis.get_offset_text()
                offset_text.set_size(size)
                offset_text.set_position((2, 1))
            else:
                def sci_notation(x, pos):
                    """Custom formatter to display labels in scientific notation"""
                    if x == 0:
                        return '0'
                    exponent = np.floor(np.log10(np.abs(x)))
                    coeff = x / 10**exponent
                    return f"{coeff:.1f} $\\times$ $10^{{{int(exponent)}}}$"

                cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_notation))

            if save_fig:
                if folder is None:
                    filename = '/home/es833/Proj42_results/' + disk + '/chi2_plots/' + a_or_i + '-' + Mth + 'Mth-dust' + dustnum + '-' + hand + '_chi2_' + style + '.png'
                else:
                    filename = f'{folder}/{disk}_{a_or_i}-{Mth}Mth-dust{dustnum}-{hand}_chi2.png'
                fig.savefig(filename, bbox_inches='tight', dpi=300)
            else:
                plt.show()

            plt.close()

            # Update the progress bar
            pbar.update(1)            




def set_chi2_heatmap(config, uJy, a_or_i, Mth, dustnum, hand, norm=False, LRT=False, save_fig=False, size=14, truncate=0, yscale=[], label=True, order_of_mag_offset=True, tag=''):
    '''yscale offers option to set limits of colourbar manually'''

    N = len(list(itertools.product(a_or_i, Mth, dustnum, hand)))

    a_or_i, Mth, dustnum, hand = ensure_list(a_or_i), ensure_list(Mth), ensure_list(dustnum), ensure_list(hand)
    
    with tqdm(total=N) as pbar:
        for a_or_i, Mth, dustnum, hand in itertools.product(a_or_i, Mth, dustnum, hand):

            diskname = f'{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}-{hand}'
            txtfile = f'/data/es833/chi2_txtfiles/{diskname}_chi2.txt'

            df = pd.read_csv(txtfile, sep='\t')
            r = df['r_planet [arcsec]'].values
            r = r[1:]
            phi = df['spiral angle [deg]'].values
            phi = phi[1:]
            chi2 = df['chi2'].values
            chi2_0 = chi2[0]
            Dchi2 = chi2_0 - chi2
            Dchi2 = Dchi2[1:]
            print(f'peak Dchi2 = {Dchi2.max()}')
            Dchi2_norm = df['dchi2/chi2 (%)'].values
            Dchi2_norm = Dchi2_norm[1:]

            if norm:
                Ds = Dchi2_norm
            else:
                Ds = Dchi2


            if truncate:
                trunc_index = np.where(np.isclose(r, truncate, atol=1e-6))[0][0] if np.any(np.isclose(r, truncate, atol=1e-6)) else print('this radius not in data')
                r = r[trunc_index:]
                phi = phi[trunc_index:]
                Ds = Ds[trunc_index:]
                print(f'(truncated) peak Dchi2 = {Ds.max()}')
                
            Nphi = np.argmax(phi)+1
            Nr = int(len(r)/Nphi)

            # this is to correct for unusual thing where chi2txtfile contains a few repeats at the end
            r, phi, Ds = r[:Nphi*Nr], phi[:Nphi*Nr], Ds[:Nphi*Nr]

            shape = Nr, Nphi #+1
            R = r.reshape(shape)
            Phi = phi.reshape(shape)
            F = Ds.reshape(shape)

            # theme
            sns.set_theme(palette='deep')
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'


            if LRT:
                LAMBDA_LR = F * chi2_0 / 100 

                dof = 6
                chi2_dist = stats.chi2(dof)
                # CDF = chi2_dist.cdf(LAMBDA_LR) 
                # p_value = 1 - CDF
                log_p_value = chi2_dist.logsf(LAMBDA_LR) # for help displaying smaller p_values
                # Replace -inf with a large negative number
                log_p_value = np.where(log_p_value < -708, -708, log_p_value) # this truncates any p_values smaller than 2^-1022 =~ 1e-308 (smallest possible float)
                
                p_value = np.exp(log_p_value)

                cmap = 'OrRd_r'
                cmap = plt.get_cmap(cmap).copy() # Replace with your colormap
                cmap.set_bad(color='white')


            else:
                if yscale:
                    Dsmin = yscale[0]
                    Dsmax = yscale[1]
                else:
                    Dsmin = Ds.min()
                    Dsmax = Ds.max()
                divnorm = colors.TwoSlopeNorm(vmin=min(Dsmin,-1e-8), vcenter=0., vmax=max(Dsmax,1e-8))
                cmap = 'coolwarm'

            ### Plotting

            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

            if LRT:
                plt.pcolormesh(Phi*np.pi/180, R, p_value, cmap=cmap, norm=LogNorm(vmin=min(p_value.min(),0.99), vmax=1), zorder=1)
                if truncate:
                    circle = plt.Circle((0, 0), truncate, transform=ax.transData._b, color='white', zorder=2)
                    ax.add_artist(circle)
            else:
                plt.pcolormesh(Phi*np.pi/180, R, F, cmap=cmap, norm=divnorm, zorder=1)
            
            # r ticks
            if r[-1] >= 0.51:
                # ax.set_yticks(1/150*np.arange(20, 80, step=10))
                # ax.set_yticklabels(np.arange(20, 80, step=10))
                ax.set_yticks(np.arange(0.1, 0.1*np.ceil(10*r[-1]) + 1e-8, step=0.1))
                
            else:      
                if truncate:                  
                    ax.set_yticks(np.arange(20/140, 0.51, step=20/140))
                    ax.set_yticklabels(np.arange(20, 71, step=20))
                else:
                    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])

            ax.tick_params(axis='both', labelsize=size)
            ax.yaxis.set_zorder(10)
            ax.set_xticks([])
            ax.grid(True, axis='y', color='black', linewidth=0.1)
            ax.set_rlabel_position(22.5)
            if save_fig == False:
                ax.set_title(diskname, pad=20)
            # r_label_position = (22.5*np.pi/180, 0.9)  # Set the label position (angle, radius)
            # ax.text(*r_label_position, r"$r ['']$", rotation=0, ha='center', va='center', fontsize=9)



            if LRT:
                # ticks = np.linspace(0, 1, 6)
                # print(log_p_value.min(), p_value.min())
                cbar = plt.colorbar()
                if label:
                    cbar.ax.set_ylabel(ylabel='$p$', rotation='horizontal', labelpad=0, fontsize=size+1)
                cbar.ax.yaxis.set_label_coords(5, 0.5)
                cbar.ax.tick_params(labelsize=size)

                # Set the formatter for the colorbar
                # formatter = LogFormatter(10, labelOnlyBase=False)
                # cbar.ax.yaxis.set_major_formatter(formatter)
            else:

                if Dsmin > 0:
                    t1 = round_to_1sf(np.array([Dsmax/4])).item()
                    ticks = np.linspace(0,4*t1,5)
                elif Dsmax < 0:
                    t2 = round_to_1sf(np.array([Dsmin/4])).item()
                    ticks = np.linspace(4*t2,0,5)
                else:
                    t1 = round_to_1sf(np.array([Dsmax/4])).item()
                    t2 = round_to_1sf(np.array([Dsmin/4])).item()
                    # t1 = 0.02
                    ticks1 = np.linspace(0,5*t1,6)
                    # t2 = -0.05
                    ticks2 = np.linspace(4*t2,t2,4)
                    ticks = np.hstack([ticks2, ticks1])

                cbar = plt.colorbar(ticks=ticks)
                if label:
                    if norm:
                        cbar.ax.set_ylabel(ylabel=r'$\dfrac{\Delta\chi^{2}}{\chi_{AO}^{2}}$ /%', rotation='horizontal', labelpad=10, fontsize=size+1) 
                    else:
                        cbar.ax.set_ylabel(ylabel=r'$\Delta\chi^{2}$', rotation='horizontal', labelpad=10, fontsize=size+1)
                cbar.ax.yaxis.set_label_coords(5, 0.5)
                cbar.ax.tick_params(labelsize=size)
                
                if order_of_mag_offset:
                    fmt = ScalarFormatter(useMathText=True, useOffset=True)
                    fmt.set_powerlimits((-1, 1))
                    cbar.ax.yaxis.set_major_formatter(fmt)
                    offset_text = cbar.ax.yaxis.get_offset_text()
                    offset_text.set_size(size)
                    offset_text.set_position((2, 1))
                else:
                    def sci_notation(x, pos):
                        """Custom formatter to display labels in scientific notation"""
                        if x == 0:
                            return '0'
                        exponent = np.floor(np.log10(np.abs(x)))
                        coeff = x / 10**exponent
                        return f"{coeff:.1f} $\\times$ $10^{{{int(exponent)}}}$"

                    cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_notation))
                            

            

            if save_fig and LRT:
                filename = f'/home/es833/Proj42_results/{config}_{uJy}uJy_simulated_observations/{diskname}LRT{tag}.png'
                fig.savefig(filename, bbox_inches='tight', dpi=300)
            elif save_fig:
                filename = f'/home/es833/Proj42_results/{config}_{uJy}uJy_simulated_observations/{diskname}chi2_polar{tag}.png'
                fig.savefig(filename, bbox_inches='tight', dpi=300)
            else:
                plt.show()

            plt.close()

            # Update the progress bar
            pbar.update(1)  
       



def yscale_setter(config, uJy, a_or_i, Mth, dustnum, hand='R'):
    '''yscale offers option to set limits of colourbar manually'''

    
    diskname = f'{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}-{hand}'
    txtfile = f'/data/es833/chi2_txtfiles/{diskname}_chi2.txt'

    df = pd.read_csv(txtfile, sep='\t')
    r = df['r_planet [arcsec]'].values
    r = r[1:]
    phi = df['spiral angle [deg]'].values
    phi = phi[1:]
    chi2 = df['chi2'].values
    chi2_0 = chi2[0]
    Dchi2 = chi2_0 - chi2
    # Dchi2 = Dchi2[1:]


    yscales = [Dchi2.min(), Dchi2.max()]

    return yscales  
