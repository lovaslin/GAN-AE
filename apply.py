#######################################################################################
## Apply a GAN-AE model to the Black-Box dataset of the LHC Olympics 2020 challenge. ##
## Use pyBumpHunter version 1.0.0 or newer.                                          ##
## Command line arguments :                                                          ##
##     --model : Name of the GAN_AE model                                            ##
##     --selec : Name of the file with the variable list to use with GAN-AE          ##
##     --njets : The number of required large jets (2 or 3)                          ##
##     --percent : The percentile of the anomaly score used for event selection      ##
##     --config : Name of configuration file with BumpHunter parameters              ##
#######################################################################################

import numpy as np
import pandas as pd
import os
import sys
import getopt
import matplotlib.pyplot as plt
import GAN_AE
import pyBumpHunter as BH

print('INITIALIZING')

# Check if the results path exits
if not os.path.exists('apply_results/'):
    os.mkdir('apply_results/', 0o755)

## Manage command line argument

# Dict with the argument names and their default values
param = {
    "model"  : 'BB1_2j_all', # Name of the model
    "selec"  : 'dijet_all_new2.txt', # The features selection file to use
    "njets"  : 2, # The number of jets to consider
    "percent": 99, # Percentile at which the selection is applied
    "config" : 'default_BH2.txt'  # The BumpHunter configuration file to use
}

# Parse the argument
arg_list = [k+'=' for k in param.keys()]
argv = sys.argv[1:]
opts, args = getopt.getopt(argv,"",arg_list)

# Update the param value
for opt, val in opts:
    if opt[2:] in param.keys():
        # Check if int
        if opt[2:] in ['njets', 'percent']:
            param[opt[2:]] = int(val)
        else:
            param[opt[2:]] = val
        print(f'{opt[2:]} = {val}')
    else:
        print('ERROR : Unknown argument {opt[2:]}')

print('\n\tparam :')
print(param)


## Initializations

# Read the features selection file
with open('selection/' + param['selec'], 'r') as f:
    var_list = eval(f.read())

# Read the GAN-AE configuration file
with open('config/' + param['config'], 'r') as f:
    bh_params = eval(f.read())

# Load the GAN-AE model
GAE = GAN_AE.GAN_AE()
GAE.load(f"models/{param['model']}/{param['model']}")

# Create a folder for the required percentile cut (if needed)
if not os.path.exists(f"apply_results/sin{param['percent']}"):
    os.mkdir(f"apply_results/sin{param['percent']}", 0o755)

# Create a folder for the required model (if needed)
mpath = f"apply_results/sin{param['percent']}/{param['model']}"
if not os.path.exists(mpath):
    os.mkdir(mpath, 0o755)

# Pick the name of the mass variable dedending on number of jets
if param['njets'] == 2:
    mass = 'mj1j2'
    rng = [2700, 5000]
else:
    mass = 'mj1j2j3'
    rng = [3000, 7000]

# Some variables
dist_all = []
stat_min = 0.15
print(mass)

## Loop over BBOX dataset

Bid = ['BBOX1', 'BBOX2', 'BBOX3']

for b in Bid:
    print(f"\n####{b}_{param['njets']}j####")
    
    rpath = f"{mpath}/{b}_{param['njets']}j"
    if not os.path.exists(rpath):
        os.mkdir(rpath, 0o755)
    
    ## Load data for BBOXb
    
    # Load the black-box b with njets
    np.random.seed(666)
    bbi = pd.read_hdf(f"data/{b}_{param['njets']}j_scalars_bkg.h5")
    
    # Check if we have signal to mix in
    has_sig = b in ['BBOX1', 'BBOX3']
    if has_sig:
        # Load and append the signal
        Nbkg = bbi.shape[0]
        bbi = bbi.append(pd.read_hdf(f"data/{b}_{param['njets']}j_scalars_sig.h5"))
        
        # Add a new columns for labels
        bbi['truth'] = np.append(np.zeros(Nbkg, dtype=int), np.ones(bbi.shape[0]-Nbkg, dtype=int))
        
        # Shuffle the dataset
        bbi = bbi.sample(frac=1)
        
        # Print original S/B ratio
        sbr = bbi['truth'][bbi['truth']==1].size / bbi['truth'][bbi['truth']==0].size
        print(f'Original S/B = {100 * sbr:.2f}%')
    
    
    ## Prepare the data
    
    # Select the mass range
    bbi = bbi[bbi[mass]>rng[0]]
    bbi = bbi[bbi[mass]<rng[1]]
    
    # Remove the dijet mass and keep it separtely
    mjj_bbi = bbi[mass].values
    bbi = bbi.drop(columns=mass)
    
    # Also get the truth labels separately if any
    if has_sig:
        truth = bbi['truth'].values
        bbi = bbi.drop(columns='truth')
    
    # Select only the features that are requested
    bbi = bbi[var_list]
    
    # Replace nan and inf value by 0
    bbi.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
    
    # Convert to numpy array
    bbi = bbi.values
    print(f'BBOX.shape={bbi.shape}')
    
    
    ## Application of the GAN-AE model
    
    # Apply the GAE preprocessing
    if GAE.use_quantile:
        Sbbi, dmin, dmax, qt = GAE.scale_data(bbi)
    else:
        Sbbi, dmin, dmax = GAE.scale_data(bbi)
        qt = None
    
    # Create an output folder for the apply method
    if not os.path.exists(rpath + '/apply'):
        os.mkdir(rpath + '/apply', 0o755)
    
    # Call the apply method
    print('Applying GAN-AE')
    GAE.apply(
        Sbbi, dmin, dmax, qt,
        var_name=var_list,
        filename=f"{rpath}/apply/{param['model']}",
        do_latent=True, do_reco=True, do_roc=False, do_auc=False
    )
    dist_bbi = GAE.distance
    
    # Do the selection at Nth percentile
    cut = np.percentile(dist_bbi, param['percent'])
    mjj_bbi_selec = mjj_bbi[dist_bbi>cut]
    print(f'    post cut stat : {mjj_bbi_selec.size}')
    print(f'    selec threshold : {cut:.4f}')
    
    # Compute signal efficiency (based on truth) if possible
    if has_sig:
        truth_selec = truth[dist_bbi>cut]
        sig_eff =  mjj_bbi_selec[truth_selec==1].size / mjj_bbi[truth==1].size
        print(f'    sig_eff : {sig_eff}')
        sbr = mjj_bbi_selec[truth_selec==1].size / mjj_bbi_selec[truth_selec==0].size
        print(f'    new S/B : {100 * sbr:.2f}%')
    else:
        print('    sig_eff : no truth available')
    
    
    ## Prepare the histograms
    
    # Data histogram
    print('Preparing histograms')
    hist_data, bins = np.histogram(mjj_bbi_selec, bins=40, range=rng)
    
    # Rebinning based on statistics per bin
    i = hist_data.size-1
    for n in hist_data[::-1]:
        if((n>0) & ((1./np.sqrt(n))>stat_min)):
            hist_data[i-1] += hist_data[i]
            hist_data[i] = -1
            bins[i] = -1
        elif(n==0):
            hist_data[i] = -1
            bins[i] = -1
        i-=1
    hist_data = hist_data[hist_data>=0]
    bins = bins[bins>=0]
    print('    Nbin={}'.format(hist_data.size))
    
    # Reference histogram
    hist_ref, _ = np.histogram(mjj_bbi, bins=bins, range=[rng[0], rng[1]])
    hist_ref = np.array(hist_ref, dtype=float)
    
    
    ## BumpHunter
    
    print('Applying BumpHunter')
    
    # Initialize DataHandler
    dh = BH.DataHandler(ndim=1, nchan=1)
    dh.set_ref(hist_ref, bins=[bins], rang=[rng], is_hist=True)
    dh.set_data(hist_data, is_hist=True)
    
    # Initialize BumpHunter
    bh = BH.BumpHunter1D(
        **bh_params
    )
    
    # Create output folder if needed
    if not os.path.exists(f"{rpath}/BH"):
        os.mkdir(f"{rpath}/BH", 0o755)
    
    # Do the BH scan
    bh.bump_scan(dh)
    print(bh.bump_info(dh))
    
    # Plot results
    F = plt.figure(figsize=(12,8))
    bh.plot_tomography(dh)
    plt.xlabel('intervalles', size=28)
    plt.ylabel('p-value locale', size=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(f"{rpath}/BH/tomography.pdf", bbox_inches='tight')
    plt.close(F)
    
    F = plt.figure(figsize=(12,10))
    pl = bh.plot_bump(dh, fontsize=28)
    pl[0].legend(fontsize=24)
    pl[1].axes.set_xlabel('masse invariante (GeV)', size=28)
    pl[0].axes.set_ylabel("nombre d'événements", size=28)
    pl[1].axes.set_ylabel('significance', size=28)
    plt.savefig(f"{rpath}/BH/bump.pdf", bbox_inches='tight')
    plt.close(F)
    
    F = plt.figure(figsize=(12,8))
    if bh.fit_param[0] is None:
        plt.title(f'p-value globale : {bh.global_Pval[0]:.3g} ({bh.significance[0]:.2f}$\sigma$)', size=28)
    else:
        plt.title(f'p-value globale : {bh.fit_Pval[0]:.3g} ({bh.fit_sigma[0]:.3f}$\sigma$)', size=28)
    bh.plot_stat()
    plt.legend(fontsize=24)
    plt.xlabel('test statistique', size=28)
    plt.ylabel("nombre de pseudo-données", size=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig(f"{rpath}/BH/BHstat.pdf", bbox_inches='tight')
    plt.close(F)
    
    # Add a truth plot if needed
    if has_sig:
        # Separate backgroud and signal
        mjj_bkg = mjj_bbi_selec[truth_selec==0]
        mjj_sig = mjj_bbi_selec[truth_selec==1]
        
        # Plot all histograms
        F = plt.figure(figsize=(12,8))
        plt.hist(
            [mjj_bkg, mjj_sig],
            bins=bins,
            range=[rng[0],rng[1]],
            stacked=True,
            label=['données (bruit de fond)', 'données (signal)']
        )
        plt.hist(
            bins[:-1],
            bins=bins,
            weights=hist_ref * bh.norm_scale[0],
            lw=2,
            histtype='step',
            label='référence'
        )
        plt.vlines(
            [bins[bh.min_loc_ar[0,0]], bins[bh.min_loc_ar[0,0] + bh.min_width_ar[0,0]]],
            0,
            hist_data.max(),
            linestyles='dashed',
            color='r',
            lw=2,
            label='Bump'
        )
        plt.xlim(dh.range[0])
        plt.legend(fontsize=24)
        plt.xlabel('masse invariante (GeV)', size=28)
        plt.ylabel("nombre d'événements", size=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.savefig(f"{rpath}/BH/bump_truth.pdf", bbox_inches='tight')
        plt.close(F)
    
    # Append the distance distribution to the global list
    dist_all.append(dist_bbi)

print('\nDoing global plot')

# Plot all the distance on one plot
F = plt.figure(figsize=(12,8))
plt.hist(
    dist_all,
    bins=50,
    histtype='step',
    lw=2,
    label=[f"{b}_{param['njets']}j" for b in Bid]
)
plt.legend(fontsize=24)
plt.xlabel('Euclidean distance', size=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.savefig(f"{mpath}/distance_all.pdf", bbox_inches='tight')
plt.close(F)


