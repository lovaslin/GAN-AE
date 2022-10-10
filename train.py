###########################################################################################
## Training script for the LHC Olympics analysis                                         ##
## Train a GAN-AE on the required dataset and test it                                    ##
## Command line arguments :                                                              ##
##     --model : the name of the GAN-AE model                                            ##
##     --input : the name of the dataset to train on (RnD or BBOXi, 2j or 3j)            ##
##     --config : file containing the configuration of the GAN-AE (in the config folder) ##
##     --selec : file containing the list of features to train on (in the input folder)  ## 
###########################################################################################


# Imports
print('Initializing')
import numpy as np
import pandas as pd
import GAN_AE
import matplotlib.pyplot as plt
import os
import getopt
import sys
import json
from scipy.spatial import distance as sd


## Manage command line argument

# Dict with the argument names and their default values
param = {
    'model' : 'RnD_2j_all',
    'input' : 'RnD_2j',
    'config': 'gae_new_big.txt',
    'selec' : 'dijet_all_new.txt'
}

# Parse the argument
arg_list = [k+'=' for k in param.keys()]
argv = sys.argv[1:]
opts, args = getopt.getopt(argv,"",arg_list)

# Update the param value
for opt, val in opts:
    if opt[2:] in param.keys():
        param[opt[2:]] = val
        print(f'{opt[2:]} = {val}')
    else:
        print('ERROR : Unknown argument {opt[2:]}')

print('\n\tparam :')
print(param)

# Get the vaiable list
with open('selection/' + param['selec'], 'r') as f:
    var_list = eval(f.read())

# Get the GAE param
with open('config/' + param['config'], 'r') as f:
    param_gae = eval(f.read())


## Load the dataset

# Load the background (based on input param)
np.random.seed(666)
print('\nloading bkg')
bkg = pd.read_hdf('data/' + param['input'] + '_scalars_bkg.h5')

# If the bkg is a BB1 or BB3, we must also mix in the signal
if param['input'][:5] in ['BBOX1', 'BBOX3']:
    bkg = bkg.append(pd.read_hdf('data/' + param['input'] + '_scalars_sig.h5'))
    bkg = bkg.sample(frac=1)

# Load the two RnD signals
print(f"loading signal with {param['input'][-2:]}")
sig1 = pd.read_hdf('data/RnD_' + param['input'][-2:] + '_scalars_sig.h5')
sig2 = pd.read_hdf('data/RnD2_' + param['input'][-2:] + '_scalars_sig.h5')


## Prepare the data

# Define the variable to decorelate from the anomaly score
if param['input'][-2:]=='2j':
    mass = 'mj1j2
    rng = [2700, 5000]
else:
    mass = 'mj1j2j3'
    rng = [3000, 7000]

# Select the mass range to train on
bkg = bkg[bkg[mass]>rng[0]]
bkg = bkg[bkg[mass]<rng[1]]
sig1 = sig1[sig1[mass]>rng[0]]
sig1 = sig1[sig1[mass]<rng[1]]
sig2 = sig2[sig2[mass]>rng[0]]
sig2 = sig2[sig2[mass]<rng[1]]

# Remove mjj from data and keep it separately
# It will be used to compute the DisCo term of the loss function
mjj_bkg = bkg[mass].values
mjj_sig1 = sig1[mass].values
mjj_sig2 = sig2[mass].values

bkg = bkg.drop(columns=mass)
sig1 = sig1.drop(columns=mass)
sig2 = sig2.drop(columns=mass)

bkg = bkg[var_list]
sig1 = sig1[var_list]
sig2 = sig2[var_list]
del var_list

# Replace nan and inf value by 0
bkg.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig1.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig2.replace([np.nan, -np.inf, np.inf], 0, inplace=True)

# Convert to numpy array and save columns names
var_names = bkg.columns
bkg = bkg.values
sig1 = sig1.values
sig2 = sig2.values

print(f'   bkg.shape={bkg.shape}')
print(f'   sig1.shape={sig1.shape}')
print(f'   sig2.shape={sig2.shape}\n')

# Compute the weights for decorrelation (based on mjj)
Hc,Hb = np.histogram(mjj_bkg,bins=500)
w_data = np.array(Hc,dtype=float)
w_data[w_data>0.0] = 1.0/w_data[w_data>0.0]
w_data[w_data==0.0] = 1.0
w_data = np.append(w_data,w_data[-1])
w_data = w_data[np.searchsorted(Hb,mjj_bkg)]
w_data *= 1000.0 # To avoid very small weights
del Hb
del Hc


## Create the GAN_AE instance and preprocess data

# GAN_AE instance
GAE = GAN_AE.GAN_AE(
    input_dim = var_names.size,
    **param_gae
)

# Apply the preprocessing (fit on bkg only)
print('Preprocessing')

if GAE.use_quantile:
    # With quantile transformer (if required)
    Sbkg, dmin, dmax, qt = GAE.scale_data(bkg)
    Ssig1, _, _, _ = GAE.scale_data(sig1, dmin=dmin, dmax=dmax, qt=qt)
    Ssig2, _, _, _ = GAE.scale_data(sig2, dmin=dmin, dmax=dmax, qt=qt)
else:
    # Without quantile transformer (otherwise)
    Sbkg, dmin, dmax = GAE.scale_data(bkg)
    Ssig1, _, _ = GAE.scale_data(sig1, dmin=dmin, dmax=dmax)
    Ssig2, _, _ = GAE.scale_data(sig2, dmin=dmin, dmax=dmax)
    qt = None

# Also need to rescale mjj
Smjj_bkg = (mjj_bkg - mjj_bkg.min()) / (mjj_bkg.max() - mjj_bkg.min())

print(f'   Sbkg.shape={Sbkg.shape}')
print(f'   Ssig1.shape={Ssig1.shape}')
print(f'   Ssig2.shape={Ssig2.shape}\n')


## Train the model and plot results

# Create a folder to put the results in (if it doesn't exists)
if not os.path.exists('train_results/' + param['model']):
    os.mkdir('train_results/'+param['model'], 0o755)
    os.mkdir('train_results/'+param['model']+'/loss', 0o755)
    os.mkdir('train_results/'+param['model']+'/sig1', 0o755)
    os.mkdir('train_results/'+param['model']+'/sig2', 0o755)

# Call the train method
print('TRAINING')
GAE.train(
    Sbkg[:100_000],
    Sbkg[100_000:200_000],
    w_data[:100_000],
    aux_data = [Smjj_bkg[:100_000], Smjj_bkg[100_000:200_000]]
)

# Plot the loss and FoM
print('Doing loss plots\n')
GAE.plot(filename='train_results/'+param['model']+'/loss/'+param['model'])

# Save the model
print("Saving model '"+param['model']+"'")

if not os.path.exists('models/'+param['model']):
    os.mkdir('models/'+param['model'], 0o755)

GAE.save(filename='models/'+param['model']+'/'+param['model'])


## Make the test sets and test the model

# Make the test set for sig1
test_data = np.empty((100_000 + Ssig1.shape[0], Sbkg.shape[1]))
test_data[:100_000] = Sbkg[200_000:300_000]
test_data[100_000:] = Ssig1

# Make the corresponding labels
label = np.append(np.zeros(100_000, dtype=int), np.ones(Ssig1.shape[0], dtype=int))

print('testing shape (sig1) :', test_data.shape)

# Call the apply method for sig1
print('TESTING')
GAE.apply(
    test_data, dmin, dmax, qt,
    var_name=var_names,
    label=label,
    filename='train_results/'+param['model']+'/sig1/'+param['model']
)

# Save the distance distribution and auc separately
sig1_dist = GAE.distance[1]
sig1_auc=GAE.auc

print('mean dist : ', sig1_dist.mean())
print('AUC : ', sig1_auc, '\n')

# Make the test set for sig2
test_data = np.empty((100_000 + Ssig2.shape[0], Sbkg.shape[1]))
test_data[:100_000] = Sbkg[200_000:300_000]
test_data[100_000:] = Ssig2

# Make the corresponding labels
label = np.append(np.zeros(100_000, dtype=int), np.ones(Ssig2.shape[0], dtype=int))

print('testing shape (sig2) :', test_data.shape)

# Call the test method for sig2
print('TESTING')
GAE.apply(
    test_data, dmin, dmax, qt,
    var_name=var_names,
    label=label,
    filename='train_results/'+param['model']+'/sig2/'+param['model']
)

# Save the distance distribution and auc separately
bkg_dist = GAE.distance[0]
sig2_dist = GAE.distance[1]
sig2_auc=GAE.auc

print('mean dist : ', sig2_dist.mean())
print('AUC : ', sig2_auc, '\n')


## Common plots

print('Doing common plots')

# Plot mass distribution after reweighting
F = plt.figure(figsize=(12,8))
plt.hist(
    mjj_bkg,
    bins=500,
    weights=w_data/1000,
    histtype='step',
    lw=2
)
plt.xlabel('invariant mass (GeV)', size=28)
plt.ylabel('weighted event count', size=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.savefig('train_results/'+param['model']+'/'+param['model']+'_mjj_weighted.pdf',bbox_inches='tight')
plt.close(F)

# Plot all distance at once
F = plt.figure(figsize=(12,8))
plt.hist(
    [bkg_dist, sig1_dist, sig2_dist],
    bins=60,
    histtype='step',
    linewidth=2,
    label=['background', 'signal 1', 'signal 2']
)
plt.legend(fontsize=24)
plt.ylabel('event count', size=28)
plt.xlabel('Euclidean distance',size=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.savefig('train_results/'+param['model']+'/'+param['model']+'_distance_all.pdf',bbox_inches='tight')
plt.close(F)

# Plot all ROC at once
Nbin = 100 # We use 100 points to make the ROC curve
roc_min = min([bkg_dist.min(),sig1_dist.min(),sig2_dist.min()])
roc_max = max([bkg_dist.max(),sig1_dist.max(),sig2_dist.max()])
step = (roc_max-roc_min)/Nbin
steps = np.arange(roc_min+step,roc_max+step,step)
roc_x = []
roc_x.append(np.array([sig1_dist[sig1_dist>th].size/sig1_dist.size for th in steps]))
roc_x.append(np.array([sig2_dist[sig2_dist>th].size/sig2_dist.size for th in steps]))
roc_y = np.array([bkg_dist[bkg_dist<th].size/bkg_dist.size for th in steps])
roc_r1 = np.linspace(0,1,100)
roc_r2 = 1-roc_r1

F = plt.figure(figsize=(12,8))
plt.plot(roc_x[0],roc_y,'-',linewidth=2,label='signal 1 auc={0:.4f}'.format(sig1_auc))
plt.plot(roc_x[1],roc_y,'-',linewidth=2,label='signal 2 auc={0:.4f}'.format(sig2_auc))
plt.plot(roc_r1,roc_r2,'--',label='random class')
plt.legend(fontsize=24)
plt.xlabel('signal efficiency',size=28)
plt.ylabel('background rejection',size=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.savefig('train_results/'+param['model']+'/'+param['model']+'_ROC_all.pdf', bbox_inches='tight')
plt.close(F)

# Dump ROC in json
roc_dict = dict()
roc_dict['bkg_rej'] = roc_y.tolist()
roc_dict['sig1_ef'] = roc_x[0].tolist()
roc_dict['sig2_ef'] = roc_x[1].tolist()
roc_dict['sig1_auc'] = sig1_auc
roc_dict['sig2_auc'] = sig2_auc
with open('train_results/'+param['model']+'/ROC_all.json', 'w') as f:
    json.dump(roc_dict,f)
del roc_dict

# Mass distribution after different cut
cut = [np.percentile(bkg_dist,50), np.percentile(bkg_dist,85)]
F = plt.figure(figsize=(12,8))
plt.hist(
    [
        mjj_bkg[200000:300000],
        mjj_bkg[200000:300000][bkg_dist>cut[0]],
        mjj_bkg[200000:300000][bkg_dist>cut[1]],
    ],
    bins=60,
    histtype='step',
    linewidth=2,
    density=True,
    label=['no cut', f'50th percentile ({cut[0]:.4f})', f'85th percentile ({cut[1]:.4f})']
)
plt.legend(fontsize=24)
plt.xlabel(mass,size=28)
plt.ylabel('normalized event count', size=24)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.savefig('train_results/'+param['model']+'/'+param['model']+'_mjj_shape.pdf', bbox_inches='tight')
plt.close(F)


## Plot JSD for many cut

# Get all the percentiles
Nth = np.arange(1, 100, 1)
th = np.percentile(bkg_dist, Nth)

# Reference uncut histogram
hist_ref, bins = np.histogram(
    mjj_bkg[200_000:300_000],
    bins=50,
    range=[rng[0], rng[1]]
)

# Loop over percentiles
jsd = []
for t in th:
    hist_cut, _ = np.histogram(
        mjj_bkg[200_000:300_000][bkg_dist>t],
        bins=bins,
        range=[rng[0], rng[1]]
    )
    jsd.append(sd.jensenshannon(hist_cut, hist_ref))
print(f'    JSD at 99th : {jsd[-1]:.5f}')

# Plot
F = plt.figure(figsize=(12,8))
plt.plot(Nth, jsd, '-', lw=2)
plt.xlabel('cut percentile', size=28)
plt.ylabel('JS distance', size=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.savefig('train_results/'+param['model']+'/JS_divergence.pdf', bbox_inches='tight')
plt.close(F)

# Dump in json
jsd_dict = dict()
jsd_dict['percentile'] = Nth.tolist()
jsd_dict['JD_div'] = jsd
with open('train_results/'+param['model']+'/JS_divergence.json', 'w') as f:
    json.dump(jsd_dict,f)

