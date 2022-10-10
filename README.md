# GAN-AE
Final version of the GAN-AE algorithm provided together with the scripts used for the LHC Olympics 2020 challenge.  
Information about the LHC Olympics 2020 challenge can be found here : https://lhco2020.github.io/homepage/

The code used to download and preprocess the data can be found here : https://gitlab.cern.ch/idinu/clustering-lhco

## Instructions

### Setting environment

We recommend to use work in a virtual environent using `venv`.  
Command line example :
```bash
python3 -m venv  .env
source .env/bin/activate
pip install -r requirement.txt
```

### Preparing the data

Please refer to the instruction given [here](https://gitlab.cern.ch/idinu/clustering-lhco) for modre details.  
The following command lines correspond to a clustering in 2 large jets.  
Command lines for the RnD dataset :
```
./LHCO.py cluster RnD -D ./data_raw/ --out_dir ./data/ --out_prefix RnD_2j --njets 2
./LHCO.py cluster RnD_3_prong -D ./data_raw/ --out_dir ./data/ --out_prefix RnD2_2j --njets 2
```

Command lines for the Black-Box dataset :
```
./LHCO.py cluster BBOX1 -K -D ./data_raw/ --out_dir ./data/ --out_prefix BBOX1_2j --njets 2
./LHCO.py cluster BBOX2 -D ./data_raw/ --out_dir ./data/ --out_prefix BBOX2_2j --njets 2
./LHCO.py cluster BBOX3 -K -D ./data_raw/ --out_dir ./data/ --out_prefix BBOX3_2j --njets 2
```

The number of required large jets can be changed to 3 using the `--njets` argument.

### Train and apply a GAN-AE model

You can use the example script `train.py` and `apply.py`.  
The hyperparameter of the GAN-AE can be set using a configuration file like the one provided in the `config` directory.  
The parameter of BumpHunter can also be set using a configuration file.
The files under `selection/` provide a list of variables to use for the traning of the GAN-AE model.  

Command line to train a new GAN-AE model on RnD dataset :
```
python3 train.py --model myModel --selec dijet_all_new2.txt --config gae_new_big.txt --input RnD_2j
```

Command line to appy the example GAN-AE model (under `models/`) to Black-Box dataset :
```
python3 apply.py --model BB1_2j_new_big3_2 --selec dijet_all_new2.txt --config default_BH2.txt --percent 99 --njets 2
```

For more information on the command line arguments, please refer to the header of the example scripts.

## The GAN_AE class

Here we describe briefly all the methods, hyperparameters and results stored in the GAN_AE class.

### Methods

* `scale_data` : Apply MinMax scaler to data (and also quantile transformer if required).  
* `restore_data` : Invert the transformation performed by the `scale_data` method.
* `build` : Build the Neural Network architecture and return them as a `tensorflow.keras` model.  
* `train` : Train the GAN-AE model according to the set hyperparameters.  
* `apply` : Apply the GAN-AE model to a test dataset and produce all required plots (latent space, ROC curves, ...).  
* `save` : Save the trained model under the `models/` directory.  
  The save files contains the hyperparameters, the optimized weights and the loss functions.  
* `load` : Load the hyperparameter, weights and loss of a saved model.  
* `plot` : Plot all terms of the loss funtions evaluated at the end of each training cycle.

### Hyperparameters

| Hyperparameter | description | | Hyperparameter | description |
| --- | --- | --- | --- | --- |
| `input_dim`  | Input size | | `Ncycle` | Number of training cycles |
| `hidden_dim`   | List of AE's hidden hidden sizes | | `NGAN` | Number of AE epochs per cycle |
| `latent_dim`   | Latent space size | | `ND` | Number of D epochs per cycle |
| `dis_dim`    | List of discriminant hidden sizes | | `batch_size` | Number of events per batch |
| `epsilon` | Reconstruction error weight in loss function | | `pretrain_AE` | Unable pretraining of AE |
| `alpha`    | DisCo weight in loss function | | `pretrain_dis` | Unable pretraining of discriminant |
| `use_quantile`    | Unable quantile transformer | | | |

### Results

* `AE_weights` : List with the weight tensors and bias vectors of the AE.  
* `dis_weights` : List with the weight tensors and bias vectors of the discriminant.  
* `loss` : Python directory containing the evolution of all the terms of the loss functions.  
* `distance` : the distance distributions obtained from the test sest.  
* `auc` : The AUC obtainedfrom the test set (only if truth labels are given).  

The variables `AE_weights`, `dis_weights` and `loss` are automatically set when calling the `train` method.  
The variables `distance` and `auc` are automatically set when calling the `test` method.


## Content of this repository
|     |     |
| --- | --- |
| `GAN_AE.py`  | Code implementing the GAN_AE class |
| `train.py`   | Script used to train a GAN-AE model on any dataset of the LHC Olympics 2020 challenge |
| `apply.py`   | Script used to apply a GAN-AE model to the Black-Box dataset |
| `config/`    | Directory containing the configuration files for GAN-AE and pyBumpHunter |
| `selection/` | Directory containing the feature selection files |
| `models/`    | Trained GAN-AE models used to produce results for the LHC Olympics 2020 challenge |
