###########################################################################
## Provides the GAN_AE class (see docstring for details).                ##
## Also provides usefull functions.                                      ##
## To use the GAN_AE class, use `import GAN_AE` in your analysis script. ##
###########################################################################

## Imports

import numpy as np
from functools import partial
import concurrent.futures as thd
import h5py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as Kl
from tensorflow.keras import models as Km
from tensorflow.keras.losses import binary_crossentropy as bc
from tensorflow.keras import optimizers as Kopt
from tensorflow.keras.initializers import glorot_normal as gn

from sklearn.metrics import auc
from sklearn import preprocessing as sp

import matplotlib.pyplot as plt


## A few parctical fuctions (for the losses)

# Define loss function for AE only model
def MeanL2Norm(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=1) / tf.cast(y_pred.shape[1], dtype=tf.float32))

# Numpy version of the Euclidean distance
def get_dist(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_pred - y_true), axis=1) / y_pred.shape[1])


# Define DisCo term
def DisCo(y_true, y_pred, var_1, var_2, power=1):
    '''
    Taken from https://github.com/gkasieczka/DisCo/blob/master/Disco_tf.py
    I just removed the 'normedweight' thing since I don't need it here.
    I also removed the alpha parameter from the loss (moved to loss weights).
    '''
    
    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
    
    yy = tf.transpose(xx)
    
    amat = tf.math.abs(xx-yy)
    
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx-yy)
    
    amatavg = tf.reduce_mean(amat, axis=1)
    bmatavg = tf.reduce_mean(bmat, axis=1)
    
    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat-minuend_1-minuend_2+tf.reduce_mean(amatavg)
    
    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat-minuend_1-minuend_2+tf.reduce_mean(bmatavg)
    
    ABavg = tf.reduce_mean(Amat*Bmat,axis=1)
    AAavg = tf.reduce_mean(Amat*Amat,axis=1)
    BBavg = tf.reduce_mean(Bmat*Bmat,axis=1)
    
    if power==1:
        dCorr = tf.reduce_mean(ABavg)/tf.math.sqrt(tf.reduce_mean(AAavg)*tf.reduce_mean(BBavg))
    elif power==2:
        dCorr = (tf.reduce_mean(ABavg))**2/(tf.reduce_mean(AAavg)*tf.reduce_mean(BBavg))
    else:
        dCorr = (tf.reduce_mean(ABavg)/tf.math.sqrt(tf.reduce_mean(AAavg)*tf.reduce_mean(BBavg)))**power
    
    return dCorr


## The GAN_AE class

class GAN_AE():
    '''
    Provides all the methods to build, train and test one or several GAN-AE models.
    
    Hyperarameters :
        input_dim : Dimention of the AE input
        hidden_dim : Dimension of hdden layers between input and latent space (AE architecture)
        latent_dim : Dimention of AE latent space (bottleneck size)
        dis_dim : dimension of the discriminator hiden layers (D architecture)
        epsilon : Combined loss function parameter
        alpha : Disco term parameter
        power : The power used when calculating DisCo term
        NGAN : Number of GAN (AE+discriminant) epochs per training cycle
        ND : number of discriminant only epochs per training cycle
        Ncycle : Total numbur of training cycle
        batch_size : The batchsize used for training 
        early_stop : Ignored argument (kept for retro-compatibility with older versions of the code)
        pretrain_AE : specify if the AE should be pretrained separatly before the GAN training
        pretrain_dis : specify if the discriminant should be pretrained separatly before the GAN training
        use_quantile : specify if quantile transformer should be used in addition to the min-max scaler
        Nmodel : Total number of trained GAN-AE model
        Nselec : Total number of selected GAN-AE model for averaging
        Nworker : Maximum number of therads to un in parallel (must be 1 for tensorflow version>1.14.0)
    '''
    
    def __init__(
        self,
        input_dim=10, hidden_dim=[7], latent_dim=5,
        dis_dim=[100, 70, 50],
        epsilon=0.2, alpha=None, power=1,
        NGAN=4, ND=10, batch_size=1024, Ncycle=60, early_stop=None, pretrain_AE=False, pretrain_dis=True,
        use_quantile=False,
        Nmodel=60, Nselec=10, Nworker=4
    ):
        
        # Initialize parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dis_dim = dis_dim
        self.epsilon = epsilon
        self.alpha = alpha
        self.power = power
        self.NGAN = NGAN
        self.ND = ND
        self.batch_size = batch_size
        self.Ncycle = Ncycle
        self.pretrain_AE = pretrain_AE
        self.pretrain_dis = pretrain_dis
        self.use_quantile = use_quantile
        self.Nmodel = Nmodel
        self.Nselec = Nselec
        self.Nworker = Nworker
        
        # Initialize results
        self.AE_weights = []
        self.dis_weights = []
        self.loss = []
        self.distance = []
        self.auc = []
        
        return
    
    # Prepare the data with min-max scaler and quantile transformer (optional)
    def scale_data(self, data, dmin=None, dmax=None, qt=None):
        '''
        MinMax scaler (rescale the data between 0 and 1).
        
        Argument :
            data : The data to be scaled (given as a numpy array)
            dmin : Array-like containing the minimum of all features (if None dmin is determined automatically)
            dmax : Array-like containing the maximum of all features (if None dmax is determined automatically)
            qt : The fiited quantile transformer used for the niformization (if None a new one is fitted)
        
        Returns :
            res : The rescaled data.
            dmin : The original minimum of all variables (to invert the scaling)
            dmax : The original maximum of all variables (to invert the scaling)
            qt : The fitted quantile transformer (to invert the uniformization)
                 Retured only if quantile transformer is riquired
        '''
        
        # Get the min and max of all variables
        if dmin is None:
            dmin = data.min(axis=0)
        if dmax is None:
            dmax = data.max(axis=0)
        
        # Apply the min-max scaling
        res = data.copy()
        for i in range(data.shape[1]):
            res[:,i] = (data[:,i] - dmin[i]) / (dmax[i] - dmin[i])
        
        # Apply uniformization with quantile transformer (if required)
        if self.use_quantile:
            if qt is None:
                qt = sp.QuantileTransformer(
                    n_quantiles=500,
                    random_state=666
                )
                qt.fit(res)
            
            res = qt.transform(res)
            return res, dmin, dmax, qt
        
        return res, dmin, dmax
    
    # Invert the min-max scaler and quantile transformer
    def restore_data(self, data, dmin, dmax, qt=None):
        '''
        Invert the MinMax scaler to restore the original data scale.
        
        Argument :
            data : The scaled data to be restored (given as a numpy array)
            dmin : The original minimum of all variables
            dmax : The original maximum of all variables
            qt : The fitted quantile transformer used for variables uniformization
                 Ignored if quantile transformer is not required
        
        Return :
            res : The restored data.
        '''
        
        # Revert uniformization with quantile transorfmer (if needed)
        if self.use_quantile:
            res = qt.inverse_transform(data)
        else:
            res = data.copy()
        
        # Revert the min-max scaling
        for i in range(res.shape[1]):
            res[:,i]= (res[:,i] * (dmax[i] - dmin[i])) + dmin[i]
        
        return res
    
    # Build the GAN-AE architecture
    def build(self, display=True):
        '''
        Method that builds a GAN-AE architecture and return its compiled components.
        
        Arguments :
            display : Specify if the detail of the built model shoud pe printed after building it.
                      Default to True.
        
        Returns :
            En : The compiled encoder keras model.
            
            AE : The compiled AE keras model (encoder+decoder).
            
            D : The compiled discriminator keras model (MLP).
            
            GAN : The full compiled GAN-AE keras model (AE+D).
        '''
        
        # Encoder input
        En_in = Kl.Input(shape=(self.input_dim,), name='En_input')
        
        # Encoder hidden layers
        N = len(self.hidden_dim)
        lay_en = []
        for i in range(N):
            lay_en.append(0)
            lay_en[i] = Kl.Dense(self.hidden_dim[i], activation='linear', kernel_initializer=gn(), name=f'encode{i+1}')
            if i==0:
                En_h = lay_en[i](En_in)
            else:
                En_h = lay_en[i](En_h)
            En_h = Kl.Dropout(0.2)(En_h)
            En_h = Kl.ReLU()(En_h)
        
        # Encoder output
        lay_en.append(0)
        lay_en[N] = Kl.Dense(self.latent_dim, activation='linear', kernel_initializer=gn(), name='En_code')
        En_co = lay_en[N](En_h)
        En = Km.Model(En_in, En_co, name='Encoder')
        
        # Decoder
        De_in = Kl.Input(shape=(self.latent_dim,), name='De_input')
        lay_de = []
        for i in range(N):
            lay_de.append(0)
            lay_de[i] = Kl.Dense(self.hidden_dim[N-i-1], activation='linear', kernel_initializer=gn(), name=f'decode{i+1}')
            if i == 0:
                De_h = lay_de[i](De_in)
            else:
                De_h = lay_de[i](De_h)
            De_h = Kl.Dropout(0.2)(De_h)
            De_h = Kl.ReLU()(De_h)
        
        lay_de.append(0)
        lay_de[N] = Kl.Dense(self.input_dim, activation='linear', kernel_initializer=gn(), name='De_outout')
        De_out = lay_de[N](De_h)
        
        # Tie the weight of the decoder hidden layer with encoder hidden layer
        for i in range(N+1):
            lay_de[N-i].kernel = tf.transpose(lay_en[i].kernel)
        
        De = Km.Model(De_in,De_out,name='Decoder')
        
        # Full generator/AE
        AE_in = Kl.Input(shape=(self.input_dim,), name='Gen_input')
        AE_mid = En(AE_in)
        AE_out = De(AE_mid)
        AE = Km.Model(AE_in, AE_out, name='Generator_AE')
        AE.compile(loss=MeanL2Norm, optimizer=Kopt.Adam(lr=0.0002, beta_1=0.5))
        
        # Discriminator
        Din  = Kl.Input(shape=(self.input_dim,))
        for i in range(len(self.dis_dim)):
            if i == 0:
                Dh  = Kl.Dense(self.dis_dim[i], activation='linear', kernel_initializer=gn(), name=f'Dis{i+1}')(Din)
            else:
                Dh  = Kl.Dense(self.dis_dim[i], activation='linear', kernel_initializer=gn(), name=f'Dis{i+1}')(Dh)
            Dh  = Kl.Dropout(0.2)(Dh)
            Dh = Kl.LeakyReLU(alpha=0.2)(Dh)
        Dout = Kl.Dense(1, activation='sigmoid', name='Dis_output')(Dh)
        D = Km.Model(Din, Dout, name='Discriminant')
        D.compile(loss=bc, optimizer=Kopt.Adam(lr=0.0002, beta_1=0.5))
        
        
        # Full GAN-AE
        GANin = Kl.Input(shape=(self.input_dim,), name='GAN_input')
        GANmid1 = En(GANin)
        GANmid2 = De(GANmid1)
        D.trainable = False
        GANout = D(GANmid2)
        if self.alpha is None:
            GAN = Km.Model([GANin], [GANout,GANmid2], name='GAN_AE')
        else:
            aux_in = Kl.Input(shape=(1,), name=('aux_input'))
            GAN = Km.Model([GANin, aux_in], [GANout, GANmid2, GANmid2], name='GAN_AE')
        
        # Custom loss function
        if self.alpha is None:
            GAN.compile(
                loss=[bc, MeanL2Norm],
                optimizer=Kopt.Adam(lr=0.0002, beta_1=0.5),
                loss_weights=[1.0, self.epsilon]
            )
        else:
            var_2 = MeanL2Norm(GANin, GANmid2)
            DisCo_p = partial(DisCo, var_1=aux_in, var_2=var_2, power=self.power)
            DisCo_p.__name__ = 'DisCo_p'
            
            GAN.compile(
                loss=[bc, MeanL2Norm, DisCo_p],
                optimizer=Kopt.Adam(lr=0.0002, beta_1=0.5),
                loss_weights=[1.0, self.epsilon, self.alpha]
            )
        
        # Display the sumary if required
        if display:
            En.summary()
            print('')
            De.summary()
            print('')
            AE.summary()
            print('')
            D.summary()
            print('')
            GAN.summary()
        
        return En,AE,D,GAN
    
    # Train one GAN-AE model
    def train(self, train_data, val_data, w_data=None, aux_data=None, ih=-1):
        '''
        Train a single GAN-AE model according to the hyperparameters defined for this instance.
        
        Arguments :
            train_data : The training dataset given as a numpy array.
            
            val_data : The validation dataset given as a numpy array. This dataset is used to
                       Evaluate the FoM at each training cycle.
            
            ih : Specify if the ethod is called from a separate thread. This argument is used only
                 when training multiple GAN-AE models simutaneously (it is then called from the
                 multi_train method).
                 If you call this method directly to train a single GAN-AE, please leave it by default.
        
        Note :
            All the given dataset must be properly scaled between 0 and 1.
            This can be done using the scale_data method before the training.
        '''
        
        # Check aux_data
        if self.alpha is not None and aux_data is None:
            print('ERROR : You must specifify a auxilary variable for the DisCo term.')
            return
        
        # Function to train the discriminator only
        def D_train(cyc, append=True):
            #D.tranable = True
            
            # Create a fake dataset using the AE
            train_fake = G.predict(train_data)
            train_full = np.concatenate((train_data, train_fake))
            
            val_fake = G.predict(val_data)
            val_full = np.concatenate((val_data, val_fake))
            
            # Create labels for D
            label_train_full = np.concatenate((np.ones(train_data.shape[0]), np.zeros(train_fake.shape[0])))
            label_val_full = np.concatenate((np.ones(val_data.shape[0]), np.zeros(val_fake.shape[0])))
            
            # Create weights for both real and fakes
            if w_data is None:
                ww = None
            else:
                ww = np.concatenate((w_data, w_data))
            
            # Train D
            histo = D.fit(
                x=train_full, y=label_train_full,
                batch_size=self.batch_size, epochs=self.ND*(cyc+1), verbose=0,
                sample_weight = ww,
                validation_data=(val_full, label_val_full),
                initial_epoch=self.ND*cyc,shuffle=True
            )
            
            # Evaluate D and append results if required
            if append :
                res_D = histo.history['loss'][-1]
                res_D2 = histo.history['val_loss'][-1]
                
                loss['D_train'].append(res_D)
                loss['D_val'].append(res_D2)
            
            # Free some memmory
            del ww
            del train_full
            del val_full
            del label_train_full
            del label_val_full
            
            return
        
        # Function to train the full GAN-AE (with D frozen)
        def GAN_train(cyc):
            #D.trainable = False
            
            # Train GAN-AE
            if self.alpha is None:
                GAN.fit(
                    x=[train_data], y=[np.ones(train_data.shape[0]), train_data],
                    batch_size=self.batch_size, epochs=self.NGAN*(cyc+1), verbose=0,
                    sample_weight = [w_data, w_data],
                    validation_data=([val_data], [np.ones(val_data.shape[0]), val_data]),
                    initial_epoch=self.NGAN*cyc, shuffle=True
                )
            else:
                histo = GAN.fit(
                    x=[train_data, aux_data[0]], y=[np.ones(train_data.shape[0]), train_data, train_data],
                    batch_size=self.batch_size, epochs=self.NGAN*(cyc+1), verbose=0,
                    sample_weight = [w_data, w_data, None],
                    validation_data=([val_data, aux_data[1]],[np.ones(val_data.shape[0]), val_data, val_data]),
                    initial_epoch=self.NGAN*cyc, shuffle=True
                )
            
            # Evaluate full GAN-AE and append loss
            if self.alpha is None :
                res_GAN = [
                    histo.history['loss'][-1], # Full loss
                    histo.history['Discriminant_loss'][-1], # D loss
                    histo.history['Decoder_loss'][-1] # AE loss
                ]
                res_GAN2 = [
                    histo.history['val_loss'][-1], # Full loss
                    histo.history['val_Discriminant_loss'][-1], # D loss
                    histo.history['val_Decoder_loss'][-1] # AE loss
                ]
            else:
                res_GAN = [
                    histo.history['loss'][-1], # Full loss
                    histo.history['Discriminant_loss'][-1], # D loss
                    histo.history['Decoder_loss'][-1], # AE loss
                    histo.history['Decoder_1_loss'][-1] # DisCo loss
                ]
                res_GAN2 = [
                    histo.history['val_loss'][-1], # Full loss
                    histo.history['val_Discriminant_loss'][-1], # D loss
                    histo.history['val_Decoder_loss'][-1], # AE loss
                    histo.history['val_Decoder_1_loss'][-1] # DisCo loss
                ]
            
            loss['GAN_train'].append(res_GAN[0])
            loss['GAN_val'].append(res_GAN2[0])
            loss['GAN_train1'].append(res_GAN[1])
            loss['GAN_val1'].append(res_GAN2[1])
            loss['GAN_train2'].append(res_GAN[2])
            loss['GAN_val2'].append(res_GAN2[2])
            if self.alpha is not None :
                loss['GAN_train3'].append(res_GAN[3])
                loss['GAN_val3'].append(res_GAN2[3])
            
            return
        
        # Build the model
        if ih==-1 :
            En,G,D,GAN = self.build()
        else:
            En,G,D,GAN = self.build(display=False)
        
        # Check if should pretrain the AE part
        if self.pretrain_AE:
            G.fit(
                x=train_data, y=train_data,
                batch_size=self.batch_size, epochs=self.NGAN, verbose=0,
                validation_data=(val_data, val_data) ,shuffle=True
            )
        
        # Check if should pretrain the D part
        if self.pretrain_dis :
            D_train(0, append=False)
        
        # Initilize loss containiner
        loss = dict()
        loss['G_train'] = []
        loss['G_val'] = []
        loss['D_train'] = []
        loss['D_val'] = []
        loss['GAN_train'] = []
        loss['GAN_val'] = []
        loss['GAN_train1'] = []
        loss['GAN_val1'] = []
        loss['GAN_train2'] = []
        loss['GAN_val2'] = []
        if self.alpha is not None:
            loss['GAN_train3'] = []
            loss['GAN_val3'] = []
        
        # Main cycle loop
        stop = int(0)
        best = 100.0
        for cyc in range(self.Ncycle):
            # D-only epochs
            D_train(cyc)
            
            # AE+D (with D forzen) epochs
            GAN_train(cyc)
            print(f"   cyc {cyc} : {loss['GAN_train'][-1]} ({FoM[-1]})")
        
        # Convert all result containers in numpy array
        loss['D_train'] = np.array(loss['D_train'])
        loss['D_val'] = np.array(loss['D_val'])
        loss['GAN_train'] = np.array(loss['GAN_train'])
        loss['GAN_val'] = np.array(loss['GAN_val'])
        loss['GAN_train1'] = np.array(loss['GAN_train1'])
        loss['GAN_val1'] = np.array(loss['GAN_val1'])
        loss['GAN_train2'] = np.array(loss['GAN_train2'])
        loss['GAN_val2'] = np.array(loss['GAN_val2'])
        if(self.alpha!=None):
            loss['GAN_train3'] = np.array(loss['GAN_train3'])
            loss['GAN_val3'] = np.array(loss['GAN_val3'])
        
        self.loss = loss
        self.FoM = FoM
        self.AE_weights = G.get_weights()
        self.dis_weights = D.get_weights()
        return
        
    
    # Apply a GAN-AE model to a test dataset
    # If label is given, they can be used to evaluate the model
    def apply(self, data, dmin, dmax, qt=None, var_name=None, label=None, filename=None, do_latent=True, do_reco=True, do_distance=True, do_roc=True, do_auc=True, ih=-1):
        '''
        Apply one GAN-AE model to a dataset in order to produce result plots.
        
        Arguments :
            data : The dataset given as a numpy array.
            
            dmin : Numpy array specifying the true minimums of the input features.
                   This is used to plot the reconstructed data in real scale.
            
            dmax : Numpy array specifying the true maximums of the input features.
                   This is used to plot the reconstructed data in real scale.
            
            qt : Quantile transformer used for variables uniformization
                 This is used to plot the reconstructed data in real scale.
            
            var_name : The x-axis labels to be used when ploting the original and/or reconstruted
                       features. If None, the names are built using the 'Var {i}' convention.
                       Defalut to None.
            
            label : Numpy array with truth label for the data.
                    The labels are used to separate the background and signal to compute the ROC
                    curve.
                    If None, the background and signal are not separated and no ROC curve is
                    computed.
                    Default to None.
            
            filename : The base name for the output files. This base is prepended to all the produced
                       files. 
                       If None, the plots are shown but not saved. Default to None.
            
            do_latent : Boolean specifying if the latent space should be ploted.
                        Default to True.
            
            do_reco : Boolean specifying if the reconstructed variables should be ploted.
                      Default to True.
            
            do_distance : Boolean specifying if the Euclidean distance distribution should be ploted.
                          The obtained distance distributions are recorded within this instance variables.
                          Default to True.
            
            do_roc : boolean specifying if the ROC curve should be ploted. This requires to give truth
                     labels.
                     Default to True.
            
            do_auc : Specify if the AUC should be recorded. Requires to give truth labels and to compute
                     the ROC curve.
                     Default to True.
        
            ih : Specify if the ethod is called from a separate thread. This argument is used only
                 when applying multiple GAN-AE models simutaneously (it is then called from the
                 multi_apply method).
                 If you call this method directly to apply a single GAN-AE, please leave it by default.
        
        Note :
            All the given dataset must be properly scaled between 0 and 1.
            This can be done using the scale_data method before the training.
        '''
        
        # Split the background and signal if labels given
        if label is not None:
            bkg = data[label==0]
            sig = data[label==1]
        
        # Build the GAN-AE model and load the weights
        En, AE, D, GAN = self.build(display=False)
        if ih == -1:
            AE.set_weights(self.AE_weights)
            D.set_weights(self.dis_weights)
        else:
            AE.set_weights(self.AE_weights[ih])
            D.set_weights(self.dis_weights[ih])
        
        # Latent space plot
        if do_latent:
            # Apply the encoder model do the data
            if label is None:
                cod_data = En.predict(data)
                Nplot = cod_data.shape[1]
            else:
                cod_bkg = En.predict(bkg)
                cod_sig = En.predict(sig)
                Nplot = cod_bkg.shape[1]
            
            # Do the plot
            F = plt.figure(figsize=(18, 5*(Nplot//3+1)))
            plt.suptitle('Latent space distribution', size='xx-large')
            for i in range(1, Nplot+1):
                plt.subplot(Nplot//3+1, 3, i)
                if label is None:
                    plt.hist(cod_data[:,i-1], bins=60,lw=2, histtype='step')
                else:
                    plt.hist(cod_bkg[:,i-1], bins=60, lw=2, histtype='step', label='background')    
                    plt.hist(cod_sig[:,i-1], bins=60, lw=2, histtype='step', label='signal')
                    plt.legend(fontsize='x-large')
                plt.xlabel(f'latent {i}', size='x-large')
                plt.xticks(fontsize='x-large')
                plt.yticks(fontsize='x-large')
            if filename is None:
                plt.show()
            else:
                plt.savefig(f'{filename}_latent_space.png', bbox_inches='tight')
                plt.close(F)
        
        # Reconstructed variables plot
        if do_reco:
            # Apply the AE model do the data
            if label is None:
                reco_data = AE.predict(data)
                Nplot = reco_data.shape[1]
                reco_data_restored = self.restore_data(reco_data, dmin, dmax, qt)
                data_restored = self.restore_data(data, dmin, dmax, qt)
            else:
                reco_bkg = AE.predict(bkg)
                reco_sig = AE.predict(sig)
                Nplot = reco_bkg.shape[1]
                reco_bkg_restored = self.restore_data(reco_bkg, dmin, dmax, qt)
                reco_sig_restored = self.restore_data(reco_sig, dmin, dmax, qt)
                bkg_restored = self.restore_data(bkg, dmin, dmax, qt)
                sig_restored = self.restore_data(sig, dmin, dmax, qt)
            
            # Check the original variable names
            if var_name is None:
                var_name = np.array([f'variable_{i}' for i in range(1,Nplot+1)])
            
            # Do the plot
            F = plt.figure(figsize=(18, 5*(Nplot//3+1)))
            plt.suptitle('Reconstucted variable distribution', size='xx-large')
            for i in range(1, Nplot+1):
                plt.subplot(Nplot//3+1, 3, i)
                if label is None:
                    plt.hist(reco_data_restored[:,i-1], bins=60, lw=2, color='b', histtype='step')
                    plt.hist(data_restored[:,i-1], bins=60, lw=2, color='b', ls='--', histtype='step')
                else:
                    plt.hist(reco_bkg_restored[:,i-1], bins=60, lw=2, color='b', histtype='step', label='background')
                    plt.hist(bkg_restored[:,i-1], bins=60, lw=2,color='b', ls='--', histtype='step')
                    plt.hist(reco_sig_restored[:,i-1], bins=60, lw=2, color='r', histtype='step', label='signal')
                    plt.hist(sig_restored[:,i-1], bins=60, lw=2,color='r', ls='--', histtype='step')
                    plt.legend(fontsize='x-large')
                plt.xlabel(var_name[i-1], size='x-large')
                plt.xticks(fontsize='x-large')
                plt.yticks(fontsize='x-large')
            if filename is None:
                plt.show()
            else:
                plt.savefig(f'{filename}_reco_variables.png', bbox_inches='tight')
                plt.close(F)
                
            F = plt.figure(figsize=(18, 5*(Nplot//3+1)))
            plt.suptitle('Reconstucted variable distribution (scaled)', size='xx-large')
            for i in range(1, Nplot+1):
                plt.subplot(Nplot//3+1, 3, i)
                if label is None:
                    plt.hist(reco_data[:,i-1], bins=60, lw=2, color='b', histtype='step')
                    plt.hist(data[:,i-1], bins=60, lw=2,color='b', ls='--', histtype='step')
                else:
                    plt.hist(reco_bkg[:,i-1], bins=60, lw=2, color='b', histtype='step', label='background')
                    plt.hist(bkg[:,i-1], bins=60, lw=2, color='b', linestyle='--', histtype='step')
                    plt.hist(reco_sig[:,i-1], bins=60, lw=2, color='r', histtype='step', label='signal')
                    plt.hist(sig[:,i-1], bins=60, lw=2, color='r', ls='--', histtype='step')
                    plt.legend(fontsize='x-large')
                plt.xlabel(var_name[i-1]+' (scaled)', size='x-large')
                plt.xticks(fontsize='x-large')
                plt.yticks(fontsize='x-large')
            if filename is None:
                plt.show()
            else:
                plt.savefig(f'{filename}_reco_variables_scaled.png', bbox_inches='tight')
                plt.close(F)
        
        # Euclidean distance plot
        if do_distance :
            # Check if we need to appy the AE model
            if not do_reco:
                if label is None:
                    reco_data = AE.predict(data)
                else:
                    reco_bkg = AE.predict(bkg)
                    reco_sig = AE.predict(sig)
            
            # Compute the Euclidean distance distribution
            if label is None:
                dist_data = get_dist(data, reco_data)
            else:
                dist_bkg = get_dist(bkg, reco_bkg)
                dist_sig = get_dist(sig, reco_sig)
            
            # Do the plot
            F = plt.figure(figsize=(12,8))
            plt.title('Euclidean distance distribution', size='xx-large')
            if label is None:
                plt.hist(dist_data, bins=60, lw=2, histtype='step')
            else:
                plt.hist(dist_bkg, bins=60, lw=2, histtype='step', label='background')
                plt.hist(dist_sig, bins=60, lw=2, histtype='step', label='signal')
                plt.legend(fontsize=24)
            plt.xlabel('Euclidean distance', size=28)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            if filename is None:
                plt.show()
            else:
                plt.savefig(f'{filename}_distance_distribution.png', bbox_inches='tight')
                plt.close(F)
        
        # Check if we should do ROC curve
        if do_roc and not do_distance:
            print("CAN'T COMPUTE THE ROC CURVE !!")
            print("Please set 'do_distance=True'")
            do_roc = False
        if do_roc and type(label)==type(None):
            print("CAN'T COMPUTE THE ROC CURVE !!")
            print('Please give truth labels.')
            do_roc = False
        
        # ROC curve plot
        if do_roc:
            # Compute the roc curve
            Nbin = 100 # We use 100 points to make the ROC curve
            roc_min = min([dist_bkg.min(), dist_sig.min()])
            roc_max = max([dist_bkg.max(), dist_sig.max()])
            step = (roc_max - roc_min) / Nbin
            steps = np.arange(roc_min + step, roc_max + step, step)
            roc_x = np.array([dist_sig[dist_sig>th].size / dist_sig.size for th in steps])
            roc_y = np.array([dist_bkg[dist_bkg<th].size / dist_bkg.size for th in steps])
            roc_r1 = np.linspace(0, 1, 100)
            roc_r2 = 1 - roc_r1
            
            # Compute AUC
            auc_sig = auc(roc_x, roc_y)
            
            # Do the plot
            F = plt.figure(figsize=(12,8))
            plt.plot(roc_x, roc_y, '-', label=f'auc={auc_sig:.4f}')
            plt.plot(roc_r1, roc_r2, '--', label='random class')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.legend(fontsize=24)
            plt.xlabel('signal efficiency', size=28)
            plt.ylabel('background rejection', size=28)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            if filename is None:
                plt.show()
            else:
                plt.savefig(f'{filename}_ROC_curve.png', bbox_inches='tight')
                plt.close(F)
        
        # Check if we should save AUC
        if do_auc and not do_roc:
            print("CAN'T COMPUTE AUC !!!")
            print("Please set 'do_roc=True'")
            do_auc = False
        
        if do_distance:
            if label is None:
                self.distance = dist_data
            else:
                self.distance = [dist_bkg,dist_sig]
                if do_auc:
                    self.auc = auc_sig
        return
    
    # Save the trained model
    def save(self, filename):
        '''
        Save all the parameters and all the trained weights of this GAN-AE instance.
        The parameters are stored in a text file and the wights of each AE and D
        models are stored in HDF format.
        
        Arguments :
            filename : The base name of the save files. The full name of the files is
                       constructed from this base.
        '''
        
        # Save the parameters, loss and FoM
        Fparam = open(f'{filename}_param.txt', 'w')
        param = dict()
        param['input_dim'] = self.input_dim
        param['hidden_dim'] = self.hidden_dim
        param['latent_dim'] = self.latent_dim
        param['dis_dim'] = self.dis_dim
        param['epsilon'] = self.epsilon
        param['alpha'] = self.alpha
        param['power'] = self.power
        param['NGAN'] = self.NGAN
        param['ND'] = self.ND
        param['batch_size'] = self.batch_size
        param['Ncycle'] = self.Ncycle
        param['pretrain_AE'] = self.pretrain_AE
        param['pretrain_dis'] = self.pretrain_dis
        param['use_quantile'] = self.use_quantile
        param['Nmodel'] = self.Nmodel
        param['Nselec'] = self.Nselec
        param['Nworker'] = self.Nworker
        
        # Check if there is no model to save
        if(self.AE_weights==[]):
            print('NO TRAINED GAN-AE MODEL FOUND !!')
            
            # Write the parameter file
            param['trained'] = 'None'
            print(param, file=Fparam)
            Fparam.close()
        
        # There is only a trained model to save
        else:
            # Build the model
            _, AE, D, _ = self.build(display=False)
            print('    saving a model')
            
            # Write the parameter file
            param['trained'] = 'single'
            print(param, file=Fparam)
            Fparam.close()
            
            # Set model weights
            AE.set_weights(self.AE_weights)
            D.set_weights(self.dis_weights)
            
            # Save the weights in HDF format
            AE.save_weights(f'{filename}_AE.h5')
            D.save_weights(f'{filename}_D.h5')
            
            # Save the loss terms in a separate file
            Floss = open(f'{filename}_loss.txt', 'w')
            loss = dict()
            for l in self.loss:
                loss[l] = list(self.loss[l])
            print(loss, file=Floss)
            Floss.close()
        
        return
    
    # Load models from files
    def load(self, filename):
        '''
        Load the parameters and weights from a set of files.
        
        Arguments : 
            filename : Base name of the files. The full name of all the files is reconstructed
                       from this base.
        '''
        
        # First, load the parameters
        Fparam = open(f'{filename}_param.txt', 'r')
        param = eval(Fparam.read())
        Fparam.close()
        
        # Restore the parameters
        self.input_dim = param['input_dim']
        self.hidden_dim = param['hidden_dim']
        self.latent_dim = param['latent_dim']
        self.dis_dim = param['dis_dim']
        self.epsilon = param['epsilon']
        self.NGAN = param['NGAN']
        self.ND = param['ND']
        self.batch_size = param['batch_size']
        self.Ncycle = param['Ncycle']
        self.pretran_AE = param['pretrain_AE']
        self.pretrain_dis = param['pretrain_dis']
        self.Nmodel = param['Nmodel']
        self.Nselec = param['Nselec']
        self.Nworker = param['Nworker']
        
        # Ensure retro-compatibility for previous versions saves
        # For that we load the new parameters only if they are present
        if 'alpha' in param.keys():
            self.alpha = param['alpha']
            self.power = param['power']
        else: # If they are not, we set them to default
            self.alpha = None
            self.power = 1
        
        if 'use_quantile' in param.keys():
            self.use_quantile = param['use_quantile']
        else:
            self.use_quantile = False
        
        # Check how many wieghts files we should load
        if param['trained'] == 'None':
            # No weights file
            print('NO WEIGHTS FILE AVAILABLE !!')
            
            # Initialize results
            self.FoM = []
            self.AE_weights = []
            self.dis_weights = []
            self.loss = []
            
        elif param['trained'] == 'single':
            # Only one model (so 2 weights files)
            _, AE, D, _ = self.build(display=False)
            AE.load_weights(f'{filename}_AE.h5')
            D.load_weights(f'{filename}_D.h5')
            self.AE_weights = AE.get_weights()
            self.dis_weights = D.get_weights()
            
            # Also one loss file
            Floss = open(f'{filename}_loss.txt', 'r')
            loss = eval(Floss.read())
            self.loss = dict()
            for l in loss:
                self.loss[l] = np.array(loss[l])
        
        self.distance = []
        self.auc = []
        
        return
    
    # Plot loss and FoM
    def plot(self, filename=None):
        '''
        Plot the loss curves and FoM curves of the trainend model.
        This method behave differently if several models have been trained.
        
        Arguments :
            filename : The base name of the save files. The full name of the files is
                       constructed from this base.
        '''
        
        # Do one plot
        def plot_one(y_train, y_val, name,title):
            # x-axis range
            x = np.arange(1, y_train.size+1, 1)
            
            # Do the plot
            F = plt.figure(figsize=(12,8))
            plt.title(title)
            if y_val is None:
                plt.plot(x, y_train, 'o-', lw=2)
            else:
                plt.plot(x, y_train, 'o-', lw=2, label='training')
                plt.plot(x, y_val, 'o-', lw=2, label='validation')
                plt.legend(fontsize=24)
            plt.xlabel('cycle', size=28)
            plt.ylabel(title, size=28)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            
            if name is None :
                plt.show()
            else:
                plt.savefig(name, bbox_inches='tight')
                plt.close(F)
            
            return
        
        # Check if the model has been trained
        if loss = []:
            print('Nothing to plot here ...')
        else:
            # Plot discriminator loss
            if filename is not None:
                fn = f'{filename}_discrim_loss.png'
            else:
                fn = None
            plot_one(self.loss['D_train'], self.loss['D_val'], fn, 'discriminator loss (crossentropy)')
            
            # Plot AE loss
            if filename is not None:
                fn = [f'{filename}_full_loss_comb.png', f'{filename}_full_t1_discrim.png', f'{filename}_full_t2_reco.png']
            else:
                fn = [None, None, None]
            plot_one(self.loss['GAN_full'], self.loss['GAN_val'], fn[0], 'full loss (combined)')
            plot_one(self.loss['GAN_train1'], self.loss['GAN_val1'], fn[1], 'full loss (binary crossentropy term)')
            plot_one(self.loss['GAN_train2'], self.loss['GAN_val2'], fn[2], 'full loss (mean Euclidean distance term)')
            
            # Check if DisCo is enabled
            if self.alpha is not None:
                if filename is not None:
                    fn = f'{filename}_full_t3_DisCo.png'
                else:
                    fn = None
                plot_one(self.loss['GAN_train3'], self.loss['GAN_val3'], fn, 'full loss (DisCo term)')
        
        return
    
    #end of GAN_AE class


