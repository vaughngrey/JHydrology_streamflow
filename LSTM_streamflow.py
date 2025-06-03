#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 15 August 2024

@author: Vaughn Grey: vaughn.grey@unimelb.edu.au 
Code of LSTM model for paper: Grey et al 2025 Journal of Hydrology, 
Harnessing the strengths of machine learning and geostatistics to improve streamflow prediction in ungauged basins; the best of both worlds

Training, validation and testing of a LSTM model, estimating daily flows for the Melbourne region
Final model

Acknowledgement: code based on LSTM model in Arsenault et.al 2023: https://doi.org/10.5194/hess-27-139-2023

"""

# %% Import packages
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
import xarray as xr

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import KFold

import math

# %% Control variables for all experiments

"""
Import trailing argument variables to process catchments in parallel jobs on UniMelb Spartan HPC system
"""
# Set up to run using Slurm scheduler on UniMelb Spartan HPC system
import sys
print(sys.argv)
#pred_rep = int(sys.argv[1]) #removed for 'whole catchment' prediction
#pred_rep = int(1)  #use for spartan troubleshooting
#print(pred_rep)

"""
All the control variables used to train the LSTM models are predefined here.
These are consistent from one experiment to the other, but can be modified as 
needed by the user.
"""

batch_size_val = 50    # Batch size used in the training - function of GPU VRAM
epoch_val = 20         # Number of epochs to train the LSTM model
window_size = 365      # Number of time steps (days) to use in the LSTM model
threshold_loss = 1     # Threshold that must be reached within 3 epochs before reset     

input_data_filename = 'LSTM_H_netcdf.nc' # input filename. Place in the same folder as the script to run

#pred_data_filename = 'LSTM_Pred_Input_rep_' + str(pred_rep) + '.nc' # note, will need to update this! 


# %% Functions to create datasets

def create_LSTM_dataset(ds):
    """
    # Particular difficulty here: All catchments have different lengths of data
    """
    
    # Number of watersheds in the dataset
    n_watersheds = ds.Qobs.shape[0]  
    
    # Number of days for each dynamic variables
    n_days = ds.Qobs.values.shape[1]
    
    # Pre-allocate the dataframes
    arr_Qobs        = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_tmax        = np.empty([n_watersheds, n_days], dtype=np.float32)
#    arr_tmin        = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_precip_mm   = np.empty([n_watersheds, n_days], dtype=np.float32)
#    arr_precip_t    = np.empty([n_watersheds, n_days], dtype=np.float32)
#    arr_vp9am       = np.empty([n_watersheds, n_days], dtype=np.float32)
#    arr_Urban_m2    = np.empty([n_watersheds, n_days], dtype=np.float32)
#    arr_Ag_m2       = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_Forest_m2   = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_Sin_Year    = np.empty([n_watersheds, n_days], dtype=np.float32)
#    arr_Cos_Year    = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_days_since  = np.empty([n_watersheds, n_days], dtype=np.float32)
    arr_soilm       = np.empty([n_watersheds, n_days], dtype=np.float32)

    
    arr_Qobs[:]         = np.nan
    arr_tmax[:]         = np.nan
#    arr_tmin[:]         = np.nan
    arr_precip_mm[:]    = np.nan
#    arr_precip_t[:]     = np.nan
#    arr_vp9am[:]        = np.nan
#    arr_Urban_m2[:]     = np.nan
#    arr_Ag_m2[:]        = np.nan
    arr_Forest_m2[:]    = np.nan
    arr_Sin_Year[:]     = np.nan
#    arr_Cos_Year[:]     = np.nan
    arr_days_since[:]   = np.nan
    arr_soilm[:]        = np.nan

    
    # Get the data from the xarray dataset within the dataframes
    arr_Qobs        = ds.Qobs.values
    arr_tmax        = ds.tmax.values
#    arr_tmin        = ds.tmin.values
    arr_precip_mm   = ds.precip_mm.values
#    arr_precip_t    = ds.precip_t.values
#    arr_vp9am       = ds.vp9am.values
#    arr_Urban_m2    = ds.Urban_m2.values
#    arr_Ag_m2       = ds.Ag_m2.values
    arr_Forest_m2   = ds.Forest_m2.values
    arr_Sin_Year    = ds.Sin_Year.values
#    arr_Cos_Year    = ds.Cos_Year.values
    arr_days_since  = ds.days_since.values
    arr_soilm       = ds.soilm.values
    
    # Stack all the dynamic data together
    arr_dynamic = np.dstack((
        arr_Qobs,
        arr_tmax,
#        arr_tmin,
        arr_precip_mm,
#        arr_precip_t,
#        arr_vp9am,
#        arr_Urban_m2,
#        arr_Ag_m2,
        arr_Forest_m2,
        arr_Sin_Year,
#        arr_Cos_Year,
        arr_days_since,
        arr_soilm
        ))
    
    # Extract all watershed descriptors
    arr_static = np.empty([n_watersheds, 18], dtype=np.float32)
    arr_static[:] = np.nan
    
    # For all LSTM model runs
 #   arr_static[:, 0]   = ds.af_2018.values
    arr_static[:, 0]   = ds.area.values
    arr_static[:, 1]   = ds.BulkDensity.values
 #   arr_static[:, 3]   = ds.ClayPC.values
    arr_static[:, 2]   = ds.DOR.values
    arr_static[:, 3]   = ds.elev.values
    arr_static[:, 4]   = ds.MAQ.values
    arr_static[:, 5]   = ds.mean_precip.values
    arr_static[:, 6]   = ds.mean_srad.values
 #   arr_static[:, 9]   = ds.mean_tmax.values
    arr_static[:, 7]  = ds.mean_tmin.values
    arr_static[:, 8]  = ds.mean_vp9am.values
    arr_static[:, 9]  = ds.SandPC.values
    arr_static[:, 10]  = ds.SiltPC.values
    arr_static[:, 11]  = ds.slope.values
    arr_static[:, 12]  = ds.tf_2018.values
    arr_static[:, 13]  = ds.ti_2018.values
    arr_static[:, 14]  = ds.Mean_Ag_km2.values
    arr_static[:, 15]  = ds.Mean_Forest_km2.values
    arr_static[:, 16]  = ds.Mean_Urban_km2.values
    arr_static[:, 17]  = ds.mean_soilm.values

    ### Spatial features for LSTM-sp runs   
    # extract a list of all variable names
    static_vars = list(ds.keys())
    
    # Lattitude / Longitude
    arr_static_LL = np.empty([n_watersheds, 2], dtype=np.float32)
    arr_static_LL[:] = np.nan
    arr_static_LL[:, 0]  = ds.lat.values
    arr_static_LL[:, 1]  = ds.lon.values

    # Euclidean distance, these start at location 9
    arr_static_euc = np.empty([n_watersheds, 114], dtype=np.float32)
    arr_static_euc[:] = np.nan
    
    for j in range(0,114):
        
        jplus = j + 9 
        arr_static_j = static_vars[jplus]
        arr_static_euc[:, j] = ds.variables[arr_static_j].values
    
    # Upstream Nested Catchment, 52 variables, these start at location 149 
    arr_static_unc = np.empty([n_watersheds, 52], dtype=np.float32)
    arr_static_unc[:] = np.nan
    
    for j in range(0,52):
        
        jplus = j + 149
        arr_static_j = static_vars[jplus]
        arr_static_unc[:, j] = ds.variables[arr_static_j].values

    # Upstream Distance, 52 variables, these start at location 202 
    arr_static_usd = np.empty([n_watersheds, 52], dtype=np.float32)
    arr_static_usd[:] = np.nan
    
    for j in range(0,52):
        
        jplus = j + 202
        arr_static_j = static_vars[jplus]
        arr_static_usd[:, j] = ds.variables[arr_static_j].values

    ### FOR BASE = do not add in any spatial variables
    ## Stack all the static data together
    #arr_static = np.hstack((
    #    arr_static,
    #    arr_static_LL,
    #    arr_static_euc,
    #    arr_static_unc,
    #    arr_static_usd
    #    ))


    return arr_dynamic, arr_static, arr_Qobs




def create_LSTM_dataset_per_watershed(arr_dynamic, arr_static, q_stds, window_size, 
                        watershed_list):
    """
    # Particular difficulty here: All catchments have different lengths of data
    """
    
    block_size = np.empty([watershed_list.shape[0]])
    block_size[:] = np.nan
    
    for i in range(0,len(watershed_list)):
        # Find position of NaN in the data block, and since data is all starting
        # at 0 and complete until the end of the series, only the ends are Nan.
        # This is only true for the meteo vars since Qobs has nans a bit everywhere.
        #print("nan search i = ")
        #print(i)
        #print(watershed_list[i])
        indices_nan = np.argwhere(~np.isnan(arr_dynamic[watershed_list[i],:,2]))[-1] #Note, in orig code this = ",2]"
        block_size[i]=(indices_nan - window_size)
            
    block_size = np.int32(block_size)
    # Preallocate the output arrays ()
    X = np.empty([
        np.sum(block_size), # Here because we want the total number of lines but it varies per catchment
        window_size,
        arr_dynamic.shape[2] - 1
        ]    
    , dtype=np.float32)
    
    X[:] = np.nan
    
    X_static = np.empty([
        np.sum(block_size),
        arr_static.shape[1]
        ]  , dtype=np.float32  
    )    
    
    X_static[:] = np.nan
    
    X_q_stds = np.empty([
        np.sum(block_size)
        ]    , dtype=np.float32
    )    
    
    X_q_stds[:] = np.nan
    
    y = np.empty([
        np.sum(block_size)
        ], dtype=np.float32
    )
    
    y[:] = np.nan

    
    block_size_cumsum = np.cumsum(block_size)

   # counter = 0
    for w in watershed_list:

        position_in_watershed_list = np.argwhere(watershed_list==w)[0][0]
        
        X_w, X_w_static, X_w_q_std, y_w = extract_watershed_block(
            arr_w_dynamic=arr_dynamic[w, 0:block_size[position_in_watershed_list]+window_size, :],
            arr_w_static=arr_static[w, :],
            q_std_w=q_stds[w],
            window_size=window_size
            )
            
        X[block_size_cumsum[position_in_watershed_list]-block_size[position_in_watershed_list]:block_size_cumsum[position_in_watershed_list],:,:]=X_w
        X_static[block_size_cumsum[position_in_watershed_list]-block_size[position_in_watershed_list]:block_size_cumsum[position_in_watershed_list], :] = X_w_static
        X_q_stds[block_size_cumsum[position_in_watershed_list]-block_size[position_in_watershed_list]:block_size_cumsum[position_in_watershed_list]] = X_w_q_std
        y[block_size_cumsum[position_in_watershed_list]-block_size[position_in_watershed_list]:block_size_cumsum[position_in_watershed_list]] = y_w
        
        
        
    return X, X_static, X_q_stds, y


def extract_watershed_block(arr_w_dynamic, arr_w_static, q_std_w, window_size):
    """
    This function extracts all series of the desired window length over all
    features for a given watershed. Both dynamic and static variables are 
    extracted.
    """
    
    # Extract all series of the desired window length for all features
    X_w = extract_windows_vectorized(
        array=arr_w_dynamic,
        sub_window_size=window_size
    )

    X_w_static = np.repeat(arr_w_static.reshape(-1,1), X_w.shape[0], axis=1).T

    X_w_q_std = np.squeeze(np.repeat(q_std_w.reshape(-1,1), X_w.shape[0], axis=1).T)

    # Get the last value of Qobs from each series for the prediction
    y_w = X_w[:, -1, 0]

    # Remove Qobs from the features
    X_w = np.delete(X_w, 0, axis=2)

    return X_w, X_w_static, X_w_q_std, y_w

def extract_windows_vectorized(array, sub_window_size):
    """
    Vectorized sliding window extractor.
    This method is more efficient than the naive sliding window extractor.

    For more info, see:
    https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    :param array: The array from which to extract windows
    :param sub_window_size: The size of the sliding window
    :return: an array of all the sub windows
    """
    max_time = array.shape[0]
    array = array.astype('float32')
    # expand_dims are used to convert a 1D array to 2D array.
    sub_windows = (
            np.expand_dims(np.arange(sub_window_size), 0) +
            np.expand_dims(np.arange(max_time - sub_window_size), 0).T
    )

    return array[sub_windows].astype('float32')


# %% Functions to create model


def nse_loss(data, y_pred):
    """

    """
        
    y_true = data[:, 0]
    q_stds = data[:, 1]
    y_pred = y_pred[:, 0]
    
    # y_true = K.print_tensor(y_true, message='y_true = ')
    # y_pred = K.print_tensor(y_pred, message='y_pred = ')
    # q_stds = K.print_tensor(q_stds, message='q_stds = ')

    eps = float(0.1)
    squared_error = (y_pred - y_true) ** 2
    weights = 1 / (q_stds + eps) ** 2
    scaled_loss = weights * squared_error

    return scaled_loss


def define_LSTM_model(window_size, n_dynamic_features, n_static_features,
                      clip_value=1, checkpoint_path='tmp.h5'):
    """
    This is where the LSTM model is actually defined. The structure here can be
    adjusted to suit particular needs. Testing is required to get optimal
    hyperparameters.
    """
    
    x_in_365 = tf.keras.layers.Input(shape=(window_size, n_dynamic_features))
    x_in_static = tf.keras.layers.Input(shape=n_static_features)

    # LSTM 365 day
    x_365 = tf.keras.layers.LSTM(512, return_sequences=True)(x_in_365)
    x_365 = tf.keras.layers.LSTM(512, return_sequences=False)(x_365)
    x_365 = tf.keras.layers.Dropout(0.3)(x_365)

    # Dense statics
    x_static = tf.keras.layers.Dense(25)(x_in_static)
    x_static = tf.keras.layers.Dropout(0.1)(x_static)
    x_static = tf.keras.layers.LeakyReLU(alpha=0.1)(x_static)

    # Concatenate the model
    x = tf.keras.layers.Concatenate()([x_365, x_static])
    x = tf.keras.layers.Dense(20)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x_out = tf.keras.layers.Dense(1)(x)


    model_LSTM = tf.keras.models.Model([x_in_365, x_in_static], [x_out])
    model_LSTM.compile(loss=nse_loss,
                           optimizer=tf.keras.optimizers.Adam(clipnorm=clip_value)
                           )

    callback = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            save_freq='epoch', 
            save_best_only=True, 
            monitor='val_loss',
            mode='min',
            verbose=1
            ), 
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            verbose=1, 
            patience=5
            ),
        tf.keras.callbacks.LearningRateScheduler(
            step_decay
            )
        ]

    return model_LSTM, callback

# Adjust learning rate as a function of the number of epochs
def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1 + epoch)/epochs_drop)
           )
   return lrate 



class training_generator(tf.keras.utils.Sequence):
    """
    Create a training generator to empty the GPU memory during training:
    """
    def __init__(self, x_set, x_set_static, x_set_q_stds, y_set, batch_size):
        self.x = x_set
        self.x_static = x_set_static
        self.x_q_stds = x_set_q_stds
        self.y = y_set

        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return (np.ceil(len(self.y) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_x_static = self.x_static[inds]
        batch_x_q_stds = self.x_q_stds[inds]
        batch_y = self.y[inds]
        
        return [np.array(batch_x), np.array(batch_x_static)], \
            np.vstack((np.array(batch_y), np.array(batch_x_q_stds))).T

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

        
class testing_generator(tf.keras.utils.Sequence):
    """
    Create a testing generator to empty the GPU memory during testing:
    """

    def __init__(self, x_set, x_set_static, batch_size):
        self.x = x_set
        self.x_static = x_set_static
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(self.x.shape[0] / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x_static = self.x_static[
            idx * self.batch_size: (idx + 1) * self.batch_size
            ]
        return [batch_x, batch_x_static], batch_x_static


# %% Run as script 
if __name__ == "__main__":

    # Load and pre-process the dataset
    ds = xr.open_dataset(input_data_filename)
    print(ds.info)
       
    # Number of watersheds in the dataset
    n_watersheds = ds.Qobs.shape[0]
    print('n_watersheds = ',n_watersheds)
    
    # Load prediction dataset 
    #dpred = xr.open_dataset(pred_data_filename)
    
    # Number of prediction sites in the dataset
    #n_predsites = dpred.site.shape[0]
    #print('n_predsites = ',n_predsites)
    
    np.random.seed(42)
    
    ### Note, the index here needs to be reduced by to account for Python indexing starting at 0 / R starting at 1
    #loo_test_index = data_rep - 1 #removed test index as no longer needed
    #loo_R_index = data_rep
    
    # Watersheds list to be used for the LOO testing
    #test_ws = loo_test_index
    #test_ws = np.array(test_ws)
    #print("LOO site = ", loo_test_index)
    # Shuffle watersheds numbers
    watersheds_ind = np.arange(0, n_watersheds)
    np.random.shuffle(watersheds_ind)
    watersheds_ind = watersheds_ind[0:n_watersheds]
    #watersheds_ind = np.delete(watersheds_ind, np.where(watersheds_ind == loo_test_index), axis=0)
     
    # K-fold 80/20 split     
    rkf = KFold(n_splits=5, shuffle=True, random_state = 94) 

    print("cv start")
    
    for i, (train_index, test_index) in enumerate(rkf.split(watersheds_ind)):
        print("cv start. i = ",i)
        # Watersheds list to be used for the training ("ws" stands for watersheds)
        train_valid_ws = watersheds_ind
        
        #train_ws = train_valid_ws[: round(n_watersheds * train_pct / 100)]
        train_ws = train_valid_ws[train_index]
        
        # Watersheds list to be used for the validation
        valid_ws = train_valid_ws[test_index]
        valid_ws = np.array(valid_ws)
        
        # %% Prepare the LSTM features (inputs) and target variable (output):
        # Create the dataset
        arr_dynamic, arr_static, arr_Qobs = create_LSTM_dataset(ds)
        
        # Get the standard deviation of the streamflow for all watersheds
        q_stds = np.nanstd(arr_Qobs, axis=1)
        
        n_var = arr_dynamic.shape[2]  # Number of dynamic variables
    
        # Scale the dynamic feature variables (not the target variable)
        scaler_dynamic = StandardScaler()  # Use standardization by mean and std
        
        # Fit the scaler using only the training watersheds
        _ = scaler_dynamic.fit_transform(arr_dynamic[train_ws, :, :].reshape(-1, n_var)[:, 1:]) 
        
        # Scale all watersheds dynamic data incl LOO site
        #for w in np.append(watersheds_ind,loo_test_index): #final, no Hydro sites excluded from watersheds_ind
        for w in watersheds_ind:
            arr_dynamic[w, :, 1:] = scaler_dynamic.transform(arr_dynamic[w, :, 1:])
        
        # Scale the static feature variables
        scaler_static = MinMaxScaler()  # Use normalization between 0 and 1    
        _ = scaler_static.fit_transform(arr_static[train_ws, :])
        
        # Scale all watersheds static data
        arr_static = scaler_static.transform(arr_static)
        
        
        # %% Prepare datasets for training and validation
        # Training dataset
        print("Training dataset")
        X_train, X_train_static, X_train_q_stds, y_train = create_LSTM_dataset_per_watershed(
            arr_dynamic=arr_dynamic, 
            arr_static=arr_static, 
            q_stds=q_stds,
            window_size=window_size, 
            watershed_list=train_ws
            )
        
        ind_nan = np.isnan(y_train)
        ind_nan_train = ind_nan
        y_train = y_train[~ind_nan]
        X_train = X_train[~ind_nan, :]
        X_train_q_stds = X_train_q_stds[~ind_nan]
        X_train_static = X_train_static[~ind_nan, :]
    
        # Validation dataset   
        print("Validation dataset")
        X_valid, X_valid_static, X_valid_q_stds, y_valid = create_LSTM_dataset_per_watershed(
            arr_dynamic=arr_dynamic, 
            arr_static=arr_static, 
            q_stds=q_stds,
            window_size=window_size, 
            watershed_list=valid_ws 
            )
        
        ind_nan = np.isnan(y_valid)
        ind_nan_val = ind_nan
        y_valid = y_valid[~ind_nan]
        X_valid = X_valid[~ind_nan, :]
        X_valid_q_stds = X_valid_q_stds[~ind_nan]
        X_valid_static = X_valid_static[~ind_nan, :]
        
        # %% Training the model
        # The LSTM model is trained until no nan loss is obtained  
        print("Training model")
        number_of_run = 1
        #number_of_run = rep_limit
        loss_table = np.empty((number_of_run, 1))
        loss_table.fill(np.nan)
        
         
        name_of_saved_model = 'LSTM_final_mods/LSTM_dailyflow_final_kfold_' + str(i) + '.h5' 
        
        # We want all, so instead of chaning code, leave this here.
        ind_train = np.squeeze(np.argwhere(y_train>-1000))
        ind_valid = np.squeeze(np.argwhere(y_valid>-1000))
        
        # Prepare pre-training
        print('pre-training')
        success = 0
        while success == 0:
            K.clear_session()  # Reset the model
    
            model_LSTM, callback = define_LSTM_model(
                window_size=window_size, 
                n_dynamic_features=X_train.shape[2],
                n_static_features=X_train_static.shape[1],
                checkpoint_path=name_of_saved_model
                )
        
            h = model_LSTM.fit(
                training_generator(
                    X_train[ind_train, :, :], 
                    X_train_static[ind_train, :], 
                    X_train_q_stds[ind_train], 
                    y_train[ind_train], 
                    batch_size=batch_size_val
                    ), 
                epochs=3, # maximum iterations to attain KGE>0 (FMIN<1)
                validation_data=training_generator(
                    X_valid[ind_valid, :, :], 
                    X_valid_static[ind_valid, :], 
                    X_valid_q_stds[ind_valid],
                    y_valid[ind_valid], 
                    batch_size=batch_size_val
                    ),
                callbacks=[callback],
                verbose=1,
            )        
    
            # We are probably out of the woods, check for NaN in case.        
            if not np.isnan(h.history['loss'][-1]):
                success = 1
                
            if h.history['loss'][-1] > threshold_loss:
                success = 0
        
        # Do the actual fitting with epoch_val epochs.
        print('model fitting')
        h = model_LSTM.fit(
            training_generator(
                X_train, 
                X_train_static, 
                X_train_q_stds, 
                y_train, 
                batch_size=batch_size_val
                ), 
            epochs=epoch_val, 
            validation_data=training_generator(
                X_valid, 
                X_valid_static, 
                X_valid_q_stds,
                y_valid, 
                batch_size=batch_size_val
                ),
            callbacks=[callback],
            verbose=1,
        )        
       
        
     
     
        
     
        print("cv end. i = ",i)

        # Delete and reload the model to free the memory
        del model_LSTM
        K.clear_session()

        
    

