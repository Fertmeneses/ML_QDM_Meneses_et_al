"""
Machine Learning models to predict Z-position from magnetic fields.

# ===========

Main functions:

    - load_data_and_gen_pars
    - quick_training
    - check_trained_model
    - train_stage1
    - train_stage2
    - train_stage3
    - train_stage4
    - train_stage5
    - train_stage6
    - train_stage7

# ===========

Functions included in Class <ML_Model>:

    - __init__
    - info
    - train_model
    - test_model

# ===========

Functions included in Class <PlotLosses>:

    - on_train_begin
    - on_epoch_end

# ===========

"""

# ============================================================================

# Required packages:

import numpy as np
import pandas as pd
import os
import json
import time
import itertools
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import InputLayer,Reshape,Dense,Dropout,Conv1D,MaxPooling1D,Flatten,GlobalAveragePooling1D
#from tensorflow.python.keras.utils.layer_utils import count_params
from keras.regularizers import l2

# Internal packages:

import MLQDM.general as ML_general
import MLQDM.timewindows as ML_twdw

# ============================================================================

# Plots configuration:

plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 10
plt.rcParams['legend.title_fontsize'] = 8
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200

plot_markers = ["o","<","s","d","v","p","P","^","*","D",">"]

# ============================================================================

def load_data_and_gen_pars(
    data_path,
    gen_pars_path,
    interp='lin_approx',
    final_stage=False
    ):
    """
    Load datasets and general parameters, which include hyperparameters and rotational frames.
    
    --- Inputs ---

    {data_path} [String]: Path for dataset .csv files containing magnetic (predictors) and Z-position
    (targets) information, each file a record that represents a segment and must have columns
    titled time_s, Bx_nT, By_nT, Bz_nT, and either zAut_m or zTrue_m (can be both). Files must have
    a name that includes either 'lin_approx' or 'phys_model'.
    {gen_pars_path} [String]: Path for .json files containing information about general (hyper)parameters,
    which must contain '_gen_pars_' in its name, and rotational frames, which must contain '_RF_' in its name.
    {interp} [String]: Choose between 'lin_approx' or 'phys_model', which sets the interpolation method for
    the Z-position labels, either Linear Approximation or Physical Model.
    {final_stage} [Boolean]: Defines the general parameters that will be loaded. Choose True only if this
    training stage is the final one.

    --- Return ---

    {data} [List]: Each element is a DataFrame with information about predictors and targets, with
    columns 'time_s', 'Bx_nT', By_nT, Bz_nT, and zAut_m and/or zTrue_m.
    {hypers} [Dictionary]: Information for hyperparameters, to be used when training the model.
    {RFs} [Dictionary]: Information for rotational frames to be used when training the model.

    """

    # Select keyword for this stage:
    stage_key = '_S7' if final_stage else '_S1to6' # Filter keyword representing the training stage

    # Load information from data files:
    files = [data_path+file for file in os.listdir(data_path) if f'{interp}_' in file]
    data = [pd.read_csv(file) for file in files]

    # Load general (hyper)parameters:
    hypers_path = [file for file in os.listdir(gen_pars_path) if
        '_gen_pars_' in file and stage_key in file][0] # Select the unique correct file
    with open(gen_pars_path+hypers_path) as f:
        hypers = json.load(f) # Load dictionary with hyper-parameters

    # Load rotational frames for this stage:
    RF_path = [file for file in os.listdir(gen_pars_path) if 
        '_RF_' in file and stage_key in file][0] # Select the unique correct file
    with open(gen_pars_path+RF_path) as f:
        RFs = json.load(f) # Load dictionary with Rotational frames

    return data, hypers, RFs

# ============================================================================

class ML_Model:
    """
    The Machine Learning Model object <ML_Model> contains an supervised artificial intelligence model
    that can be trained and tested with <Time_Wdw> objects. It predicts the z-position of the elevator
    based on magnetic field signals. It can work in arbitrary rotational frames.

    --- Attributes ---
    
    {self.hyps} [Dictionary]: Contains all the hyper-parameters information to build the Keras model, 
    with the following keys, and also examples for values:
        * "Time_Window_pp": 20
        * "Magnetic_Components": ["Bx","Bz"] # Order does not matter
        * "Loss_Function": "MAE"
        * "Last_Activation_Function": 'linear'
        * "Batch_Size": 128,
        * "Epochs": 50
        * "Training_p_val": 0.2 # Validation dataset fraction
        * "Early_Stop_Monitor": "val_loss" # Criterium for early stopping during training.
        * "Earlt_Stop_Min_Delta": 0 # Minimum improvement to prevent early stopping.
        * "Early_Stop_Patience": 10 # Epochs during which the early stopping happens if no
        improvement is registered.
        * "Early_Stop_Start_From_Epoch": 20 # First epochs in which early stopping can't occur.
        * "Early_Stop_Restore_Best_Weights": True # Allows to restore best weights when the
        training is early stopped.
        * "z_thres": 1 # z-value threshold for accuracy, in [m]
        * "Activation_Function": 'tanh'
        * "Optimizer": 'adam'
        * "Learning_Rate": 0.0003
        * "RF": [[np.array([0.41,0.75,0.52]),90,'RF2']] # List with rotational frames, each element is a list with
        the format [Numpy array,String,Numeric]; meaning [rotation axis in (x,y,z) format, rotation angle in degree,
        RF name]. If RF name is 'RF1', then the data in kept in the original frame (defined as "RF1").
        * "Convolutional_Network": True # Allow to use Convolutional Layers before the Dense layers
        * "Conv_Layers": [[64,5],[128,5]] # List with Conv. layers, in order, in the format [filters,kernel size]
        * "Pool_Layers": [None,None] # List with pooling layers, in order, each element is the size (None=ignore layer)
        * "Flatten_Average": If True, just flattens the input data. If False, makes a global averaging for the many Conv filters.
        * "Dens_Layers": [1000,500] # List with dense layers, in order, each element is the number of neurons
        * "Dropout_Fraction": 0.2 # Fraction for dropout layers after each dense layer (0=ignore layer)
        * "Model_Name": "S1_Conv_5_5" # Usually contains the Stage and architecture information
        * "Full_Name": "S1_Conv_5_5_wdw2s_Bx_By_" # Contains all the information
        * "Train_Segms": "segm1segm3" # Specifies the segments used in training.
        * "Test_Segms": "segm1segm3" # Specifies the segments used in testing.
        * "p_train": 1 # Float between 0 and 1; fraction of the training dataset that is going
        to be used. If 1: entire training dataset.
        * "seed": 0 # Random initialization seed.
        * "N_augm": 1 # Number of times the original training dataset will be augmented.
        * "noise": [1,1,1] # List comprised of 3 float numbers; noise intensity for data
        augmentation in the [Bx,By,Bz] components, in [nT] units. Only relevant if N_augm>0.
    {self.model} [Keras object]: Machine Learning model.
    {self.seed} [Integer]: seed used in weights initialization when building the model.
    """

    # =============

    def __init__(
        self, 
        full_hyp, 
        seed=0, 
        load_model=None
        ):
        """
        Initiate ML_Model object with basic information.

        --- Inputs ---

        {full_hyp} [Dictionary]: Hyperparameters information.
        {load_model} [None or Keras object]: If None, starts a new model. Else, load a previous model.
        {seed} [Integer]: seed for weight initialization.

        --- Return ---

        Create the ML_Model object.

        """        
        self.hyps = full_hyp # Incorporate hyper-parameters information
        self.seed = seed # Record seed
        # Start building model:
        if load_model is not None:
            self.model = tf.keras.models.load_model(load_model)
        else:
            self.model = Sequential() # Initiate Keras object
            self.model.add(InputLayer(shape=(self.hyps["Time_Window_pp"],
                len(self.hyps["Magnetic_Components"])))) # First (input) layer
            # Add Convolutional, Pooling and Flattening layers (if any):
            if self.hyps["Convolutional_Network"]:
                for i in range(len(self.hyps["Conv_Layers"])):
                    self.model.add(Conv1D(
                        filters=self.hyps["Conv_Layers"][i][0],
                        kernel_size=self.hyps["Conv_Layers"][i][1],
                        activation=self.hyps["Activation_Function"],
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
                        kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
                    ))
                    if self.hyps["Pool_Layers"][i] is not None:
                        self.model.add(MaxPooling1D(pool_size=self.hyps["Pool_Layers"][i]))
                self.model.add(Flatten()) if self.hyps["Flatten_Average"] else self.model.add(GlobalAveragePooling1D()) 
            else:
                self.model.add(Flatten())
            # Add Dense layers (and Dropout, if any):
            for i in range(len(self.hyps["Dens_Layers"])):
                self.model.add(Dense(
                    self.hyps["Dens_Layers"][i],
                    activation=self.hyps["Activation_Function"],
                    kernel_initializer='random_normal'))
                if self.hyps["Dropout_Fraction"]:
                    self.model.add(Dropout(self.hyps["Dropout_Fraction"]))
            # Last layers:
            self.model.add(Dense(1, activation=self.hyps["Last_Activation_Function"],
                                kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
                                ))
            # Compile model:
            self.model.compile(
                loss=self.hyps["Loss_Function"],
                optimizer=self.hyps["Optimizer"],
            )
    
    # =============

    def info(
        self
        ):
        """
        Print in screen information about the ML model.
        """
        print('-'*30)
        print('Full name:',self.hyps["Full_Name"])
        print("Time windows' points:",self.hyps["Time_Window_pp"])
        print("Input magnetic components:",self.hyps["Magnetic_Components"])
        print("Model's trainable parameters:",self.model.count_params())
        print('-'*30)
    
    # =============

    def train_model(
        self,
        time_wdws,
        RF_info,
        savefigs_path=None,
        save_format='png',
        exp_model_path=None,
        device='/GPU:0'
        ):
        """
        Train the ML model based on provided <Time_Wdw> objects and specified rotational frame. 
        Data is randomly split between training and validation datasets.
        
        --- Inputs ---
        
        {time_wdws} [List]: Each element is a <Time_Wdw> object, the list is the entire dataset 
        that will be split into validation and training.
        {RF_info} [List]: Each element is a rotational frame, a list with the format
        [Numpy array,Numeric,String]; meaning [rotation axis in (X,Y,Z) format, rotation angle
        in degree, RF name]. If RF[2]='RF1', then the data is kept in the original frame.
        {savefigs_path} [String]: If specified, save the training figures in .png format by default.
        {save_format} [String]: Choose the saving format, don't include the dot.
        {exp_model_path} [String]: If specified, exports the ML model.
        {device} [String]: Choose between '/GPU:0' and '/CPU:0' as the hardware resource for
        training.
        
        --- Return ---
        
        Train the ML model stored in the <self.model> attribute, updating weights and biases.

        {results} [Dictionary]: Contains information about the model and its results.

        """

        # Prepare data according to magnetic components and rotational frame:
        mag_data, z_data = [], [] # Initiate predictors and targets 
        for time_wdw in time_wdws:
            time_wdw.rotate_frame(RF_info)
            mag_data.append(time_wdw.matrix_format(self.hyps["Magnetic_Components"])) # Array format [samples,wdw_pp,channels], units [nT]
            z_data.append(time_wdw.z_labels) # z-position labels, units [m]
        all_mag_data = np.concatenate(mag_data,axis=0) # Concatenate all arrays, units [nT]
        all_z_data = np.concatenate(z_data,axis=0) # Concatenate all arrays, units [m]
        # Normalize magnetic data:
        all_mag_data /= time_wdws[0].norm_value # Adimensional units

        # Split training/validating datasets:
        msk = np.random.rand(all_mag_data.shape[0]) < 1-self.hyps["Training_p_val"] # Mask for training indexes
        X_train, Y_train = all_mag_data[msk], all_z_data[msk] # Training predictors and targets
        X_val, Y_val = all_mag_data[~msk], all_z_data[~msk] # Validating predictors and targets

        # Prepare callbacks:
        early_stop = keras.callbacks.EarlyStopping(
            monitor=self.hyps["Early_Stop_Monitor"],
            min_delta=self.hyps["Early_Stop_Min_Delta"],
            mode='min',
            patience=self.hyps["Early_Stop_Patience"],
            start_from_epoch=self.hyps["Early_Stop_Start_From_Epoch"],
            restore_best_weights=self.hyps["Early_Stop_Restore_Best_Weights"],
            verbose=1
            )

        # Train the model:
        with tf.device(device):
            history = self.model.fit(
                X_train,Y_train,
                validation_data=(X_val, Y_val),
                epochs=self.hyps["Epochs"],
                callbacks=[plot_losses,early_stop],
                batch_size=self.hyps["Batch_Size"],
                verbose=1
                )

        # Save the training figure:
        history_frame = pd.DataFrame(history.history) # Generate training dataframe
        epochs = range(1,len(history_frame)+1) # (Range) Effective epochs
        if savefigs_path is not None:
            fig_name = f'Training_{self.hyps["Full_Name"]}'
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(epochs,history_frame['val_loss'],'-s', label="val_loss", color='gray', alpha=0.8)
            ax.plot(epochs,history_frame['loss'],'-o', label="loss", color='brown', alpha=0.8)
            ax.set(xlabel='Epochs',ylabel=f'Loss function {self.hyps["Loss_Function"]} [m]')
            ax.legend()
            fig.tight_layout()
            save_file(savefigs_path+fig_name,save_format=save_format)
            plt.close(fig)

        # Predictions on validation dataset:
        z_val_pred = np.squeeze(self.model.predict(X_val,verbose=1)) # Get the predictions [m]
        AE_val = np.transpose(np.abs(Y_val-z_val_pred)) # Array with absolute errors [m]
        # Calculate accuracy according to different threshold:
        acc_z = sum(AE_val<self.hyps["z_thres"])/len(AE_val)*100 # [%]
        print('-'*30)
        print(f'VALIDATING; Accuracy using {self.hyps["z_thres"]}m threshold: {np.round(acc_z,1)}%')
        print('-'*30)
            
        # Export the ML model:
        if exp_model_path is not None:
            self.model.save(exp_model_path+self.hyps["Full_Name"]+'.keras')
      
        # Generate dictionary with basic results:
        res_train = {
                "Full_Name": self.hyps["Full_Name"],
                "Wdw_pp": self.hyps["Time_Window_pp"],
                "Mag_Comps": "".join(self.hyps["Magnetic_Components"]),
                "RF": RF_info[2],
                "Seed": self.seed,
                "Train_Segms": self.hyps["Train_Segms"],
                "Batch": self.hyps["Batch_Size"],
                "Epochs": epochs[-1],
                "p_val": self.hyps["Training_p_val"],
                "Activ_func": self.hyps["Activation_Function"],
                "Optimizer": self.hyps["Optimizer"],
                "Learn_rate": self.hyps["Learning_Rate"],
                "Model_Name": self.hyps["Model_Name"],
                "Best_MAE_val": min(history_frame['val_loss']), # [m]
                "Best_MAE_train": min(history_frame['loss']), # [m]
                "z_thres": self.hyps["z_thres"], # [m]
                "Acc_Val_z": acc_z, # [%]
                "Trainable_pars": self.model.count_params()
                }
        # Then optional parameters:
        for par in ['p_train','N_augm','noise']:
            if par in self.hyps:
                if par=='noise':
                    res_train['Bx_noise'] = self.hyps[par][0]
                    res_train['By_noise'] = self.hyps[par][1]
                    res_train['Bz_noise'] = self.hyps[par][2]
                else:
                    res_train[par] = self.hyps[par]
        
        return res_train

    # =============

    def test_model(
        self,
        time_wdws,
        RF_info,
        return_preds=False,
        savefigs_path=None,
        save_format='png'
        ):
        """
        Test the ML model (already trained) on provided <Time_Wdw> objects and specified
        rotational frame. Returns the performance results and, if requested, the ground truth
        and predicted arrays.
        
        --- Inputs ---

        {time_wdws} [List]: each element is a <Time_Wdw> object,  the list is the entire testing dataset.
            {RF_info} [List]: Each element is a rotational frame, a list with the format
        [Numpy array,Numeric,String]; meaning [rotation axis in (X,Y,Z) format, rotation angle
        in degree, RF name]. If RF[2]='RF1', then the data is kept in the original frame.
        {return_preds} [Boolean]: If True, returns the ground truth and predicted arrays.
        {savefigs_path} [String]: If specified, save the training figures in .png format by default.
        {save_format} [String]: Choose the saving format, don't include the dot.
        {exp_model_path} [String]: If specified, exports the ML model.
        
        --- Return ---
        
        {results} [Dictionary]: Contains information about the model and its results.

        Optional, only if {return_preds}=True:

        {t_data} [List]: Each element is a Numpy array representing the time, correlated with
        z-position, for a single segment. Units [s].
        {z_data} [List]: Each element is a Numpy array representing the ground truth for z-position
        for a single segment. Units [m].
        {z_preds} [List] Each element is a Numpy array representing the predictions for z-position
        for a single segment. Units [m].

        """

        # Prepare data according to magnetic components and rotational frame:
        segms = [time_wdw.name for time_wdw in time_wdws]
        self.hyps["Test_Segms"] = "".join(segms)
        mag_data, z_data, t_data = [], [], [] # Initiate predictors, targets and time
        for time_wdw in time_wdws:
            time_wdw.rotate_frame(RF_info)
            mag_data.append(time_wdw.matrix_format(self.hyps["Magnetic_Components"])) # Array format [samples,wdw_pp,channels], units [nT]
            z_data.append(time_wdw.z_labels) # z-position labels, units [m]
            t_data.append(time_wdw.time[time_wdw.pp:]) # Time, units [s]
        all_mag_data = np.concatenate(mag_data,axis=0) # Concatenate all arrays, units [nT]
        all_z_data = np.concatenate(z_data,axis=0) # Concatenate all arrays, units [m]
        all_t_data = np.concatenate(t_data,axis=0) # Concatenate all arrays, units [s]
        # Normalize magnetic data:
        all_mag_data /= time_wdws[0].norm_value # Adimensional units
        for vec in mag_data:
            vec /= time_wdws[0].norm_value # Adimensional units
        
        # Obtain predictions and calculate performance metrics for ALL data:
        z_pred = np.squeeze(self.model.predict(all_mag_data,verbose=1)) # Get the predictions [m]
        AE_test = np.transpose(np.abs(all_z_data-z_pred)) # Array with absolute errors [m]
        # Calculate accuracy according to different threshold:
        acc_z = sum(AE_test<self.hyps["z_thres"])/len(AE_test)*100 # [%]
        print('-'*30)
        print(f'TESTING; Accuracy using {self.hyps["z_thres"]}m threshold: {np.round(acc_z,1)}%')
        print('-'*30)
        # Obtain predictions for each segment:
        z_preds = [np.squeeze(self.model.predict(mag_data[i],verbose=2)) for i in range(len(mag_data))] # [m]

        # Save the predictions figure:
        if savefigs_path is not None:
            for i in range(len(t_data)):
                fig_name = f'Testing_{self.hyps["Full_Name"]}_{time_wdws[i].name}'
                fig, ax = plt.subplots(figsize=(8,3))
                ax.plot(t_data[i],z_data[i],'--',color='teal',lw=0.5,label='Actual',alpha=0.8)
                ax.plot(t_data[i],z_preds[i],'-r',lw=0.5,label='Predictions',alpha=0.8)
                for z in np.append(0,4.1+np.arange(0,3.7*6+0.1,3.7)):
                    ax.axhline(z,ls='-',lw=0.5,alpha=0.2)         
                ax.set(xlabel='Time [s]',ylabel='z-pos [m]')
                ax.legend()
                fig.tight_layout()
                save_file(savefigs_path+fig_name,save_format=save_format)
                plt.show()

        # Prepare testing results:
        res_test = {
            "Test_Segms": self.hyps["Test_Segms"],
            "MAE_test": AE_test.mean(),
            "Acc_Test_z": acc_z,
        }

        if return_preds:
            return res_test, t_data, z_data, z_preds
        else:
            return res_test

    # =============

# ============================================================================

class PlotLosses(keras.callbacks.Callback):
    """
    This class will be used to plot live charts of the ML training process.
    Adapted from: [Github] stared/live_loss_plot_keras.ipynb, link:
    https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e.

    Called during ML training as a callback, it plots the live progress of the
    loss function for both training and validating datasets, as a function of
    epochs. The plot is updated for every single epoch. 
    """

    # =============

    def on_train_begin(
        self,
        logs={}
        ):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []        
        self.logs = []

    # =============

    def on_epoch_end(
        self, 
        epoch, 
        logs={}
        ):
        # Prepare data:
        self.logs.append(logs)
        self.x.append(self.i+1)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        self.fig = plt.figure(figsize=(6,3))

        # Plot figure:
        clear_output(wait=True)
        if not all(elem is None for elem in self.val_losses):
            plt.plot(self.x, self.val_losses,'-s', label="val_loss", color='gray', alpha=0.8)        
        plt.plot(self.x, self.losses,'-o', label="loss", color='brown', alpha=0.8)
        plt.legend()
        plt.xlabel("Epochs"), plt.ylabel("Loss (MAE) [m]")
        #plt.yscale('log') 
        plt.show()
        
        # =============

plot_losses = PlotLosses() # Initiate the class, will be used during future ML training

# ============================================================================

def quick_training(
    full_hyp,
    t_wdws_train,
    t_wdws_test,
    RF,
    N_models=100,
    quick_epochs=5,
    total_epochs=100,
    seed=0
    ):
    """
    Xxxx
    """

    # Screen display:
    print('-'*10,' QUICK TIMING TEST MODE ','-'*10)
    print('Running dry test for 5 epochs...')
    time.sleep(1)

    start = time.time() # Start tracking the time
    
    # Prepare the ML model:   
    keras.backend.clear_session() # Clear memory
    full_hyp['Epochs'] = quick_epochs # Set number of epochs for training
    ML_model = ML_Model(full_hyp,seed=seed)

    # Train and test the ML model:
    _ = ML_model.train_model(t_wdws_train,RF)
    _ = ML_model.test_model(t_wdws_test,RF)

    end = time.time() # Stop tracking the time

    # Calculate elapsed time:
    elapsed_time = end - start # [s]
    single_model_est_time = elapsed_time/5*total_epochs/60 # [min]

    # Print estimated times:
    print('\n','-'*40,'\n')
    print(f'Trainable models in this training session:',N_models)
    print(f'Estimated training time per model ({total_epochs} epochs): {np.round(single_model_est_time,1)} min')
    print(f'Estimated total time for training {N_models} models: {np.round(single_model_est_time*N_models/60,1)} hours')
    print('\n','-'*10,' IMPORTANT ','-'*10,'\n')
    print('If you want to train all models, select quick_timing_test=False')

# ============================================================================

def check_trained_model(
    ref_path=None
    ):
    """
    Compiles the information for the already trained models according to a reference file
    and returns a list with all those models.

    --- Inputs ---

    {ref_path} [String]: Path for the results' file of the already trained model. The format
    of the file must be equal to the one produced in any of the traning function train_stageX.

    --- Return ---

    {trained_models} [List]: Each element is the full name of an already trained model. 

    """
    if ref_path is not None:
        pd_ref = pd.read_csv(ref_path)
        trained_models = list(pd_ref["Full_Name"]) # Extract the full names of trained ML models
    else:
        trained_models = [] # An empty list means that there are no already trained models

    return trained_models

# ============================================================================        

# For the future training stages, prepare the standard information to initiate...
# ... the DataFrame which summarizes in the results:

standard_info_df_ML = [
'Full_Name','Wdw_pp','Mag_Comps','RF','Seed',
'Train_Segms','Test_Segms',
'Batch','Epochs','p_val','Activ_func','Optimizer',
'Learn_rate','Model_Name','Best_MAE_val',
'Best_MAE_train','MAE_test',
'z_thres','Acc_Val_z','Acc_Test_z'
]

# Augmented data options:
augmented_info_df_ML = [
'p_train','N_augm','Bx_noise','By_noise','Bz_noise',
]

# ============================================================================

def train_stage1(
    B_comps_opts,
    arch_opts,
    RF_opts,
    t_wdws_train,
    t_wdws_test,
    seed_opts=[0],
    interpolation=None,
    savefigs_path=None,
    exp_model_path=None,
    results_path='./',
    check_rep_model=None,
    quick_timing_test=False
    ):
    """
    Training procedure for Stage 1, focusing on magnetic components. It trains many
    ML models using different options for magnetic components, architectures, seeds
    and rotational frames. It returns and exports a dataframe with the results.

    --- Inputs ---

    {B_comps_opts} [List]: Each element is a list in which the different magnetic
    components are included as elements. Options: 'Bx', 'By', 'Bz', 'B'.
    {arch_opts} [List]: Each element is a dictionary with the ML model hyper-parameters.
    {RF_opts} [List]: Each element is a list with the format [Numpy array,Numeric,String];
    meaning [rotation axis in (x,y,z) format, rotation angle in degree, RF name].
    If RF[2]='RF1', then the data is kept in the original frame.
    {t_wdws_train} [List]: Each element is a <Time_Wdw> object that will be used for
    the training process.
    {t_wdws_test} [List]: Each element is a <Time_Wdw> object that will be used for
    the testing process.
    {seed_opts} [List]: each element is an integer meaning a random initialization seed.
    {interpolation} [String or None]: Approach used for interpolating the target data in 
    the ML training process. If provided, it will be included in the output files' names.
    {savefigs_path} [String]: If specified, saves the training figures.
    {save_format} [String]: Choose the saving format, 'png' by default, don't include the dot.
    {exp_model_path} [String]: If specified, exports the ML model in .keras format.
    {results_path} [String]: Path for exporting results.
    {check_rep_model} [String or None]: If provided, use a .csv file as a reference, same format
    as the output of this function, to skip any already trained model.
    {quick_timing_test} [Boolean]: If True, ONLY runs a dry test (no savings of any type) for 5 epochs
    and estimate the total training times.

    --- Return ---

    {df_results} [Pandas dataframe]: information about the ML models' results.

    In addition, exports a .csv file containing the {df_results} information
    
    """

    # Load a list with already trained models, if any:
    trained_models = check_trained_model(ref_path=check_rep_model)

    # Initiate dataframe:
    df_results = pd.DataFrame(columns=standard_info_df_ML)

    # Prepare time windows:
    segms_train = [time_wdw.name for time_wdw in t_wdws_train] # List with training segments' names
    segms_test = [time_wdw.name for time_wdw in t_wdws_test] # List with testing segments' names
    for subset in [t_wdws_train,t_wdws_test]:
        for t_wdw in subset: 
            t_wdw.window_data()

    # Prepare all training options:
    train_options = list(itertools.product(
        *[B_comps_opts,arch_opts,RF_opts,seed_opts])
    )   

    # Train all models:

    N_models = 0 # Initiatie auxiliar count
    for (B_comps,arch,RF,seed) in train_options:

        # Prepare rotational frame and hyper-parameter options for ML model:
        full_hyp = arch.copy() # Initiate Full Hyper-parameters dictionary
        full_hyp["Magnetic_Components"] = B_comps
        full_hyp["Train_Segms"] = "".join(segms_train)
        full_hyp["Test_Segms"] = "".join(segms_test)
        full_hyp["Full_Name"] = arch["Model_Name"]+'_'+'_'.join([b for b in B_comps])
        full_hyp["Full_Name"] += f'_{RF[2]}_seed{seed}_{"".join(segms_train)}'

        # Train and test the model (if not trained yet):
        if full_hyp["Full_Name"] not in trained_models:
            N_models += 1 # Update number of trainable models            

            if not quick_timing_test:

                # Train the model:
                keras.backend.clear_session()
                ML_model = ML_Model(full_hyp,seed=seed)
                res_train = ML_model.train_model(
                    t_wdws_train,RF,savefigs_path=savefigs_path,exp_model_path=exp_model_path)

                # Test the model:
                res_test = ML_model.test_model(
                    t_wdws_test,RF,savefigs_path=savefigs_path)

                # Update results:
                results = dict(list(res_train.items())+list(res_test.items()))
                df_results.loc[len(df_results)] = results

                # Export .csv file (overwrites in everystep):
                if interpolation is None:
                    name = f'ML_S1_{len(B_comps_opts)}magcomps_{len(seed_opts)}seeds'
                else:
                    name = f'ML_S1_{interpolation}_{len(B_comps_opts)}magcomps_{len(seed_opts)}seeds'
                name += f'_{len(RF_opts)}RFs_wdw{t_wdws_train[0].pp}pp.csv'
                df_results.to_csv(results_path+name, index=False)

    # If prompted, make an estimation of the total training time based on a quick test (full run skipped):
    if quick_timing_test:
        quick_training(full_hyp,t_wdws_train,t_wdws_test,RF,
            N_models=N_models,total_epochs=full_hyp['Epochs'],seed=seed)
            
    return df_results

# ============================================================================

def train_stage2(
    pp_opts,
    arch_opts,
    RF_opts,
    t_wdws_train,
    t_wdws_test,
    seed_opts=[0],
    interpolation=None,
    savefigs_path=None,
    exp_model_path=None,
    results_path='./',
    check_rep_model=None,
    quick_timing_test=False
    ):
    """
    Training procedure for Stage 2, focusing on time windows' length. It trains many
    ML models using different options for magnetic components, architectures, seeds
    and rotational frames. It returns and exports a dataframe with the results.

    --- Inputs ---

    {pp_opts} [List]: Each element is an integer meaning the number of points for each
    time window.
    {arch_opts} [List]: Each element is a dictionary with the ML model hyper-parameters.
    {RF_opts} [List]: Each element is a list with the format [Numpy array,Numeric,String];
    meaning [rotation axis in (x,y,z) format, rotation angle in degree, RF name].
    If RF[2]='RF1', then the data is kept in the original frame.
    {t_wdws_train} [List]: Each element is a <Time_Wdw> object that will be used for
    the training process.
    {t_wdws_test} [List]: Each element is a <Time_Wdw> object that will be used for
    the testing process.
    {seed_opts} [List]: each element is an integer meaning a random initialization seed.
    {interpolation} [String or None]: Approach used for interpolating the target data in 
    the ML training process. If provided, it will be included in the output files' names.
    {savefigs_path} [String]: If specified, saves the training figures.
    {save_format} [String]: Choose the saving format, 'png' by default, don't include the dot.
    {exp_model_path} [String]: If specified, exports the ML model in .keras format.
    {results_path} [String]: Path for exporting results.
    {check_rep_model} [String or None]: If provided, use a .csv file as a reference, same format
    as the output of this function, to skip any already trained model.
    {quick_timing_test} [Boolean]: If True, ONLY runs a dry test (no savings of any type) for 5 epochs
    and estimate the total training times.
    
    --- Return ---

    {df_results} [Pandas dataframe]: information about the ML models' results.

    In addition, exports a .csv file containing the {df_results} information
    
    """

    # Load a list with already trained models, if any:
    trained_models = check_trained_model(ref_path=check_rep_model)

    # Initiate dataframe:
    df_results = pd.DataFrame(columns=standard_info_df_ML)

    # First, get the segments' names (time windows):
    segms_train = [time_wdw.name for time_wdw in t_wdws_train] # List with training segments' names
    segms_test = [time_wdw.name for time_wdw in t_wdws_test] # List with testing segments' names

    # Prepare all training options:
    train_options = list(itertools.product(
        *[pp_opts,arch_opts,RF_opts,seed_opts])
    )   

    # Train all models:

    N_models = 0 # Initiatie auxiliar count
    for (pp,arch,RF,seed) in train_options:
        # Check if {arch} is not appropriate for {pp}: 
        if arch["Convolutional_Network"]:
            if np.sum([filt_kern[1] for filt_kern in arch["Conv_Layers"]])>pp:
                continue # Skip this training 

        # Prepare rotational frame and hyper-parameter options for ML model:
        full_hyp = arch.copy() # Initiate Full Hyper-parameters dictionary
        full_hyp["Time_Window_pp"] = pp
        full_hyp["Train_Segms"] = "".join(segms_train)
        full_hyp["Test_Segms"] = "".join(segms_test)
        full_hyp["Full_Name"] = arch["Model_Name"]+f'_{pp}pp'
        full_hyp["Full_Name"] += f'_{RF[2]}_seed{seed}_{"".join(segms_train)}'
        B_comps = ''.join([b for b in full_hyp["Magnetic_Components"]])

        # Train and test the model (if not trained yet):
        if full_hyp["Full_Name"] not in trained_models:
            N_models += 1 # Update number of trainable models
            
            if not quick_timing_test:

                # Prepare Time windows:
                for t_wdw in t_wdws_train:
                    t_wdw.pp = pp # Define time window's points
                    t_wdw.window_data() # Window data
                for t_wdw in t_wdws_test:
                    t_wdw.pp = pp # Define time window's points
                    t_wdw.window_data() # Window data

                # Train the model:
                keras.backend.clear_session()
                ML_model = ML_Model(full_hyp,seed=seed)
                res_train = ML_model.train_model(
                    t_wdws_train,RF,savefigs_path=savefigs_path,exp_model_path=exp_model_path)

                # Test the model:
                res_test = ML_model.test_model(
                    t_wdws_test,RF,savefigs_path=savefigs_path)

                # Update results:
                results = dict(list(res_train.items())+list(res_test.items()))
                df_results.loc[len(df_results)] = results

                # Export .csv file (overwrites in everystep):  
                if interpolation is None:
                    name = f'ML_S2_{len(pp_opts)}ppOptions_{len(seed_opts)}seeds'
                else:
                    name = f'ML_S2_{interpolation}_{len(pp_opts)}ppOptions_{len(seed_opts)}seeds'
                name += f'_{len(RF_opts)}RFs_comps{B_comps}.csv'
                df_results.to_csv(results_path+name, index=False)

    # If prompted, make an estimation of the total training time based on a quick test (full run skipped):
    if quick_timing_test:
        # Prepare Time windows:
        for t_wdw in t_wdws_train:
            t_wdw.pp = pp # Define time window's points
            t_wdw.window_data() # Window data
        for t_wdw in t_wdws_test:
            t_wdw.pp = pp # Define time window's points
            t_wdw.window_data() # Window data
        # Perform quick test:
        quick_training(full_hyp,t_wdws_train,t_wdws_test,RF,
            N_models=N_models,total_epochs=full_hyp['Epochs'],seed=seed)
            
    return df_results

# ============================================================================

def train_stage3(
    Conv_opts,
    pool_opts,
    conversion1D_opts,
    Dense_opts,
    gen_hyps,
    RF_opts,
    t_wdws_train,
    t_wdws_test,
    seed_opts=[0],
    interpolation=None,
    savefigs_path=None,
    exp_model_path=None,
    results_path='./',
    check_rep_model=None,
    quick_timing_test=False
    ):
    """
    Training procedure for Stage 3, focusing on main ML architectures. It trains many
    ML models using different options for ML architectures, seeds and rotational frames.
    It returns and exports a dataframe with the results.

    Note: all ML architecture will include several blocks, in this order:
    1. Convolutional block, including Convolutional Neural Networks (NN) with arbitrary
    number of filters and kernel sizes, and optionally pooling layers.
    2. 1D-conversion layer, which flattens the previous input, either by 'Average' or 'Global'
    functions.
    3. Dense block, including Dense NN with arbitrary number of neurons.

    --- Inputs ---

    {Conv_opts} [List]: Each element is an option for the Convolutional block, which must have
    the format of a list containing as many elements as CNN, each single element with the
    format [filters,kernel].
    {pool_opts} [List]: Each element is an option for the pooling size following each CNN.
    The values can be a single integer or None, in which case no pooling is made.
    {conversion1D_opts} [List]: Each element is an option for the 1D-conversion layer. The values
    must be either 'Flatten' (flattening average) or 'Glob' (global average).
    {Dense_opts} [List]: Each element is an option for the Dense block, which must have
    the format of a list containing as many elements as DNN, each single element an integer
    that represents the number of neurons.
    {gen_hyps} [Dictionary]: Information for hyperparameters.
    {RF_opts} [List]: Each element is a list with the format [Numpy array,Numeric,String];
    meaning [rotation axis in (x,y,z) format, rotation angle in degree, RF name].
    If RF[2]='RF1', then the data is kept in the original frame.
    {t_wdws_train} [List]: Each element is a <Time_Wdw> object that will be used for
    the training process.
    {t_wdws_test} [List]: Each element is a <Time_Wdw> object that will be used for
    the testing process.
    {seed_opts} [List]: each element is an integer meaning a random initialization seed.
    {interpolation} [String or None]: Approach used for interpolating the target data in 
    the ML training process. If provided, it will be included in the output files' names.
    {savefigs_path} [String]: If specified, saves the training figures.
    {save_format} [String]: Choose the saving format, 'png' by default, don't include the dot.
    {exp_model_path} [String]: If specified, exports the ML model in .keras format.
    {results_path} [String]: Path for exporting results.
    {check_rep_model} [String or None]: If provided, use a .csv file as a reference, same format
    as the output of this function, to skip any already trained model.
    {quick_timing_test} [Boolean]: If True, ONLY runs a dry test (no savings of any type) for 5 epochs
    and estimate the total training times.
    
    --- Return ---

    {df_results} [Pandas dataframe]: information about the ML models' results.

    In addition, exports a .csv file containing the {df_results} information
    
    """

    # Load a list with already trained models, if any:
    trained_models = check_trained_model(ref_path=check_rep_model)

    # Initiate dataframe:
    df_results = pd.DataFrame(columns=standard_info_df_ML)

    # Prepare time windows:
    segms_train = [time_wdw.name for time_wdw in t_wdws_train] # List with training segments' names
    segms_test = [time_wdw.name for time_wdw in t_wdws_test] # List with testing segments' names
    for subset in [t_wdws_train,t_wdws_test]:
        for t_wdw in subset: 
            t_wdw.window_data()

    # Generate all possible combinations for ML architecture:
    arch_opts = []
    # Convolutional options:
    for C in Conv_opts:
        arch = {} # Initiate dictionary
        arch_conv = {"Conv_Layers": C}
        name_C = '_'.join(['C'+str(filter_kernel[1]) for filter_kernel in C]) # Name segment
        # Pooling options:
        for P in pool_opts:
            name_pool = 'NP' if P is None else f'P{P}' # Name
            arch_pool = {"Pool_Layers": [P for _ in C]} # Pooling layers
            # 1D-conversion options:
            for conver in conversion1D_opts:
                flat = True if conver=='Flatten' else False
                arch_1D = {"Flatten_Average": flat}
                for D in Dense_opts:
                    arch_dense = {"Dens_Layers": D}
                    name_D = '_'.join(['D'+str(n) for n in D]) # Name segment
                    # Combine options and update the list:
                    full_name = f'S3_{name_C}_{name_pool}_{conver}_{name_D}'
                    arch_opts.append( # NP option
                        dict(list(gen_hyps.items())+
                             list(arch_conv.items())+
                             list(arch_pool.items())+
                             list(arch_1D.items())+
                             list(arch_dense.items())+
                             list({"Model_Name": full_name}.items())
                             ))

    # Prepare all training options:
    train_options = list(itertools.product(
        *[arch_opts,RF_opts,seed_opts])
    )   

    # Train all models:

    N_models = 0 # Initiatie auxiliar count
    for (arch,RF,seed) in train_options:

        # Prepare rotational frame and hyper-parameter options for ML model:
        full_hyp = arch.copy() # Initiate Full Hyper-parameters dictionary
        full_hyp["Train_Segms"] = "".join(segms_train)
        full_hyp["Test_Segms"] = "".join(segms_test)
        full_hyp["Full_Name"] = arch["Model_Name"]+'_'.join(
            [b for b in full_hyp["Magnetic_Components"]])
        full_hyp["Full_Name"] += f'_{RF[2]}_seed{seed}_{"".join(segms_train)}'

        # Train and test the model (if not trained yet):
        if full_hyp["Full_Name"] not in trained_models:
            N_models += 1 # Update number of trainable models

            if not quick_timing_test:

                # Train the model:
                keras.backend.clear_session()
                ML_model = ML_Model(full_hyp,seed=seed)
                res_train = ML_model.train_model(
                    t_wdws_train,RF,savefigs_path=savefigs_path,exp_model_path=exp_model_path)

                # Test the model:
                res_test = ML_model.test_model(
                    t_wdws_test,RF,savefigs_path=savefigs_path)

                # Update results:
                results = dict(list(res_train.items())+list(res_test.items()))
                df_results.loc[len(df_results)] = results

                # Export .csv file (overwrites in everystep):
                if interpolation is None:
                    name = f'ML_S3_{len(arch_opts)}archs_{len(seed_opts)}seeds_{len(RF_opts)}RFs.csv'
                else:
                    name = f'ML_S3_{interpolation}_{len(arch_opts)}archs_{len(seed_opts)}seeds_{len(RF_opts)}RFs.csv'
                df_results.to_csv(results_path+name, index=False)

    # If prompted, make an estimation of the total training time based on a quick test (full run skipped):
    if quick_timing_test:
        quick_training(full_hyp,t_wdws_train,t_wdws_test,RF,
            N_models=N_models,total_epochs=full_hyp['Epochs'],seed=seed)
            
    return df_results

# ============================================================================

def train_stage4(
    activ_opts,
    optim_opts,
    lr_opts,
    gen_hyps,
    RF_opts,
    t_wdws_train,
    t_wdws_test,
    seed_opts=[0],
    interpolation=None,
    savefigs_path=None,
    exp_model_path=None,
    results_path='./',
    check_rep_model=None,
    quick_timing_test=False
    ):
    """
    Training procedure for Stage 4, focusing on global hyperparameters. It trains many
    ML models using different options for seeds and rotational frames. It returns and
    exports a dataframe with the results.

    --- Inputs ---

    {activ_opts} [List]: Each element is a string or keras object representing an activation
    function, which will be used in all layers except the last one.
    {optim_opts} [List]: Each element is a string or keras object representing the global
    optimizer.
    {lr_opts} [List]: Each element is a float number representing the learning rate.
    {gen_hyps} [Dictionary]: Information for hyperparameters.
    {RF_opts} [List]: Each element is a list with the format [Numpy array,Numeric,String];
    meaning [rotation axis in (x,y,z) format, rotation angle in degree, RF name].
    If RF[2]='RF1', then the data is kept in the original frame.
    {t_wdws_train} [List]: Each element is a <Time_Wdw> object that will be used for
    the training process.
    {t_wdws_test} [List]: Each element is a <Time_Wdw> object that will be used for
    the testing process.
    {seed_opts} [List]: each element is an integer meaning a random initialization seed.
    {interpolation} [String or None]: Approach used for interpolating the target data in 
    the ML training process. If provided, it will be included in the output files' names.
    {savefigs_path} [String]: If specified, saves the training figures.
    {save_format} [String]: Choose the saving format, 'png' by default, don't include the dot.
    {exp_model_path} [String]: If specified, exports the ML model in .keras format.
    {results_path} [String]: Path for exporting results.
    {check_rep_model} [String or None]: If provided, use a .csv file as a reference, same format
    as the output of this function, to skip any already trained model.
    {quick_timing_test} [Boolean]: If True, ONLY runs a dry test (no savings of any type) for 5 epochs
    and estimate the total training times.
    
    --- Return ---

    {df_results} [Pandas dataframe]: information about the ML models' results.

    In addition, exports a .csv file containing the {df_results} information
    
    """

    # Load a list with already trained models, if any:
    trained_models = check_trained_model(ref_path=check_rep_model)

    # Initiate dataframe:
    df_results = pd.DataFrame(columns=standard_info_df_ML)

    # Prepare time windows:
    segms_train = [time_wdw.name for time_wdw in t_wdws_train] # List with training segments' names
    segms_test = [time_wdw.name for time_wdw in t_wdws_test] # List with testing segments' names
    for subset in [t_wdws_train,t_wdws_test]:
        for t_wdw in subset: 
            t_wdw.window_data()

    # Prepare all training options:
    train_options = list(itertools.product(
        *[activ_opts,optim_opts,lr_opts,RF_opts,seed_opts])
    )   

    # Train all models:

    N_models = 0 # Initiatie auxiliar count
    for (activ,optim,lr,RF,seed) in train_options:

        # Prepare hyper-parameter options for model:
        full_hyp = gen_hyps.copy() # Initiate Full Hyper-parameters dictionary
        full_hyp["Activation_Function"] = activ
        full_hyp["Optimizer"] = optim
        full_hyp["Learning_Rate"] = lr
        full_hyp["Train_Segms"] = "".join(segms_train)
        full_hyp["Test_Segms"] = "".join(segms_test)
        fullname = f"activ{activ}_optim{optim}_lr{lr}_{RF[2]}_seed{seed}"
        fullname += f'_{"".join(segms_train)}'
        full_hyp["Full_Name"] = fullname

        # Train and test the model (if not trained yet):
        if full_hyp["Full_Name"] not in trained_models:
            N_models += 1 # Update number of trainable models

            if not quick_timing_test:

                # Train the model:
                keras.backend.clear_session()
                ML_model = ML_Model(full_hyp,seed=seed)
                res_train = ML_model.train_model(
                    t_wdws_train,RF,savefigs_path=savefigs_path,exp_model_path=exp_model_path)

                # Test the model:
                res_test = ML_model.test_model(
                    t_wdws_test,RF,savefigs_path=savefigs_path)

                # Update results:
                results = dict(list(res_train.items())+list(res_test.items()))
                df_results.loc[len(df_results)] = results

                # Export .csv file (overwrites in everystep):
                if interpolation is None:
                    name = f'ML_S4_{len(seed_opts)}seeds_{len(RF_opts)}RFs.csv'
                else:
                    name = f'ML_S4_{interpolation}_{len(seed_opts)}seeds_{len(RF_opts)}RFs.csv'
                df_results.to_csv(results_path+name, index=False)

    # If prompted, make an estimation of the total training time based on a quick test (full run skipped):
    if quick_timing_test:
        quick_training(full_hyp,t_wdws_train,t_wdws_test,RF,
            N_models=N_models,total_epochs=full_hyp['Epochs'],seed=seed)
            
    return df_results


# ============================================================================

def train_stage5(
    filter_opts,
    drop_opts,
    gen_hyps,
    RF_opts,
    t_wdws_train,
    t_wdws_test,
    seed_opts=[0],
    interpolation=None,
    savefigs_path=None,
    exp_model_path=None,
    results_path='./',
    check_rep_model=None,
    quick_timing_test=False
    ):
    """
    Training procedure for Stage 5, focusing on the fine architecture. It trains many
    ML models using different options for seeds and rotational frames. It returns and
    exports a dataframe with the results.

    --- Inputs ---

    {filter_opts} [List]: Each element is an option for the Convolutional block, which must have
    the format of a list containing as many elements as CNN, each single element with the
    format [filters,kernel].
    {drop_opts} [List]: Each element is a float number representing the dropout fraction in all
    Dense layers.
    {gen_hyps} [Dictionary]: Information for hyperparameters.
    {RF_opts} [List]: Each element is a list with the format [Numpy array,Numeric,String];
    meaning [rotation axis in (x,y,z) format, rotation angle in degree, RF name].
    If RF[2]='RF1', then the data is kept in the original frame.
    {t_wdws_train} [List]: Each element is a <Time_Wdw> object that will be used for
    the training process.
    {t_wdws_test} [List]: Each element is a <Time_Wdw> object that will be used for
    the testing process.
    {seed_opts} [List]: each element is an integer meaning a random initialization seed.
    {interpolation} [String or None]: Approach used for interpolating the target data in 
    the ML training process. If provided, it will be included in the output files' names.
    {savefigs_path} [String]: If specified, saves the training figures.
    {save_format} [String]: Choose the saving format, 'png' by default, don't include the dot.
    {exp_model_path} [String]: If specified, exports the ML model in .keras format.
    {results_path} [String]: Path for exporting results.
    {check_rep_model} [String or None]: If provided, use a .csv file as a reference, same format
    as the output of this function, to skip any already trained model.
    {quick_timing_test} [Boolean]: If True, ONLY runs a dry test (no savings of any type) for 5 epochs
    and estimate the total training times.
    
    --- Return ---

    {df_results} [Pandas dataframe]: information about the ML models' results.

    In addition, exports a .csv file containing the {df_results} information
    
    """

    # Load a list with already trained models, if any:
    trained_models = check_trained_model(ref_path=check_rep_model)

    # Initiate dataframe:
    df_results = pd.DataFrame(columns=standard_info_df_ML)

    # Prepare time windows:
    segms_train = [time_wdw.name for time_wdw in t_wdws_train] # List with training segments' names
    segms_test = [time_wdw.name for time_wdw in t_wdws_test] # List with testing segments' names
    for subset in [t_wdws_train,t_wdws_test]:
        for t_wdw in subset: 
            t_wdw.window_data()

    # Prepare all training options:
    train_options = list(itertools.product(
        *[filter_opts,drop_opts,RF_opts,seed_opts])
    )   

    # Train all models:

    N_models = 0 # Initiatie auxiliar count
    for (filters,drop,RF,seed) in train_options:

        # Prepare hyper-parameter options for model:
        full_hyp = gen_hyps.copy() # Initiate Full Hyper-parameters dictionary
        full_hyp["Conv_Layers"] = filters
        full_hyp["Dropout_Fraction"] = drop
        full_hyp["Train_Segms"] = "".join(segms_train)
        full_hyp["Test_Segms"] = "".join(segms_test)
        str_filters = '_'.join([str(f[0]) for f in filters])
        fullname = f"Filters{str_filters}_dropout{drop}"
        fullname += f"_{RF[2]}_seed{seed}_{''.join(segms_train)}"
        full_hyp["Full_Name"] = fullname

        # Train and test the model (if not trained yet):
        if full_hyp["Full_Name"] not in trained_models:
            N_models += 1 # Update number of trainable models

            if not quick_timing_test:

                # Train the model:
                keras.backend.clear_session()
                ML_model = ML_Model(full_hyp,seed=seed)
                res_train = ML_model.train_model(
                    t_wdws_train,RF,savefigs_path=savefigs_path,exp_model_path=exp_model_path)

                # Test the model:
                res_test = ML_model.test_model(
                    t_wdws_test,RF,savefigs_path=savefigs_path)

                # Update results:
                results = dict(list(res_train.items())+list(res_test.items()))
                df_results.loc[len(df_results)] = results

                # Export .csv file (overwrites in everystep):
                if interpolation is None:
                    name = f'ML_S5_{len(seed_opts)}seeds_{len(RF_opts)}RFs.csv'
                else:
                    name = f'ML_S5_{interpolation}_{len(seed_opts)}seeds_{len(RF_opts)}RFs.csv'
                df_results.to_csv(results_path+name, index=False)

    # If prompted, make an estimation of the total training time based on a quick test (full run skipped):
    if quick_timing_test:
        quick_training(full_hyp,t_wdws_train,t_wdws_test,RF,
            N_models=N_models,total_epochs=full_hyp['Epochs'],seed=seed)
            
    return df_results

# ============================================================================

def train_stage6(
    p_train_opts,
    N_augm_opts,
    noise_opts,
    gen_hyps,
    RF_opts,
    t_wdws_train,
    t_wdws_test,
    seed_opts=[0],
    interpolation=None,
    savefigs_path=None,
    exp_model_path=None,
    results_path='./',
    check_rep_model=None,
    quick_timing_test=False
    ):
    """
    Training procedure for Stage 6, focusing on data augmentation and the size of the original
    training dataset. It trains many ML models using different options for seeds and rotational
    frames. It returns and exports a dataframe with the results.

    --- Inputs ---

    {p_train_opts} [List]: Each element is a float number representing the fraction of the
    original training dataset that will be used.
    {N_augm_opts} [List]: Each element is an integer number representing the number of times
    that the original training dataset will be augmented (0 means no augmentation).
    {noise_opts} [List]: Each element is a list with three float numbers in the format [bx,by,bz],
    representing the average noise intensity for the Bx,By,Bz magnetic fields when augmenting
    the original dataset. Units are [nT]. For N_augm_opts=[0], this input doesn't matter.
    {gen_hyps} [Dictionary]: Information for hyperparameters.
    {RF_opts} [List]: Each element is a list with the format [Numpy array,Numeric,String];
    meaning [rotation axis in (x,y,z) format, rotation angle in degree, RF name].
    If RF[2]='RF1', then the data is kept in the original frame.
    {t_wdws_train} [List]: Each element is a <Time_Wdw> object that will be used for
    the training process.
    {t_wdws_test} [List]: Each element is a <Time_Wdw> object that will be used for
    the testing process.
    {seed_opts} [List]: each element is an integer meaning a random initialization seed.
    {interpolation} [String or None]: Approach used for interpolating the target data in 
    the ML training process. If provided, it will be included in the output files' names.
    {savefigs_path} [String]: If specified, saves the training figures.
    {save_format} [String]: Choose the saving format, 'png' by default, don't include the dot.
    {exp_model_path} [String]: If specified, exports the ML model in .keras format.
    {results_path} [String]: Path for exporting results.
    {check_rep_model} [String or None]: If provided, use a .csv file as a reference, same format
    as the output of this function, to skip any already trained model.
    {quick_timing_test} [Boolean]: If True, ONLY runs a dry test (no savings of any type) for 5 epochs
    and estimate the total training times.
    
    --- Return ---

    {df_results} [Pandas dataframe]: information about the ML models' results.

    In addition, exports a .csv file containing the {df_results} information
    
    """

    # Load a list with already trained models, if any:
    trained_models = check_trained_model(ref_path=check_rep_model)

    # Initiate dataframe:
    df_results = pd.DataFrame(columns=standard_info_df_ML+augmented_info_df_ML)

    # Prepare time windows:
    segms_train = [time_wdw.name for time_wdw in t_wdws_train] # List with training segments' names
    segms_test = [time_wdw.name for time_wdw in t_wdws_test] # List with testing segments' names
    # For now, only window the test dataset, the training dataset will vary depending on the augmentation:
    for t_wdw in t_wdws_test: 
        t_wdw.window_data()

    # Prepare all training options:
    train_options = list(itertools.product(
        *[p_train_opts,N_augm_opts,noise_opts,RF_opts,seed_opts])
    )   

    # Train all models:

    N_models = 0 # Initiatie auxiliar count
    for (p_train,N_augm,noise,RF,seed) in train_options:

        # Prepare hyper-parameter options for model:
        full_hyp = gen_hyps.copy() # Initiate Full Hyper-parameters dictionary
        full_hyp["Train_Segms"] = "".join(segms_train)
        full_hyp["Test_Segms"] = "".join(segms_test)
        fullname = f"ptrain{p_train}_Naugm{N_augm}_"
        fullname += f"noiseBx{noise[0]}_By{noise[1]}_Bz{noise[2]}_nT"
        fullname += f"_{RF[2]}_seed{seed}_{''.join(segms_train)}"
        full_hyp["Full_Name"] = fullname

        # Train and test the model (if not trained yet):
        if full_hyp["Full_Name"] not in trained_models:
            N_models += 1 # Update number of trainable models

            if not quick_timing_test:

                # Reduce original training dataset:
                t_wdws_train_reduced = [
                ML_twdw.copy_and_reduce_TimeWdw(time_wdw,p_train) for time_wdw in t_wdws_train]        
                # Augmentate and window training dataset:
                for t_wdw in t_wdws_train_reduced:
                    t_wdw.window_data(N_augm=N_augm,
                        Bx_noise=noise[0], By_noise=noise[1], Bz_noise=noise[2])
                                
                # Train the model:
                keras.backend.clear_session()
                ML_model = ML_Model(full_hyp,seed=seed)
                res_train = ML_model.train_model(
                    t_wdws_train_reduced,RF,savefigs_path=savefigs_path,exp_model_path=exp_model_path)

                # Test the model:
                res_test = ML_model.test_model(
                    t_wdws_test,RF,savefigs_path=savefigs_path)

                # Update results:
                results = dict(list(res_train.items())+list(res_test.items()))
                df_results.loc[len(df_results)] = results

                # Export .csv file (overwrites in everystep):
                if interpolation is None:
                    name = f'ML_S6_{len(seed_opts)}seeds_{len(RF_opts)}RFs.csv'
                else:
                    name = f'ML_S6_{interpolation}_{len(seed_opts)}seeds_{len(RF_opts)}RFs.csv'
                df_results.to_csv(results_path+name, index=False)

    # If prompted, make an estimation of the total training time based on a quick test (full run skipped):
    if quick_timing_test:
        # Reduce original training dataset:
        t_wdws_train_reduced = [
        ML_twdw.copy_and_reduce_TimeWdw(time_wdw,p_train) for time_wdw in t_wdws_train]        
        # Augmentate and window training dataset:
        for t_wdw in t_wdws_train_reduced:
            t_wdw.window_data(N_augm=N_augm,
                Bx_noise=noise[0], By_noise=noise[1], Bz_noise=noise[2])
        # Perform test:
        quick_training(full_hyp,t_wdws_train_reduced,t_wdws_test,RF,
            N_models=N_models,total_epochs=full_hyp['Epochs'],seed=seed)
            
    return df_results

# ============================================================================

def train_stage7(
    full_hyps,
    t_wdws_train,
    t_wdws_test,
    RF_opts,
    seed_opts=[0],
    interpolation=None,
    savefigs_path=None,
    exp_model_path=None,
    results_path='./',
    check_rep_model=None,
    quick_timing_test=False
    ):
    """
    Train a single model in many rotational frames, export the results and predictions.

    --- Inputs ---

    {full_hyps} [Dictionary]: Hyperparameters information for the ML model.
    If RF[2]='RF1', then the data is kept in the original frame.
    {t_wdws_train} [List]: Each element is a <Time_Wdw> object that will be used for
    the training process.
    {t_wdws_test} [List]: Each element is a <Time_Wdw> object that will be used for
    the testing process.
    {RF_opts} [List]: Each element is a list with the format [Numpy array,Numeric,String];
    meaning [rotation axis in (x,y,z) format, rotation angle in degree, RF name].
    {seed_opts} [List]: each element is an integer meaning a random initialization seed.
    {interpolation} [String or None]: Approach used for interpolating the target data in 
    the ML training process. If provided, it will be included in the output files' names.
    {savefigs_path} [String]: If specified, saves the training figures.
    {save_format} [String]: Choose the saving format, 'png' by default, don't include the dot.
    {exp_model_path} [String]: If specified, exports the ML model in .keras format.
    {results_path} [String]: Path for exporting results.
    {check_rep_model} [String or None]: If provided, use a .csv file as a reference, same format
    as the output of this function, to skip any already trained model.
    {quick_timing_test} [Boolean]: If True, ONLY runs a dry test (no savings of any type) for 5 epochs
    and estimate the total training times.
    
    --- Return ---

    {df_results} [Pandas dataframe]: information about the ML models' results.

    In addition, exports:
    - General .csv file containing the {df_results} information,
    - For each ML model trained in a single rotational frame, one .csv file containing
    the time, predictions and ground truth Z-positions for each time.

    """

    # Load a list with already trained models, if any:
    trained_models = check_trained_model(ref_path=check_rep_model)

    # Initiate dataframe:
    df_results = pd.DataFrame(columns=standard_info_df_ML+augmented_info_df_ML)

    # Prepare time windows:
    segms_train = [time_wdw.name for time_wdw in t_wdws_train] # List with training segments' names
    segms_test = [time_wdw.name for time_wdw in t_wdws_test] # List with testing segments' names
    for subset in [t_wdws_train,t_wdws_test]:
        for t_wdw in subset: 
            t_wdw.window_data()

    # Update ML model information:
    ML_name = f'MLmodel_{full_hyps["Model_Name"]}' # Model name
    full_hyps["Train_Segms"] = "".join(segms_train) # Add training segments information
    full_hyps["Test_Segms"] = "".join(segms_test) # Add testing segments information
    ML_name += f'_{full_hyps["Train_Segms"]}' # Update name

    # Reduce training dataset (if requested):
    if full_hyps["p_train"] < 1:
        t_wdws_train_reduced = [copy_and_reduce_TimeWdw(time_wdw,p_train) for time_wdw in t_wdws_train]
        ML_name += f'_ptrain{full_hyps["p_train"]}' # Update name
    else: 
        t_wdws_train_reduced = t_wdws_train

    # Window training data, with data augmentation if requested:
    for t_wdw in t_wdws_train_reduced:
        t_wdw.window_data(N_augm=full_hyps["N_augm"],Bx_noise=full_hyps["noise"][0], 
                          By_noise=full_hyps["noise"][1], Bz_noise=full_hyps["noise"][2])
    # Update ML model's information: 
    ML_name += f'_Naugm{full_hyps["N_augm"]}' # Update name
    if full_hyps["N_augm"] > 0:
        ML_name += f'_noise_{full_hyps["noise"][0]}_' # Update name
        ML_name += f'{full_hyps["noise"][1]}_{full_hyps["noise"][2]}_nT' # Update name    

    # Window testing data:
    for t_wdw in t_wdws_test: 
        t_wdw.window_data()    

    # Prepare all training options:
    train_options = list(itertools.product(
        *[RF_opts,seed_opts])
    )

    # Train all models:

    N_models = 0 # Initiatie auxiliar count
    predictions = {} # Initiate
    for (RF,seed) in train_options:
        hyps = full_hyps.copy() # Initiate Full Hyper-parameters dictionary
        hyps["Full_Name"] = ML_name + f"_{RF[2]}_seed{seed}_{''.join(segms_train)}" # Add full name for the model
        if hyps["Full_Name"] not in trained_models:
            N_models += 1 # Update number of trained models
            if not quick_timing_test:

                # Train the model:
                keras.backend.clear_session()
                ML_model = ML_Model(hyps,seed=seed)
                res_train = ML_model.train_model(
                    t_wdws_train_reduced,RF,savefigs_path=savefigs_path)

                # Test the model and obtain ground truth and predictions, in [m]:
                res_test, t, true_z, pred_z = ML_model.test_model(
                    t_wdws_test,RF,return_preds=True,savefigs_path=savefigs_path)        
                predictions[RF[2]] = [t, true_z, pred_z] # Update dictionary for current RF; [s],[m],[m]

                # Update general results:
                results = dict(list(res_train.items())+list(res_test.items())) # Compile results
                df_results.loc[len(df_results)] = results # Add new row in dataframe

                # Export .csv file (overwrites in everystep):
                if interpolation is None:
                    name = f'ML_S7_{ML_name}_{hyps["Epochs"]}epochs.csv'
                else:
                    name = f'ML_S7_{interpolation}_{ML_name}_{hyps["Epochs"]}epochs.csv'
                df_results.to_csv(results_path+name, index=False)

                # Export predictions file:
                interp_info = f'_interp_{interpolation}_' if interpolation is not None else '_' # Interpolation info
                for i in range(len(t)):
                    np.savetxt(
                        results_path+f"Preds{interp_info}{ML_name}_{RF[2]}_seed{seed}_test{i}.csv",
                        np.transpose(np.array([t[i],true_z[i], pred_z[i]])),
                        delimiter=",",header='time[s]_trueZ[m]_predZ[m]',fmt='%.7e')

    # If prompted, make an estimation of the total training time based on a quick test (full run skipped):
    if quick_timing_test:
        quick_training(hyps,t_wdws_train,t_wdws_test,RF,
            N_models=N_models,total_epochs=hyps['Epochs'],seed=seed)

    return df_results