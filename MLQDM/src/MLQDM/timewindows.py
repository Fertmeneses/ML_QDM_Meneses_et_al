"""

Process magnetic field and Z-position timeseries into time windows.

# ===========

Main functions:

    - prepare_time_windows
    - summarize_TW_segments
    - rotate_3D
    - copy_and_reduce_TimeWdw

# ===========

Functions included in Class <Time_Wdw>:

    - __init__
    - store_orig_data
    - window_data
    - rotate_frame
    - info
    - plot_instances
    - matrix_format

# ===========

"""

# ============================================================================

# Required packages:

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Internal packages:

import MLQDM.general as ML_general

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

class Time_Wdw:
    """
    The Time Window object <Time_Wdw> can process an input dataframe with information about time, magnetic fields,
    z-position labels and optionally the ground truth z-position labels, all given in the Laboratory rotational
    frame, defined as RF1. <Time_Wdw> can return numpy arrays representing time windows for the magnetic projections
    Bx, By and Bz, in the requested rotational frame. All time windows will be correlated to z-position labels, and
    eventually to the ground truth labels, if they were provided in the input dataframe.

    --- Attributes ---
    
    GENERAL:
    <self.name> [String]: Name for the <Time_Wdw> object.
    <self.pp> [Integer]: Number of points for each time window.
    <self.gr_tr> [Boolean]: Specifies whether the ground truth labels are included (True) or not (False).
    <self.rotFrame> [List]: Format [Numpy array,Numeric,String]; specifies the rotated frame that is currently 
    active for time windowing. The elements of the list are the following: [rotation axis in (x,y,z) format,rotation
    angle in degree units, RF name]. By default, the rotated frame is the original lab frame, then [None,None,'RF1'].
    <self.norm_value> [Float]: Normalizing value for future reference, in [nT] units.
    ORIGINAL DATA:
    <self.time> [Numpy array]: Time vector from the original data, before windowing. Units: s.
    <self.z> [Numpy array]: Z-position vector from the original data, before windowing. Units: m.
    <self.Bx_RF1> [Numpy array]: Bx vector from the original data in RF1, before windowing. Units: nT.
    <self.By_RF1> [Numpy array]: By vector from the original data in RF1, before windowing. Units: nT.
    <self.Bz_RF1> [Numpy array]: Bz vector from the original data in RF1, before windowing. Units: nT.
    <self.B_RF1> [Numpy array]: Calculated B (scalar) vector from the original data, before windowing. Units: nT.
    WINDOWED DATA:
    The following attributes are Numpy arrays containing the time windows' information. Each time window instance is
    stored as a row with self.pp points. Then, the matrix has a size (N x self.pp), where N is the number of time windows. 
    <self.time_wdw> [Numpy array]: Time matrix. Units: s.
    <self.Bx_wdw_RF1> [Numpy array]: Bx matrix in RF1. Units: nT.
    <self.By_wdw_RF1> [Numpy array]: By matrix in RF1. Units: nT.
    <self.Bz_wdw_RF1> [Numpy array]: Bz matrix in RF1. Units: nT.
    <self.B_wdw> [Numpy array]: B (scalar) matrix (rotational invariant). Units: nT.
    <self.Bx_wdw_RFX> [Numpy array] Bx matrix in the current rotated frame self.rotFrame, if any. Units: nT.
    <self.By_wdw_RFX> [Numpy array] By matrix in the current rotated frame self.rotFrame, if any. Units: nT.
    <self.Bz_wdw_RFX> [Numpy array] Bz matrix in the current rotated frame self.rotFrame, if any. Units: nT.
    <self.z_labels> [Numpy array]: Z-position vector from the original data, after windowing. Units: m.
    Note: windowed data can be augmented artificially, in which case the new augmented data will replace the previous
    attributes, but original data (not windowed) will remain the same.
    <self.augm_N> [Integer]: Specifies how many times the data was artificially augmented (if there
    is no augmentation, then augm_N=0).
    <self.augm_noise> [List]: Format [Bx_noise, By_noise, Bz_noise]; each element specifies the noise level
    which was used to augmentate the data, in [nT], for the [Bx,By,Bz] original data in RF1.
    """

    # =============

    def __init__(
        self,
        pp,
        name,
        gr_tr=False
        ):
        """
        Initiate Time_Wdw object with basic information.

        --- Inputs ---

        {pp}: Integer; number of points for each time window.
        {name}: String; name for the <Time_Wdw> object, it may contain the segment number.
        {gr_tr}: Boolean; specifies whether the <Time_Wdw> object is labeled according to
        the ground truth (True) or not (False, default).

        --- Return ---

        Create the Time_Wdw object.

        """
        self.name = name # Name
        self.pp = pp # Number of points for every time window
        self.gr_tr = gr_tr # Informs whether ground truth is available (True) or not (False)
        self.rotFrame = [None,None,'RF1'] # At the beginning, there is no rotated frame

    # =============

    def store_orig_data(
        self, 
        df
        ):
        """
        Receives a dataframe with the measurements data (t,Bx,By,Bz,z), eventually z_true as 
        well, in RF1. If <self.gr_tr>=True, then it will be assumed that the dataframe 
        includes the ground truth labels.

        --- Inputs ---        
        
        {df} [Pandas dataframe]: It must have the following columns: 'Time_s', 'Bx_nT', 'By_nT',
        'Bz_nT', 'zAut_m' and, optionally, 'zTrue_m'.

        --- Return ---

        Generate or update:

        <self.time>, <self.Bx_RF1>, <self.By_RF1>, <self.Bz_RF1>, <self.B_RF1>, <self.z>
        """

        # Prepare and store original data, except z: t, Bx, By, Bz, B (scalar)
        t, Bx, By, Bz = df['Time_s'], df['Bx_nT'], df['By_nT'], df['Bz_nT'] # [s,nT,nT,nT]
        self.time = t.to_numpy() # Time [s]
        self.Bx_RF1 = Bx.to_numpy() # Bx fields in RF1 [nT]
        self.By_RF1 = By.to_numpy() # By fields in RF1 [nT]
        self.Bz_RF1 = Bz.to_numpy() # Bz fields in RF1 [nT]
        self.B_RF1 = np.sqrt(self.Bx_RF1**2 + self.By_RF1**2 + self.Bz_RF1**2) # Scalar B field [nT]

        # Store z-position, using the ground truth when available:
        z = df['zTrue_m'] if self.gr_tr else df['zAut_m'] # [m]
        self.z = z.to_numpy() # Z-position [m]
    
    # =============

    def window_data(
        self, 
        N_augm=0, 
        Bx_noise=1, 
        By_noise=1, 
        Bz_noise=1
        ):
        """
        Generates time windows in the original dataframe RF1, with predictors
        (magnetic fields) and targets (z-pos) from the original data according to the 
        attributes <self.pp> and <self.dp>, with the possibility of augmentating the data
        using random noise.

        --- Inputs ---

        {N_augm} [Integer]: Specifies how many times the data will be augmented. If 0 (default),
        there will be no augmentation.
        The following parameters are only valid when {N_augm}>0.
        {Bx_noise} [Float]: Value for the standard deviation of the Bx noise in the random normal
        distribution, for data augmentation, units [nT].
        {By_noise} [Float]: Value for the standard deviation of the By noise in the random normal
        distribution, for data augmentation, units [nT].
        {Bz_noise} [Float]: Value for the standard deviation of the Bz noise in the random normal
        distribution, for data augmentation, units [nT].

        WARNING: This function will only work if the original data was previously stored.

        --- Return ---

        Generate or update:

        <self.time_wdw>, <self.Bx_wdw_RF1>, <self.By_wdw_RF1>, <self.Bz_wdw_RF1>, <self.B_wdw>, 
        <self.z_labels>, <self.augm_N>, <self.augm_noise>.
        """

        # Store the [time, Bx, By, Bz, B, z] time windows and labels in RF1:
        N = len(self.z)-self.pp # Total number of time windows
        self.time_wdw = np.array([np.array(self.time[i:i+self.pp]) for i in range(N)]) # Store times [s]
        self.Bx_wdw_RF1 = np.array([self.Bx_RF1[i:i+self.pp] for i in range(N)]) # Store Bx fields windows in RF1 [nT]
        self.By_wdw_RF1 = np.array([self.By_RF1[i:i+self.pp] for i in range(N)]) # Store By fields windows in RF1 [nT]
        self.Bz_wdw_RF1 = np.array([self.Bz_RF1[i:i+self.pp] for i in range(N)]) # Store Bz fields windows in RF1 [nT]
        self.B_wdw = np.array([self.B_RF1[i:i+self.pp] for i in range(N)]) # Store scalar B fields windows [nT]
        self.z_labels = self.z[self.pp:] # z-position labels, for time windows [m]

        # Augmentate data if required:
        self.augm_N = N_augm # State how many times the data was augmented
        self.augm_noise = [Bx_noise,By_noise,Bz_noise] # State the augmentation noise levels [nT]
        if N_augm:
            # Initiate augmentated data with the original windows:
            Bx_wdw_augm = np.copy(self.Bx_wdw_RF1) # [nT]
            By_wdw_augm = np.copy(self.By_wdw_RF1) # [nT]
            Bz_wdw_augm = np.copy(self.Bz_wdw_RF1) # [nT]
            B_wdw_augm = np.copy(self.B_wdw) # [nT]
            z_labels_augm = np.copy(self.z_labels) # [m]
            # Determine noise factor for scalar B:
            B_noise = np.sqrt(Bx_noise**2+By_noise**2+Bz_noise**2) # [nT]
            # Augmentate data
            for i in range(N_augm):
                # Generate random noise matrices:
                new_Bx_wdw = self.Bx_wdw_RF1+np.random.normal(size=self.Bx_wdw_RF1.shape)*Bx_noise # [nT]
                new_By_wdw = self.By_wdw_RF1+np.random.normal(size=self.By_wdw_RF1.shape)*By_noise # [nT]
                new_Bz_wdw = self.Bz_wdw_RF1+np.random.normal(size=self.Bz_wdw_RF1.shape)*Bz_noise # [nT]
                new_B_wdw = self.B_wdw+np.random.normal(size=self.B_wdw.shape)*B_noise # [nT]
                # Augmentate windowed data:
                Bx_wdw_augm = np.vstack([Bx_wdw_augm,new_Bx_wdw]) # [nT]
                By_wdw_augm = np.vstack([By_wdw_augm,new_By_wdw]) # [nT]
                Bz_wdw_augm = np.vstack([Bz_wdw_augm,new_Bz_wdw]) # [nT]
                B_wdw_augm = np.vstack([B_wdw_augm,new_B_wdw]) # [nT]
                z_labels_augm = np.hstack([z_labels_augm,self.z_labels]) # [m]
            # Update attributes:    
            self.Bx_wdw_RF1 = Bx_wdw_augm # Update windowed data [nT]
            self.By_wdw_RF1 = By_wdw_augm # Update windowed data [nT]
            self.Bz_wdw_RF1 = Bz_wdw_augm # Update windowed data [nT]
            self.B_wdw = B_wdw_augm # Update windowed data [nT]
            self.z_labels = z_labels_augm # Update windowed data [m]

    # =============

    def rotate_frame(
        self,
        RF_info
        ):
        """
        Generates time windows in a rotated frame, labeled as "RFX" in the object's attributes.
        Only one rotated frame can be recorded at a time.   

        --- Inputs ---    

        {RF_info} [List]: Format [Numpy array,Numeric,String]; specifies the rotated frame that is currently stored. 
        The elements of the list are the following: [rotation axis in (x,y,z) format,rotation angle in degree units,
        RF name]. If RF_info[2] == 'RF1', then the data is kept in the original frame.

        --- Return ---

        Generate or update:

        <self.rotFrame>, <self.Bx_wdw_RFX>, <self.By_wdw_RFX>, <self.Bz_wdw_RFX>.

        """

        # Process variables:
        n_vec, theta_deg, RF_name = RF_info[0], RF_info[1], RF_info[2]
        self.rotFrame = [n_vec,theta_deg,RF_name] # RFX information: rotation axis, rotation angle [degree], name
        # Keep original frame:
        if RF_name == 'RF1':
            self.Bx_wdw_RFX = self.Bx_wdw_RF1
            self.By_wdw_RFX = self.By_wdw_RF1
            self.Bz_wdw_RFX = self.Bz_wdw_RF1
        else:
            # Rotate original information (RF1), window by window:
            # {vector}: Vector to be rotated, must be a numpy array with three rows (x,y,z).
            # Initiate:
            self.Bx_wdw_RFX = np.empty(shape=(self.Bx_wdw_RF1.shape))
            self.By_wdw_RFX = np.empty(shape=(self.By_wdw_RF1.shape))
            self.Bz_wdw_RFX = np.empty(shape=(self.Bz_wdw_RF1.shape))
            for i in range(len(self.Bx_wdw_RF1)):
                v_rot = rotate_3D(np.array([self.Bx_wdw_RF1[i],self.By_wdw_RF1[i],self.Bz_wdw_RF1[i]]),
                                  n_vec,theta_deg) # Rotated fields Bx', By', Bz' [nT]
                self.Bx_wdw_RFX[i] = v_rot[0] # [nT]
                self.By_wdw_RFX[i] = v_rot[1] # [nT]
                self.Bz_wdw_RFX[i] = v_rot[2] # [nT]
    
    # =============

    def info(
        self
        ):
        """
        Prints in screen a summary of the object.
        """
        print('-'*30)
        print('Time window object - Summary:')
        print('Name:',self.name)
        print('Points in each time window:',self.pp)
        print('Ground truth available:',self.gr_tr)
        if 'z_labels' in dir(self):
            print('Number of windows:',len(self.z_labels))
        print('Rotated Frame:',self.rotFrame[2])
        if 'augm_N' in dir(self):
            print(f'Data was augmented {self.augm_N} times.')
            print(f'Augmentation noise levels in RF1 (Bx,By,Bz): ({self.augm_noise}) nT')
        else:
            print('No data augmentation')
        print('-'*30)

    # =============

    def plot_instances(
        self,
        stride_pp=None,
        start_wdw=0,
        RF=False,
        instances=5,
        savefig=None,
        save_format='png',
        figsize=(6,4)
        ):
        """
        Plots some instances of time windows along with z-position labels.
        
        --- Inputs ---
        
        {stride_pp} [Integer]: Number of points by which the time windows' first point will be spaced in the plot.
        If None (default), then it will be chosen automatically.
        {start_pp} [Integer]: First time window to start the plot. Default: 0 (first window).
        {RF_name} [Boolean]: If False (default), the original rotational frame (RF1) is used. If True, then the
        current <self.rotFrame> is used.
        {instances} [Integer]: Number of instances to be plotted. Default: 5.
        {savefig} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
        {save_format} [String]: saving format for the figure,don't include the dot.
        {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

        --- Return ---

        Single plot with several instances of time windows, containing magnetic fields and the target Z-positions.

        """
        # Prepare data:
        if stride_pp is None:
            stride_pp = max(1,round(self.pp*0.8))
        pp_i, pp_f = start_wdw, start_wdw+stride_pp*instances # Boundary indexes
        t_data = self.time_wdw[pp_i:pp_f:stride_pp] # [s]
        z_data = self.z_labels[pp_i:pp_f:stride_pp] # [m]
        if RF:
            Bx_data = self.Bx_wdw_RFX[pp_i:pp_f:stride_pp] # [nT]
            By_data = self.By_wdw_RFX[pp_i:pp_f:stride_pp] # [nT]
            Bz_data = self.Bz_wdw_RFX[pp_i:pp_f:stride_pp] # [nT]
            RF_title = self.rotFrame[2]
        else:
            Bx_data = self.Bx_wdw_RF1[pp_i:pp_f:stride_pp] # [nT]
            By_data = self.By_wdw_RF1[pp_i:pp_f:stride_pp] # [nT]
            Bz_data = self.Bz_wdw_RF1[pp_i:pp_f:stride_pp] # [nT]
            RF_title = 'Laboratory'

        # Plot figure:
        colors = plt.cm.viridis(np.linspace(0, 1, instances))
        fig, (ax_Bz, ax_By, ax_Bx, ax_z) = plt.subplots(4,1,figsize=figsize)
        # Time windows:
        for i in range(instances):
            ax_Bx.scatter(t_data[i],Bx_data[i],color=colors[i],alpha=0.8)
            ax_By.scatter(t_data[i],By_data[i],color=colors[i],alpha=0.8)
            ax_Bz.scatter(t_data[i],Bz_data[i],color=colors[i],alpha=0.8)
            ax_z.scatter(t_data[i][-1],z_data[i],color=colors[i],alpha=0.8)
        # Floor references:
        label = 'Levels'
        for z in np.append(0,4.1+np.arange(0,3.7*6+0.1,3.7)):
            ax_z.axhline(z,ls='--',alpha=0.2,label=label)
            label = None # Prevent further labeling
        # Configuration:
        ax_z.legend(), ax_z.set(xlabel='Time [s]',ylabel='z labels [m]')
        ax_z.set_ylim([np.min(z_data)-1,np.max(z_data)+1]) # z limits [m]
        ax_z.spines['right'].set_visible(False), ax_z.spines['top'].set_visible(False)
        axes_B = [ax_Bx, ax_By, ax_Bz]; labels_B = ['Bx','By','Bz']
        for i in range(3):
            axes_B[i].set(ylabel=f'{labels_B[i]} [nT]')
            axes_B[i].set_xticks([])
            for spin in ['right','bottom','top']:
                axes_B[i].spines[spin].set_visible(False)
        for ax in axes_B+[ax_z]:
            ax.set_xlim([t_data[0][0]-1,t_data[-1][-1]+1])
        plt.suptitle(f'{self.name} ; Rotational Frame: {RF_title}')
        fig.tight_layout()
        if savefig:
            save_file(savefig,save_format=save_format)

    # =============

    def matrix_format(
        self,
        mag_comps
        ):
        """
        Returns a Numpy array with the magnetic signals in the current rotated frame RFX, in a matrix format
        suitable for Machine Learning models.
        
        --- Inputs ---
        
        {mag_comps} [List]: Each element is a string representing the magnetic components that will be used, 
        must choose from options "Bx", "By", "Bz" and/or "B".
        
        --- Return ---
        
        Numpy array with shape [samples,time_wdw_pp,channels], in [nT] units, each channel is a magnetic component.
        The order of magnetic components is always Bx, By, Bz, B (ignoring not required components).

        """

        mag_data = [] # Initiate list with all magnetic numpy arrays
        if "Bx" in mag_comps:
            mag_data.append(self.Bx_wdw_RFX)
        if "By" in mag_comps:
            mag_data.append(self.By_wdw_RFX)
        if "Bz" in mag_comps:
            mag_data.append(self.Bz_wdw_RFX)
        if "B" in mag_comps:
            mag_data.append(self.B_wdw)
            
        return np.stack(mag_data,axis=2)

# ============================================================================

def prepare_time_windows(
    data,
    wdw_pp,
    train_segm=[0],
    plot_instances=True,
    stride_pp=None,
    start_wdw=0,
    RF=False,
    instances=5,
    save_name=None,
    save_format='png',
    figsize=(6,4)
    ):
    """
    Create Time_Wdw objects based on data containing both predictors and targets,
    and separate the entire information into training and testing datasets.

    --- Inputs ---

    {data} [List]: Each element is a DataFrame with information about predictors and targets, with
    columns 'time_s', 'Bx_nT', By_nT, Bz_nT, and zAut_m and/or zTrue_m.
    {wdw_pp} [Integer]: Number of points for each time window.
    {train_segm} [List]: Indexes correlated with the {data} list, indicating which data elements
    will be assigned to the training dataset, the others will be assigned to the testing dataset.
    {plot_instances} [Boolean]: If True, plot instances of Time Windows along with Z-position labels.

    The following inputs are only relevant when {plot_instances}=True:

    {start_pp} [Integer]: First time window to start the plot. Default: 0 (first window).
    {instances} [Integer]: Number of instances to be plotted. Default: 5.
    {savefig} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    {t_wdws_train} [List]: Each element is a Time_Wdw object, representing a segment of the
    original data. This collection is the training dataset.
    {t_wdws_test} [List]: Each element is a Time_Wdw object, representing a segment of the
    original data. This collection is the testing dataset.

    In addition, prints on screen information about the Time Windows and number of points and
    time-length for both training and testing dataset.

    If {plot_instances}=True, plot instances of Time Windows along with Z-position labels.

    """
    
    # Load time windows from different segments within the data:
    t_wdws = [] # Initiate list for all segments
    norm_aux = 0 # Initiate auxiliar value for normalization.
    for df in data:
        # Determine time range:
        t_range = f"time_{int(df['Time_s'].iloc[0]/60)}_to_{int(df['Time_s'].iloc[-1]/60)}min"
        # Generate time window and append it to the list:
        t_wdws.append(Time_Wdw(wdw_pp, t_range, gr_tr='zTrue_m' in df.columns)) # Initiate object
        t_wdws[-1].store_orig_data(df) # Store original data 
        norm_aux = np.max([norm_aux,np.max(t_wdws[-1].B_RF1)])

    # Set normalizing value for future reference [nT]
    for t_wdw in t_wdws:
        t_wdw.norm_value = norm_aux/np.sqrt(3) 

    # Separate into training and testing datasets:
    t_wdws_train = [t_wdws[i] for i in train_segm]
    t_wdws_test = [t_wdws[i] for i in range(len(t_wdws)) if i not in train_segm]

    # Print summary:
    print('Summary for time windows within training and testing datasets:\n')
    print('Number of points for every time window:',wdw_pp)
    print(f'Time resolution: {t_wdw.time[1]-t_wdw.time[0]:.4f} s')
    print('\n','-'*20,' Training dataset ','-'*20,'\n')
    summarize_TW_segments(t_wdws_train)
    print('\n','-'*20,' Testing dataset ','-'*20,'\n')
    summarize_TW_segments(t_wdws_test)

    # Plot instances of Time Windows if requested:
    if plot_instances:
        # Make a full copy of the first training time window and window the data:
        t_wdw_copy = copy_and_reduce_TimeWdw(t_wdws_train[0],1) # Make a full copy
        t_wdw_copy.window_data() # Window data
        # Plot instances:
        t_wdw_copy.plot_instances(
            start_wdw=start_wdw,stride_pp=stride_pp,instances=instances,
            savefig=save_name,save_format=save_format,figsize=figsize)

    return t_wdws_train, t_wdws_test

# ============================================================================

def summarize_TW_segments(
    t_wdws
    ):
    """
    Summarizes the number of points and time duration for a collection of time windows.

    --- Inputs ---

    {t_wdws} [List]: Each element is a Time_Wdw object, integrating the same dataset.
    All Time Windows must have the same number of points.

    --- Return ---

    Prints on screen information about the number of point and time duration for each
    individual time window, and also for the entire collection.

    """
    N = 0 # Initiate total number of points for the t_wdws collection
    for wdw in t_wdws:
        # Check time resolution:
        dt = wdw.time[1]-wdw.time[0] # [s]
        # Print on screen the number of points and time duration for the current time window:
        print(f'{wdw.name}: {len(wdw.z)} points, {int(len(wdw.z)*dt/60)} min')
        # Update the total number of points:
        N += len(wdw.z)
    # Print on screen the number of points and time duration for the total collection of time window:
    print(f'Total points/time: {N} / {np.round(N*dt/3600,2)} hours')

# ============================================================================

def rotate_3D(
    vector,
    n,
    theta_deg
    ):
    """
    Rotate a vector or matrix by an angle theta around the axis n.

    --- Inputs ---
    
    {vector} [List]: Vector or matrix to be rotated, must be a numpy array with three rows (x,y,z).
    {n} [List]: Cartesian coordinates for the rotation axis, in format [X,Y,Z].
    {theta_deg} [Float]: Rotation angle, units [degree].
    
    --- Outputs ---
    
    {vector_R} [Numpy array]: Rotated vector or matrix.

    """

    # Normalise u and convert angle to [rad]:
    if not isinstance(n, np.ndarray):
        n = np.array(n) # Convert to numpy array if necessary
    u = n/np.linalg.norm(n) # Normalised unit vector
    ux, uy, uz = u[0], u[1], u[2] # Cartesian components
    theta = theta_deg/360*2*np.pi # Rotation angle [rad]
    # Define the rotation matrix:
    c = np.cos(theta)
    q = 1-c
    s = np.sin(theta)
    R = np.array(
        [[c+ux**2*q,ux*uy*q-uz*s,ux*uz*q+uy*s],
        [uy*ux*q+uz*s,c+uy**2*q,uy*uz*q-ux*s],
        [uz*ux*q-uy*s,ux*uy*q+ux*s,c+uz**2*q]])

    return np.matmul(R,vector)

# ============================================================================

def copy_and_reduce_TimeWdw(
    time_wdw,
    fraction
    ):
    """
    Copy the basic attributes of a Time_Wdw object, except for its time windows if any,
    reduce the original data to a fraction (cutting off from the last chronological data)
    and returns a new Time_Wdw object. Ignores any rotated data.

    --- Inputs ---

    {time_wdw} [Time_Wdw object]: Original Time Window that will be copied. Information must
    be already stored, except for time windows (which will not be copied).
    {fraction} [Float]: Fraction between 0 and 1 that sets how much of the original training
    dataset is going to be copied.

    --- Return ---

    {new_time_wdw} [Time_Wdw object]: Copy of the original {time_wdw}, but only keeping
    a fraction {fraction} of the data.

    """
    # Initiate object and copy attributes:
    new_time_wdw = Time_Wdw(time_wdw.pp, time_wdw.name, gr_tr=time_wdw.gr_tr)
    N_max = int(len(time_wdw.time)*fraction) # Last point to be preserved
    new_time_wdw.time = time_wdw.time[:N_max] # [s]
    new_time_wdw.Bx_RF1 = time_wdw.Bx_RF1[:N_max] # [nT]
    new_time_wdw.By_RF1 = time_wdw.By_RF1[:N_max] # [nT]
    new_time_wdw.Bz_RF1 = time_wdw.Bz_RF1[:N_max] # [nT]
    new_time_wdw.B_RF1 = time_wdw.B_RF1[:N_max] # [nT]
    new_time_wdw.z = time_wdw.z[:N_max] # Z-position [m]
    new_time_wdw.norm_value = time_wdw.norm_value # [nT]

    return new_time_wdw

# ============================================================================