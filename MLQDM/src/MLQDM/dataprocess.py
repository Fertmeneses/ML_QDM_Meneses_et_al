"""
Data processing: magnetic fields and Z-position.

# ===========

Main functions:

    - plot_magdata
    - plot_magdata_single_component
    - compare_corrected_data
    - obtain_Bx_levels
    - labels_from_file
    - hist_speeds
    - phys_model_z
    - interp_z
    - interp_orig_z_vs_t
    - compare_z_vs_t_approaches
    - process_automatic_labels
    - compute_z_vs_t
    - comp_man_aut_labels
    - confMatrix_perform_labeling
    - show_parking_mismatch
    - correct_parking_mismatch
    - export_predictors_targets
    - plot_all_data_elevator_position

# ===========

Functions included in Class <accel_data>:

    - __init__
    - info
    - add_events_info
    - add_events_times
    - plot_raw_data
    - compare_same_flight_type

# ===========

"""

# ============================================================================

# Required packages:

import numpy as np
import pandas as pd
import copy
import scipy.signal
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from collections import Counter
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from  matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
import seaborn as sns
import os

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

def plot_magdata(
    data,
    label='Magnetic data',
    res_Bx=5,
    color='teal',
    save_name=None,
    save_format='png',
    figsize=(8,9)
    ):
    """
    Plot magnetic timeseries for the Bx,By,Bz components as a function of time and
    magnetic (intensities) histograms. 

    --- Inputs ---

    {data} [Numpy array]: the rows are the array representing the time [s], and magnetic
    components Bx [nT], By [nT] and Bz [nT], in that order.
    {label} [String]: Text for the legend's label representing the magnetic data.
    {res_Bx} [Float]: width of each bin for the magnetic histograms. Units [nT].
    {color} [Matplotlib color]: color for the curves in the plots.
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    Plots organized in 3 rows and 2 columns:
    Left column: timeseries for magnetic data.
    Right column: histogram for magnetic intensity values.
    First, Second, Third rows: magnetic components Bx, By, Bz, respectively.

    """

    # Define binning for histograms:
    bins = round((np.max(data[:,1])-np.min(data[:,1]))/res_Bx) # Number of bins for histogram, based on Bx resolution
    # Plot original timseries and histograms:
    fig, ((ax11, ax12), (ax21, ax22),(ax31,ax32)) = plt.subplots(3,2,figsize=figsize)
    ax_timeseries, ax_hist = [ax11,ax21,ax31], [ax12,ax22,ax32]
    for i,comp in enumerate(['Bx','By','Bz']):
        # Timeseries:
        ax_timeseries[i].plot(data[:,0]/60,data[:,i+1],'-',color=color,lw=1,alpha=0.8)
        ax_timeseries[i].set(xlabel='Time [min]',ylabel=f"{comp} signal [nT]")    
        # Histogram:
        ax_hist[i].hist(data[:,i+1],bins=bins,label=label,color=color,alpha=0.8)
        ax_hist[i].legend(title = f"{comp} component")
        ax_hist[i].set(xlabel=f'{comp} signal [nT]',ylabel='Counts')   
        ax_hist[i].set_xlim([np.min(data[:,1]),np.max(data[:,1])])    
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format)  

# ============================================================================

def plot_data_single_component(
    data,
    comp,
    label='Magnetic data',
    res_bin=5,
    levels=None,
    save_name=None,
    color='navy',
    save_format='png',
    title=None,
    figsize=(8,2.5)
    ):
    """
    Plot magnetic timeseries for the Bx,By,Bz components as a function of time and
    magnetic (intensities) histograms. 

    --- Inputs ---

    {data} [Numpy array]: the rows are the array representing the time [s], and magnetic
    components Bx [nT], By [nT] and Bz [nT], in that order.
    {comp} [String]: choose between 'Bx', 'By', 'Bz', which selects the magnetic component to be analyzed.
    {label} [String]: Text for the legend's label representing the magnetic data.
    {res_bin} [Float]: width of each bin for the magnetic histograms. Units [nT].
    {levels} [Numpy array]: each element is a magnetic value associated to a parking position of the elevator,
    units [nT].
    {color} [Matplotlib color]: color for the curves in the plots.
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {title} [String]: general title for the plot. 
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    Plots organized in 2 columns:
    Left column: timeseries for the chosen magnetic component data.
    Right column: histogram for magnetic intensity values.

    """

    time_data = data[:,0] # [s]
    mag_data = data[:,1] if comp == 'Bx' else (data[:,2] if comp == 'By' else data[:,3]) # [nT]
    bins = round((np.max(mag_data)-np.min(mag_data))/res_bin) # Number of bins for histogram, based on Bx resolution
    fig, (ax_timeseries,ax_hist) = plt.subplots(1,2,figsize=figsize)
    # Timeseries:
    ax_timeseries.plot(time_data/60,mag_data,'-',color=color,lw=0.5,alpha=0.8)
    for x in levels:
        ax_timeseries.axhline(x,lw=0.5,ls='--',alpha=0.8,color='blue')   
    ax_timeseries.set(xlabel='Time [min]',ylabel=f'{comp} Signal [nT]')    
    # Histograms:
    ax_hist.hist(mag_data,bins=bins,label=label,color=color,alpha=0.8)
    for x in levels:
        ax_hist.axvline(x,lw=0.5,ls='--',alpha=0.6,color='blue')             
    #ax_hist.legend()
    ax_hist.set(xlabel=f'{comp} Signal [nT]',ylabel='Counts')
    ax_hist.set_xlim([np.min(mag_data),np.max(mag_data)])    
    # General config:
    plt.suptitle(title)
    plt.legend()
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format)

# ============================================================================

def compare_corrected_data(
    data_old,
    data_new,
    comp,
    res_bin=5,
    levels=None,
    save_name=None,
    save_format='png',
    figsize=(8,2.5)
    ):
    """
    Compare the original and corrected magnetic data for a single magnetic component, 
    both timeseries and histogram of magnetic intensity values. 

    --- Inputs ---

    {data_old} [Numpy array]: the rows are the array representing the time [s] and magnetic 
    components Bx [nT], By [nT] and Bz [nT], in that order, for the original data.
    {data_new} [Numpy array]: the rows are the array representing the time [s] and magnetic 
    components Bx [nT], By [nT] and Bz [nT], in that order, for the corrected data.
    {comp} [String]: choose between 'Bx', 'By', 'Bz', which selects the magnetic component to be analyzed.
    {res_bin} [Float]: width of each bin for the magnetic histograms. Units [nT].
    {levels} [Numpy array]: each element is a magnetic value associated to a parking position of the elevator,
    units [nT].
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    Plots organized in 2 columns, comparing original and corrected data:
    Left column: timeseries for the chosen magnetic component data.
    Right column: histogram for magnetic intensity values.

    """

    # Prepare data:
    time_data_old = data_old[:,0] # [s]
    time_data_new = data_new[:,0] # [s]
    mag_data_old = data_old[:,1] if comp == 'Bx' else (data_old[:,2] if comp == 'By' else data_old[:,3]) # [nT]
    mag_data_new = data_new[:,1] if comp == 'Bx' else (data_new[:,2] if comp == 'By' else data_new[:,3]) # [nT]
    bins = round((np.max(mag_data_old)-np.min(mag_data_old))/res_bin) # Number of bins for histogram, based on Bx resolution

    # Prepare figure layout:
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 5)
    ax_hist, ax_timeseries = plt.subplot(gs[:,2:]), plt.subplot(gs[:,:2])
    ax_timeseries_lvl = ax_timeseries.twinx() # Right axis for elevator's levels

    # Timeseries:
    ax_timeseries.plot(time_data_old/60,mag_data_old,'-',color='teal',lw=1,alpha=0.5,label='Original')
    ax_timeseries.plot(time_data_new/60,mag_data_new,'-',color='navy',lw=1,alpha=0.5,label='Corrected')
    if comp == 'Bx':
        for level in levels:
            ax_timeseries.axhline(level,color='blue',ls='--',lw=0.5,alpha=0.5)
        ylim_ = ax_timeseries.get_ylim()
        ax_timeseries_lvl.set_ylim(ylim_[0],ylim_[1])
        ax_timeseries_lvl.set_yticks(levels, range(len(levels)),color='blue')
        ax_timeseries_lvl.set_ylabel('Elevator level',color='blue')    
    else:
        ax_timeseries_lvl.set_yticks([])            
    ax_timeseries.set(xlabel='Time [min]',ylabel=f'{comp} Signal [nT]')

    # Histograms:
    ax_hist.hist(mag_data_old,bins=bins,label='Original',color='teal',alpha=0.5)
    ax_hist.hist(mag_data_new,bins=bins,label='Corrected',color='navy',alpha=0.5)
    if comp == 'Bx':
        for level in levels:
            ax_hist.axvline(level,lw=0.5,ls='--',alpha=0.6,color='blue')
    ax_hist.legend()
    ax_hist.set(xlabel=f'{comp} Signal [nT]',ylabel='Counts')
    ax_hist.set_xlim([min(np.min(mag_data_new),np.min(mag_data_old)),
                      max(np.max(mag_data_new),np.max(mag_data_old))])    
    # General config:
    plt.suptitle(f'{comp} Component')
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format)

# ============================================================================

def obtain_Bx_levels(
    data,
    N_lvls,
    pp_Bx=100,
    pol_Bx=3,
    pp_dBx=30,
    pol_dBx=3,
    thres_rel_frac_dBx=1/6,
    bin_fraction=0.5,
    peak_dist=18,
    peak_height=3,
    save_name=None,
    save_format='png',    
    figsize=(6,4)
    ):
    """
    Identifies the average Bx signals for all parking levels. The parking levels assignment is very
    sensitive to the input parameters, so you must check the results and iterate until you are happy
    with them.
    
    --- Inputs: magnetic signal, smoothing and derivating ---
    
    The raw input data (time,Bx,By,Bz) is smoothed and derivated in order to identify the traveling events
    and the parking intervals.
    
    {data} [Numpy array with shape (N,4)]: each row represents a time entry and the columns represent 
    time [s], Bx [nT], By [nT] and Bz [nT], respectively.
    {N_lvls} [Integer > 0]: number of levels (lift parking positions) for the dataset {data}. If you
    input a wrong number (compared to the actual levels), the labeling will be inaccurate.
    {pp_Bx} [Integer > 0]: number of points for smoothing the Bx magnetic signal, according to the 
    Savitzky-Golay filter.
    {pol_Bx} [Integer >= 0], order of the polynomial used for smoothing the Bx magnetic signal. 
    Note: {pol_Bx} must be less than {pp_Bx}.
    {pp_dBx} [Integer > 0]: number of points for smoothing the dBx/dt derivative signal, according to the
    Savitzky-Golay filter.
    {pol_dBx} [Integer >= 0]: order of the polynomial used for smoothing the dBx/dt derivative signal.
    Note: {pol_Bx} must be less than {pp_Bx}.
    {thres_rel_frac_dBx} [Float]: fraction of the maximum amplitude in the dBx/dt derivative signal, which
    will be the threshold to define traveling events.
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.
    
    --- Inputs: parking levels assignment ---
    
    Using the parking intervals, a histogram will be built from the mean Bx values and the parking levels
    will be assigned. Then, using the traveling events and assuming a constant travel velocity for each
    interval, the z-position vs time will be computed. 
    
    {bin_fraction} [Float]: fraction of the total number of parking intervals that. It will define the number
    of bins in the Bx histogram.
    {peak_dist} [Integer > 0]: minimum distance between peaks in the level assignment.
    {peak_height} [Integer > 0]: minimum height for peaks in the level assignment.
    
    --- Return ---
    
    {lvls_central_Bx} [Numpy array with length {N_lvls}]: average Bx values assigned to each level, 
    from 0 to {N_lvls}. Units: [nT].

    Histogram plot for the magnetic intensity values of the Bx component and the identified levels. 

    """

    ## PART 1: Smooth and derivate raw data; Identify traveling events and parking intervals. 
    
    # Smooth original Bx curve:
    mag_smooth = savgol_filter(data[:,1], pp_Bx, pol_Bx) # Smoothed magnetic curve [nT]
    # Time step size:
    time = data[:,0] # Time vector [s]
    dt = time[1]-time[0] # Time step [s]    
    # Compute vector of forward differences:
    mag_diff = np.diff(mag_smooth)/dt # Derivative [nT/s]
    mag_diff_smooth = savgol_filter(mag_diff, pp_dBx, pol_dBx) # Smoothed version [nT/s]
    # Apply threshold filter:
    min_diff, max_diff = np.min(mag_diff_smooth), np.max(mag_diff_smooth) # Identify max/min values [nT/s]
    thres_diff = np.min([np.abs(max_diff),np.abs(min_diff)])*thres_rel_frac_dBx # Define threshold value [nT/s]
    mag_diff_smooth[np.abs(mag_diff_smooth) < thres_diff] = 0 # Apply threshold
    # Identify traveling events:
    travel_index = np.append(np.abs(mag_diff_smooth) > 0, False) # Traveling intervals [time indexes]
    travel_diff = np.append(np.diff(travel_index*1),False) # Events (start or finish traveling) [indexes]
    event_times = np.append(time[0],time[travel_diff!=0]) # Filter event times [s]
    t_i_travel = event_times[1::2] # Time flags for start traveling [s]
    t_f_travel = event_times[2::2] # Time flags for stop traveling [s]
    # Identify parking times:
    t_i_park = np.append(event_times[0],t_f_travel[:-1]) # Time flags for initial parking time [s]    
    t_f_park = np.append(event_times[1],t_i_travel[1:]) # Time flags for final parking time [s]
    # Identify parking intervals and mean values:
    i_park = np.array((t_i_park-time[0])/dt,dtype=int) # Index flags for initial parking times
    f_park = np.array((t_f_park-time[0])/dt,dtype=int) # Index flags for final parking times
    mean_mag = np.array([np.mean(mag_smooth[i_park[i]:f_park[i]]) for i in range(len(t_i_park))])
    
    ## PART 2: Assign parking levels and compute z-position vs time. 
    
    # Fit the mean magnetic values distribution and assign levels for all intervals:
    bins = int(len(mean_mag)*bin_fraction) # Define number of bins
    # Make a histogram with Bx values:
    Bx_counts, Bx_edges = np.histogram(mean_mag,bins=bins) # [counts, nT]
    # Obtain the central Bx value for each bin:
    Bx_values = np.array([0.5*(Bx_edges[i+1]+Bx_edges[i]) for i in range(len(Bx_edges)-1)]) # [nT]
    # Find and plot peaks for the mean Bx values histogram:
    index_peaks, height_peaks = scipy.signal.find_peaks(Bx_counts,distance=peak_dist,height=peak_height)
    fig = plt.figure(figsize=figsize)
    plt.plot(Bx_values[index_peaks],height_peaks['peak_heights'],'o',label='Peaks',alpha=0.5)
    plt.hist(mean_mag,bins=bins,label='Hist',alpha=0.5)
    plt.title(f'Proposed {N_lvls} levels',fontsize=8)
    plt.legend()
    plt.xlabel('Bx values [nT]',fontsize=8)
    plt.ylabel('Counts: mean Bx during parking',fontsize=8)
    fig.tight_layout();
    # If the number of identified levels is not correct, exit the function:
    if len(index_peaks) != N_lvls:
        print('Wrong input parameters: the number of identified levels does not match the real levels.')
        return None
    # Assign a level to each parking interval:
    lvls_central_Bx = Bx_values[index_peaks] # Central Bx values for levels, from 0 to {N_lvls}
    
    return lvls_central_Bx

# ============================================================================

def labels_from_file(
    file
    ):
    """
    Process labels (elevator parking positions) based on a manual registry.
    Note: The distance between levels 0-1 is 4.1m. The distance for any other pair of adjacent levels is 3.7m.
    
    --- Inputs ---
    
    {file} [String]: path to the label's file in .xlsx or .csv format, which must have 6 header lines and 
    then columns representing the "Starting Time [s]", "Starting Floor", "Final floor", "Final time [s]"
    and "Lift velocity [floors/s]".
    
    ---Outputs---
    
    {pos_t} [Numpy array with shape (2,N)]: the first row contains the lift event times
    (N records), and the second row contains the correlated elevator positions (floors).
    {speeds} [Dictionary]; keys are tuples (initial,final) levels, values are the speed instances
    found in the dataset. Speed values are always positive.
    
    """

    # Read labels:
    new_names_cols = {
    'Starting Time [s]':'time_i',
    'Starting Floor':'floor_i', 
    'Final floor':'floor_f',
    'Final time [s]':'time_f',
    'Lift velocity [floors/s]':'v_lift'
                      } 
    labels = pd.read_excel(file,header=6) if 'xlsx' in file else pd.read_csv(file,header=6)
    labels = labels.rename(columns=new_names_cols) # Change column names
     
    # Process the elevator position and convert the levels to signals:
    times, levels, speeds = [labels.time_i[0]], [labels.floor_i[0]], {}
    floors = list(set(labels.floor_i).union(set(labels.floor_f))) # Identify all floors in the file.
    # Go through each travelling instance:
    for floor_i in floors: # Initiate dictionary
        for floor_j in floors:
            speeds[(int(floor_i),int(floor_j))] = [] # Here (i,i) keys are included, they will be deleted later
    for i in range(1,len(labels)): # Populate dictionary
        # Obtain initial and final levels and times:
        times.extend([labels.time_i[i],labels.time_f[i]])
        levels.extend([labels.floor_i[i],labels.floor_f[i]])
        # Calculate speed:
        z_init = 0 if labels.floor_i[i] == 0 else 4.1+3.7*(labels.floor_i[i]-1) # [m]
        z_end = 0 if labels.floor_f[i] == 0 else 4.1+3.7*(labels.floor_f[i]-1) # [m]
        speed = abs(z_end-z_init)/(labels.time_f[i]-labels.time_i[i]) # [m/s]
        speeds[(labels.floor_i[i],labels.floor_f[i])].append(speed)
    # Remove all invalid (i,i) and empty keys in speeds:
    for i in floors: 
        speeds.pop((i,i))
    for key in list(speeds.keys()):
        if len(speeds[key])==0:
            speeds.pop(key)

    # Obtain z-positions from levels:
    z_levels = [0 if level == 0 else np.round(4.1+3.7*(level-1),2) for level in levels] # [m]

    # Combine the time and position arrays:
    pos_t = np.array([times,z_levels]) # [s,m]
    
    # Return the processed data:
    return pos_t, speeds

# ============================================================================

def hist_speeds(
    all_speeds,
    save_name=None,
    save_format='png',
    figsize=(8,6)
    ):
    """
    Plots the speed distribution for the elevator grouped by the number of traveled floors.
    
    --- Inputs ---
    
    {all_speeds} [Dictionary]: keys are tuples (initial,final) levels, values are the speed instances
    found in the dataset. Speed values are always positive.
    {title} [String]: title for the plot.
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    Note: The distance between levels 0-1 is 4.1m. The distance for any other pair of adjacent levels is 3.7m.
    
    --- Return ---
    Plot for the elevator speed distribution classified according to the number of travelled levels.

    """
    # Convert to numpy and adjust units if floors' height was provided, changing from [floors/s] to [m/s]:
    v = {key: np.array(all_speeds[key]) for key in all_speeds} # [m/s]

    # Identify "jumps", defined by the number of floors travelled in a single trip
    jumps = set([abs(key[1]-key[0]) for key in v])
    # Colors and auxiliar variables:
    colors = plt.cm.viridis(np.linspace(0, 1, len(jumps)))
    # Determine maximum and minimum values, then bin distribution:
    v_min = np.min([np.min(v[key]) for key in v]) # Minimum velocity [floors/s]
    v_max = np.max([np.max(v[key]) for key in v]) # Maximum velocity [floors/s]
    dv = 0.05 # Bin's width [floors/s]
    bins = np.arange(v_min, v_max + dv, dv) # Arrangement of bins
    # Plot figure:
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(jumps), 1, hspace=0, wspace=0)
    axes = gs.subplots(sharex='col', sharey='row')
    for counter,jump in enumerate(jumps):
        avg_v = [] # Initiate a list to calculate average speed for the current N-level flights:
        for key in v:
            # Select data that matches the current number of travelled levels:
            if abs(key[0]-key[1]) == jump:
                # Add data to the plot:
                axes[counter].hist(v[key],bins=bins,color=colors[counter],alpha=0.6)
                # Update the average speed list:
                for speed in v[key]:
                    avg_v.append(speed)
        avg_v = np.mean(avg_v) # Calculate average speed for these N-level flights from list
        axes[counter].text(0.8, 0.75, f'Traveling {jump} floors', transform = axes[counter].transAxes)
        axes[counter].text(0.8, 0.45, f'Avg. speed: {avg_v:.2f} m/s', transform = axes[counter].transAxes,fontsize=8)
        axes[counter].set_xlim([v_min-1*dv,v_max+3*dv])
        axes[counter].axes.yaxis.set_visible(False)
        for spin in ['right','left','top']:
            axes[counter].spines[spin].set_visible(False)
    fig.supxlabel('Speed [m/s]'), fig.supylabel('Ocurrences')
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format) 

# ============================================================================

def phys_model_z(
    t_total,
    z_total,
    dt=0.1
    ):
    """
    Calculates the elevator z-position according to the physical model.

    --- Inputs ---

    {t_total} [Float]: Total traveling time, units [s].
    {z_total} [Float]: Total traveling distance, units [m].
    Note: Adjacent levels are 3.7m apart, except for the lowest levels 1-2, which are 4.1m apart.
    {dt} [Float]: Time resolution for the generated data, units [s].

    --- Return ---

    {z} [Numpy array]: Position timeseries, units [s].

    """

    # Prepare parameters:
    if z_total == 3.7 or z_total == 4.1:
        Dt = t_total/5 # Time for single step [s]
        a = 25/4*z_total/t_total**2 # Max. acceleration [m/s^2]
    elif t_total <= 6:
        Dt = t_total/6 # Time for single step [s]
        a = 216/35*z_total/t_total**2 # Max. acceleration [m/s^2]
        t_V = 0 # Time traveling at terminal velocity [s]
        pp_V = 0 # Number of points traveling at terminal speed
    else:
        Dt = 1.0 # Time for single step [s]
        a = z_total/(2*t_total-37/6) # Max. acceleration [m/s^2]
        t_V = t_total - 6 # Time traveling at terminal velocity [s]
        pp_V = int(t_V/dt) # Number of points traveling at terminal speed
    pp_t = int(Dt/dt) # Number of points per linear acceleration segment
    Dtp = Dt/2 # Time to switch to opposite accel., only 1-level flights [s]

    # Generate position timeseries from physical model:

    # STEP 1 (common), from a=0 to a=amax:
    z1 = 1/6*np.arange(0,(pp_t-0.1)*dt,dt)**3/Dt # [s^2]

    # STEP 2 (common), constant a=amax:
    z2 = 1/6*Dt**2+1/2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)+1/2*np.arange(0,(pp_t-0.1)*dt,dt)**2 # [s^2]

    # Case (a): N-level flights, with N>1:
    if z_total != 3.7 and z_total != 4.1:
        # STEP 3a [N>1], from a=amax to a=0:
        z3_a = 8/6*Dt**2+3/2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)-1/6*(Dt-np.arange(0,(pp_t-0.1)*dt,dt))**3/Dt # [s^2]

        # STEP 4a [N>1], constant a=0:
        z4_a = 17/6*Dt**2+2*Dt*np.arange(0,(pp_V-0.1)*dt,dt) # [s^2]

        # STEP 5a [N>1], from a=0 to a=-amax:
        z5_a = 17/6*Dt**2+2*Dt*t_V+2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)-1/6*np.arange(0,(pp_t-0.1)*dt,dt)**3/Dt # [s^2]

        # STEP 6a [N>1], constant a=-amax:
        z6_a = 14/3*Dt**2+2*Dt*t_V+3/2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)-1/2*np.arange(0,(pp_t-0.1)*dt,dt)**2 # [s^2]

        # STEP 7a [N>1], from a=-amax to a=0:
        z7_a = 35/6*Dt**2+2*Dt*t_V-1/6*(Dt-np.arange(0,(pp_t-0.1)*dt,dt))**3/Dt # [s^2]

        # Concatenate all acceleration steps:
        z = a*np.concatenate([z1,z2,z3_a,z4_a,z5_a,z6_a,z7_a,[z7_a[-1]]]) # [m]

    # Case (b): 1-level flights:
    else:
        # STEP 3b [N=1], a=amax to a=-amax:
        z3_b = (7/6*Dt**2-1/6*Dtp**2+3/2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)+1/2*Dtp*np.arange(0,(pp_t-0.1)*dt,dt)+
        1/6*(Dtp-np.arange(0,(pp_t-0.1)*dt,dt))**3/Dtp) # [s^2]

        # STEP 4b [N=1], constant a=-amax:
        z4_b = (7/6*Dt**2+2/3*Dtp**2+3*Dt*Dtp+3/2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)-
        1/2*np.arange(0,(pp_t-0.1)*dt,dt)**2) # [s^2]

        # STEP 5b [N=1], from a=-amax to a=0:
        z5_b = 7/3*Dt**2+2/3*Dtp**2+3*Dt*Dtp-1/6*(Dt-np.arange(0,(pp_t-0.1)*dt,dt))**3/Dt # [s^2]

        # FINAL POSITION:
        zf_b = 7/3*Dt**2+2/3*Dtp**2+3*Dt*Dtp # [s^2]

        # Concatenate all acceleration steps:
        z = a*np.concatenate([z1,z2,z3_b,z4_b,z5_b,[zf_b]]) # [m]

    return z

# ============================================================================

def interp_z(
    t1,
    t2,
    z1,
    z2,
    dt=0.1,
    approach='linear_approx'
    ):
    """
    Interpolates the position of the elevator z(t) for traveling event using
    the previous and next parking positions. Two approaches can be used: either
    the linear approximation 'linear_approx' or a physical model based on the
    acceleration profile 'phys_model' (see notebook Acceleration_data_model).
    If the initial and final parking positions are the same, meaning that this
    is a parking event, then the approach 'linear_approx' is used.

    --- Inputs ---

    {t1} [Float]: Initial time of the traveling event, units [s].
    {t2} [Float]: Final time of the traveling event, units [s].
    {z1} [Float]: Initial parking position, units [m].
    {z2} [Float]: Final parking position, units [m].
    {dt} [Float]: Time resolution for the generated data, units [s].
    {approach} [String]: Either 'linear_approx' or 'phys_model', it determines
    which kind of model will be used for interpolation.

    --- Return ---

    {t} [Numpy array]: Timeseries for the interpolated time, units [s].
    It always includes the boundary {t1}, and {t2} only if t2 = t1+n*dt, 
    with n an integer.
    {z} [Numpy array]: Timeseries for the interpolated position, units [m].
    It includes the boundaries {z1} and {z2}.

    """

    # Prepare variables:
    t_total = np.round(t2-t1,4) # Total traveling time [s]
    z_total = np.round(z2-z1,4) # Total traveling distance [m]
    t = np.arange(t1, t2-dt/2, dt) # Time vector [s]
    
    # Parking event:
    if z_total == 0:
        z = np.ones(len(t))*z1 # Position vector [m]

    # Traveling event:
    elif 'phys_model':
        # Linear approximation:
        if approach == 'linear_approx':
            z = z1 + (t-t1)*z_total/t_total # Position vector [m]
        # Physical model based on aceleration:
        else:
            z = z1 + phys_model_z(t_total,z_total,dt=dt)
            # Patch the last point/s in z-vector if there is a mismatch with the time vector:
            if len(z) < len(t):
                z = np.append(z,np.linspace(z[-1],z2,len(t)-len(z))) # [m]
            elif len(z) > len(t):
                z = z[:len(t)] # [m]

    else:
        raise ValueError('Error: choose between either approach="linear_approx" or "phys_model".')
    return t, z

# ============================================================================

def interp_orig_z_vs_t(parking_t_pos,time_res,approach='linear_approx'):
    """
    Interpolates the original parking level positions into a continuous position,
    using either a linear approximation or a physical model based on the acceleration
    progile (see notebook Acceleration_data_model).

    --- Inputs ---
    
    {parking_t_pos} [Numpy array with shape (2,N)]: the first row contains the elevator event times
    (N records) indicating initial and final parking intervals [s], and the second row contains the 
    correlated parking levels [m].
    {time_res} [Float]: time resolution during measurements, which will be used as the time unit
    for interpolation [s].
    {approach} [String]: Either 'linear_approx' or 'phys_model', it determines which kind of model
    will be used for interpolation.
    
    --- Return ---
    
    {t} [Numpy array]: full time records for the elevator trajectory according to the
    interpolation model [s].
    {z} [Numpy array]: full z-position records for the elevator trajectory according to the
    interpolation model [m].
    
    """
    
    # Initiate time and position vectors:
    t, z = np.empty(0), np.empty(0) # [s], [m]

    # Interpolate segment by segment and update the {t} and {z} vectors:
    for i in range(len(parking_t_pos[0])-1):
        # Identify parameters:
        t1 = parking_t_pos[0][i] # Initial segment time [s]
        t2 = parking_t_pos[0][i+1] # Final segment time [s]
        z1 = parking_t_pos[1][i] # Initial segment position [m]
        z2 = parking_t_pos[1][i+1] # Final segment position [m]    
        # Interpolation:
        t_segm, z_segm = interp_z(t1,t2,z1,z2,dt=time_res,approach=approach)
        # Update {z} and {t}:
        t = np.append(t,t_segm) # [s]
        z = np.append(z,z_segm) # [m]

    return t, z

# ============================================================================

def compare_z_vs_t_approaches(
    t_lin,
    z_lin,
    t_phys,
    z_phys,
    i1,
    i2,
    save_name=None,
    save_format='png',
    figsize=(8,3)   
    ):
    """
    Compare in a plot two different approaches for interpolating the elevator parking positions: the linear
    approximation and the physical model based on the acceleration profile. Only a segment of the timeseries
    will be plot.

    --- Inputs ---

    {t_lin} [Numpy array]: Time vector for the linear approximation [s].
    {z_lin} [Numpy array]: Z-Position vector for the linear approximation [s].
    {t_phys} [Numpy array]: Time vector for the physical model [s].
    {z_phys} [Numpy array]: Z-Position vector for the physical model [s].
    {i1} [Integer]: Index for the left-most boundary within the timeseries.
    {i2} [Integer]: Index for the right-most boundary within the timeseries.
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.
    
    --- Return ---
    
    Plot a segment of the position timeseries comparing the linear approximation and
    physical model approaches.
    
    """

    # Plot figure:    
    fig = plt.figure(figsize=(8,3))
    plt.plot(t_lin[i1:i2],z_lin[i1:i2],'o',alpha=0.6,markersize=4,label='Linear')
    plt.plot(t_phys[i1:i2],z_phys[i1:i2],'s',markersize=4,alpha=0.6,label='Phys.')

    # Configuration and saving options:
    plt.legend()
    plt.xlabel('Time [s]'), plt.ylabel('Z-Position [m]')
    plt.title('Comparison between linear approximation and physical model')
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format) 

# ============================================================================

def process_automatic_labels(
    mag_data_path,
    lvls_Bx,
    approach='linear_approx',
    header=0,
    sep='\t'
    ):
    """
    Process the magnetic data to automatically identify Z-labels, based on the 'Bx' component correlation.

    --- Inputs ---

    {mag_data_path} [String]: Path for magnetic information, each file represents a segment and must have
    4 columns for time [s], Bx [nT], By [nT], Bz [nT], in that order, recorded in the laboratory frame, 
    in which the Bx component has a clear correlation with the elevator position.
    {lvls_Bx}: [Numpy array]: Average Bx values assigned to each parking level, units [nT].
    {header} [Integer]: Number of header lines in the magnetic files.
    {sep} [String]: Character or regex pattern to treat as the delimiter in the magnetic files.
    {approach} [String]: Either 'linear_approx' or 'phys_model', it determines which kind of model
    will be used for interpolation.
    
    --- Return ---

    {t_segms_aut} [List]: Each element is a one-dimensional Numpy array representing the time of
    a magnetic file, units [s]. They are sorted in chronological time-order.
    {z_segms_aut} [List]: Each element is a one-dimensional Numpy array representing the elevator
    z-positibon automatic labels of a magnetic file, units [m]. They are correlated with {t_segms_aut}.
    {data_segms} [List]: Each element is 4-rows Numpy array representing the time [s], Bx [nT], 
    By [nT], Bz [nT], in that row-order, of a magnetic file. They are sorted in chronological time-order.
    """
    # Identify measurements files for segments (avoid the full record):
    files = [mag_data_path+file for file in os.listdir(mag_data_path) if "_full" not in file]
    # Initiate auxiliar variables for each segment:
    data_aux, t_aux, z_aux, segms_init_time = [], [], [], []
    # Process data for each file (=segment):
    for file in files:
        # Load dataframes and convert them to numpy arrays, with columns time[s],Bx[nT],By[nT],Bz[nT]:
        data = pd.read_csv(file,header=header,sep=sep).to_numpy()
        data_aux.append(data)
        # Compute z-position vs time:
        time, z_pos = compute_z_vs_t(data,lvls_Bx,approach=approach) 
        t_aux.append(time) # Store current time array [s]
        z_aux.append(z_pos) # Store z-position array for current segment [m]
        segms_init_time.append(time[0]) # Store initial time for the segment [s]
            
    # Sort segments according to chronological order:
    data_segms, t_segms_aut, z_segms_aut = [], [], [] # Initiate
    sort_index = np.argsort(segms_init_time) # Index keys
    for i in sort_index:
        t_segms_aut.append(np.round(t_aux[i],4)) # Round to avoid numerical trash[s]
        z_segms_aut.append(z_aux[i]) # [m] 
        data_segms.append(data_aux[i]) # [s,nT,nT,nT]

    return t_segms_aut, z_segms_aut, data_segms

# ============================================================================

def compute_z_vs_t(
    data,
    lvls_Bx,
    pp_Bx=100,
    pol_Bx=3,
    pp_dBx=30,
    pol_dBx=3,
    thres_rel_frac_dBx=1/6,
    approach='linear_approx'
    ):
    """
    Computes the z-position as a function of time based on the Bx values previously assigned to each level.
    The raw input data (time,Bx,By,Bz) is smoothed and derivated in order to identify the traveling events
    and the parking intervals. Then, each parking interval is assigned to one level based on the Bx signal
    and the traveling trajectory is calculated using either the 'linear_approx' or 'phys_model' approach.
    Note: The distance between the lowest two levels is 4.1m, for any other pair of adjacent levels is 3.7m.
    
    --- Inputs ---
    
    {data} [Numpy array with shape (N,4)]: each row represents a time entry and the columns represent 
    time [s], Bx [nT], By [nT] and Bz [nT], respectively.
    {lvls_Bx} [Numpy array]: average Bx values assigned to each level, from 0 to the maximum level,
    in steps of 1 level.
    {pp_Bx} [Integer > 0]: number of points for smoothing the Bx magnetic signal, according to the 
    Savitzky-Golay filter.
    {pol_Bx} [Integer >= 0], order of the polynomial used for smoothing the Bx magnetic signal. 
    Note: {pol_Bx} must be less than {pp_Bx}.
    {pp_dBx} [Integer > 0]: number of points for smoothing the dBx/dt derivative signal, according to the
    Savitzky-Golay filter.
    {pol_dBx} [Integer >= 0]: order of the polynomial used for smoothing the dBx/dt derivative signal.
    Note: {pol_Bx} must be less than {pp_Bx}.
    {thres_rel_frac_dBx} [Float]: fraction of the maximum amplitude in the dBx/dt derivative signal, which
    will be the threshold to define traveling events.
    {approach} [String]: Either 'linear_approx' or 'phys_model', it determines which kind of model
    will be used for interpolation.

    --- Return ---

    {t} [List]: Each element is a one-dimensional Numpy array representing a time vector from a magnetic
    file, units [s].
    {z} [List]: Each element is a one-dimensional Numpy array representing a Z-position vector from a magnetic
    file, according to the automatic labels algorithm, units [m].
    
    """
    ## PART 1: Smooth and derivate raw data; Identify traveling events and parking intervals. 
    
    # Smooth original Bx curve:
    mag_smooth = savgol_filter(data[:,1], pp_Bx, pol_Bx) # Smoothed magnetic curve [nT]
    # Time step size:
    time = data[:,0] # Time vector [s]
    dt = np.round(time[1]-time[0],4) # Time step, rounded to avoid any numerical trash [s]
    # Compute vector of forward differences:
    mag_diff = np.diff(mag_smooth)/dt # Derivative [nT/s]
    mag_diff_smooth = savgol_filter(mag_diff, pp_dBx, pol_dBx) # Smoothed version [nT/s]
    # Apply threshold filter:
    max_diff, min_diff = np.max(mag_diff_smooth), np.min(mag_diff_smooth) # Identify max/min values [nT/s]
    thres_diff = np.min([np.abs(max_diff),np.abs(min_diff)])*thres_rel_frac_dBx # Define threshold value [nT/s]
    mag_diff_smooth[np.abs(mag_diff_smooth) < thres_diff] = 0 # Apply threshold
    # Identify traveling events:
    travel_index = np.append(np.abs(mag_diff_smooth) > 0, False) # Traveling intervals [time indexes]
    travel_diff = np.append(np.diff(travel_index*1),False) # Events (start or finish traveling) [indexes]
    event_times = np.append(time[0],time[travel_diff!=0]) # Filter event times [s]
    # If the recording starts when the elevator is parked, everything is ok:
    if len(event_times[1::2]) == len(event_times[2::2]):
        t_i_travel = event_times[1::2] # Time flags for start traveling [s]
        t_f_travel = event_times[2::2] # Time flags for stop traveling [s]
        # Identify parking times:
        t_i_park = np.append(event_times[0],t_f_travel[:-1]) # Time flags for initial parking time [s]    
        t_f_park = np.append(event_times[1],t_i_travel[1:]) # Time flags for final parking time [s]
    # If the recording starts when the elevator is already traveling, then the first traveling event...
    # ...must be ignored because we don't know in which floor it started:
    else:
        t_i_travel = event_times[2::2] # Time flags for start traveling [s]
        t_f_travel = event_times[3::2] # Time flags for stop traveling [s]
        # Identify parking times:
        t_i_park = np.append(event_times[1],t_f_travel[:-1]) # Time flags for initial parking time [s]    
        t_f_park = np.append(event_times[2],t_i_travel[1:]) # Time flags for final parking time [s]    
    
    # Identify parking intervals and mean values:
    i_park = np.array((t_i_park-time[0])/dt,dtype=int) # Index flags for initial parking times
    f_park = np.array((t_f_park-time[0])/dt,dtype=int) # Index flags for final parking times
    mean_mag = np.array([np.mean(mag_smooth[i_park[i]:f_park[i]]) for i in range(len(t_i_park))])
        
    ## PART 2: Assign parking levels and interpolate z-position vs time.
    
    # Identify parking positions:

    parking_z = [] # Initiate labels list:
    for value in mean_mag:
        difference_array = np.absolute(lvls_Bx-value) # Calculate the difference array: 
        level = difference_array.argmin() # Find the index of minimum element (=level) from the array
        z_level = 0 if level == 0 else np.round(4.1+3.7*(level-1),2) # [m]
        parking_z.append(z_level) # Convert level to z-position and compute parking level [m]

    # Interpolate z-position vs time:

    t, z = np.empty(0), np.empty(0) # Initiate time and position vectors [s], [m]
    # Interpolate segment by segment and update the {t} and {z} vectors:
    for i in range(len(parking_z)-1):
        # Parking event:
        # Identify parameters:
        t1 = t_i_park[i] # Initial segment time [s]
        t2 = t_f_park[i] # Final segment time [s]
        z1 = parking_z[i] # Initial segment position [m]
        z2 = parking_z[i] # Final segment position [m]    
        if t2 <= t1:
            print(t1,t2,z1,z2)
        # Interpolation:
        t_segm, z_segm = interp_z(t1,t2,z1,z2,dt=dt,approach=approach)
        # Update {z} and {t}:
        t = np.append(t,t_segm) # [s]
        z = np.append(z,z_segm) # [m]

        # Traveling event:
        # Identify parameters:
        t1 = t_f_park[i] # Initial segment time [s]
        t2 = t_i_park[i+1] # Final segment time [s]
        z1 = parking_z[i] # Initial segment position [m]
        z2 = parking_z[i+1] # Final segment position [m]
        # Interpolation:
        t_segm, z_segm = interp_z(t1,t2,z1,z2,dt=dt,approach=approach)
        # Update {z} and {t}:
        t = np.append(t,t_segm) # [s]
        z = np.append(z,z_segm) # [m]

    # Last parking event:
    t1 = t_i_park[-1] # Initial segment time [s]
    t2 = t_f_park[-1] # Final segment time [s]
    z1 = parking_z[-1] # Initial segment position [m]
    z2 = parking_z[-1] # Final segment position [m]    
    # Interpolation:
    t_segm, z_segm = interp_z(t1,t2,z1,z2,dt=dt,approach=approach)
    # Update {z} and {t}:
    t = np.append(t,t_segm) # [s]
    z = np.append(z,z_segm) # [m]

    return t, z

# ============================================================================

def comp_man_aut_labels(
    t_man,
    z_man,
    t_aut,
    z_aut,
    t_zoom=None,
    savefig=None,
    save_format='png'
    ):
    """
    Compares the z-position labels for manual and automatic methods.
    
    --- Inputs ---
    
    {t_man} [Numpy array]: time for the manual labels. Units: [s].
    {z_man} [Numpy array]: z-position for the manual labels. Units: [m].
    {t_aut} [Numpy array]: time for the automatic labels. Units: [s].
    {z_aut} [Numpy array]: z-position for the automatic labels. Units: [m].
    {t_zoom}: [List or None]: If provided in the format [t_min,t_max], set values for the
    zoomed-in plot. If None, they will be calculated automatically. Units [s].
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.

    """

    # Find minimum and maximum times for comparison:
    min_time = np.max([t_man[0],t_aut[0]]) # Min time [s]
    max_time = np.min([t_man[-1],t_aut[-1]]) # Max time [s]
    # Relate them to the corresponding indexes:
    i_man = np.abs(t_man-min_time).argmin()+1 # Minimum index, man labels
    f_man = np.abs(t_man-max_time).argmin()-1 # Maximum index, man labels
    i_aut = np.abs(t_aut-min_time).argmin()+1 # Minimum index, automatic labels
    f_aut = np.abs(t_aut-max_time).argmin()-1 # Maximum index, automatic labels
    delta_i = np.min([f_man-i_man,f_aut-i_aut]) # Minimum index interval
    # Compare z-pos vs time:
    t_comp = t_man[i_man:i_man+delta_i] # Time vector [s]
    z_comp = z_man[i_man:i_man+delta_i]-z_aut[i_aut:i_aut+delta_i] # z-pos difference [m]
    # Prepare comparison figure:
    man_line, aut_line = ':', '-' # Line styles for manual and automatic labels
    if not isinstance(t_zoom,(list,np.ndarray)): 
        t_zoom = [t_comp[-1500],t_comp[-1]] # Time limits for zoomed plot version [s]
    deltat = t_comp[-1]-t_comp[0] # Total comparison time for general plot [s]
    time_scale = 1 if deltat<600 else (1/60 if deltat<7200 else 1/3600) # Scaling factor

    fig, axes = plt.subplots(2,2,figsize=(8,6))
    
    # Subplot: Full time range, labels
    axes[0][0].plot(t_man[i_man:f_man]*time_scale,
                    z_man[i_man:f_man],man_line,
                    label='Manual',alpha=0.6)
    axes[0][0].plot(t_aut[i_aut:f_aut]*time_scale,
                    z_aut[i_aut:f_aut],aut_line,
                    label='Automatic',alpha=0.6)
    axes[0][0].legend()
    y_min, y_max = axes[0][0].get_ylim() # Prepare to highlight the zoom-in region
    rectangle_1 = patches.Rectangle((t_zoom[0], y_min), t_zoom[1]-t_zoom[0], y_max-y_min, edgecolor=None,
                                    facecolor="orange", alpha=0.25)
    axes[0][0].add_patch(rectangle_1)
    axes[0][1].set_facecolor("#fff1d9ff")
    
    # Subplot: Full time range, errors
    axes[1][0].plot(t_comp*time_scale,z_comp,aut_line,label='Manual vs Automatic',alpha=0.8,color='red')
    axes[1][0].legend()
    for level in [-3.76/2,3.76/2]:
        axes[1][0].axhline(level,color='k',ls='--',lw=1,alpha=0.6)
    y_min, y_max = axes[1][0].get_ylim() # Prepare to highlight the zoom-in region        
    rectangle_2 = patches.Rectangle((t_zoom[0]*time_scale, y_min), (t_zoom[1]-t_zoom[0])*time_scale,
        y_max-y_min, edgecolor=None,facecolor="orange", alpha=0.25)
    axes[1][0].add_patch(rectangle_2)
    axes[1][1].set_facecolor("#fff1d9ff")
    
    # Subplot: Zoom-in range, labels
    axes[0][1].plot(t_man[i_man:f_man]*time_scale,
                    z_man[i_man:f_man],man_line,label='Manual',alpha=0.7)
    axes[0][1].plot(t_aut[i_aut:f_aut]*time_scale,
                    z_aut[i_aut:f_aut],aut_line,label='Automatic',alpha=0.7)
    axes[0][1].legend()
    axes[0][1].set_xlim(t_zoom[0]*time_scale,t_zoom[1]*time_scale)
    
    # Subplot: Zoom-in range, errors
    axes[1][1].plot(t_comp*time_scale,z_comp,aut_line,label='Manual vs Automatic',alpha=0.8,color='red')
    axes[1][1].legend()
    axes[1][1].set_xlim(t_zoom[0]*time_scale,t_zoom[1]*time_scale)
    for level in [-3.76/2,3.76/2]:
            axes[1][1].axhline(level,color='k',ls='--',lw=1,alpha=0.6)    
    # General config:
    t_units = '[s]' if time_scale==1 else ('[min]' if time_scale==1/60 else ['h'])
    axes[1][0].set(xlabel=f'Time {t_units}',ylabel='Z difference [m]')
    axes[1][1].set_xlabel(f'Time {t_units}')
    axes[0][0].set_ylabel('Z position [m]')
    fig.tight_layout();
    if savefig:
        ML_general.save_file(savefig,save_format=save_format)           

# ============================================================================

def confMatrix_perform_labeling(
    t_segms_man,
    z_segms_man,
    t_segms_aut,
    z_segms_aut,
    binning=1,
    err_tol=0.5,
    savefig=None,
    save_format='png'
    ):
    """
    Plots a figure for the automatic labeling method performance, including a general performance
    plot and a confusion matrix. In all cases, both manual and automatic labels are additionally
    assigned to discrete values, namely the "position bins", which have a uniform size and
    are represented by their central value minus half of the bin, starting at -bin/2. For example,
    a 3m-binning means that positions z1=0.8m and z2=4.6m are converted into z1'=0m (bin -1.5m to
    1.5m) and z2'=6m (bin 4.5 to 7.5m).
    
    General performance plot: for each manual label-bin, an accuracy bar for the automatic labeling 
    method is plot, representing the percentage of "correct" values within an error tolerance
    {err_tol} when compared to the manual labels.

    Confusion matrix: Rows and columns represent the manual and automatic label-bins, respectively. The 
    value for each intersection (row=i,col=j) is the percentage of data-points that were labeled as bin=j
    according to the automatic method, and as bin=i according to the manual method, normalized by the
    number of data-points in bin=i according to the manual method. Values close to 100 in the diagonal i=j
    are indicators of good matching between the two methods.

    --- Inputs ---

    {t_segms_man} [List]: each element is a Numpy array, representing the time vector for the 
    manual labels. Units: [s].
    {t_segms_aut} [List]: each element is a Numpy array, representing the time vector for the
    automatic labels. Units: [s].
    {z_segms_man} [List]: each element is a Numpy array, representing the Z-position for the 
    manual labels. Units: [m].
    {z_segms_aut} [List]: each element is a Numpy array, representing the Z-position for the
    automatic labels. Units: [m].
    {binning} [Float]: uniform size for each z-position bin. Units: [m].
    {err_tol} [Float]: tolerance error for the general performance analysis. Units: [m].
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.

    """

    # Initiate dataframe to allocate labels, binning and errors:
    tol_str = f'err_tol_{err_tol}m' # Name for tolerance-error column
    Z_err_segm = [] # Initiate auxiliar variable
    # Iterate over the manual labels (ground truth) and compare with the automatic labels
    for i in range(len(t_segms_man)):
        # Identify relevant time-data and find min/max times for comparison::
        t_man, t_aut = t_segms_man[i], t_segms_aut[i] # [s]
        min_t = np.max([t_man[0],t_aut[0]]) # Min time [s]
        max_t = np.min([t_man[-1],t_aut[-1]]) # Max time [s]
        # Relate min/max times to the corresponding indexes:
        i_man = (np.abs(t_man - min_t)).argmin()+1 # Minimum index, man labels
        f_man = (np.abs(t_man - max_t)).argmin()-1 # Maximum index, man labels
        i_aut = (np.abs(t_aut - min_t)).argmin()+1 # Minimum index, aut labels
        f_aut = (np.abs(t_aut - max_t)).argmin()-1 # Maximum index, aut labels  
        delta_i = np.min([f_man-i_man,f_aut-i_aut]) # Minimum index interval
        # Identify relevant position-data and min/max z-values:
        z_man = z_segms_man[i][i_man:i_man+delta_i] # Relevant manual labels [m]
        z_aut = z_segms_aut[i][i_aut:i_aut+delta_i] # Relevant automatic labels [m]  
        z_min = np.min([np.min(z_aut),np.min(z_man)]) # Min z-position label [m]
        z_max = np.max([np.max(z_aut),np.max(z_man)]) # Max z-position label [m]
        # Associate positions to bins (middle values):
        z_man_bin = np.round(z_man-(z_man-binning/2)%binning,4) + binning/2 # Manual labels [m]
        z_aut_bin = np.round(z_aut-(z_aut-binning/2)%binning,4) + binning/2 # Automatic labels [m]
        # Compare z-pos vs time:
        z_comp = z_man-z_aut # z-pos difference [m]
        # Generate current dataframe and update the general one:
        Z_err_segm.append(pd.DataFrame({
            'z_man': z_man, 'z_aut': z_aut, # Label positions [m]
            'bin_mid_man': z_man_bin, 'bin_mid_aut': z_aut_bin, # Middle bin positions [m]
            'err_abs': np.abs(z_comp), # Absolute error [m]
            tol_str: np.abs(z_comp)>err_tol # True/False Tolerance [%]
            }))
    # Concatenate all segments' results:    
    Z_err = Z_err_segm[0]
    for i in range(1,len(Z_err_segm)):
        Z_err = pd.concat([Z_err, Z_err_segm[i]], ignore_index=True)
        
    # Prepare performance results:
    mid_bin_vals = list(set(Z_err['bin_mid_man'])) # [m]
    bin_err_abs, bin_tol = [], [] # Initiate auxiliar variables [m], [%]
    for val in mid_bin_vals:
        cond = Z_err['bin_mid_man']==val # Condition for current bin
        bin_err_abs.append(np.mean(Z_err[cond]['err_abs'])) # Mean |z_aut-z_man| [m]
        bin_tol.append(int(100-sum(Z_err[cond][tol_str])/len(Z_err[cond])*100)) # Mean accuracy [%]
    perf = pd.DataFrame({
        "mid_bin": mid_bin_vals,
        'err_abs': bin_err_abs,
        tol_str: bin_tol
    })

    # Prepare confusion matrix, rows are manual labels, columns are automatic labels:
    conf = np.zeros((len(mid_bin_vals),len(mid_bin_vals))) # Initiate
    for i in range(len(mid_bin_vals)):
        bin_man = mid_bin_vals[i] # Manual label mid-value-bin [m]
        df_bin = Z_err[Z_err['bin_mid_man']==bin_man] # Dataframe restricted to current bin
        N_man = len(df_bin) # Total number of manual labels in current bin
        for j in range(len(mid_bin_vals)):
            bin_aut = mid_bin_vals[j] # Automatic label mid-value-bin [m]
            conf[i,j] = len(df_bin[df_bin['bin_mid_aut']==bin_aut])/N_man # [%]
    conf = (conf*100).astype(int) # Turn into [%] and round to integer values
   
    # Prepare figure:
    fig, (ax_perf, ax_conf) = plt.subplots(1,2,figsize=(8,4))
    floors_h = 4 # Reference marking for levels [m]

    ### Plot Performance:
    bars = ax_perf.bar(perf['mid_bin'],perf[tol_str]) # Main plot
    # Adjust gradient colors:
    rg_cmap = LinearSegmentedColormap.from_list('rg',["r", "y", "g"], N=101) # Green-Red gradient     
    y_min, y_max = ax_perf.get_ylim()
    grad = np.atleast_2d(np.linspace(0, 1, 256)).T
    ax_perf = bars[0].axes  # axis handle
    lim = ax_perf.get_xlim()+ax_perf.get_ylim()
    for bar in bars:
        bar.set_zorder(1)  # put the bars in front
        bar.set_facecolor("none")  # make the bars transparent
        x, _ = bar.get_xy()  # get the corners
        w, h = bar.get_width(), bar.get_height()  # get the width and height    
        c_map = ML_general.truncate_colormap(rg_cmap, min_val=0,
                                  max_val=(h-y_min)/(y_max-y_min)) # Define a new color map.    
        ax_perf.imshow(grad, extent=[x, x+w, h, y_min], aspect="auto", zorder=0,
                       cmap=c_map) # Let the imshow only use part of the color map
    ax_perf.axis(lim)
    ax_perf.set(xlabel='Manual label position [m]',ylabel='Automatic labeling accuracy [%]')
    ax_perf.set_xticks(np.arange(0, max(mid_bin_vals)+0.1, floors_h).astype(int))
    ax_perf.set_title('Agreement')
    
    # Plot Confusion matrix:
    wg_cmap = LinearSegmentedColormap.from_list('wg',["w", "y", "g"], N=256) # Green-Red gradient        
    sns.heatmap(conf, annot=True, annot_kws={"size": 6}, cmap=wg_cmap, ax=ax_conf,
                cbar_kws={'label': 'Labels distribution [%]'})
    if binning == 1:
        ticks_pos = np.arange(0, conf.shape[0], floors_h)+0.5
        ticks_labels = (np.arange(0, conf.shape[0], floors_h)).astype(int)
    else:
        ticks_pos = np.arange(0, conf.shape[0], binning)+0.5
        ticks_labels = (np.arange(0, conf.shape[0], binning)*binning+0.5).astype(int)
    ax_conf.set_xticks(ticks_pos), ax_conf.set_yticks(ticks_pos)
    ax_conf.set_xticklabels(ticks_labels), ax_conf.set_yticklabels(ticks_labels)
    ax_conf.set(xlabel='Automatic labels [m]', ylabel='Manual labels [m]')
    ax_conf.set_title('Confusion matrix')

    plt.suptitle(f'Binning {binning}m ; Tolerance {err_tol}m')
    fig.tight_layout();
    if savefig:
        ML_general.save_file(savefig,save_format=save_format)  

# ============================================================================

def show_parking_mismatch(
    t_segms_man,
    z_segms_man,
    t_segms_aut,
    z_segms_aut
    ):
    """
    Plots the z-position vs time where there is a mismatch in parking levels between the Automatic and
    Manual methods. The mismatch is identified by an exact difference in the z-positions equal to the height
    of the floors.
    Note: The distance between levels 0-1 is 4.1m. The distance for any other pair of adjacent levels is 3.7m.
    
    --- Inputs ---
    
    {t_segms_man} [List]: each element is a Numpy array, representing the time vector for the 
    manual labels. Units: [s].
    {t_segms_aut} [List]: each element is a Numpy array, representing the time vector for the
    automatic labels. Units: [s].
    {z_segms_man} [List]: each element is a Numpy array, representing the Z-position for the 
    manual labels. Units: [m].
    {z_segms_aut} [List]: each element is a Numpy array, representing the Z-position for the
    automatic labels. Units: [m].

    """
    for i in range(len(t_segms_man)):
        t_man, t_aut = t_segms_man[i], t_segms_aut[i] # Times [s]
        z_man, z_aut = z_segms_man[i], z_segms_aut[i] # z-Positions [m]
        # Find minimum and maximum times for comparison:
        min_time = np.max([t_man[0],t_aut[0]]) # Min time [s]
        max_time = np.min([t_man[-1],t_aut[-1]]) # Max time [s]
        # Relate them to the corresponding indexes:
        i_man = (np.abs(t_man - min_time)).argmin()+1 # Minimum index, man labels
        f_man = (np.abs(t_man - max_time)).argmin()-1 # Maximum index, man labels
        i_aut = (np.abs(t_aut - min_time)).argmin()+1 # Minimum index, automatic labels
        f_aut = (np.abs(t_aut - max_time)).argmin()-1 # Maximum index, automatic labels
        delta_i = np.min([f_man-i_man,f_aut-i_aut]) # Minimum index interval
        # Compare z-pos vs time and extract indexes for mismatched parking positions:
        t_comp = t_man[i_man:i_man+delta_i] # Time vector [s]
        z_comp = z_man[i_man:i_man+delta_i]-z_aut[i_aut:i_aut+delta_i] # z-pos difference [m]
        idx = (abs(z_comp) == 4.1) + (abs(z_comp) == 3.7) # Indexes for mismatched parking labels
        if sum(idx):
            fig = plt.figure(figsize=(8,2))
            plt.plot(t_comp[idx],z_comp[idx],'o',markersize=1)
            plt.title(f'Parking levels mistach in Segment {i+1}/{len(t_segms_man)}')
            fig.tight_layout()
        else:
            print(f'No parking levels mismatch in Segment {i+1}/{len(t_segms_man)}')

# ============================================================================

def correct_parking_mismatch(
    t_segms_man,
    z_segms_man,
    t_segms_aut,
    z_segms_aut,
    mismatch_thres=5):
    """
    Corrects the z-positions vs time in which there is a mismatch in parking levels between the Automatic and
    Manual methods. The mismatch is identified by an exact difference in the z-positions equal to the height
    of the floors. The corrections are carried out in the mismatched parking levels and also in the neighbouring
    traveling regions. The corrections are always LINEAR.
    Note: The distance between levels 0-1 is 4.1m. The distance for any other pair of adjacent levels is 3.7m.
    
    --- Inputs ---
    
    {t_segms_man} [List]: each element is a Numpy array, representing the time vector for the 
    manual labels. Units: [s].
    {t_segms_aut} [List]: each element is a Numpy array, representing the time vector for the
    automatic labels. Units: [s].
    {z_segms_man} [List]: each element is a Numpy array, representing the Z-position for the 
    manual labels. Units: [m].
    {z_segms_aut} [List]: each element is a Numpy array, representing the Z-position for the
    automatic labels. Units: [m].
    {mismatch_thres} [Float]: threshold for mismatching time in identifying traveling events. Units: [s].

    --- Outputs ---
    
    {z_segms_superv_aut} [List]: each element is a Numpy array, representing the Z-position for the 
    automatic labels including corrections. Units: [m].
    
    """

    z_segms_aut_superv = [] # Initiate
    for i_segm in range(len(t_segms_man)):
        t_man, t_aut = t_segms_man[i_segm], t_segms_aut[i_segm] # Times [s]
        z_man, z_aut = z_segms_man[i_segm], z_segms_aut[i_segm] # z-Positions [m]
        z_superv_aut = np.copy(z_aut) # Copy z_aut [m] 
        # Find minimum and maximum times for comparison:
        min_time = np.max([t_man[0],t_aut[0]]) # Min time [s]
        max_time = np.min([t_man[-1],t_aut[-1]]) # Max time [s]
        # Relate them to the corresponding indexes:
        i_man = (np.abs(t_man - min_time)).argmin()+1 # Minimum index, man labels
        f_man = (np.abs(t_man - max_time)).argmin()-1 # Maximum index, man labels
        i_aut = (np.abs(t_aut - min_time)).argmin()+1 # Minimum index, automatic labels
        f_aut = (np.abs(t_aut - max_time)).argmin()-1 # Maximum index, automatic labels
        delta_i = np.min([f_man-i_man,f_aut-i_aut]) # Minimum index interval
        # Compare z-pos vs time and extract indexes for mismatched parking positions:
        t_comp = t_man[i_man:i_man+delta_i] # Time vector [s]
        z_comp = z_man[i_man:i_man+delta_i]-z_aut[i_aut:i_aut+delta_i] # z-pos difference [m]
        idx = (abs(z_comp) == 4.1) + (abs(z_comp) == 3.7)
        # Identify individual mismatched parking regions and adjacent traveling regions:
        idx_flags = np.diff(idx).nonzero()[0]+1 # Indexes in which a True region in {idx} starts
        for i_idx in range(0,len(idx_flags)-1,2):
            # Automatic labels:            
            park_i_aut, park_f_aut = i_aut+idx_flags[i_idx], i_aut+idx_flags[i_idx+1] # Indexes for parking region:
            trav_f_aut = park_i_aut-1 # Initiate index for last point in traveling event
            z = z_aut[trav_f_aut] # Identify position at the auxiliar index
            while z == z_aut[park_i_aut]: # Look until z_pos changes (traveling)
                trav_f_aut -= 1 # Update
                z = z_aut[trav_f_aut]
            trav_i_aut = trav_f_aut-1 # Initiate index for first point in traveling event
            z = z_aut[trav_i_aut]
            while z != z_aut[trav_i_aut+1]: # Look until z_pos remains stable (parking)
                trav_i_aut -= 1 # Update
                z = z_aut[trav_i_aut]
            # Manual labels:
            park_i_man, park_f_man = i_man+idx_flags[i_idx], i_man+idx_flags[i_idx+1] # Manual labels
            trav_f_man = park_i_man-1 # Initiate index for last point in traveling event
            z = z_man[trav_f_man] # Identify position at the auxiliar index
            while z == z_man[park_i_man]: # Look until z_pos changes (traveling)
                trav_f_man -= 1 # Update
                z = z_man[trav_f_man]
            trav_i_man = trav_f_man-1 # Initiate index for first point in traveling event
            z = z_man[trav_i_man]
            while z != z_man[trav_i_man+1]: # Look until z_pos remains stable (parking)
                trav_i_man -= 1 # Update
                z = z_man[trav_i_man]
            # Correct Automatic labels: traveling event is ok, but parking level is wrong
            dt = t_man[trav_f_man]-t_aut[trav_f_aut] # Event mismatch [s]
            if abs(dt)<mismatch_thres: # Less than 5 seconds mismatch means the traveling was well identified
                # Correct parking segment:
                z_superv_aut[trav_f_aut+1:park_f_aut] += z_man[i_man+idx_flags[i_idx]]-z_aut[i_aut+idx_flags[i_idx]]
                # Correct traveling event:
                replace = np.linspace(z_aut[trav_i_aut],z_superv_aut[park_i_aut],trav_f_aut-trav_i_aut+1)
                z_superv_aut[trav_i_aut:trav_f_aut+1] = replace
            # Correct Automatic labels: traveling event is missing
            else:
                z_superv_aut[trav_i_man-i_man+i_aut:i_aut+idx_flags[i_idx+1]] = z_man[trav_i_man:park_f_man]
            fig = plt.figure(figsize=(4,3))
            plt.plot(t_man[trav_i_man:park_f_man],z_man[trav_i_man:park_f_man],
                     '--k',label='Manual',alpha=0.6,lw=2)
            plt.plot(t_aut[trav_i_aut:park_f_aut],z_aut[trav_i_aut:park_f_aut],
                     '-r',label='Aut - Orig',alpha=0.6,lw=1)
            plt.plot(t_aut[trav_i_aut:park_f_aut],z_superv_aut[trav_i_aut:park_f_aut],
                     '-g',label='Aut - Superv',alpha=0.8,lw=1)            
            plt.xlabel('Time [s]'), plt.ylabel('Position [m]')
            plt.legend()
            fig.tight_layout()
        z_segms_aut_superv.append(z_superv_aut)
    return z_segms_aut_superv

# ============================================================================

def export_predictors_targets(
    data_segms,
    t_segms_aut,
    z_segms_aut,
    z_segms_aut_superv,
    path='.',
    prefix='None',
    ):
    """
    """
    # Prepare data:
    all_df = [] # Initiate
    base_name = f'{prefix}_data_zAut' if prefix is not None else 'data_zAut' # Base name 
    dt = np.round(t_segms_aut[0][1]-t_segms_aut[0][0],4) # Time resolution [s]
    
    # Generate a dataframe for each segment including all relevant data:
    for i,data in enumerate(data_segms):
        name = base_name + f'_segm{i+1}'
        t_aut = t_segms_aut[i] # Post-processed [s]
        z_aut = z_segms_aut[i] # Automatic labels [m]
        # Identify the common data (exclude times in which not all information is available):
        i1 = round((t_aut[0]-data[0,0])/dt) # Minimum index for common data
        i2 = round((data[-1,0]-t_aut[-1])/dt) # Distance to last index, for common data
        t, Bx, By, Bz = data[i1:-i2,0],data[i1:-i2,1],data[i1:-i2,2],data[i1:-i2,3] # [s,nT,nT,nT]
        # Write dataframe:
        if i<len(z_segms_aut_superv): 
            df = pd.DataFrame(columns=['Time_s','Bx_nT','By_nT','Bz_nT','zAut_m','zTrue_m']) # Initiate        
            df['zTrue_m'] = z_segms_aut_superv[i] # [m]
        else:
            df = pd.DataFrame(columns=['Time_s','Bx_nT','By_nT','Bz_nT','zAut_m']) # Initiate
        df['Time_s'] = t_aut # [s]
        df['Bx_nT'], df['By_nT'], df['Bz_nT'] = Bx, By, Bz # [nT]
        df['zAut_m'] = z_aut # [m]
        all_df.append(df)
        df.to_csv(f'{path}/{name}.csv', index=False) # Export results

# ============================================================================

def plot_all_data_elevator_position(
    input_path,
    keyword=None,
    header=0,
    sep=',',
    # time,
    # position,
    res_bin=0.1,
    levels=np.append(0,4.1+np.arange(0,3.7*6+0.1,3.7)),
    label='Position data',
    color='#00520fcc',
    save_name=None,
    save_format='png',
    figsize=(8,3)
    ):
    """
    Using information recorded in the input files, plot the elevator position as a function
    of time and its histogram. 

    --- Inputs ---

    {input_path} [String]: Path for elevator position files. Each file must have a time column titled
    'Time_s' and either a position column titled 'zTrue_m' or 'zAut_m' (or both).
    {keyword} [String or None]: If provided, only accept files that include the {keyword} in their names. 
    {header} [Integer]: Number of header lines in the magnetic files.
    {sep} [String]: Character or regex pattern to treat as the delimiter in the magnetic files.


    {time} [Numpy array]: Time in units [s].
    {position} [Numpy array]: Position in units [s].


    {res_bin} [Float]: width of each bin for the elevator position histogram. Units [m].
    {levels} [Numpy array]: position of the parking levels. Units [m]. 
    {label} [String]: Text for the legend's label representing the magnetic data.
    {color} [Matplotlib color]: color for the curves in the plots.
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    Plots organized in 2 columns:
    Left column: timeseries for elevator position data.
    Right column: histogram for elevator positions, in y-logarithmic scale.

    """

    # Read data:
    files = sorted([file for file in os.listdir(input_path) if keyword in file])
    all_df = [pd.read_csv(input_path+file,header=header,sep=sep) for file in files]

    # Prepare data:
    t, z = np.zeros(0), np.zeros(0) # [s], [m]
    for df in all_df:
        t = np.append(t,df['Time_s']) # Update time [s]
        if 'zTrue_m' in df:
            z = np.append(z,df['zTrue_m']) # Update position [m]
        else:
            z = np.append(z,df['zAut_m']) # Update position [m]
    bins = round((np.max(z)-np.min(z))/res_bin) # Number of bins for histogram, based on Bx resolution

    # Prepare figure:
    fig, (ax_timeseries,ax_hist) = plt.subplots(1,2,figsize=figsize)

    # Timeseries:
    ax_timeseries.plot(t/60,z,'-',color=color,lw=0.5,alpha=0.8)   
    ax_timeseries.set(xlabel='Time [min]',ylabel="Car position [m]")

    # Histogram:
    ax_hist.hist(z,bins=bins,color=color,lw=0.5,alpha=0.8,label=label)
    ax_hist.set(xlabel="Car position [m]",ylabel='Counts [log]')
    # Levels:
    for level in levels:
        ax_timeseries.axhline(level,color='gray',ls='--',lw=0.5,alpha=0.8) 
    ax_hist.set_yscale('log')
    ax_hist.legend()

    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format)  

# ============================================================================

class accel_data:
    """
    Object containing information about the acceleration experimental data for the elevator.

    --- Attributes ---

    {self.name} [String]: Idenfitier.
    {self.comment} [String or None]: Comment about the data.
    {self.time} [Numpy array]: timeseries for Time, units [s].
    {self.ax} [Numpy array]: timeseries for Acceleration along X-direction, units [m/s^2].
    {self.ay} [Numpy array]: timeseries for Acceleration along Y-direction, units [m/s^2].
    {self.az} [Numpy array]: timeseries for Acceleration along Z-direction, units [m/s^2].
    {self.a} [Numpy array]: timeseries for Acceleration magnitude, units [m/s^2].
    {self.events} [List]: each element is a list indicating the [initial,final] parking level.
    The order of the events must be correlated with the acceleration data.
    {self.event_stats} [Counter]: Count the number of flights sharing the same number of
    travelled levels, direction matters.
    {self,events_times} [List]: each element is a list indicating the [initial,final] indexes for
    the traveling events. The order of the events must be correlated with the acceleration data.
    """

    # =============

    def __init__(
        self,
        name,
        path,
        sep='\t',
        header=0,
        columns=['Time (s)',
        'Linear Acceleration x (m/s^2)',
        'Linear Acceleration y (m/s^2)',
        'Linear Acceleration z (m/s^2)',
        'Absolute acceleration (m/s^2)'],
        comment=None,
        ):
        """
        Initiate accel_data object with basic information and load the data.

        --- Inputs ---

        {name} [String]: Identifier.
        {path} [String]: Path for the .csv file.
        {sep} [String]: Character or regex pattern to treat as the delimiter.
        {header} [Integer, Sequence of Int., ‘infer’ or None]: Row number(s) 
        containing column labels and marking the start of the data (zero-indexed).
        {columns} [List]: Each element is the name for the column containing information
        about time [units s], acceleration along X axis (a_x), a_y, a_z, and absolute
        acceleration [units m/s^2], in that order.
        {comment} [String or None]: Comment about the data.

        --- Return ---
        
        Create the accel_data object with the following attributes:
        <self.name>, <self.comment>, <self.time>, <self.ax>, <self.ay>, <self.az>, <self.a>

        """

        # Basic information:
        self.name = name
        self.comment = comment

        # Load data and redefine relevant column names:
        data = pd.read_csv(path,sep=sep,header=header)
        new_names_cols = {
            columns[0]:'time_s', # [s]
            columns[1]:'ax_mpers2', # [m/s^2]
            columns[2]:'ay_mpers2', # [m/s^2]
            columns[3]:'az_mpers2', # [m/s^2]
            columns[4]:'a_mpers2' # [m/s^2]
        }
        data = data.rename(columns=new_names_cols) # Change column names

        # Update attributes with data information:
        self.time = data['time_s'].to_numpy() # [s]
        self.ax = data['ax_mpers2'].to_numpy() # [m/s^2]
        self.ay = data['ay_mpers2'].to_numpy() # [m/s^2]
        self.az = data['az_mpers2'].to_numpy() # [m/s^2]
        self.a = data['a_mpers2'].to_numpy() # [m/s^2]

    # =============

    def info(
        self
        ):
        """
        Print basic information (attributes) about self.
        """
        print(
            '-'*10,' accel_data object ','-'*10,
            colored(f'\nName: {self.name}','green'),
            '\nComments:',self.comment,
            )
        print(f'Recording duration: {self.time[-1]:.2f} s')

        if 'events' in dir(self):
            print(f'Available information for {len(self.events)} traveling events, stats for travelled levels:')
            for (flight,counts) in sorted(self.event_stats.items()):
                print(f'   {flight} levels flight: {counts} counts.')
        else:
            print(colored('No information available for traveling events.','red'))

        if 'events_times' in dir(self):
            print('There is available information for the timing of each event.')
        else:
            print(colored('There is NOT available information for the timing of each event.','red'))

    # =============

    def add_events_info(
        self,
        events
        ):
        """
        Adds information about the traveling events.
        
        --- Inputs ---

        {events} [List]: each element is a list indicating the [initial,final] parking level,
        ranging from 0 to 7. The order of the events must be correlated with the acceleration
        data.

        --- Return ---
        
        Updates the <self.events> and <self.event_stats> attribute.

        """

        # Update information and then calculate the statistics for traveling events:
        self.events = events
        stats = [event[0]-event[1] for event in events] # Number of travelled levels
        stats = Counter(stats) # Update to a dictionary
        self.event_stats = stats

    # =============

    def add_events_times(
        self,
        events_times
        ):
        """
        Adds information about the timing for traveling events.
        
        --- Inputs ---

        {events_times} [List]: each element is a list indicating the [initial,final] indexes for
        the traveling events. The order of the events must be correlated with the acceleration data.

        --- Return ---
        
        Updates the <self.events_times> attribute.

        """

        # Update information and then calculate the statistics for traveling events:
        self.events_times = events_times

    # =============

    def plot_raw_data(
        self,
        comps=['ax','ay','az'],
        i_range=[0,-1],
        save_name=None,
        save_format='png',    
        figsize=(6,3),
        label_font=4
        ):
        """
        Plot the raw data: acceleration vs time.
        
       --- Inputs ---

        {comps} [List]: Choose which components are going to be plot altogether in the same
        graph. Options are 'ax', 'ay', 'az' and 'a'.
        {i_range} [List]: First and second elements are the index boundaries for the plot.
        {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
        {save_format} [String]: saving format for the figure,don't include the dot.
        {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.
        {label_font} [Float]: Font size for the flights labels, only if available.

        --- Return ---
        
        Plot for acceleration(s) as a function of time. If there is available information about the
        events, the travelling events are highlighted and the number of travelled flights are labeled.

        """

        # Copy index range to avoid future issues and turn negative indexes into positive:
        i_range_ = copy.copy(i_range) 
        for i,index in enumerate(i_range_):
            if index<0:
                i_range_[i] = len(self.time)+index # Update range 
        # Prepare data:
        time = self.time[i_range_[0]:i_range_[1]] # [s]
        data = {'ax':self.ax[i_range_[0]:i_range_[1]],
                'ay':self.ay[i_range_[0]:i_range_[1]],
                'az':self.az[i_range_[0]:i_range_[1]],
                'a':self.a[i_range_[0]:i_range_[1]]} # [m/s^2]

        colors = plt.cm.viridis(np.linspace(0, 1, 4))
        a_colors = {'ax':colors[0], 'ay':colors[1], 'az':colors[2], 'a':colors[3]}

        # Plot main figure:
        fig = plt.figure(figsize=figsize)
        max_a = 0. # Initiate maximum acceleration in the plot, useful later
        for comp in comps:
            plt.plot(self.time[i_range_[0]:i_range_[1]], data[comp],
            label=comp, color=a_colors[comp], alpha=0.8)
            max_a = np.max([max_a,np.max(data[comp])]) # Update maximum value [m/s^2]
        # Add events information if available:
        if 'events_times' in dir(self):
            label = 'Traveling'
            for i,event in enumerate(self.events_times):
                if (event[0] >= i_range_[0]) and (event[1] <= i_range_[1]):
                    plt.axvspan(self.time[event[0]], self.time[event[1]],
                        alpha=0.2, color='yellow',label=label)
                    label = None # Avoid repeating labels
                # Add flight label (how many levels during travel) if available:
                if 'events' in dir(self):
                    flight = self.events[i][1]-self.events[i][0] # Number of travelled levels
                    color = 'green' if flight>0 else 'red' # Set color (green >0, else red)
                    flight = '+'+str(flight) if flight>0 else str(flight) # Format to string
                    plt.annotate(flight, xy=(self.time[event[0]],max_a*1.01),
                        color=color,fontsize=label_font) # Annotate
        # Additional configuration:
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s$^2$]')
        plt.title(self.name)
        plt.ylim([plt.gca().get_ylim()[0],max_a*1.2])
        fig.tight_layout();
        if save_name:
            ML_general.save_file(save_name,save_format=save_format) 

            # =============

    def compare_same_flight_type(
        self,
        comp='az',
        N_bins=100,
        a_thres=None,
        save_name=None,
        save_format='png',    
        figsize=(6,3),
        ):
        """
        Xxxx
        
       --- Inputs ---

        {comp} [String]: Choose which acceleration component is going to be plot, options
        are 'ax', 'ay', 'az' and 'a'.
        {N_bins} [Integer]: Number of bins for the histogram.
        {a_thres} [Float or None]: If provided, identifies the most frequented value in the histograms for
        both positive and negative acceleration, excluding the interval [-a_thres,a_thres], units [m/s^2].
        {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
        {save_format} [String]: saving format for the figure,don't include the dot.
        {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

        --- Return ---
        
        Xxxx

        """

        colors = plt.cm.viridis(np.linspace(0, 1, 4)) # Auxiliar definition
        a_colors = {'ax':colors[0], 'ay':colors[1], 'az':colors[2], 'a':colors[3]}
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.events_times))) # Update definition
        if 'events' not in dir(self):
            print('You need to load information about the events firts.')
            return None
        else:
            flights = set([event[1]-event[0] for event in self.events]) # Collect all type of flights
            # Plot one figure for each flight type comparing all events:
            for flight in flights:
                valid_events = [i for i in range(len(self.events)) if self.events[i][1]-self.events[i][0]==flight]
                fig, (ax_timeseries,ax_hist) = plt.subplots(1,2,figsize=figsize)

                # Plot timeseries:
                hist_data = np.zeros(0) # Initiate [m/s^2]
                for i,event in enumerate(list(self.events_times[i] for i in valid_events)):
                    # Prepare data:
                    time = self.time[event[0]:event[1]]-self.time[event[0]] # Starts from zero [s]
                    data = {'ax':self.ax[event[0]:event[1]],
                            'ay':self.ay[event[0]:event[1]],
                            'az':self.az[event[0]:event[1]],
                            'a':self.a[event[0]:event[1]]} # [m/s^2]
                    hist_data = np.append(hist_data,data[comp]) # Save acceleration data for histogram [m/s^2]
                    #label = comp if i==0 else None # Set acceleration component label
                    ax_timeseries.plot(time, data[comp], color=colors[i], alpha=0.8, label=i+1)

                # Plot histograms:
                hist = ax_hist.hist(hist_data,bins=N_bins,label=comp,color=a_colors[comp],alpha=0.8)
                if a_thres:
                    bin_dist = hist[1][1]-hist[1][0] # Bins' separation [m/s^2]
                    # Prepare positive acceleration data:
                    counts_pos = [hist[0][i] for i in range(len(hist[0])) if hist[1][i]>a_thres]
                    edges_pos = [hist[1][i] for i in range(len(hist[1])) if hist[1][i]>a_thres]
                    a_mode_pos = edges_pos[np.argmax(counts_pos)] # Most frequent positive acceleration [m/s^2]
                    # Prepare negative acceleration data:
                    counts_neg = [hist[0][i] for i in range(len(hist[0])) if hist[1][i]<-a_thres]
                    edges_neg = [hist[1][i] for i in range(len(hist[1])) if hist[1][i]<-a_thres]
                    a_mode_neg = edges_neg[np.argmax(counts_neg)] # Most frequent positive acceleration [m/s^2]
                    # Plot vertical lines and values for most frequent positive and negative accelerations:
                    for a in [a_mode_pos,a_mode_neg]:
                        ax_hist.axvline(a+bin_dist/2,ls='--',lw=0.5,color='red')
                        ax_hist.annotate(f'{a:.2f} m/s$^2$', xy=(a+bin_dist/2,ax_hist.get_ylim()[1]*0.3),
                            color='red',rotation=90) # Annotate

                # Additional configuration:
                ax_timeseries.legend(title=f"{comp} events",fontsize=6,ncol=2)
                ax_timeseries.set(xlabel='Time [s]',ylabel='Acceleration [m/s$^2$]')
                ax_hist.set(xlabel=f'Acceleration [m/s$^2$]',ylabel='Counts')
                fig.suptitle(f'{self.name} ; {flight}-level flights, {len(valid_events)} events')
                fig.tight_layout()
                if save_name:
                    ML_general.save_file(f'{save_name}_{flight}-level_flights',save_format=save_format)    



