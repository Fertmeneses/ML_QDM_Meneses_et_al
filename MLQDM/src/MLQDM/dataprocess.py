"""
Data processing: acceleration measurements.

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



