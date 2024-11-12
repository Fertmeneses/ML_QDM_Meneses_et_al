"""
Physical model for elevator's movement.

# ===========

Functions included in Class <elev_motion_model>:

    - __init__
    - info
    - model_a_from_tV
    - model_v_from_tV
    - model_x_from_tV
    - gen_motion_from_tV
    - compare_totaltime_with_exp
    - fit_amax_deltat_tV_for_exp

"""

# ============================================================================

# Required packages:

from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

class elev_motion_model:
    """
    Object containing information about XXXX.

    --- Attributes ---
    {self.name} [String]: Idenfitier.
    {self.max_accel} [Float]: Maximum Z-acceleration in either direction [m/s^2].
    {self.delta_t} [Float]: Interval's duration for accelerating from 0 to +-{max_accel} or vice-versa,
    valid for N-level flights, N>1. Units [s].
    {self.delta_tprime} [Float]: Interval's duration for accelerating from 0 to +-{max_accel} or vice-versa,
    valid for 1-level flights. Units [s].
    {self.time_res} [Float]: Time resolution for the output timeseries.
    <self.pp_delta_t> [Integer]: Number of points for the Delta t intervals, according to the time resolution.
    <self.pp_delta_tprime> [Integer]: Number of points for the Delta t' intervals, according to the time resolution.
    <self.a_trip> [List]: The first and second elements are Numpy arrays describing the time [s] and
    Z-acceleration [m/s^2] timeseries, respectively.
    <self.v_trip> [List]: The first and second elements are Numpy arrays describing the time [s] and
    Z-velocity [m/s] timeseries, respectively.
    <self.x_trip> [List]: The first and second elements are Numpy arrays describing the time [s] and
    Z-position [m] timeseries, respectively.
    <self.h> [Float]: Uniform separation between adjacent levels.
    Note: Make sure that you choose {h} to match the assumption in the
    experimental results {total_time_exp}.
    <self.N_lvls> [Integer]: Number of levels in the building.

    All the following dictionaries have the number of travelled levels as keys:

    <self.model_tV_theory> [Dictionary]: Values are the calculated time intervals
    t_V traveling at terminal velocity, according to <self.h>, units [s].
    <self.model_totaltime_theory> [Dictionary]: Values are the calculated total
    times t_total for the trip, according to <self.h>, units [s].
    <self.model_amax_exp> [Dictionary]: Values are the fitted maximum acceleration, 
    according to experimental results, units [m/s^2].
    <self.model_deltat_exp> [Dictionary]: Values are the fitted interval times for
    linear acceleration, according to experimental results, units [s].
    <self.model_tV_exp> [Dictionary]: Values are the fitted interval times
    traveling at terminal velocity, according to experimental results, units [s].

    """

    # =============

    def __init__(
        self,
        name=None,
        max_accel=0.87,
        delta_t=1.0,
        delta_tprime=0.6,
        time_res=0.1,
        h=3.76,
        N_lvls=8
        ):
        """
        Initiate elev_motion_model object with the basic model parameters.

        --- Inputs ---

        {name} [String or None]: Identifier.
        {max_accel} [Float]: Maximum Z-acceleration in either direction [m/s^2].
        {delta_t} [Float]: Interval's duration for accelerating from 0 to +-{max_accel} or vice-versa,
        valid for N-level flights, N>1. Units [s].
        {delta_tprime} [Float]: Interval's duration for accelerating from 0 to +-{max_accel} or vice-versa,
        valid for 1-level flights. Units [s].
        {time_res} [Float]: Time resolution for the output timeseries.
        {h} [Float]: Uniform separation between adjacent levels.
        Note: Make sure that you choose {h} to match the assumption in the
        experimental results {total_time_exp}.
        {N_lvls} [Integer]: Number of levels in the building.

        --- Return ---
        
        Create the accel_data object and make theoretical fittings.

        """

        # Identification information and basic model's parameters:
        self.name = name
        self.max_accel = max_accel # [m/s^2]
        self.delta_t = delta_t # [s]
        self.delta_tprime = delta_tprime # [s]
        self.time_res = time_res # [s]
        self.h = h # [m]
        self.N_lvls = N_lvls # [levels]

        # Prepare time discretization:
        pp_delta_t = round(self.delta_t/self.time_res,5)
        pp_delta_tprime = round(self.delta_tprime/self.time_res,5)
        # Trigger error if required:
        if not pp_delta_t.is_integer() or not pp_delta_tprime.is_integer():
            raise ZeroDivisionError('delta_t and delta_t_prime must be both multiple of time_res!')
        # Assign attributes
        self.pp_delta_t = int(pp_delta_t)
        self.pp_delta_tprime = int(pp_delta_tprime)

        # Fit the time intervals traveling at terminal velocity according to the...
        # ...model and the avergage distance between adjacent levels:

        delta_N= np.arange(2,N_lvls) # List for all N-level flights
     
        # Initiate and prepare auxiliar terms:
        model_tV_theory = {}
        model_totaltime_theory = {}
        term_1 = 35/6*max_accel*(delta_t)**2 # Term in numerator [m]
        term_2 = 2*max_accel*delta_t # Denominator [m/s]

        # Calculate t_V for each N-level flight:
        model_tV_theory[1] = 0 # [s]
        model_totaltime_theory[1] = 4*delta_t+2*delta_tprime # [s]
        for N in delta_N:
            model_tV_theory[N] = np.round((N*h-term_1)/term_2,2) # [s]
            model_totaltime_theory[N] = np.round(6*delta_t+model_tV_theory[N],2) # [s]

        # Update attributes:
        self.model_tV_theory = model_tV_theory
        self.model_totaltime_theory = model_totaltime_theory

    # =============

    def info(
        self
        ):
        """
        Print basic information (attributes) about self.
        """
        print(
            '-'*10,' elev_motion_model object ','-'*10,
            colored(f'\nName: {self.name}','green')
            )
        print('\n----- Basic information -----')
        print(f'Maximum Z-acceleration (either direction) {self.max_accel} m/s^2.')
        print(f'Time to accelerate from 0 to max. accel. or vice-versa in N-level flights, N>1: {self.delta_t} s.')
        print(f'Time to accelerate from 0 to max. accel. or vice-versa in 1-level flights: {self.delta_tprime} s.')
        print(f'Time resolution for generated data {self.time_res} m.')
        print(f'Average separation between adjacent levels: {self.h}m.')
        print(f'Number of levels in the bulding: {self.N_lvls}')

        print('\n----- Theoretical fitting -----')
        print('Total trip duration only adjusting t_V:')
        display(pd.DataFrame(data={
            'N-level flight':list(self.model_tV_theory.keys()),
            'Total time [s]': list(self.model_totaltime_theory.values()),
            'Traveling time at tV [s]': list(self.model_tV_theory.values())
            }))

        if 'model_amax_exp' in dir(self):
            print('\n----- Experimental fitting -----')
            print('Parameters to match experimental results:')
            display(pd.DataFrame(data={
                'N-level flight':list(self.model_amax_exp.keys()),
                'Max. accel. [m/s^2]': list(self.model_amax_exp.values()),
                'Lin. accel. time [s]': list(self.model_deltat_exp.values()),
                'Term. Vel. time [s]': list(self.model_tV_exp.values())
                }))

    # =============

    def model_a_from_tV(
        self,
        t_V,
        going_up=True
        ):
        """
        Calculate the acceleration timeseries for the given traveling time at terminal velocity {t_V}
        (t_V=0 means traveling only one level), according to the physical model.

        --- Inputs ---

        {t_V} [Float]: Duration for the interval traveling at constant terminal velocity, units [s].
        If only traveling a single level, then choose t_V=0.
        {going_up} [Boolean]: If True, the elevator travels upwards. If False, it travels downwards.

        --- Return ---

        Generate or update:
        <self.a_trip> [List]: The first and second elements are Numpy arrays describing the
        time and Z-acceleration timeseries, respectively.

        """
        # Set general parameters:
        i = 1 if going_up else -1 # Traveling direction (upwards is positive)
        amax = i*self.max_accel # Maximum acceleration in traveling direction [m/s^2]
        Dt = self.delta_t # Delta t interval length [s]
        Dtp = self.delta_tprime # Delta t' interval length [s]
        pp_t = self.pp_delta_t # Number of points during Delta t interval
        pp_tp = self.pp_delta_tprime # Number of points during Delta t' interval
        pp_V = int(t_V/self.time_res) # Number of points traveling at terminal speed
        dt = self.time_res # Time resolution [s]
        
        # Generate acceleration timeseries from physical model:

        # STEP 1 (common), from a=0 to a=amax:
        a1 = np.arange(0,(pp_t-0.1)*dt,dt)/Dt # [adim]

        # STEP 2 (common), constant a=amax:
        a2 = np.ones(pp_t) # [adim]

        # Case (a): N-level flights, with N>1, equivalent to t_V>0:
        if t_V > 0:
            # STEP 3a [N>1], from a=amax to a=0:
            a3_a = (Dt-np.arange(0,(pp_t-0.1)*dt,dt))/Dt # [adim]

            # STEP 4a [N>1], constant a=0:
            a4_a = 0*np.ones(pp_V) # [adim]

            # STEP 5a [N>1], from a=0 to a=-amax:
            a5_a = -np.arange(0,(pp_t-0.1)*dt,dt)/Dt # [adim]

            # STEP 6a [N>1], constant a=-amax:
            a6_a = -np.ones(pp_t) # [adim]

            # STEP 7a [N>1], from a=-amax to a=0:
            a7_a = -(Dt-np.arange(0,(pp_t-0.1)*dt,dt))/Dt # [adim]

            # Concatenate all acceleration steps:
            a = amax*np.concatenate([a1,a2,a3_a,a4_a,a5_a,a6_a,a7_a,np.array([0])]) # [m/s^2]

        # Case (b): 1-level flights, equivalent to t_V=0:
        else:
            # STEP 3b [N=1], a=amax to a=-amax:
            a3_b = (Dtp-np.arange(0,(2*pp_tp-0.1)*dt,dt))/Dtp # [adim]

            # STEP 4b [N=1], constant a=-amax:
            a4_b = -np.ones(pp_t) # [adim]

            # STEP 5b [N=1], from a=-amax to a=0:
            a5_b = -(Dt-np.arange(0,(pp_t-0.1)*dt,dt))/Dt # [adim]

            # Concatenate all acceleration steps:
            a = amax*np.concatenate([a1,a2,a3_b,a4_b,a5_b,np.array([0])]) # [m/s^2]

        # Store the time and acceleration timeseries as attributes:
        t = np.arange(0,(len(a)-0.1)*dt,dt) # Time vector [s]
        self.a_trip = [t,a] # [s],[m/s^2]

    # =============

    def model_v_from_tV(
        self,
        t_V,
        going_up=True
        ):
        """
        Calculate the velocity timeseries for the given traveling time at terminal velocity {t_V}
        (t_V=0 means traveling only one level), according to the physical model.

        --- Inputs ---

        {t_V} [Float]: Duration for the interval traveling at constant terminal velocity, units [s].
        If only traveling a single level, then choose t_V=0.
        {going_up} [Boolean]: If True, the elevator travels upwards. If False, it travels downwards.

        --- Return ---

        Generate or update:
        <self.vel_trip> [List]: The first and second elements are Numpy arrays describing the
        time and Z-velocity timeseries, respectively.

        """
        # Set general parameters:
        i = 1 if going_up else -1 # Traveling direction (upwards is positive)
        amax = i*self.max_accel # Maximum acceleration in traveling direction [m/s^2]
        Dt = self.delta_t # Delta t interval length [s]
        Dtp = self.delta_tprime # Delta t' interval length [s]
        pp_t = self.pp_delta_t # Number of points during Delta t interval
        pp_tp = self.pp_delta_tprime # Number of points during Delta t' interval
        pp_V = int(t_V/self.time_res) # Number of points traveling at terminal speed
        dt = self.time_res # Time resolution [s]
        
        # Generate velocity timeseries from physical model:

        # STEP 1 (common), from a=0 to a=amax:
        v1 = 1/2*np.arange(0,(pp_t-0.1)*dt,dt)**2/Dt # [s]

        # STEP 2 (common), constant a=amax:
        v2 = 1/2*Dt+np.arange(0,(pp_t-0.1)*dt,dt) # [s]

        # Case (a): N-level flights, with N>1, equivalent to t_V>0:
        if t_V > 0:
            # STEP 3a [N>1], from a=amax to a=0:
            v3_a = 2*Dt-1/2*(Dt-np.arange(0,(pp_t-0.1)*dt,dt))**2/Dt # [s]

            # STEP 4a [N>1], constant a=0:
            v4_a = 2*Dt*np.ones(pp_V) # [s]

            # STEP 5a [N>1], from a=0 to a=-amax:
            v5_a = 2*Dt-1/2*np.arange(0,(pp_t-0.1)*dt,dt)**2/Dt # [s]

            # STEP 6a [N>1], constant a=-amax:
            v6_a = 3/2*Dt-np.arange(0,(pp_t-0.1)*dt,dt) # [s]

            # STEP 7a [N>1], from a=-amax to a=0:
            v7_a = 1/2*(Dt-np.arange(0,(pp_t-0.1)*dt,dt))**2/Dt# [s]

            # Concatenate all acceleration steps:
            v = amax*np.concatenate([v1,v2,v3_a,v4_a,v5_a,v6_a,v7_a,np.array([0])]) # [m/s]

        # Case (b): 1-level flights, equivalent to t_V=0:
        else:
            # STEP 3b [N=1], a=amax to a=-amax:
            v3_b = 3/2*Dt+1/2*Dtp-1/2*(Dtp-np.arange(0,(2*pp_tp-0.1)*dt,dt))**2/Dtp # [m/s]

            # STEP 4b [N=1], constant a=-amax:
            v4_b = 3/2*Dt-np.arange(0,(pp_t-0.1)*dt,dt) # [m/s]

            # STEP 5b [N=1], from a=-amax to a=0:
            v5_b = 1/2*(Dt-np.arange(0,(pp_t-0.1)*dt,dt))**2/Dt # [m/s]

            # Concatenate all acceleration steps:
            v = amax*np.concatenate([v1,v2,v3_b,v4_b,v5_b,np.array([0])]) # [m/s]

        # Store the time and acceleration timeseries as attributes:
        t = np.arange(0,(len(v)-0.1)*dt,dt) # Time vector [s]
        self.v_trip = [t,v] # [s],[m/s]

    # =============

    def model_x_from_tV(
        self,
        t_V,
        going_up=True
        ):
        """
        Calculate the position timeseries for the given traveling time at terminal velocity {t_V}
        (t_V=0 means traveling only one level), according to the physical model.

        --- Inputs ---

        {t_V} [Float]: Duration for the interval traveling at constant terminal velocity, units [s].
        If only traveling a single level, then choose t_V=0.
        {going_up} [Boolean]: If True, the elevator travels upwards. If False, it travels downwards.

        --- Return ---

        Generate or update:
        <self.pos_trip> [List]: The first and second elements are Numpy arrays describing the
        time and Z-position timeseries, respectively.

        """
        # Set general parameters:
        i = 1 if going_up else -1 # Traveling direction (upwards is positive)
        amax = i*self.max_accel # Maximum acceleration in traveling direction [m/s^2]
        Dt = self.delta_t # Delta t interval length [s]
        Dtp = self.delta_tprime # Delta t' interval length [s]
        pp_t = self.pp_delta_t # Number of points during Delta t interval
        pp_tp = self.pp_delta_tprime # Number of points during Delta t' interval
        pp_V = int(t_V/self.time_res) # Number of points traveling at terminal speed
        dt = self.time_res # Time resolution [s]
        
        # Generate position timeseries from physical model:

        # STEP 1 (common), from a=0 to a=amax:
        x1 = 1/6*np.arange(0,(pp_t-0.1)*dt,dt)**3/Dt # [s^2]

        # STEP 2 (common), constant a=amax:
        x2 = 1/6*Dt**2+1/2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)+1/2*np.arange(0,(pp_t-0.1)*dt,dt)**2 # [s^2]

        # Case (a): N-level flights, with N>1, equivalent to t_V>0:
        if t_V > 0:
            # STEP 3a [N>1], from a=amax to a=0:
            x3_a = 8/6*Dt**2+3/2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)-1/6*(Dt-np.arange(0,(pp_t-0.1)*dt,dt))**3/Dt # [s^2]

            # STEP 4a [N>1], constant a=0:
            x4_a = 17/6*Dt**2+2*Dt*np.arange(0,(pp_V-0.1)*dt,dt) # [s^2]

            # STEP 5a [N>1], from a=0 to a=-amax:
            x5_a = 17/6*Dt**2+2*Dt*t_V+2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)-1/6*np.arange(0,(pp_t-0.1)*dt,dt)**3/Dt # [s^2]

            # STEP 6a [N>1], constant a=-amax:
            x6_a = 14/3*Dt**2+2*Dt*t_V+3/2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)-1/2*np.arange(0,(pp_t-0.1)*dt,dt)**2 # [s^2]

            # STEP 7a [N>1], from a=-amax to a=0:
            x7_a = 35/6*Dt**2+2*Dt*t_V-1/6*(Dt-np.arange(0,(pp_t-0.1)*dt,dt))**3/Dt # [s^2]

            # Concatenate all acceleration steps:
            x = amax*np.concatenate([x1,x2,x3_a,x4_a,x5_a,x6_a,x7_a,[x7_a[-1]]]) # [m]

        # Case (b): 1-level flights, equivalent to t_V=0:
        else:
            # STEP 3b [N=1], a=amax to a=-amax:
            x3_b = (7/6*Dt**2-1/6*Dtp**2+3/2*Dt*np.arange(0,(2*pp_tp-0.1)*dt,dt)+1/2*Dtp*np.arange(0,(2*pp_tp-0.1)*dt,dt)+
            1/6*(Dtp-np.arange(0,(2*pp_tp-0.1)*dt,dt))**3/Dtp) # [s^2]

            # STEP 4b [N=1], constant a=-amax:
            x4_b = (7/6*Dt**2+2/3*Dtp**2+3*Dt*Dtp+3/2*Dt*np.arange(0,(pp_t-0.1)*dt,dt)-
            1/2*np.arange(0,(pp_t-0.1)*dt,dt)**2) # [s^2]

            # STEP 5b [N=1], from a=-amax to a=0:
            x5_b = 7/3*Dt**2+2/3*Dtp**2+3*Dt*Dtp-1/6*(Dt-np.arange(0,(pp_t-0.1)*dt,dt))**3/Dt # [s^2]

            # FINAL POSITION:
            xf_b = 7/3*Dt**2+2/3*Dtp**2+3*Dt*Dtp # [s^2]

            # Concatenate all acceleration steps:
            x = amax*np.concatenate([x1,x2,x3_b,x4_b,x5_b,[xf_b]]) # [m]

        # Store the time and acceleration timeseries as attributes:
        t = np.arange(0,(len(x)-0.1)*dt,dt) # Time vector [s]
        self.x_trip = [t,x] # [s],[m]

    # =============

    def gen_motion_from_tV(
        self,
        t_V,
        going_up=True,
        plot_motion=True,
        save_name=None,
        save_format='png',    
        figsize=(6,6)
        ):
        """
        Calculate the position, velocity and acceleration timeseries for the given traveling
        time at terminal velocity {t_V} (t_V=0 means traveling only one level), according to
        the physical model.

        --- Inputs ---

        {t_V} [Float]: Duration for the interval traveling at constant terminal velocity, units [s].
        If only traveling a single level, then choose t_V=0.
        {going_up} [Boolean]: If True, the elevator travels upwards. If False, it travels downwards.
        {plot_motion} [Boolean]: If True, plot the generated position, velocity and acceleration.
        The following inputs are only relevant if {plot_motion}=True.
        {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
        {save_format} [String]: saving format for the figure,don't include the dot.
        {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

        --- Return ---

        Generate or update:
        <self.pos_trip> [List]: The first and second elements are Numpy arrays describing the
        time and Z-position timeseries, respectively.
        <self.vel_trip> [List]: The first and second elements are Numpy arrays describing the
        time and Z-velocity timeseries, respectively.
        <self.a_trip> [List]: The first and second elements are Numpy arrays describing the
        time and Z-acceleration timeseries, respectively.

        If {plot_motion}=True, plot the position, velocity and acceleration along the Z-axis as a function
        of time, identifying in the title the final position.

        """

        self.model_x_from_tV(t_V,going_up=going_up) # Update position timersies
        self.model_v_from_tV(t_V,going_up=going_up) # Update velocity timersies
        self.model_a_from_tV(t_V,going_up=going_up) # Update acceleration timersies

        if plot_motion:
            fig, (ax_x,ax_v,ax_a) = plt.subplots(3,1,sharex=True,figsize=figsize)

            # Physical model plots:
            ax_x.plot(self.x_trip[0],self.x_trip[1],'o',color='green',alpha=0.9,label='Phys. model')
            ax_v.plot(self.v_trip[0],self.v_trip[1],'o',color='blue',alpha=0.9,label='Phys. model')
            ax_a.plot(self.a_trip[0],self.a_trip[1],'o',color='orange',alpha=0.9,label='Phys. model')

            # Linear approximation plots:
            ax_x.plot(self.x_trip[0],np.linspace(self.x_trip[1][0],self.x_trip[1][-1],len(self.x_trip[0])),
                's',markersize=3,color='gray',alpha=0.6,label='Linear approx.')
            ax_v.plot(self.v_trip[0],np.ones(len(self.v_trip[0]))*(
                self.x_trip[1][-1]-self.x_trip[1][0])/(self.v_trip[0][-1]-self.v_trip[0][0]),
                's',markersize=3,color='gray',alpha=0.6,label='Linear approx.')

            # Axes configuration:
            ax_x.set(ylabel='Position [m]')
            ax_v.set(ylabel='Velocity [m/s]')
            ax_a.set(xlabel='Time [s]', ylabel='Acceleration [m/s$^2$]')
            for _,(ax,title) in enumerate(zip([ax_x,ax_v,ax_a], ['Position','Velocity','Acceleration'])):
                ax.legend(title=title)

            # General configuration and saving option:
            title = f'Z-motion from $t_V$ = {t_V} s ; output: $x_f$ = {self.x_trip[1][-1]:.2f} m, '
            title += f'$t_f$ = {self.v_trip[0][-1]:.2f} s'
            fig.suptitle(title) 
            fig.tight_layout()
            if save_name:
                ML_general.save_file(save_name,save_format=save_format)

    # =============

    def compare_totaltime_with_exp(
        self,
        total_time_exp,
        save_name=None,
        save_format='png',    
        figsize=(4,3)
        ):
        """
        Compares for the total traveling times for each number of
        levels according to both experimental and modelled results.

        --- Inputs ---

        {total_time_exp} [Dictionary]: Keys are the number of travelled
        levels, values are the total traveling times [s].
        {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
        {save_format} [String]: saving format for the figure,don't include the dot.
        {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

        --- Return ---

        Plot a comparison for the total traveling times for each number of
        levels according to both experimental and modelled results.

        """
         
        # Prepare figure:
        fig, ax = plt.subplots(figsize=figsize)

        # Plot model results:
        N_lvls_model = list(self.model_totaltime_theory.keys())
        times_model = list(self.model_totaltime_theory.values()) # [s]
        ax.scatter(N_lvls_model,times_model,
            marker='o',color='green',alpha=0.9,label='Phys. model')

        # Plot experimental results:
        N_lvls_exp = list(total_time_exp.keys())
        times_exp = list(total_time_exp.values()) # [s]
        ax.scatter(N_lvls_exp,times_exp,
            marker='s',color='gray',alpha=0.9,label='Experiments')

        # Axes configuration:
        ax.set(xlabel='Number of travelled levels', ylabel='Total travelled time [s]')
        ax.legend()

        # General configuration and saving option:
        fig.suptitle(f'Comparison for total travelled times, assuming h={self.h} m') 
        fig.tight_layout()
        if save_name:
            ML_general.save_file(save_name,save_format=save_format)

    # =============

    def fit_amax_deltat_tV_for_exp(
        self,
        total_time_exp,
        ):
        """
        Based on the experimental results, adjust the parameters amax, deltat
        and tV to match both the total traveling time and distance.

        --- Inputs ---

        {total_time_exp} [Dictionary]: Keys are the number of travelled
        levels, values are the total traveling times [s].

        --- Return ---

        Generate or update:
        All the following dictionaries have the number of travelled
        levels as keys:

        <self.model_amax_exp> [Dictionary]: Values are the fitted maximum 
        acceleration, units [m/s^2].
        <self.model_deltat_exp> [Dictionary]: Values are the fitted interval
        times for linear acceleration [s].
        <self.model_tV_exp> [Dictionary]: Values are the fitted interval times
        traveling at terminal velocity [s].

        """
         
        # Initiate and prepare auxiliar terms:
        model_amax_exp = {} # Maximum acceleration [m/s^2]
        model_deltat_exp = {} # Time interval for linear acceleration [s]
        model_tV_exp = {} # Time interval traveling at terminal velocity [s]

        # First fit deltat and tV according to the experimental total time:
        for N in total_time_exp:
            if N == 1:
                model_tV_exp[N] = np.round(0,2) # [s]
                model_deltat_exp[N] = np.round(total_time_exp[N]/5,2) # [s]
            elif total_time_exp[N] <= 6:
                model_tV_exp[N] = np.round(0,2) # [s]
                model_deltat_exp[N] = np.round(total_time_exp[N]/6,2) # [s]
            else:
                model_tV_exp[N] = np.round(total_time_exp[N]-6,2) # [s]
                model_deltat_exp[N] = np.round(1,2) # [s]

        # Then fit amax according to the number of travelled levels:
        for N in total_time_exp:
            Dx = self.h*N # Total travelled distance [m]
            if N == 1:
                model_amax_exp[N] = np.round(Dx*25/(4*total_time_exp[N]**2),2) # [m/s^2]
            elif total_time_exp[N] <= 6:
                model_amax_exp[N] =  np.round(Dx*216/(35*total_time_exp[N]**2),2) # [m/s^2]
            else:
                model_amax_exp[N] =  np.round(Dx/(2*total_time_exp[N]-37/6),2) # [m/s^2]

        # Update attributes:
        self.model_amax_exp = model_amax_exp
        self.model_deltat_exp = model_deltat_exp
        self.model_tV_exp = model_tV_exp
        print(colored('Experimentally fitted amax, deltat and tV have been updated.','blue'))

    # =============

# ============================================================================

# ============================================================================

# ============================================================================



