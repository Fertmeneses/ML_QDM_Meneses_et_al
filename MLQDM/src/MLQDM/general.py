"""
General tools.

# ===========

Functions:

    - find_between
    - save_file
    - truncate_colormap
    - prepare_random_rot_frames

"""

# ============================================================================

# Required packages:

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

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

def find_between(
    name,
    first,
    last
    ):
    """
    Extracts a substring from a string-type object.

    --- Inputs ---
    {name} [string]: Original string that will be analyzed.
    {first},{last} [string]: substrings contained in {name} that will act as boundaries for extracting the substring between them.
    If providing an empty string for {last}, then the last boundary is ignored and a substring is extracted from {first}
    to the end of {name}. 
    If providing an empty string for {first}, then the first boundary is ignored and a substring
    is extracted from he beginning of {name} to {last}.
    
    --- Outputs ---
    [string] The substring between the boundaries. Notice that {first} and {last} are the FIRST substrings found in {name},
    e.g. find_between("a1ba2b","a","b") will return "1" and not "2".
    """

    start = name.index(first) + len(first) # Identifies the first boundary 
    if last == '': # If no last boundary is provided, return the substring from start to the end of {name}.
        return name[start:]
    end = name.index(last,start) # If the last boundary is provided, identify its index in the {sname} string

    return name[start:end]

# ============================================================================

def save_file(
    filename,
    save_format='png'
    ):
    """
    Saves a plot into an image file.
    
    --- Inputs ---

    {filename} [String]: name for the file name, without any extension.
    {save_format} [String]: extension for the file, don't include the dot (e.g. 'png').
    
    """
    if save_format == 'svg':
        plt.savefig(f'{filename}.svg',format="svg",transparent=True) 
    else:
        plt.savefig(f'{filename}.{save_format}') 

# ============================================================================

def truncate_colormap(
    cmap, 
    min_val=0.0, 
    max_val=1.0, 
    n=100
    ):
    """
    Truncates a colormap according to boundary values.
    
    --- Inputs ---
    
    {cmap} [Colormap]: original colormap that will be truncated.
    {min_val,max_val} [Float]: minimum and maximum boundaries for truncation.
    {n} [Integer]: number of points for the new (truncated) colormap.

    ---Outputs---
    
    {new_cmap} [Colormap]; new colormap.

    """
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
        cmap(np.linspace(min_val, max_val, n)))
    return new_cmap

# ============================================================================

def prepare_random_rot_frames(
    theta_pp,
    rand_angle=5,
    min_theta=10,
    max_theta=90,
    dphi_dtheta_pp=10/90,
    min_sep_phi=20,
    include_lab_frame=True,
    random_seed=0,
    ):
    """
    Generates a semi-random collection of rotational frames based on regular
    grid for polar (theta) and azimuth (phi) rotational angles which define the
    rotational axes. 
    Notes: 
    - The rotation angle is fixed to 90 degrees.
    - The polar angles can be constrained within {min_theta} and {max_theta}
    limits, but the azimuth angles are always within the [0,359] degree range.

    --- Inputs ---

    {theta_pp} [Integer]: Number of options for the polar angle (values are 
    determined later according to the other parameters).
    {rand_angle} [Float]: Maximum random variation for both polar and azimuth angles
    based on the regular grid, units [degree].
    {min_theta} [Float]: Minimum polar angle for the grid, units [degree].
    {max_theta} [Float]: Maximum polar angle for the grid, units [degree].
    {dphi_dtheta_pp} [Float]: Fraction of points of azimuth angle options per
    degree in the polar angle. For example, if {dphi_dtheta_pp}=0.2, then for 
    theta=20 there will be 20*0.2=4 options for the azimuth angles.
    {min_sep_phi} [Float]: Minimum separation for the randomly generated azimuth angles
    for a given polar angle, units [degree].
    {include_lab_frame} [Boolean]: If True, include the original laboratory frame
    (no rotation) in the output collection.
    {random_seed} [Integer]: Seed for the random generation of polar and azimuth angles.

    --- Return ---

    Collection of rotational frames condensed in a dictionary:

    {RFs} [Dictionary]: 
    Keys are strings with the format 'n_{X}_{Y}_{Z}_rot90deg', where X, Y, Z are the
    Cartesian components of the rotating axis. 
    Values are lists with information about the rotational frame, in the the format 
    [List,Numeric,String], which represent the [rotation axis in (X,Y,Z) format,
    rotation angle in degree units,RF name].

    """
    # Random see initialization:
    np.random.seed(random_seed)

    # Prepare theta-phi grid:
    theta_basic = np.linspace(min_theta, max_theta, theta_pp) + np.random.random(theta_pp)*rand_angle # [deg]
    # Check feasibility of {min_sep_phi}:
    if 360/(max_theta*dphi_dtheta_pp) < min_sep_phi:
        raise ValueError('Error: dphi_dtheta_pp is too large compared to max_theta.') 

    # Boundary and conversion functions:
    bounder_theta = lambda x: min_theta if x<min_theta else (max_theta if x>max_theta else int(x)) # Function to constrain theta
    bounder_phi = lambda x: int(x) % 359 # Function to reduce phi to equivalent angles
    deg_to_rad = lambda x: np.round(x/360*2*np.pi) # [rad]
    v_bounder_theta = np.vectorize(bounder_theta)
    v_bounder_phi = np.vectorize(bounder_phi)

    # Build the final theta-phi grid:
    theta = v_bounder_theta(theta_basic) # [deg]
    phi_basic = {th: np.linspace(0,359,int(np.ceil(dphi_dtheta_pp*th)))+ # Main grid
                 np.random.random(int(np.ceil(dphi_dtheta_pp*th)))*rand_angle+ # Random variation
                 np.random.random()*360 # Random phase
                 for th in theta} # [deg]
    phi = {th: v_bounder_phi(phi_basic[th]) for th in theta} # [deg]
    wrong_sep = True # Initiate
    while wrong_sep: # Separate points if necessary:
        wrong_sep = False # Reset condition, must be verified through the entire array
        for th in phi:
            for i in range(0,len(phi[th])):
                for j in range(i+1,len(phi[th])):
                    if abs(phi[th][i]-phi[th][j]) % 359 < min_sep_phi:
                        wrong_sep = True # The entire array will be for one more iteration
                        phi[th][j] = (phi[th][j] + min_sep_phi/2) % 359 # Correct phi position [deg]

    # Final conversion of phi to [rad]:
    deg_to_rad = {th: np.round(th*2*np.pi/360,2) for th in theta} # [deg]: [rad]
    theta_rad = {th: np.round(th*2*np.pi/360,2) for th in theta} # [deg]: [rad]
    phi_rad = {th: np.round(phi[th]*2*np.pi/360,2) for th in theta} # [rad]

    # Build the collection of rotating axes:
    n = []
    for th in theta:
        th_rad = theta_rad[th] # Polar angle [rad]
        for ph_rad in phi_rad[th]:
            n.append(np.round(np.array([
                np.sin(th_rad)*np.cos(ph_rad), # Lab x-component
                np.sin(th_rad)*np.sin(ph_rad), # Lab y-component
                np.cos(th_rad)] # Lab z-component)
                              ),3).tolist())

    # Define rotational frames in a dictionary, with the name as key, rotational axis, angle [degree] and name as values:
    RFs = {'n_'+'_'.join(str(comp) for comp in n_axis) + '_rot90deg':
        [n_axis,90,'n_'+'_'.join(str(comp) for comp in n_axis) + '_rot90deg']
        for n_axis in n
        }

    if include_lab_frame:
        # Add the laboratory frame:
        RFs['n_1_0_0_rot90deg'] = [[0,0,1],0,'n_1_0_0_rot90deg'] # Original RF

    return RFs