"""
Analysis for general results and predictions from Machine Learning.

# ===========

Main functions:

    - load_results_and_overview
    - plot_gen_test_vs_val
    - group_results
    - plot_best_results_and_archs
    - keep_best_archs
    - plot_performance_vs_Naugm_vs_ptrain
    - load_preds
    - analyze_acc_fixed_Zthres
    - plot_preds_single_RF

# ===========

"""

# ============================================================================

# Required packages:

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # remove warning
import matplotlib.pyplot as plt

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

color_min = {'lin_approx': 'cyan', 'phys_model': 'lightgreen'}
color_min_edge = {'lin_approx': 'blue', 'phys_model': 'green'}
color_bar = {'lin_approx': 'royalblue', 'phys_model': 'darkgreen'}
markers = {'lin_approx': 'd', 'phys_model': '*'}
bar_width = {'lin_approx': 0.5, 'phys_model': 0.4}
color_Bx = '#008affff'
color_By = '#001bffff'
color_Bz = '#8d00ffff'
color_Zpred = '#b50000e5'
color_Ztrue = '#58a79cb2'

# ============================================================================

def load_results_and_overview(
    res_file,
    group_var,
    interpolation='Unknown',
    eval_col="Acc_Test_z",
    xlabel_metric=None,
    x_lims=[0,100],
    lim_N_legends=32,
    save_name=None,
    save_format='png',
    figsize=(6,3) 
    ):
    """
    Load general ML results and plot an overview of the models' performance.

    --- Inputs ---

    {res_file} [String]: Path for the general results' file, must be a .csv and have the same output
    format as the one provided in any of training functions train_stageX in the MLmodel.py module.
    {group_var} [String]: Training variable, must much one of the columns within the general results file.
    {interpolation} [String]: Approach used for interpolating the target data in the ML training
    process. It only affects the title of the overview plot.
    {eval_col} [String]: Evaluation metric, must much one of the columns within the general results file.
    {xlabel_metric} [String]: If provided, the x-label for the plot, which describes the metric.
    {x_lims} [List]: Limits for the x-axis, with the format [xmin,xmax].
    {lim_N_legends} [Integer]: Maximum number of labels allowed in order to plot the full names in the legend.
    If there are more labels than this limits, they are identified with a generic number. 
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    {res} [DataFrame]: Same information as the general results file, turned into Dataframe.

    In addition, plot a histogram counting the {eval_col} results from all ML models
    grouped by the {group_var} variable.

    """

    # Load general results:
    res = pd.read_csv(res_file)

    # Get group names, define colors and labels:
    group_name = res[group_var].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(group_name)))
    labels = ([var for var in group_name] if len(group_name)<lim_N_legends 
        else [f'#{i+1}' for i in range(len(group_name))])

    # Plot the histogram
    fig, ax = plt.subplots(figsize=figsize)
    for i, group in enumerate(group_name):
        data = res[res[group_var]==group][eval_col]
        binning = np.max([int(np.ceil((max(data)-min(data))/2)),1]) # (minimum=1)
        data.hist(
            bins=binning,grid=False,color=colors[i],
            edgecolor=colors[i]*0.9,alpha=0.7,label=labels[i],ax=ax)
    ax.set_xlim(x_lims)

    # Additional configuration:
    cols_legend = int(np.ceil(len(group_name)/8))
    size_legend = plt.rcParams['legend.fontsize']*5/(4+cols_legend)  
    ax.legend(ncol=cols_legend,fontsize=size_legend)
    ax.set_title(f'Results of ML models grouped by {group_var} ; interpolation {interpolation}')
    if xlabel_metric is not None:
        ax.set_xlabel(xlabel_metric)
    else:
        ax.set_xlabel(f'Evaluation metric: {eval_col}')
    ax.set_ylabel('Frequency')    
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format)

    return res

# ============================================================================

def plot_gen_test_vs_val(
    df_results,
    test_col="Acc_Test_z",
    train_col="Acc_Val_z",
    gral_metric='Metric',
    interpolation='Unknown',
    group_by=None,
    lim_N_legends=32,
    save_name=None,
    save_format='png',
    figsize=(6,4)    
    ):
    """

    Plot the results according to the training and testing results, to show
    overfitting or underfitting. If requested, the plot will be replicated for
    different grouping features.

    --- Inputs ---

    {df_results} [DataFrame]: Information about the results, in the format provided
    by the function load_results_and_overview.
    {test_col} [String]: Column for the results according to the testing dataset.
    {train_col} [String]: Column for the results according to the training dataset.
    {gral_metric} [String or None]: If provided, describes the general metric used
    in both training and testing datasets in the plot/s.
    {interpolation} [String]: Approach used for interpolating the target data in the ML training
    process. It only affects the title of the overview plot.
    {group_by} [List or None]: If a list is provided, each element represents a grouping
    variable that must be one of the columns in {df_results}. For each feature, a 
    results plot will be shown, grouped by that feature.
    {lim_N_legends} [Integer]: Maximum number of labels allowed in order to plot the full names in the legend.
    If there are more labels than this limits, they are identified with a generic number.
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    Plot the evaluation results comparing the testing and training datasets.
    If {group_by} is not None, there will one plot per element in {group_by}.
    Else, there will be a single plot combining all ML results.

    """

    # Case: results are grouped by elements in {group_by}:
    if group_by is not None:
        for feature in group_by:
            # Prepare data and grouping variable:
            group_name = df_results[feature].unique()
            colors = plt.cm.viridis(np.linspace(0, 1,len(group_name)+1))
            labels = ([var for var in group_name] if len(group_name)<lim_N_legends 
                else [f'#{i+1}' for i in range(len(group_name))])
            # Plot results:
            fig, ax = plt.subplots(figsize=figsize)    
            for i, group in enumerate(group_name):
                data = df_results[df_results[feature]==group]
                ax.plot(data[test_col],data[train_col],plot_markers[i%len(plot_markers)],
                        color=colors[i],alpha=0.7,label=labels[i])
            # Plot identity line:
            xmin,xmax = ax.get_xlim()    
            ID = np.linspace(xmin,xmax,3)
            ax.plot(ID,ID,'--k')
            # Additional configuration:
            ax.set(xlabel=f"{gral_metric} in Testing",ylabel=f"{gral_metric} in Training")
            ax.set_title(f'Results grouped by {feature} ; interpolation {interpolation}')
            cols_legend = int(np.ceil(len(group_name)/8))
            size_legend = plt.rcParams['legend.fontsize']*5/(4+cols_legend)
            ax.legend(title=feature,ncol=cols_legend,fontsize=size_legend)
            fig.tight_layout()
            if save_name:
                ML_general.save_file(save_name+f'_{feature}',save_format=save_format)

    # Case: results all together
    else:
        fig, ax = plt.subplots(figsize=figsize)    
        # Plot results:
        ax.scatter(df_results[test_col],df_results[train_col],
                   color='orange',edgecolor='gray',alpha=0.7)
        # Plot identity line:
        xmin,xmax = ax.get_xlim()    
        ID = np.linspace(xmin,xmax,3)
        ax.plot(ID,ID,'--k')
        ax.set(xlabel=f"{gral_metric} in Testing",ylabel=f"{gral_metric} in Training")
        ax.text(xmin,max(df_results[train_col])*0.95, "Overfitting ", ha='left')
        ax.text(xmax, ax.get_ylim()[0]*1.01, "Underfitting ", ha='right')
        ax.set_title(f'All results ; interpolation {interpolation}')
        fig.tight_layout()
        if save_name:
            ML_general.save_file(save_name+f'_All',save_format=save_format)

# ============================================================================

def group_results(
    df_res_dict,
    group_by,
    eval_col='Acc_Test_z',
    metric='Metric',
    i_order=None,
    print_summary=True,
    score_thres=None,
    y_lims=[0,100],
    save_name=None,
    save_format='png',
    figsize=(8,4)    
    ):
    """
    Plot the results of ML models according to the provided interpolation approaches, 
    evaluated to the chosen metric, and grouped by the chosen variable.

    --- Inputs ---

    {df_res_dict} [Dictionary]: Each key is an interpolation approach, 'lin_approx' and/or
    'phys_model'. Each value is a Dataframe Information for general ML results, same 
    format as the output of function "load_results_and_overview".
    {group_by} [String]: Variable to group results, must be a column in {df_res}.
    {eval_col} [String]: Evaluation criterion by which the ML models get their scores, must be
    a column in {df_res}
    {metric} [String]: Name for the evaluation metric, it only affects labels in the plot.
    {i_order} [Dictionary or None]: If provided, each key is an interpolation approach. Each value
    must be a list by which elements within the {df_res_dict} grouped-results will be sorted
    as they appear in the provided list. Then, {i_order} must contain all unique values contained in
    the column {group_by} and nothing else. If {i_order}=None, sorting is automatic.
    {print_summary} [Boolean]: If True, displays on screen the best variable in the grouped-results,
    along with its lowest and highest scores.
    {score_thres} [Float or None]: If provided, plot a horizontal line in the results plot,
    indicating a threshold score, related to the evaluation metric {eval_col}.
    {y_lims} [List]: Limits for the y-axis, related to {eval_col}, with the format [ymin,ymax].
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    Plot a range of evaluation results for each family of ML models sharing the same variable according
    to the chosen feature {group_by}. The lowest result from each group is highlighted as a blue
    diamond, while all other results are included in a shaded bar.
    If only one interpolation approach was provided in {df_res_dict}, then the plot just displays
    those results. If the two interpolation approaches were provided, then both results are displayed
    in the same plot.

    In addition, displays in screen a summary for the best variable according to the proposed
    metric {eval_col}, along with its highest and lowest scores, for each interpolation approach.

    """

    # Check that a valid interpolation approach was provided:
    if not ('lin_approx' or 'phys_model' in df_res_dict):
        raise ValueError('Error: either or both "linear_approx" or "phys_model" must be provided.') 

    # Group dataframe and get min/max results:
    gr_res, gr_res_max, gr_res_min = {}, {}, {} # Initiate
    for approach in df_res_dict:
        gr_res[approach] = df_res_dict[approach].groupby(group_by)
        i_sort = i_order[approach] if i_order is not None else None
        gr_res_max[approach] = gr_res[approach][eval_col].max().reindex(index=i_sort)
        gr_res_min[approach] = gr_res[approach][eval_col].min().reindex(index=i_sort)

    # Determine best magnetic components boundaries:
    best_choice, low_score, high_score = {}, {}, {} # Initiate
    for approach in df_res_dict:
        best_choice[approach] = gr_res_min[approach].sort_values(ascending=False).index[0]
        low_score[approach] = np.round(gr_res_min[approach].sort_values(ascending=False).iloc[0],1)
        high_score[approach] = np.round(gr_res_max[approach].sort_values(ascending=False).iloc[0],1)

    # Print summary on screen:
    if print_summary:
        for approach in df_res_dict: 
            print(f'\nInterpolation approach: {approach}.')
            print(f'Best choice (based on lowest group-score): {best_choice[approach]}')
            print(f'Lowest score for {best_choice[approach]}: {low_score[approach]}')
            print(f'Highest score for {best_choice[approach]}: {high_score[approach]}\n')

    # Plot results:
    fig, ax = plt.subplots(figsize=figsize)
    for approach in df_res_dict:
        ax.scatter(gr_res_min[approach].index,gr_res_min[approach],
            color=color_min[approach],edgecolor=color_min_edge[approach],
            marker=markers[approach],label=f'{approach}: lowest score',alpha=1)
        ax.bar(gr_res_min[approach].index, gr_res_max[approach]-gr_res_min[approach],
            bottom=gr_res_min[approach],width=bar_width[approach],
            color=color_bar[approach],alpha=0.3,label=f'{approach}: all results')
    # Include scoring threshold, if any:
    if score_thres is not None:
        ax.axhline(80,ls='--',alpha=0.6,lw=1,color='red',label='Threshold')

    # Additional configuration:
    ax.set_ylim(y_lims)
    ax.set(xlabel=group_by,ylabel=metric)
    ax.set_title(f'ML results grouped by {group_by}')
    ax.legend(loc='lower right',ncol=3)
    fig.tight_layout() 
    if save_name:
        ML_general.save_file(save_name,save_format=save_format)

# ============================================================================

def plot_best_results_and_archs(
    df_res,
    group_var,
    xlabel=None,
    eval_col='Acc_Test_z',
    metric='Metric',
    interpolation='Unknown',
    save_name=None,
    save_format='png',
    figsize=(5,6)
    ):
    """
    Plot the best metric results among many ML model architectures, for each value
    within a chosen variable. In addition, indicate the best architecture for each
    of those values in a different plot.

    --- Inputs ---

    {df_res} [DataFrame]: Information for general ML results, same format as the output of
    function "load_results_and_overview".
    {group_var} [String]: Variable to group results, must be a column in {df_res}.
    {xlabel} [String or None]: Name for the grouping variable to be displayed in the plot
    as the x-label. If None, then just {group_var} is displayed.
    {eval_col} [String]: Evaluation criterion by which the ML models get their scores, must be
    a column in {df_res}.
    {metric} [String]: Name for the evaluation metric, it only affects labels in the plot.
    {interpolation} [String]: Approach used for interpolating the target data in the ML training
    process. It only affects the title of the overview plot.
    {save_name} [String]: filename for the figure (don't include the extension). If None, 
    no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for 
    the entire figure (2 plots included).

    --- Return ---

    Plot the best metric results among many ML model architectures, for each value
    within {group_var}, for each rotational frame.

    In addition, plot the best architecture for each value within {group_var}, for each
    rotational frame.

    """

    res_best = pd.DataFrame(columns=list(df_res.columns)) # Initiate
    # Choose best architecture for each value in {group_var} and RF:
    for value in df_res[group_var].unique():
        for RF in df_res[df_res[group_var]==value]['RF'].unique():
            data = df_res[(df_res[group_var]==value) & (df_res['RF']==RF)]
            best_idx = data[eval_col].idxmax()
            res_best.loc[len(res_best)] = df_res.loc[best_idx]
            
    # Get group names and define colors
    group_name = res_best['RF'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(group_name)+1))
    
    # Plot:
    fig, (ax_acc,ax_arch) = plt.subplots(2,1,figsize=figsize)
    for i, group in enumerate(group_name):
        data = res_best[res_best['RF']==group]
        ax_acc.plot(data[group_var],data[eval_col],['-o','-s','-d'][i],color=colors[i],
                    alpha=0.7,label=group)
        ax_arch.plot(data[group_var],data["Model_Name"],['o','s','d'][i],color=colors[i],
                     alpha=0.7,label=group)
    ax_acc.set_title(f'Best metric result for each {group_var} ; Interpolation {interpolation} ')
    ax_acc.set_ylabel(metric)
    ax_arch.set_title(f'Best architecture for each {group_var} ; Interpolation {interpolation} ')
    ax_arch.set_ylabel('Architecture')
    for ax in [ax_acc,ax_arch]:
        xlabel = xlabel if xlabel is not None else group_var
        ax.set_xlabel(xlabel)
        ax.legend(group_name)
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format)

# ============================================================================

def keep_best_archs(
    df_res,
    variable,
    eval_col='Acc_Test_z'
    ):
    """
    Find the best architectures for each value within the chosen feature {variable},
    according to the chosen metric {eval_col}.

    --- Inputs ---

    {df_res} [DataFrame]: Information for general ML results, same format as the output of
    function "load_results_and_overview".
    {variable} [String]: Variable from which the ML models will compete to have the best
    result according to {eval_col} and the best will be be chosen. {variable} must be a
    column in {df_res}.
    {eval_col} [String]: Evaluation criterion by which the ML models get their scores, must be
    a column in {df_res}.   

    --- Return ---

    {df_best_archs}[DataFrame]: Information for best ML results, similar to {df_res} but
    only including the best model for each value of {variable}.

    """

    # Prepare list with all possible values within the feature {variable}:
    var_list = sorted(set(df_res[variable])) # Identify all options for number of points in windows

    # Find the best architecture for each value in {var_list}:
    best_archs = {} # Initiate dictionary
    for var in var_list:
        # Identify subset of architectures with the current value of {variable}:
        res_var_subset = df_res[df_res[variable]==var]
        # Make a set with the architectures within the subset:
        archs = sorted(set(res_var_subset['Model_Name']))
        # Obtain best architecture according the lowest value in {eval_col} among all RF:
        lowest_eval, best_arch = 0, 'None' # Initiate
        for arch in archs:
            # Identify minimum metric value for this architecture:
            min_eval = np.min(res_var_subset[res_var_subset['Model_Name']==arch][eval_col])
            # Update lowest metric if necessary:
            if min_eval > lowest_eval:
                lowest_eval = min_eval # [%]
                best_arch = arch
        best_archs[var] = best_arch # Register the best architecture for the current value of {variable}

    # Make a new dataset with the best architecture for each value in {var_list}:
    df_best_archs = pd.concat([df_res[(df_res[variable]==var) & (df_res['Model_Name']==best_archs[var])] 
        for var in var_list])

    return df_best_archs

# ============================================================================

def plot_performance_vs_Naugm_vs_ptrain(
    df_res,
    eval_col='Acc_Test_z',
    metric='Metric',
    interpolation='Unknown',
    save_name=None,
    save_format='png',
    figsize=(6,4)
    ):
    """
    Plot the ML performance according to the chosen evaluation metric as a function of
    both the data augmentation level and the training dataset fraction.

    --- Inputs ---

    {df_res} [DataFrame]: Information for general ML results, same format as the output of
    function "load_results_and_overview".
    {eval_col} [String]: Evaluation criterion by which the ML models get their scores, must be
    a column in {df_res}.
    {metric} [String]: Name for the evaluation metric, it only affects labels in the plot.
    {interpolation} [String]: Approach used for interpolating the target data in the ML training
    process. It only affects the title of the overview plot.
    {save_name} [String]: filename for the figure (don't include the extension). If None, 
    no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    Plot the metric results as a function of the training dataset fraction (how much was used from
    the original dataset) and the augmentation level (who many times the training dataset was 
    replicated).

    """
    
    # Determine metric per tuple (p_train,N_augm) as the worst performance for each ML model:
    gr_res = df_res.groupby(['p_train','N_augm']) # Group by tuple
    gr_res_min = gr_res[eval_col].min().sort_index(ascending=False) # Determine lowest metric per tuple
    # Generate new correlated vectors for p_train, N_augm, metric:
    tuple_idx = gr_res_min.index.to_numpy() # indexes
    p_train_vec = np.array([tuple_idx[i][0] for i in range(len(tuple_idx))]) # p_train values
    N_augm_vec = np.array([tuple_idx[i][1] for i in range(len(tuple_idx))]) # N_augm values
    eval_vec = np.array([gr_res_min.iloc[i] for i in range(len(tuple_idx))]) # Metric values

    # Plot figure:
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        p_train_vec,p_train_vec*(1+N_augm_vec),
        c=eval_vec, vmin=min(eval_vec), marker='d',
        vmax=max(eval_vec), s=40, cmap='RdYlGn', alpha=0.9
        ) # Main plot

    # Colorbar configuration:
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel(f'{metric}', rotation=90)

    # Plot identity line:
    xmax = max(p_train_vec)*1.1
    ID = np.linspace(0,xmax,3)
    plt.plot(ID,ID,'-k',lw=0.5,alpha=0.2,label='No augmentation')
    label = 'Augm. levels' # Set only one label for the augmentation lines
    for i in (set(N_augm_vec)-{0}):
        plt.plot(
            ID,(i+1)*ID,':k',lw=1,alpha=0.2,label=label
            )
        label = None # Stop labeling the following augmentation lines

    # Additional configuration:
    ax.legend()
    ax.set_title(f'{metric} vs training frac. & data augm. ; Interpolation {interpolation} ')
    ax.set_xlim([0,xmax])
    ax.set_ylim([0,max(p_train_vec*(1+N_augm_vec))*1.1])
    ax.set_xlabel('Original training dataset fraction')
    ax.set_ylabel('Total (augmented) dataset fraction')
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format)

# ============================================================================

def load_preds(
    files,
    skiprows=1,
    header=None,
    sep=',',
    t_col=0,
    z_true_col=1,
    z_pred_col=2,
    interpolation='Unknown',
    park_lvls = np.round(np.append([0],4.1+3.7*(np.arange(1,8)-1)),1),
    plot_sample=True,
    save_name=None,
    save_format='png',
    figsize=(8,4)
    ):
    """
    Load the ML predictions, along with the ground truth labels, each prediction set
    identified by its rotational frame. The predictions are divided into two outputs:
    one grouping all Z-positions associated to parking intervals (ground truth),
    the other grouping all Z-positions, including parking and interpolated intervals.

    --- Inputs ---

    {files} [List]: Each element is a path for a prediction .csv file, which must
    include the following substring in its name: "_n_{X}_{Y}_{Z}_rot", where 
    X, Y, Z are the coordinates for the rotating axis.
    Each file must contain the columns with time [s], Z-position ground truth [m]
    and Z-position predictions [m].
    {skiprows} [Integer]: Number of initial lines that must be skipped when reading 
    the .csv files.
    {header} [Integer or None]: Number of header lines in the .csv files.
    {sep} [String]: Character or regex pattern to treat as the delimiter in the magnetic files.
    {t_col} [Integer]: Column number (starting from 0) in each .csv file associated
    with the time vector.
    {z_true_col} [Integer]: Column number (starting from 0) in each .csv file associated
    with the ground truth values for Z-position.
    {z_pred_col} [Integer]: Column number (starting from 0) in each .csv file associated
    with the predicted values for Z-position.
    {interpolation} [String]: Approach used for interpolating the target data in the ML training
    process. It only affects the title of the overview plot.
    {park_lvls} [Numpy array]: Each element is a parking Z-position (ground truth), units [m].
    {plot_sample} [Boolean]: If True, show sample plots (using only one rotational frame) of
    the Z-position predictions and ground truth as a function of time, for both tracking and
    parking groups.
    The following parameters are only relevant when {plot_sample}=True:
    {save_name} [String]: filename for the figure (don't include the extension). If None, 
    no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for 
    the figure.

    --- Return ---

    The following outputs are [Dictionaries] for the prediction results, each key a different
    prediction set associated to a rotational frame. The key format is "pol{X}_azim{Y}", with
    X and Y the polar and azimuth angles in the Laboratory frame, respectively, for the
    rotating axis used in the rotational frame operation, units [degree].
    For each key, the value is a Dataframe with columns "Time_s", "Truth_m" and "Preds_m",
    representing the time [s], ground truth Z-positions [m] and Z-prediction [m]. The ground
    truth refers to either the Parking positions or all positions (parking and interpolation).

    {preds_park} [Dictionary]: Only considers the parking intervals according to the ground
    truth.
    {preds_track} [Dictionary]: Considers all Z-positions, including the parking and
    interpolated intervals, everything considered as the ground truth.

    In addition, if {plot_sample}=True, make two sample plots, using a single rotational
    frame, with the tracking and parking groups. Each plot will show the Z-position 
    predictions and ground truth as a function of time.

    """
    
    # Obtain the names of all rotational frames:
    all_names = set(ML_general.find_between(file,'_n_','_rot') for file in files)

    # Group predictions that share the same rotational frames (different testing datasets):
    preds = {name: [file for file in files if name in file] for name in all_names}

    # Assign polar and azimuth angles to each n_axis:
    pol_azi = {} # Initiate
    for name in all_names:
        # Identify XYZ Cartesian coordinates in the Laboratory frame:
        n_axis = [float(number) for number in name.split('_')]
        # Assign polar angle:
        pol = int(np.arccos(n_axis[2]/np.sqrt((n_axis[0]**2+n_axis[1]**2+n_axis[2]**2)))/
        np.pi*180 % 180) # Polar angle [deg]
        # Assign azimuth angle:
        if (n_axis[0]**2+n_axis[1]**2) != 0:
            azi = int(np.sign(n_axis[1])*np.arccos(n_axis[0]/np.sqrt((n_axis[0]**2+n_axis[1]**2)))/
                np.pi*180 % 360) # Azimuth angle [deg]
        else:
            azi = 0 # Azimuth angle [deg], case theta=0
        pol_azi[name] = [pol,azi] # Polar [deg] and azimuth [deg] angles

    # Load all predictions and separate the tracking (all) and parking (partial) groups:
    preds_track = {} # Initiate
    for name in preds:
        df_sets = [] # Initiate list with results from different testing datasets
        for file in preds[name]:
            # Load data from a single testing dataset:
            data = pd.read_csv(file,skiprows=skiprows,header=header)
            # Obtain time, Z-predictions and ground truth Z-positions:
            time = data[t_col] # Time [s]
            Z_true = data[z_true_col] # Ground truth Z-positions [m]
            Z_pred = data[z_pred_col] # Z-predictions [m]
            # Add Dataframe to the list:
            df_sets.append(pd.DataFrame(
                data={'Time_s': time, 'Preds_m': Z_pred, "Truth_m": Z_true}))
        # Concatenate all Dataframes with different testing datasets:
        RF_name = f'pol{pol_azi[name][0]}_azim{pol_azi[name][1]}'
        preds_track[RF_name] = pd.concat(df_sets).reset_index() # Concatenate all results
        preds_track[RF_name] = preds_track[RF_name].sort_values('Time_s') # Sort chronologically
 
    # Prepare the parking levels information:
    park_lvls = sorted(park_lvls) # Sort parking levels by ascending order (just in case)
    N_lvl = {park_lvl: i+1 for i,park_lvl in enumerate(park_lvls)} # Name the levels, starting on 1
    # Build the parking predictions:
    preds_park = {} # Initiate
    for RF in preds_track:
        # Using the ground truth, obtain the parking indexes:
        i_park = preds_track[RF]["Truth_m"].apply(lambda x: x in park_lvls)
        # Select those rows which belong to parking intervals:
        preds_park[RF] = preds_track[RF].loc[i_park]
        # Convert float values to labels:
        preds_park[RF]["Level_GT"] = preds_park[RF]["Truth_m"].apply(
        lambda x: N_lvl[x]) # Add the ground truth level

    # Plot samples for Z-position predictions and ground truth vs time, if requested:
    if plot_sample:

        # Tracking group:
        df = preds_track[RF_name] # Prepare sample data
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df['Time_s']/60,df['Preds_m'],'o',markersize=3,
            color='red',alpha=0.7,label='Predictions')
        ax.plot(df['Time_s']/60,df['Truth_m'],'-',lw=1.2,
            color='green',alpha=0.7,label='Ground truth')
        ax.set_title(f'Tracking mode: predictions vs ground truth; interpolation {interpolation} ')
        ax.set_ylabel('Z-position [m]')
        ax.set_xlabel('Time [min]')
        ax.legend()
        fig.tight_layout()
        if save_name:
            ML_general.save_file(save_name+'_tracking',save_format=save_format)

        # Parking group:
        df = preds_park[RF_name] # Prepare sample data
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df['Time_s']/60,df['Preds_m'],'o',markersize=3,
            color='red',alpha=0.7,label='Predictions')
        ax.plot(df['Time_s']/60,df['Truth_m'],'s',markersize=1,
            color='cyan',alpha=1,label='Ground truth')
        ax.set_title(f'Parking mode: predictions vs ground truth; interpolation {interpolation} ')
        ax.set_ylabel('Z-position [m]')
        ax.set_xlabel('Time [min]')
        ax.legend()
        fig.tight_layout()
        if save_name:
            ML_general.save_file(save_name+'_parking',save_format=save_format)       

    return preds_park, preds_track

# ============================================================================

def analyze_acc_fixed_Zthres(
    preds_park,
    preds_track,
    z_thres=1.0,
    score_thres=None,
    y_lims=None,
    save_name=None,
    save_format='png',
    figsize=(8,3)
    ):
    """
    Analyze the prediction results from the same ML algorithm trained in many rotational
    frames, and compute the parking and tracking accuracies using a fixes position
    tolerance value.

    --- Inputs ---

    {preds_park} [Dictionary]: Prediction results for parking intervals. Each key represents
    a rotational frame, the value is a Dataframe with columns "Time_s", "Truth_m" and "Preds_m".
    {preds_track} [Dictionary]: Prediction results for general tracking. Each key represents
    a rotational frame, the value is a Dataframe with columns "Time_s", "Truth_m" and "Preds_m".
    {z_thres} [Float]: Position tolerance value, used to determine which predictions are correct.
    {score_thres} [Float or None]: If provided, plot a horizontal line in the plots, indicating a
    threshold score for both tracking and parking accuracies.
    {y_lims} [List]: Limits for the accuracy y-axes, with the format [ymin,ymax].
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    A figure containing two plots side by side. Each plot shows the results for the
    included approaches in {preds_park} and {preds_track}.

    Left plot: General parking and tracking accuracies.
    Right plot: Parking accuracy per level

    """

    # Prepare figures:
    fig, (ax_gral,ax_lvl) = plt.subplots(1,2,figsize=figsize,width_ratios=[1, 2.5])
    
    # Process data:
    for approach in preds_park:
        print(f'\n--------- Interpolation approach: {approach} ---------\n')

        # Identify parking and tracking data in the current interpolation approach:
        data_park = preds_park[approach] # Parking data
        data_track = preds_track[approach] # Tracking data
        # Identify parking levels:
        lvls = sorted(set(data_park[list(data_park.keys())[0]]['Level_GT']))

        # Collect results in every rotational frame:
        acc_park, acc_track = [], [] # Initiate accuracy results
        park_acc_lvl = {lvl: [] for lvl in lvls} # Initiate parking detailed accuracy

        for RF in data_park:
            # Parking accuracy:
            park_correct = np.abs(data_park[RF]['Preds_m']-data_park[RF]['Truth_m'])<z_thres # Boolean predictions list
            acc_park.append(np.round(sum(park_correct)/len(park_correct)*100,2)) # Fraction of correct parking predictions [%]

            # Tracking accuracy:
            track_correct = np.abs(data_track[RF]['Preds_m']-data_track[RF]['Truth_m'])<z_thres # Boolean predictions list
            acc_track.append(np.round(sum(track_correct)/len(track_correct)*100,2)) # Fraction of correct tracking predictions [%]
            
            # Parking accuracy in details:
            for lvl in sorted(set(data_park[RF]['Level_GT'])):
                # Select appropriate data:
                data = data_park[RF][data_park[RF]['Level_GT']==lvl]
                # Evaluate predictions:
                park_lvl_correct = np.abs(data['Preds_m']-data['Truth_m'])<z_thres # Boolean predictions list
                # Calculate parking accuracy in the current parking level and update dictionary:
                acc_park_lvl = np.round(sum(park_lvl_correct)/len(park_lvl_correct)*100,2) # [%]
                park_acc_lvl[lvl].append(acc_park_lvl) # [%]

        # Official performance (lowest results):
        acc_park_lowest = np.min(acc_park) # Parking lowest accuracy [%]
        acc_track_lowest = np.min(acc_track) # Tracking lowest accuracy [%]
        # Print results in screen:
        print(f'Overall PARKING accuracy: {acc_park_lowest}%\n')
        print(f'Overall TRACKING accuracy: {acc_track_lowest}%\n')

        # Prepare data for [Accuracy per level] plot:
        accs = ['Tracking','Parking']
        min_values = np.array([acc_track_lowest,acc_park_lowest]) # [%]
        max_values = np.array([np.max(acc_track),np.max(acc_park)]) # [%]

        # Add data to [General accuracy] plot:
        ax_gral.scatter(
            accs,min_values,
            color=color_min[approach],edgecolor=color_min_edge[approach],
            marker=markers[approach],label=f'{approach}: lowest score',alpha=1)
        ax_gral.bar(
            accs,max_values-min_values,bottom=min_values,
            width=bar_width[approach],color=color_bar[approach],
            alpha=0.3,label=f'{approach}: all results')

        # Prepare data for [Accuracy per level] plot:
        low_park_acc_lvl = np.array([np.min(park_acc_lvl[lvl]) for lvl in lvls]) # Lowest parking accuracy per level [%]
        high_park_acc_lvl = np.array([np.max(park_acc_lvl[lvl]) for lvl in lvls]) # Highest parking accuracy per level [%]

        # Add data to [Accuracy per level] plot:
        ax_lvl.scatter(
            lvls,low_park_acc_lvl,
            color=color_min[approach],edgecolor=color_min_edge[approach],
            marker=markers[approach],label=f'{approach}: lowest score',alpha=1)
        ax_lvl.bar(
            lvls,high_park_acc_lvl-low_park_acc_lvl,bottom=low_park_acc_lvl,
            width=bar_width[approach],color=color_bar[approach],
            alpha=0.3,label=f'{approach}: all results')
    
    # General features for plot:

    # Include scoring threshold, if any:
    if score_thres is not None:
        for ax in [ax_gral,ax_lvl]:
            ax.axhline(80,ls='--',alpha=0.6,lw=1,color='red',label='Threshold')

    # Set axes, labels and legend:
    ax_lvl.set(xlabel='Parking level',ylabel='Parking accuracy [%]')
    ax_gral.set(xlabel='Accuracy type',ylabel='Overall accuracy [%]')
    for ax in [ax_gral,ax_lvl]:
        ax.legend()
        if y_lims is not None:
            ax.set_ylim(y_lims)
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name,save_format=save_format)      

# ============================================================================

def plot_preds_single_RF(
    preds_track,
    mag_files,
    RF_number=0,
    i_lims = [0,-1],
    save_name=None,
    save_format='png',
    figsize=(8,3)
    ):
    """
    For a chosen rotational frame, plot the magnetic predictors vs time in a single plot, and the 
    Z-predictions and ground truth vs time in a separate plot.

    --- Inputs ---

    {preds_track} [Dictionary]: Prediction results for general tracking, same format as in function 
    "load_preds". It must contain a single key representing the rotational frame, its value a Dataframe
    with columns "Time_s", "Truth_m" and "Preds_m".
    {mag_files} [List]: Each element is a string containing the path to the magnetic data. Be careful
    to select the same segments as in the testing dataset (time is sorted automatically). Each magnetic
    file must be in .csv format, with columns "Time_s", "Bx_nT", "By_nT" and "Bz_nT".
    {RF_number} [Integer]: Index order identifying the rotational frames within the {preds_track}.
    {i_lims} [List or None]: Index limits for plotting figures. If None, plot all data. If a list is provided,
    it must have the format [i_min,i_max] representing the index range for the plot.
    {save_name} [String]: filename for the figure (don't include the extension). If None, no figure is saved.
    {save_format} [String]: saving format for the figure,don't include the dot.
    {figsize} [Tuple]: 2 integer-elements indicating the width and height dimensions for the figure.

    --- Return ---

    Figures for the magnetic predictors (top) and Z-predictions vs ground truth (bottom).

    """

    # Identify Rotational frame and select data:
    RF = list(preds_track.keys())[RF_number] # Format 'polX_azimY', with X and Y in [degrees]   
    preds_selec =  preds_track[RF] # Select relevant data
    print(f'Chosen frame: {RF}')

    # Load magnetic files from testing dataset:
    mag_data = pd.concat([pd.read_csv(file) for file in mag_files]) # Load DataFrames
    mag_data = mag_data.sort_values('Time_s') # Sort them according to time

    # Extract time and magnetic fields:
    t = mag_data['Time_s'].to_numpy() # Time [s]
    Bx = mag_data['Bx_nT'].to_numpy() # X-component of magnetic field [nT]
    By = mag_data['By_nT'].to_numpy() # Y-component of magnetic field [nT]
    Bz = mag_data['Bz_nT'].to_numpy() # Z-component of magnetic field [nT]

    # Rotate magnetic data:
    pol = float(ML_general.find_between(RF,'pol','_'))/180*np.pi # Polar angle for rotating vector [rad]
    azim =  float(ML_general.find_between(RF,'azim',''))/180*np.pi # Azimuth angle for rotating vector [rad]
    alpha = 90 # Rotating angle [degree]
    n_vec = np.array( # Rotating vector, XYZ format in Laboratory frame
        [np.sin(pol)*np.cos(azim), 
         np.sin(pol)*np.sin(azim), 
         np.cos(pol)])
    Bx_rot, By_rot, Bz_rot = ML_twdw.rotate_3D(
        np.array([Bx,By,Bz]),n_vec,alpha) # Rotated fields Bx', By', Bz' [nT]

    # Obtain Z-predictions and check that the time is correct:
    preds = preds_selec.sort_values('Time_s')
    t_pred = preds['Time_s'].to_numpy() #  Time [s]
    Z_pred = preds['Preds_m'].to_numpy() # Z-Predictions [m]
    Z_true = preds['Truth_m'].to_numpy() # Ground truth for Z-positions [m]

    # Reduce magnetic fields to those times in common with the predictions:
    indexes = np.isin(t, t_pred) # Identify indexes
    t, Bx_rot, By_rot, Bz_rot = t[indexes], Bx_rot[indexes], By_rot[indexes], Bz_rot[indexes] # Reduce arrays

    # Check i_lims:
    if i_lims[0]>len(t):
        print('Lower index boundary is too high, converted to i=0.')
        i_lims[0] = 0
    if i_lims[1]>len(t):
        print('Upper index boundary is too high, converted to i=-1.')
        i_lims[0] = -1
    i1,i2 = i_lims

    # Plot figure for magnetic fields:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t[i1:i2],(Bx_rot-Bx_rot.min())[i1:i2],color=color_Bx)
    ax.plot(t[i1:i2],(By_rot-By_rot.max())[i1:i2],color=color_By)
    ax.plot(t[i1:i2],(Bz_rot-Bz_rot.max()-(By_rot.max()-By_rot.min()))[i1:i2],color=color_Bz)
    ax.set_xticks([]) # Remove x_ticks
    ax.set_yticks([20]) # Set an arbitrary number to align this plot with the next one
    for spine in ['right','top','bottom']:
        ax.spines[spine].set_visible(False)
    ax.set_ylabel('Magnetic signal')
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name+'_mag_fields',save_format=save_format)  

    # Plot figure for Z-predictions:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_pred[i1:i2]/60,Z_pred[i1:i2],color=color_Zpred,alpha=1)
    ax.plot(t_pred[i1:i2]/60,Z_true[i1:i2],ls='--',color=color_Ztrue,alpha=0.8)
    for z in np.append(0,4.1+np.arange(0,3.7*6+0.1,3.7)):
        ax.axhline(z,lw=0.5,alpha=0.3,color='brown')
    for spine in ['right','top']:
        ax.spines[spine].set_visible(False)
    ax.set(xlabel='Time [min]', ylabel='Z-Position [m]')
    fig.tight_layout()
    if save_name:
        ML_general.save_file(save_name+'_Zpreds',save_format=save_format)  

# ============================================================================

    #     #df_total = df_total.reset_index() # Decouple time and z_true
    #     Z_abs_diff[name] = np.abs(df_total[df_total.columns[2]]-df_total[df_total.columns[1]]) # [m]
    #     acc_1m[name] = np.round(sum(Z_abs_diff[name]<1)/len(Z_abs_diff[name])*100,2) # Accuracy for 1-m-threshold
    #     acc_0p5m[name] = np.round(sum(Z_abs_diff[name]<0.5)/len(Z_abs_diff[name])*100,2) # Accuracy for 1-m-threshold
        

    # # Identify the parking intervals (equal in all rotational frames):
    # pred1 = preds[all_names[0]]









#     {z_thres} [Float]: Z-position threshold to compute correct predictions according
    #to the absolute difference between predicted and ground truth values.

    # Values for each 
    #key are boolean-type Numpy arrays representing correct (True) or incorrect (False)
    #predictions, correlated with {t_col}, according to {z_thres}.