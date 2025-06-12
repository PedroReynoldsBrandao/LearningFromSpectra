# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:07:53 2025

@author: pedro.brandao
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
import seaborn as sns


def plot_rmsecv_barplot(search, stdy, figsize=(4, 3), vmin=None, vmax=None):
    """
    Plot barplot of RMSECVs for all pipelines from a GridSearchCV search.

    Parameters
    ----------
    search : fitted GridSearchCV or similar object
        Must have .cv_results_ with 'mean_test_score' key.
    stdy : float
        Training response, for normalization.
    figsize : tuple, default (5, 3.33)
        Figure size for the plot.

    Returns
    -------
    None (displays a plot)
    """
    results = search.cv_results_
    mean_test_scores = results['mean_test_score']
    iterations = list(range(1, len(mean_test_scores) + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    rmsecvs = np.sqrt(-mean_test_scores)
    scatter = ax.scatter(iterations, rmsecvs, c=rmsecvs, cmap='viridis_r', edgecolor='black', vmax=vmax, vmin=vmin)

    min_idx = np.nanargmin(rmsecvs)
    # ax.bar(min_idx + 1, rmsecvs[min_idx], width=0.001*len(iterations), color='C1')
    plt.plot(min_idx + 1,rmsecvs[min_idx], 'o', markersize=14,
             markerfacecolor='none', markeredgecolor='C1', markeredgewidth=2, label='Best')

    # Baselines and best marker
    ax.axhline(y=stdy, color='blue', linestyle='--', linewidth=1)
    ax.axhline(y=stdy * np.sqrt(0.5), color='C0', linestyle='--', linewidth=1)
    ax.axhline(y=rmsecvs[min_idx], color='C1', linestyle='--', linewidth=1)

    # ax.set_ylim([0, 2*stdy])
    ax.set_xlabel('Pipelines')
    ax.set_xticks([])
    ax.set_title('GridSearchCV')

    # After plotting, get the rightmost x position for text placement
    x_text = ax.get_xlim()[1]

    # Prepare texts
    percent_explained = 100 - ((rmsecvs[min_idx] / stdy) ** 2 * 100)
    texts = [
        (stdy, 'No variance explained', 'blue'),
        (stdy * np.sqrt(0.5), '50% variance explained', 'C0'),
        (rmsecvs[min_idx], f'{percent_explained:.1f}% variance explained', 'C1')
    ]

    for y, text, color in texts:
        ax.text(x_text*1.05, y, text, color=color, va='center', ha='left', fontsize=10)

    # Optionally, expand xlim so all texts fit
    ax.set_xlim(ax.get_xlim()[0], x_text + 2)
    ax.set_ylim([0,stdy*1.2])
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RBFInterpolator



def plot_hyperparam_contour(
    search,
    y_train,
    param_x,
    param_y,
    score_label='RMSECV',
    n_iter=1000,
    figsize=(9, 9),
    vmax=None,
    vmin=None
):
    results = search.cv_results_
    params = results['params']
    mean_test_scores = results['mean_test_score']
    
    rmsecvs = np.sqrt(-mean_test_scores)
    min_rmsecv = np.nanmin(rmsecvs)
    
    results_df = pd.DataFrame(params)
    results_df['mean_test_score'] = rmsecvs

    x_values = np.array([param[param_x] for param in params])
    y_values = np.array([param[param_y] for param in params])
    scores = results_df['mean_test_score'].values

    def is_all_positive_numeric(arr):
        try:
            arr = np.array(arr, dtype=float)
            return np.issubdtype(arr.dtype, np.number) and np.all(arr > 0)
        except Exception:
            return False

    x_numeric = is_all_positive_numeric(x_values)
    y_numeric = is_all_positive_numeric(y_values)

    plt.figure(figsize=figsize)
    cbar = None

    std_val = float(y_train.std())

    if x_numeric and y_numeric:
        x_log = np.log10(x_values.astype(float))
        y_log = np.log10(y_values.astype(float))

        x_lin = np.linspace(x_log.min(), x_log.max(), n_iter)
        y_lin = np.linspace(y_log.min(), y_log.max(), n_iter)
        x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

        rbf_interpolator = RBFInterpolator(
            np.column_stack((x_log, y_log)), scores, kernel='linear'
        )
        grid_scores = rbf_interpolator(
            np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
        ).reshape(x_mesh.shape)
        x_mesh_plot = 10 ** x_mesh
        y_mesh_plot = 10 ** y_mesh

        contour = plt.contourf(
            x_mesh_plot, y_mesh_plot, grid_scores,
            levels=20, cmap='viridis_r', alpha=0.8, vmax=vmax, vmin=vmin
        )
        sc = plt.scatter(x_values.astype(float), y_values.astype(float), c=scores, cmap='viridis_r', edgecolor='black', vmax=vmax, vmin=vmin)

        # Mark the best point with a red circle
        best_idx = np.argmin(scores)
        plt.plot(x_values[best_idx], y_values[best_idx], 'o', markersize=14, markerfacecolor='none', markeredgecolor='C1', markeredgewidth=2, label='Best')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        cbar = plt.colorbar(contour, label=score_label)
        
    else:
        pivot = results_df.pivot_table(index=param_y, columns=param_x, values='mean_test_score')
        ax = sns.heatmap(
            pivot,
            annot=False,
            fmt=".2f",
            cbar_kws={"label": score_label},
            vmax=vmax,
            vmin=vmin,
            cmap='viridis_r'
        )
        cbar = ax.collections[0].colorbar
        plt.xlabel(param_x, fontsize=16)
        plt.ylabel(param_y, fontsize=16)
        plt.tight_layout()  # This will fix layout for heatmap

        # Mark the best point with a red circle
        # Find indices of the minimum in the pivot table
        min_idx = np.unravel_index(np.nanargmin(pivot.values), pivot.shape)
        # Get the corresponding x and y values from the pivot table's columns and index
        best_x = pivot.columns[min_idx[1]]
        best_y = pivot.index[min_idx[0]]
        
        # Draw a circle at the center of the best cell
        ax.plot(
            min_idx[1] + 0.5,  # x position (add 0.5 to center on cell)
            min_idx[0] + 0.5,  # y position (add 0.5 to center on cell)
            'o',
            markersize=14,
            markerfacecolor='none',
            markeredgecolor='C1',
            markeredgewidth=2,
            label='Best'
        )

    # ---- ADDED LINES ON COLORBAR ----
    if cbar is not None:
        percent_explained = 100 - ((min_rmsecv / std_val) ** 2 * 100)
        cbar.ax.hlines(min_rmsecv, xmin=0, xmax=1, colors='C1', linestyles='dotted', linewidth=2, transform=cbar.ax.get_yaxis_transform())
        cbar.ax.text(1.5, min_rmsecv, 'Best:'+f'{percent_explained:.1f}% variance explained', color='C1', va='center', ha='left', transform=cbar.ax.get_yaxis_transform())
        
        cbar.ax.hlines(std_val * np.sqrt(0.5), xmin=0, xmax=1, colors='C0', linestyles='dotted', linewidth=2, transform=cbar.ax.get_yaxis_transform())
        cbar.ax.text(1.5, std_val * np.sqrt(0.5), f"50% variance explained", color='C0', va='center', ha='left', transform=cbar.ax.get_yaxis_transform())
        
        cbar.ax.hlines(std_val, xmin=0, xmax=1, colors='C0', linestyles='dotted', linewidth=2, transform=cbar.ax.get_yaxis_transform())
        cbar.ax.text(1.5, std_val, f"0% variance explained", color='C0', va='center', ha='left', transform=cbar.ax.get_yaxis_transform())
    plt.title(score_label)
    plt.show()