# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:13:25 2025

@author: pedro.brandao

Function for PCA


"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from transformations import apply_dilution_factor
from spectrum_plotter import plotpartialspectrum_R2
from sklearn.preprocessing import StandardScaler

# Define the main function
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb

from matplotlib.colors import LinearSegmentedColormap

def process_pca(
    number_of_PCs=2, data="Standard", prepreprocess=StandardScaler(), include_time=0,
    data2use_group="Assay_ori", data2use='all', days2include="all", discrimination='Assay',
    personalized_discrimination=False, colors=['C0', 'C2', 'C1', 'C3', 'C4', 'C5', 'C6'],
    rgb_colors=False, PCs2plot=[1, 2], negatePC1=0, negatePC2=0, legend=1, dotsize=50,
    arrow_factor=2, arr_head=0.05, figsize=(7, 6), xlim=None, ylim=None, monitordata=None,
    fluorodata=None, absdata=None, absextdata=None, taxondata=None, show_spectral_heatmaps=False,
    showloads=True, ax=False, marker = 'o', font = 10, cmap4spectra = 'YlGn'
):
    """
    Process and visualize PCA for given data.
    """
    data_map = {
        "Standard": monitordata, "Abs": absdata, "Absext": absextdata,
        "Metagenome": taxondata, "2DF": fluorodata, "MiniSpec": fluorodata
    }
    if data not in data_map:
        raise ValueError("Unsupported data type.")
    
    df = data_map[data].dropna()
    if data2use != 'all':
        df = df[df.index.get_level_values(data2use_group).isin(data2use)]
    if days2include != 'all':
        df = df[df.index.get_level_values('Day').isin(days2include)]
    
    data_raw = df[data_map[data].columns]
    if include_time:
        data_raw["Time"] = df.index.get_level_values("Day")
    if data == 'Standard' and personalized_discrimination:
        data_raw = data_raw.drop(columns=[discrimination])
    
    pipeline = Pipeline([('Pre-pre-process', prepreprocess)])
    data_transf = pd.DataFrame(pipeline.fit_transform(data_raw), columns=data_raw.columns, index=data_raw.index)
    
    pca = PCA(n_components=number_of_PCs)
    pca_result = pca.fit_transform(data_transf)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    if negatePC1:
        loadings[:, PCs2plot[0] - 1] *= -1
        pca_result[:, PCs2plot[0] - 1] *= -1
    if negatePC2:
        loadings[:, PCs2plot[1] - 1] *= -1
        pca_result[:, PCs2plot[1] - 1] *= -1
    
    pca_df = pd.DataFrame(pca_result[:, [PCs2plot[0] - 1, PCs2plot[1] - 1]], columns=PCs2plot, index=df.index)
    pca_df2 = pd.DataFrame(loadings[:, [PCs2plot[0] - 1, PCs2plot[1] - 1]], columns=PCs2plot, index=data_transf.columns)
    
    plt.figure(figsize=figsize)
    plot_data = pca_df.join(monitordata[discrimination]) if personalized_discrimination else pca_df
    plot_data = plot_data.sort_values(by=discrimination)
    
    sns.scatterplot(data=plot_data, x=PCs2plot[0], y=PCs2plot[1], hue=discrimination, palette=[to_rgb(c) for c in colors] if rgb_colors else colors, legend=legend, s=dotsize, marker = marker)
    plt.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='grey', linestyle='--', linewidth=0.5)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f'PCA Biplot - {data}' + (f' colored by {discrimination}' if personalized_discrimination else ''))
    plt.xlabel(f'PC{PCs2plot[0]} ({pca.explained_variance_ratio_[PCs2plot[0] - 1]:.2%} explained variance)')
    plt.ylabel(f'PC{PCs2plot[1]} ({pca.explained_variance_ratio_[PCs2plot[1] - 1]:.2%} explained variance)')
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    
    if (data == "Standard" or data == "Metagenome") and showloads:
        for i in range(pca_df2.shape[0]):
            plt.arrow(0, 0, pca_df2.iloc[i, 0] * arrow_factor, pca_df2.iloc[i, 1] * arrow_factor, color='red', alpha=0.5, head_width=arr_head)
            if font != 0:
                plt.text(pca_df2.iloc[i, 0] * 1.1 * arrow_factor, pca_df2.iloc[i, 1] * 1.1 * arrow_factor, pca_df2.index[i], color='black', fontsize=font)
    
    if legend: plt.legend(title=discrimination)
    plt.show()
    
    if show_spectral_heatmaps:
        for pc in PCs2plot:
            if data == "2DF":
                plotpartialspectrum_R2(data_raw, pca_df2.T, pc, vmin=-1, vmax=1, cmap=cmap4spectra)
                plt.title(f"PC{pc}")
                plt.show()
            elif data == "MiniSpec":
                varx = np.arange(0, len(fluorodata.columns), 1)
                plt.figure(figsize=(6, 2))
                plt.bar(varx, pca_df2.T.loc[pc], width=0.5, color='C1')
                plt.plot(varx, fluorodata.mean() / 1000, c='C7')
                plt.xticks([0, 65, 119], ['EE 470/475-800 nm', 'EE 530/475-800 nm', 'EE 640/475-800 nm'], ha='left')
                plt.axvline(x=0, color='b', linestyle=':', label='EE 470/475-800 nm')
                plt.axvline(x=65, color='g', linestyle=':', label='EE 530/475-800 nm')
                plt.axvline(x=119, color='r', linestyle=':', label='EE 640/475-800 nm')
                plt.ylabel(f'Loading on {pc}')
                plt.title(f"PC{pc}")
                plt.show()
            else:
                varx = [int(item[-6:-3]) if data == "Abs" else int(item[-3:]) for item in pca_df2.index]
                values = pca_df2.T.loc[pc]
                
                # Normalize separately for negative and positive values
                neg = values[values < 0]
                pos = values[values >= 0]
                
                # Create colormaps: yellow (neg) to green (zero) for negatives, yellow (zero) to green (pos) for positives
                # For negatives: yellow (more negative) to green (towards zero)
                neg_cmap = LinearSegmentedColormap.from_list('neg_yellow_green', ['yellow', 'olive'])
                # For positives: yellow (zero) to green (more positive)
                pos_cmap = LinearSegmentedColormap.from_list('pos_yellow_green', ['olive', 'green'])
                
                colors = []
                for v in values:
                    if v < 0 and len(neg) > 0:
                        norm = (v - neg.min()) / (0 - neg.min() + 1e-12)  # Negative: -max (yellow) to 0 (green)
                        colors.append(neg_cmap(norm))
                    elif v >= 0 and len(pos) > 0:
                        norm = (v - 0) / (pos.max() - 0 + 1e-12)  # Positive: 0 (yellow) to max (green)
                        colors.append(pos_cmap(norm))
                    else:
                        colors.append('grey')  # fallback for rare cases
                
                plt.figure(figsize=(1.5, 1))
                plt.bar(varx, values, width=3, color=colors)
                plt.xlabel('Wavelength (nm)')
                plt.ylabel(f'Loading on {pc}')
                plt.title(f"PC{pc}")
                plt.show()

    return pca_df,pca_df2

import matplotlib.colors as mcolors

def to_rgb(color):
    """Convert color code, name, hex, or RGB tuple to normalized RGB tuple."""
    if isinstance(color, str):  # Matplotlib color code, name, or hex
        try:
            return mcolors.to_rgb(color)  # Handles named colors and hex
        except ValueError:
            if color.startswith('C'):  # Handle Matplotlib 'C0', 'C1', etc.
                return mpl_color_to_rgb(color)
            raise ValueError(f"Unknown color code: {color}")
    elif isinstance(color, tuple) and len(color) == 3:  # RGB tuple
        if all(0 <= channel <= 1 for channel in color):
            return color  # Already normalized
        elif all(0 <= channel <= 255 for channel in color):
            return tuple(channel / 255 for channel in color)  # Normalize
        else:
            raise ValueError(f"RGB values must be in the range 0-1 or 0-255: {color}")
    else:
        raise ValueError(f"Invalid color format: {color}")

def mpl_color_to_rgb(code):
    """
    Converts a Matplotlib standard color code (e.g., 'C0', 'C1') to an RGB tuple.

    Args:
        code (str): The Matplotlib color code (e.g., 'C0').

    Returns:
        tuple: An RGB tuple (e.g., (0, 0, 0)).
    """
    # Get the color from Matplotlib's default color cycle
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_index = int(code[1:]) % len(color_list)  # Handle cyclic color cycle

    # Convert the color name to RGB
    return mcolors.to_rgb(color_list[color_index])


from matplotlib.patches import Ellipse
import scipy.stats as stats


def pca_spiruline(df_ori, variables, discrimination, scaler=StandardScaler(), data2use=None, 
                  plot_spectral_loadings=False, label=None, hotel=False, legend=True,
                  point_colors=None, figsize=(6,4.5), n_comps=2, components_to_plot=(1, 2),
                  negatePC1 = False, negatePC2 = False):
    """
    Perform PCA with flexible coloring, Hotelling’s T² outlier detection, and loadings.

    Parameters:
    - df: pandas DataFrame with multi-index.
    - variables: list of columns to include in PCA.
    - discrimination: str, index level used to color PCA scores.
    - data2use: dict, e.g., {'Batch_ID': ['batch1','batch2']} to filter observations.
    - plot_spectral_loadings: bool, if True plots loadings on mean absorbance spectrum.
    - label: str or None, index level used to label each point on PCA scores.
    - hotel: bool, if True performs Hotelling’s T² outlier detection.
    - point_colors: str (colormap) or list of color strings.
    - figsize: tuple, size of figures in inches.
    - n_comps: int, number of components to keep in PCA.
    - components_to_plot: tuple of two integers, components to plot (1-based index).

    Returns:
    - fig, ax: PCA score plot.
    - fig_loadings, axs_loadings (optional): Loadings plot (if requested).
    - outliers_dict (optional): Dict with 95% and 99% outlier Obs_IDs (if hotel=True).
    """
    df = df_ori.copy()
    
    # Filter based on data2use
    if data2use:
        for level, values in data2use.items():
            mask = df.index.get_level_values(level).isin(values)
            df = df[mask]

    # Scale the data
    X = df[variables].values
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_comps)
    scores = pca.fit_transform(X_scaled)
    pcx, pcy = scores[:, components_to_plot[0] - 1], scores[:, components_to_plot[1] - 1]

    # Extract index levels
    index_df = df.index.to_frame(index=False)
    group_labels = index_df[discrimination].values
    label_values = index_df[label].values if label else [None] * len(df)
    obs_ids = index_df['Obs_ID'].values

    # DataFrame for plotting
    scores_df = pd.DataFrame({'PCX': pcx, 'PCY': pcy, 
                              discrimination: group_labels,
                              'Label': label_values,
                              'Obs_ID': obs_ids})

    # Handle point_colors
    unique_groups = sorted(np.unique(group_labels))
    num_groups = len(unique_groups)
    
    if point_colors is None:
        palette = sns.color_palette('Set2', n_colors=num_groups)
    elif isinstance(point_colors, str):
        cmap = plt.get_cmap(point_colors)
        palette = [cmap(i / max(1, num_groups - 1)) for i in range(num_groups)]
    elif isinstance(point_colors, list):
        palette = point_colors[:num_groups]
    else:
        raise ValueError("point_colors must be None, a colormap string, or a list of color strings.")
    
    # Ensure `unique_groups` and `group_labels` use consistent types for mapping
    color_mapping = {val: color for val, color in zip(unique_groups, palette)}
    
    # Map colors to `group_labels`, ensuring no type mismatch
    scores_df['Color'] = [color_mapping[val] for val in group_labels]

    # Plot PCA scores
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.scatter(scores_df['PCX'], scores_df['PCY'], 
               c=scores_df['Color'], s=50, edgecolor='k')

    # Label each point if requested
    if label:
        for i in range(len(scores_df)):
            ax.text(scores_df['PCX'][i], scores_df['PCY'][i], str(scores_df['Label'][i]),
                    fontsize=8, ha='right', va='bottom')

    # Hotelling’s T² Outlier Detection
    outliers_dict = None
    if hotel:
        cov = np.cov(scores.T)
        inv_cov = np.linalg.inv(cov)
        mean_scores = np.mean(scores, axis=0)
        T2 = np.array([ (s - mean_scores).T @ inv_cov @ (s - mean_scores) for s in scores ])

        T2_95 = stats.chi2.ppf(0.95, df=2)
        T2_99 = stats.chi2.ppf(0.99, df=2)

        outliers_95, outliers_99 = [], []

        for i, t2 in enumerate(T2):
            if t2 > T2_99:
                outliers_99.append(scores_df['Obs_ID'][i])
                color = 'red'
            elif t2 > T2_95:
                outliers_95.append(scores_df['Obs_ID'][i])
                color = 'orange'
            else:
                continue

            # ax.text(scores_df['PCX'][i], scores_df['PCY'][i], 
            #         f"{scores_df['Obs_ID'][i]}", fontsize=9, 
            #         ha='left', va='top', color=color, weight='bold')

        # Draw Hotelling’s ellipses
        for conf, ls, color in zip([0.95, 0.99], ['--', ':'], ['orange', 'red']):
            radius = np.sqrt(stats.chi2.ppf(conf, df=2))
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order][:2], vecs[:, order][:2]

            # Debug prints to check variable states
            print(f"vals: {vals}")
            print(f"vecs: {vecs}")
            print(f"vecs[:, 0]: {vecs[:, 0]}")
            print(f"type(vecs[:, 0]): {type(vecs[:, 0])}")
            print(f"shape(vecs[:, 0]): {vecs[:, 0].shape}")

            # Fix the input to np.arctan2
            theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2 * radius * np.sqrt(vals)
            ellipse = Ellipse(xy=mean_scores[:2], width=width, height=height,
                              angle=theta, edgecolor=color, fc='None', 
                              lw=2, ls=ls, label=f'Hotelling {int(conf*100)}%')
            ax.add_patch(ellipse)

        outliers_dict = {'95%': outliers_95, '99%': outliers_99}

    # Legend
    if legend:
        if num_groups <= 10:
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=color_mapping[grp], markersize=8, 
                       markeredgecolor='k', label=str(grp)) 
                       for grp in unique_groups]
            ax.legend(handles=handles, title=discrimination)
        else:
            ax.legend(title=discrimination)

    # Emphasize main axes
    ax.axhline(0, color='black', lw=1.5)
    ax.axvline(0, color='black', lw=1.5)

    ax.set_title('PCA Score Plot')
    ax.set_xlabel(f'PC{components_to_plot[0]} ({pca.explained_variance_ratio_[components_to_plot[0] - 1]*100:.1f}%)')
    ax.set_ylabel(f'PC{components_to_plot[1]} ({pca.explained_variance_ratio_[components_to_plot[1] - 1]*100:.1f}%)')
    ax.grid(True)
    
    if negatePC1:
        ax.invert_xaxis()
    if negatePC2:
        ax.invert_yaxis()

    plt.tight_layout()
    

    
    plt.show()

    # Spectral Loadings Plot: Separate Figures for PC1 and PC2
    if plot_spectral_loadings:
        loadings = pca.components_[:2]
        mean_spectrum = df[variables].mean()
    
        for i, pc_label in enumerate(['PC1', 'PC2']):
            fig, ax_loading = plt.subplots(figsize=(3, 2), dpi=300)  # Small figure, paper-friendly
            loading_vals = loadings[i]
            if negatePC2:
                loading_vals *=-1
            
            colors = [((0/255, 120/255, 0/255) if val > 0 else (255/255, 204/255, 0/255)) for val in loading_vals]
            ax_loading.bar(variables, loading_vals, color=colors, edgecolor=colors, alpha=0.6)
            ax2 = ax_loading.twinx()
            ax2.plot(variables, mean_spectrum, color='k', label='Mean Spectrum')
            ax_loading.set_title(f'{pc_label}', fontsize=9)
            ax_loading.set_ylabel('Loadings', fontsize=8)
            ax_loading.set_xlabel('Wavelength', fontsize=8)
            ax_loading.grid(True, linewidth=0.3)
            ax_loading.tick_params(axis='both', labelsize=7)
            ax2.set_ylabel('Mean Absorbance', fontsize=8)
            ax2.tick_params(axis='y', labelsize=7)
            plt.tight_layout()
            # plt.gca().invert_xaxis()
            

            plt.show()


    return outliers_dict


