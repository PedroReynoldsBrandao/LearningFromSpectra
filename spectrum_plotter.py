# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:56:48 2024

@author: pedro.brandao
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

    

def average_pooling(data, pool_size, ex, em):
    """
    Apply average pooling to the data.
    
    Parameters:
    data (np.ndarray): 2D array of the spectrum data.
    pool_size (int): Size of the pooling window.
    
    Returns:
    np.ndarray: Pooled data.
    """
    output_shape = (data.shape[0] // pool_size, data.shape[1] // pool_size)
    pooled_data = np.zeros(output_shape)
    avg_ex = np.zeros(output_shape[0])
    avg_em = np.zeros(output_shape[1])
      
    for i in range(output_shape[0]):
        avg_ex[i] = np.mean(ex[i*pool_size:(i+1)*pool_size])
        
        for j in range(output_shape[1]):
            pooled_region = data[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]      
            pooled_data[i, j] = np.mean(pooled_region)
            
            avg_em[j] = np.mean(em[j*pool_size:(j+1)*pool_size])

            
    return pooled_data, avg_ex, avg_em


def plot_spectrum(df2plot, spectrum_index, pool_size=False,log=False):
    """
    Plots the original and pooled fluorescence spectrum as heat maps with contour lines.
    
    If pool_size == False, just plots the provided spectrum
    
    Parameters:
    df (pd.DataFrame): Dataframe containing the fluorescence spectra.
    spectrum_index (int): Index of the spectrum to plot.
    pool_size (int): Size of the pooling window.
    """
    pooled_df = []
    
    # Extract the column names
    column_names = df2plot.columns
    excitation_wavelengths = []
    emission_wavelengths = []
    
    # Parse the excitation and emission wavelengths from the column names
    for name in column_names:
        excitation, emission = name.split(' / ')
        excitation_wavelengths.append(int(excitation.replace('EEP ', '').replace('nm', '').strip()))
        emission_wavelengths.append(int(emission.replace('nm', '').strip()))
    
    # Get unique excitation and emission wavelengths
    unique_excitation = sorted(set(excitation_wavelengths))
    unique_emission = sorted(set(emission_wavelengths))
    
    # Create a 2D array for the original heatmap data
    original_data = np.zeros((len(unique_excitation), len(unique_emission)))
    
    # Fill the original heatmap data
    for i, exc in enumerate(unique_excitation):
        for j, em in enumerate(unique_emission):
            if em > exc:
                col_name = f'EEP {exc} / {em} nm'
                original_data[i, j] = df2plot.loc[spectrum_index, col_name]
                if log:
                    original_data[i, j] = np.log(df2plot.loc[spectrum_index, col_name]+0.1)
    

    if log:
        vmin = 0
        vmax = 7
    else:
        vmin = 0
        vmax = 1000
    
    # Create the heatmap for original data
    # plt.figure(figsize=(10, 8))
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (4,4)

    if pool_size:
        # plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        
        
    contourf = plt.contourf(unique_emission,unique_excitation, original_data, #levels = [0,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1],
                 cmap='turbo', origin='lower',vmin=vmin,vmax=vmax)
    
    
    CS1 = plt.contour(unique_emission,unique_excitation,original_data, levels=contourf.levels, colors='white')#, linestyles='dashed')
    # plt.clabel(CS1, inline=1, fontsize=10, fmt='%1.1f')
    # plt.gca().invert_yaxis()  # Invert the x-axis for emission wavelengths
    # plt.xlabel('Emission Wavelength (nm)')
    # plt.ylabel('Excitation Wavelength (nm)')
    plt.yticks([400,600])
    
    # plt.title(f'Original Spectrum (Spectrum Index: {spectrum_index})')
    
    # Apply average pooling to the original data
               
    if pool_size:
        pooled_data, avg_ex, avg_em = average_pooling(original_data, pool_size,unique_excitation,unique_emission)
        
    # Create the heatmap for pooled data
    
        plt.subplot(1, 2, 2)
        plt.contourf(avg_em, avg_ex, pooled_data, cmap='turbo', origin='lower')
        # CS2 = plt.contour(pooled_data, levels=10, colors='white', linestyles='dashed')
        # plt.gca().invert_yaxis() 
        # plt.clabel(CS2, inline=1, fontsize=10, fmt='%1.1f')
        # plt.xlabel('Emission Wavelength (nm)')
        # plt.ylabel('Excitation Wavelength (nm)')
        # plt.title(f'Pooled Spectrum (Spectrum Index: {spectrum_index}, Pool Size: {pool_size})')
        
        plt.show()
    
        columns = []
        values = []
        
        # Unravel the pooled data
        for i, exc in enumerate(avg_ex):
            for j, em in enumerate(avg_em):
                if em > exc:
                    em = int(em)
                    exc = int(exc)
                    col_name = f'EEP {exc} / {em} nm'
                    columns.append(col_name)
                    values.append(pooled_data[i, j])
        
        pooled_df = pd.Series(data=values,index=columns,name=spectrum_index)
    
    return pooled_df




def plot_partialspectrum(df2plot_ref,df2plot, spectrum_index=False,vmin=0,vmax=1000, log=False, contours = True, scale=False,
                         lims = False, background = 0.1, figsize = (1.8,1.8), colorbar = False):
    """

    """
    plt.rcParams['figure.dpi'] = 250

    plt.rcParams['figure.figsize'] = figsize

    # Extract the column names
    column_names = df2plot_ref.columns
    excitation_wavelengths = []
    emission_wavelengths = []
    
    if log:
        vmax = 7
    
    # Parse the excitation and emission wavelengths from the column names
    for name in column_names:
        excitation, emission = name.split(' / ')
        excitation_wavelengths.append(int(excitation.replace('EEP ', '').replace('nm', '').strip()))
        emission_wavelengths.append(int(emission.replace('nm', '').strip()))
    
    # Get unique excitation and emission wavelengths
    unique_excitation = sorted(set(excitation_wavelengths))
    unique_emission = sorted(set(emission_wavelengths))
    
    # Create a 2D array for the original heatmap data
    data2plot = np.zeros((len(unique_excitation), len(unique_emission))) - background
    if spectrum_index:
        datafromdf = df2plot.loc[spectrum_index].squeeze()
    else:
        datafromdf = df2plot.max()
    
    # Fill the original heatmap data
    for i, exc in enumerate(unique_excitation):
        for j, em in enumerate(unique_emission):
            if em > exc:
                col_name = f'EEP {exc} / {em} nm'
                if col_name in datafromdf.index:
                    if log:
                        data2plot[i, j] = np.log(datafromdf[col_name]+ 1)
                    else:
                        data2plot[i, j] = datafromdf[col_name]

    # Create the heatmap for reference data
    # plt.figure(figsize=(10, 8))

    if log:
        vmax = 7
    if scale:
        contourf = plt.contourf(unique_emission,unique_excitation, data2plot, levels = 10,
                 cmap='turbo', origin='lower')
    else:
            
        contourf = plt.contourf(unique_emission,unique_excitation, data2plot, levels = 5, #levels = [0,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1],
                     cmap='turbo', origin='lower',vmin=vmin,vmax=vmax)
        
    if contours:
        CS1 = plt.contour(unique_emission,unique_excitation,data2plot, levels=contourf.levels, colors='white')#, linestyles='dashed')
    # plt.clabel(CS1, inline=1, fontsize=10, fmt='%1.1f')
    # plt.xlabel('Emission Wavelength (nm)')
    # plt.ylabel('Excitation Wavelength (nm)')
    if spectrum_index:
        plt.title(f'{spectrum_index}')
    if colorbar:
        plt.colorbar()
    if lims:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    plt.show()


    
    
def plot_partialspectrum2(ax,df2plot_ref,df2plot, spectrum_index=False,vmin=0,vmax=7,lims=False,
                          contours=False,log = True, background = 0.1):
    """

    """
    
    if not ax:
        figsize = (3.5, 3)
        fig, ax = plt.subplots(figsize=figsize)
        
    if log:
        vmax = 7   
    # Extract the column names
    column_names = df2plot_ref.columns
    excitation_wavelengths = []
    emission_wavelengths = []
    
    # Parse the excitation and emission wavelengths from the column names
    for name in column_names:
        excitation, emission = name.split(' / ')
        excitation_wavelengths.append(int(excitation.replace('EEP ', '').replace('nm', '').strip()))
        emission_wavelengths.append(int(emission.replace('nm', '').strip()))
    
    # Get unique excitation and emission wavelengths
    unique_excitation = sorted(set(excitation_wavelengths))
    unique_emission = sorted(set(emission_wavelengths))
    
    # Create a 2D array for the original heatmap data
    data2plot = np.zeros((len(unique_excitation), len(unique_emission))) - background
    if spectrum_index:
        datafromdf = df2plot.loc[spectrum_index].squeeze()
    else:
        datafromdf = df2plot.max()
    
    # Fill the original heatmap data
    for i, exc in enumerate(unique_excitation):
        for j, em in enumerate(unique_emission):
            if em > exc:
                col_name = f'EEP {exc} / {em} nm'
                if col_name in df2plot.columns:
                    if log:
                        data2plot[i, j] = np.log(datafromdf[col_name]+0.1)
                    else:
                        data2plot[i, j] = datafromdf[col_name]
                        
    # levels = [-0.1,-0.05,-0.025,-0.01,0,0.01,0.025,0.05,0.1]

    # Create the heatmap for reference data
    # ax.figure(figsize=(10, 8))


        
    contourf = ax.contourf(unique_emission,unique_excitation,
                 data2plot, cmap='turbo',levels=7,
                 origin='lower',vmax = vmax,vmin = vmin)
    
    if contours:
        ax.contour(unique_emission,unique_excitation,data2plot, levels=contourf.levels, colors='white')#, linestyles='dashed')
    
    # plt.contour(unique_emission,unique_excitation,data2plot, levels=contourf.levels, colors='white')#, linestyles='dashed')
    # plt.clabel(CS1, inline=1, fontsize=10, fmt='%1.1f')
    
    spectrum_used = np.round(len(df2plot.columns)/len(df2plot_ref.columns)*100,1)
    
    ax.set_xlabel('Emission Wavelength (nm)')
    ax.set_ylabel('Excitation Wavelength (nm)')
    ax.set_title(f'Spectrum used: {spectrum_used} %')
    
    if lims:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])

    
def plotpartialspectrum_R2(df2plot_ref,df2plot, spectrum_index,vmin,vmax,cmap,levels = 5):
    """

    """
    
    # Extract the column names
    column_names = df2plot_ref.columns
    excitation_wavelengths = []
    emission_wavelengths = []
    
    # Parse the excitation and emission wavelengths from the column names
    for name in column_names:
        excitation, emission = name.split(' / ')
        excitation_wavelengths.append(int(excitation.replace('EEP ', '').replace('nm', '').strip()))
        emission_wavelengths.append(int(emission.replace('nm', '').strip()))
    
    # Get unique excitation and emission wavelengths
    unique_excitation = sorted(set(excitation_wavelengths))
    unique_emission = sorted(set(emission_wavelengths))
    
    # Create a 2D array for the original heatmap data
    data2plot = np.zeros((len(unique_excitation), len(unique_emission))) #-0.001
    
    datafromdf = df2plot.loc[spectrum_index]
    
    # Fill the original heatmap data
    for i, exc in enumerate(unique_excitation):
        for j, em in enumerate(unique_emission):
            if em > exc:
                col_name = f'EEP {exc} / {em} nm'
                if col_name in df2plot.columns:
                    # data2plot[i, j] = np.log(datafromdf[col_name].iloc[0]+0.1)
                    data2plot[i, j] = datafromdf[col_name]

    data2plot[data2plot<vmin] = vmin
    # Create the heatmap for reference data
    # plt.figure(figsize=(1.5,1.5))
    plt.figure(figsize=(4,3))
        
    contourf = plt.contourf(unique_emission,unique_excitation, data2plot,levels = levels, #[0,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1],
                 cmap=cmap, origin='lower',vmin=vmin,vmax=vmax)
    plt.colorbar()
    CS1 = plt.contour(unique_emission,unique_excitation,data2plot, levels=contourf.levels, colors='white')#, linestyles='dashed')
    # plt.clabel(CS1, inline=1, fontsize=10, fmt='%1.1f')
    
    # plt.xlabel('Emission Wavelength (nm)')
    # plt.ylabel('Excitation Wavelength (nm)')
    # plt.show()
    

# from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator

def plot_meanpartialspectrum(df, df_transformed, days, assays, sampleinfo,
                             log = False,  contours = True, scale = False, 
                             lims = False,assay_id = 'Assay',vmin=0,vmax=1000,
                             title = False):

    for assay in assays:
        
        df_transformed2plot = df_transformed[df_transformed.index.get_level_values(assay_id) == assay]
        
        for day in days:
            if title:
                name = title
            else:
                name = 'Day ' + str(day) + ' of ' + assay
            df2plot = df_transformed2plot[df_transformed2plot.index.get_level_values('Day') == day]
            df2plot = pd.DataFrame(data=df2plot.mean(axis=0)).T.rename(index={0:name})
            if log:
                vmax = 7
            plot_partialspectrum(df, df2plot, spectrum_index=name, log = log, 
                                 contours = contours, scale = scale, lims=lims,
                                 vmin=vmin,vmax=vmax)
    


def plotmeanEEMs(
    factors,
    discriminator,
    plotdiffs,
    logtransf,
    days2use,
    varxID,
    df,
    colorbar=False,
    dataindex=False,
    contours=True
):
    """
    Plots the mean 2D fluorescence spectra (EEMs) for selected assays and days.

    Parameters:
        factors (list): List of group/category names (assays) to plot.
        discriminator (str): Column name used to identify groups (e.g., 'Assay').
        plotdiffs (bool): If True, plot the difference from the first day.
        logtransf (bool): If True, apply log transformation to the input data.
        days2use (list): List of days (values in 'Day' column) to plot.
        varxID (list): List of column names for the fluorescence data (EEM vector).
        df (pd.DataFrame): DataFrame containing the data.
        colorbar (bool): Whether to show a colorbar.
        dataindex (bool): If True, reset index before plotting.
        contours (bool): Whether to overlay contour lines.
    """
    plt.rcParams['figure.dpi'] = 300
    df_work = df.copy()
    if dataindex:
        df_work.reset_index(inplace=True)

    ex = np.arange(250, 795, 5)
    em = np.arange(260, 805, 5)
    ynew = np.arange(250, 790.5, 5)
    xnew = np.arange(260, 800.5, 5)

    for assay in factors:
        df_assay = df_work[df_work[discriminator] == assay].copy()
        x_aux = None  # Used for difference plotting

        for i, day in enumerate(days2use):
            df_day = df_assay[df_assay['Day'] == day]
            if df_day.empty:
                continue

            X = df_day[varxID].copy()
            if logtransf:
                X[X < 1] = 1
                X = np.log(X)

            x_mean = X.mean().T
            if i == 0:
                x_aux = x_mean.copy()
            if plotdiffs:
                x_plot = x_mean - x_aux
                x_aux = x_mean.copy()
            else:
                x_plot = x_mean

            if not x_plot.isna().any():
                # Build EEM grid
                z = np.zeros((len(ex), len(em)))
                idx = 0
                for i_ex, ex_val in enumerate(ex):
                    for j_em, em_val in enumerate(em):
                        if ex_val >= em_val:
                            z[i_ex, j_em] = 0
                        else:
                            z[i_ex, j_em] = x_plot.iloc[idx]
                            idx += 1

                # Interpolate EEM data for smooth plotting
                f = RegularGridInterpolator((ex, em), z, method='linear')
                new_coords = np.array([[y, x] for y in ynew for x in xnew])
                data1 = f(new_coords).reshape(len(ynew), len(xnew))

                plt.figure(figsize=(3, 3))
                
                n_levels = 7 if logtransf else 10
                
                contourf = plt.contourf(xnew, ynew, data1, levels=n_levels , cmap='turbo')
                if colorbar:
                    plt.colorbar()
                if contours:
                    plt.contour(xnew, ynew, data1, levels=contourf.levels, colors='white')

                fig_title = f"2DF of T{day} of\n{assay}"
                plt.title(fig_title)
                plt.xlabel("Emission (nm)")
                plt.ylabel("Excitation (nm)")
                plt.tight_layout()
                plt.show()
    
        
import matplotlib.cm as cm


def plotmeanAbsSpecs(
    factors, discriminator, days2use, varxID, df, legend=True, ymax=1, 
    plotdiffs=False, logtransf=False, dataindex=False, color_palette=None
):
    """
    Plots the mean absorbance spectra for selected assays and days.

    Parameters:
        factors (list): List of group/category names (assays) to plot.
        discriminator (str): Column name (or index name) used to identify groups (e.g., 'Assay').
        days2use (list): List of days to plot.
        varxID (list): List of column names for absorbance wavelengths.
        df (pd.DataFrame): DataFrame containing the data.
        legend (bool): Whether to display the legend. Default is True.
        ymax (float): Y-axis maximum for absorbance. Default is 1.
        plotdiffs (bool): If True, plot differences from the first day. Default is False.
        logtransf (bool): If True, apply log transformation to the spectra. Default is False.
        dataindex (bool): If True, use index for group selection. Default is False.
        color_palette (str or None): Name of colormap to use, or None for default colors.
    """
    # Extract numeric wavelengths from column names
    ex = [int(wave[7:10]) for wave in varxID]

    for assay in factors:
        # Select data for the current assay
        if dataindex:
            df4plot = df[df.index.get_level_values(discriminator).isin([assay])].copy()
        else:
            df4plot = df[df[discriminator].isin([assay])].copy()

        plt.rcParams['figure.dpi'] = 250
        plt.figure(figsize=(4, 2.5))

        # Generate a color map if requested
        if color_palette:
            cmap = cm.get_cmap(color_palette)
            colors = [cmap(i / max(1, len(days2use) - 1)) for i in range(len(days2use))]
        else:
            colors = [None] * len(days2use)  # Use default color cycle

        x_aux = None  # Used for difference plotting

        for day_idx, day in enumerate(days2use):
            # Select data for the current day
            if dataindex:
                df4plot_aux = df4plot[df4plot.index.get_level_values('Day').isin([day])]
            else:
                df4plot_aux = df4plot[df4plot['Day'].isin([day])]

            if df4plot_aux.empty:
                continue

            X = df4plot_aux[varxID].copy()

            if logtransf:
                X = np.log(X + 1)  # Avoid log(0)

            x = X.mean()

            if plotdiffs:
                if x_aux is None:
                    x_aux = x.copy()
                x_plot = x - x_aux
                x_aux = x.copy()
            else:
                x_plot = x

            plt.scatter(
                ex, x_plot, label=str(day), s=8,
                color=colors[day_idx]
            )

        plt.xlim([300, 800])
        plt.ylim([0, ymax])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absorbance (a.u.)")
        plt.title(str(assay))
        if legend:
            plt.legend(title="Day")
        plt.tight_layout()
        plt.show()
        
def getEEMpositions(ex1,ex2,em1,em2,step):
    
    positions = [] # EEP coordinates in the EEM
    waves = [] 
    ids = []
    ex = np.arange(ex1,ex2+step,step)
    em = np.arange(em1,em2+5,5)
    
    iteration = 0
        
    for j in range(len(em)):
        for i in range(len(ex)):
            if ex[i] < em[j]:
                positions.append([i,j])
                waves.append([ex[i],em[j]])   
                ids.append('EEP '+str(ex[i])+' / '+str(em[j])+' nm')
                iteration +=1
                
    translations = pd.DataFrame(columns = ['ex','em'],data=waves)
    translations['coords'] = positions      
    translations['ids'] = ids    
    
    return positions,translations
