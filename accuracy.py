# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:00:58 2024

@author: pedro.brandao
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import seaborn as sns

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
Trainsub = str.maketrans("t","ₜ")
badchars = ["(",")","/"," "]



from sklearn.metrics import make_scorer, mean_squared_error, r2_score


def rmsecv_scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    return rmse(y_true,y_pred)

def q2_scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    return r2(y_true,y_pred)


def r2(y_true,y_pred):
    RSS = np.sum((y_true - y_pred.reshape(y_true.shape))**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - RSS/TSS
    # print(r2)
    return r2

def rmse(y_true,y_pred):
    RSS = np.sum((y_true - y_pred.reshape(y_true.shape))**2)
    rmse = np.sqrt(RSS/len(y_true))
    return rmse


''' Scoring for a specific class '''

def accuracy_for_classes(y_true, y_pred, class_labels):
    accuracies = []
    for class_label in class_labels:
        class_indices = (y_true == class_label)
        class_accuracy = (y_pred[class_indices] == y_true[class_indices]).sum()
        accuracies.append(class_accuracy)
    return sum(accuracies) / len(accuracies)  # Average accuracy across classes




def plotAccScatters(y_train,y_predtrain,y_test,y_pred,output2predict,
                valcolor,rsq_train,rsq,rmseT,rmseP,sampledata,assay):
    
    
    fig8,ax = plt.subplots(figsize=(0.75, 0.25))
    ax.set_axis_off()  
    plt.text(0,0.5,
            # "R2t".translate(SUP).translate(Trainsub)+" = "+str(np.round(rsq_train,2))+
            # "\nRMSET = "+str(np.round(rmseT,2))+'\n'+
            "R\u00b2 = "+str(np.round(rsq,2))+
            "\nRMSEP = "+str(np.round(rmseP,2)))
    plt.rcParams['figure.dpi'] = 300
    plt.show()
    
    # figsize = (3.5, 3)
    figsize = (1.75, 1.75)
    fig, ax1 = plt.subplots(figsize=figsize)
    
    y = y_test.join(y_pred)
    y = sampledata.join(y)
    
    # Define markers and colors
    markers = ['o','s','^','d',
               'o','s','^','d',
               'o','s','^','d',
               'o','s','^','d',
               'o','s','^','d',
               'o','s','^','d',
               'o','s','^','d']


    colors = valcolor

    # Convert all colors to normalized RGB
    colors = [to_rgb(c) for c in colors]
    
    unique_mlalg = sampledata[assay].sort_values().unique()
    
    # Create a dictionary for markers and colors
    marker_map = {alg: marker for alg, marker in zip(unique_mlalg, markers)}
    color_map = {alg: color for alg, color in zip(unique_mlalg, colors)}
    
    for alg in unique_mlalg:
        marker = marker_map[alg]
        color = color_map[alg]
        subset = y[y[assay] == alg]
    
        sns.scatterplot(data=subset, x=output2predict, y=output2predict + ' - Model',
                        marker=marker,size=1, color=color,label=str(alg),legend=False)
    
    stdy = float(np.std(y_test.values))
    yplotmax = float(np.max(y_test)) * 1.15
    yplotmin = float(np.min(y_test)*.85)
    plt.rcParams['figure.dpi'] = 300
    plt.plot([yplotmin, yplotmax], [yplotmin, yplotmax], '--', color='k', linewidth=0.75)  # Y = PredY line
    plt.plot([yplotmin, yplotmax], [yplotmin + stdy, yplotmax + stdy], '--', color='k', linewidth=0.75)
    plt.plot([yplotmin, yplotmax], [yplotmin - stdy, yplotmax - stdy], '--', color='k', linewidth=0.75)
    
    # plt.xlim([0, yplotmax])
    # plt.ylim([-stdy, yplotmax])
    plt.ylabel(output2predict + ' - Model')
    plt.xlabel(output2predict + ' - Experimental')
    
    # Extract handles and labels for the legend
    handles, labels = ax1.get_legend_handles_labels()
    specid_handles = [plt.Line2D([0], [0], marker=marker_map[alg], color='w', markerfacecolor=color_map[alg], markersize=10) for alg in unique_mlalg]
    specid_labels = unique_mlalg
  
    plt.show()
    
    # Create a separate figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(2, 2))
    ax_legend.legend(specid_handles,specid_labels,  title=assay, loc='center')
    ax_legend.axis('off')
    plt.show()

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


def plotAccScatters_justtest(ax,y_test,y_pred,output2predict,model_ID,
                rsq,rmseP,sampledata,label,showlegend,train_test_or_cv,tag,
                valcolor=['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'],
                valmarkers=['o','s','^','d','o','s','^','d','o','s','o','^'],
                rgb_colors=False):
    
    
    if not ax:
        # print("No axis given")
        figsize = (3.5, 3)
        figsize = (1.75, 1.5)
        fig, ax = plt.subplots(figsize=figsize)
    
    y = y_test.join(y_pred)
    
    y = y.sort_index(level=tag)
    # y = sampledata.join(y).dropna()
    
    
    
    # Define markers and colors
    
    markers = valmarkers

    colors = valcolor
    
    # unique_mlalg = y[label].unique()
    
    unique_mlalg = y.index.get_level_values(label).unique()
    
    # unique_mlalg = [str(year) for year in unique_mlalg]
    
    # Create a dictionary for markers and colors
    marker_map = {alg: marker for alg, marker in zip(unique_mlalg, markers)}
    if rgb_colors:
        color_map = {alg: tuple(value / 255 for value in color) for alg, color in zip(unique_mlalg, colors)}
    else:
        color_map = {alg: color for alg, color in zip(unique_mlalg, colors)}
    
    
    for alg in unique_mlalg:
        marker = marker_map[alg]
        color = color_map[alg]
        # subset = y[y[label] == alg]
        subset = y[y.index.get_level_values(label) == alg]
    
        sns.scatterplot(ax=ax,data=subset, x=output2predict, y=output2predict + ' - Model',
                        marker=marker, color=color,label=str(alg),legend= False)
    
    stdy =  float(np.std(y_test.values))
    yplotmax = float(np.max(y_test)*1.15)
    yplotmin = float(np.min(y_test)*.85)
    
    ax.plot([yplotmin, yplotmax], [yplotmin, yplotmax], '--', color='k', linewidth=0.75)  # Y = PredY line
    ax.plot([yplotmin, yplotmax], [yplotmin + stdy, yplotmax + stdy], '--', color='k', linewidth=0.75)
    ax.plot([yplotmin, yplotmax], [yplotmin - stdy, yplotmax - stdy], '--', color='k', linewidth=0.75)
    
    # ax.set_xlim([0, yplotmax])
    # ax.set_ylim([-stdy, yplotmax])
    # ax.set_xlim([50,62])
    # ax.set_ylim([50,62])
    ax.set_ylabel(output2predict + ' - Model')
    ax.set_xlabel(output2predict + ' - Experimental')
    
    if train_test_or_cv == "CV":
    
        ax.set_title('CV result for \n' + model_ID+
                "\nQ\u00b2 = "+str(np.round(rsq,2))+
                "\nRMSECV = "+str(np.round(rmseP,2)))
        
    elif train_test_or_cv == "Train" :
        
        ax.set_title('Train result for \n' + model_ID+
                "\nR\u00b2 = "+str(np.round(rsq,2))+
                "\nRMSET = "+str(np.round(rmseP,2)))  
        
    elif train_test_or_cv == "Test":
        
        ax.set_title('Test result for \n' + model_ID+
                "\nR\u00b2 = "+str(np.round(rsq,2))+
                "\nRMSEP = "+str(np.round(rmseP,2)))     
    # plt.show()
    
    if showlegend:
    
        # Extract handles and labels for the legend
        handles, labels = ax.get_legend_handles_labels()
        specid_handles = [plt.Line2D([0], [0], marker=marker_map[alg], color='w', 
                                     markerfacecolor=color_map[alg], markersize=10) for alg in unique_mlalg]
        specid_labels = unique_mlalg
    
        
        plt.show()
        
        # Create a separate figure for the legend
        fig_legend, ax_legend = plt.subplots(figsize=(2, 2))
        ax_legend.legend(specid_handles,specid_labels,  title=tag, loc='center')
        ax_legend.axis('off')
        plt.show()

from sklearn.model_selection import cross_val_predict
import pandas as pd

def get_modelling_results(grid_search_results,pipeline,X_ori_train,y_ori_train,group_kfold,groups,
                   output2predict):

    r2train_list = []
    q2_list = []
    rmseT_list = []
    rmseCV_list = []
    
    for params, mean_score, scores in zip(grid_search_results['params'],
                                          grid_search_results['mean_test_score'],
                                          grid_search_results['split0_test_score']):

        pipeline.set_params(**params)  # Set parameters for the classifier

        # Get predictions for the current parameter setting
        y_predtrainCV = cross_val_predict(pipeline,  X_ori_train, y_ori_train, cv=group_kfold, groups=groups)
        
        model = pipeline.fit(X_ori_train,y_ori_train)  
        y_predtrain = model.predict(X_ori_train)
        
        rmseCV = rmse(y_ori_train.values, y_predtrainCV.values)
        q2 = r2_score(y_ori_train.values, y_predtrainCV.values)
        
        rmseT = rmse(y_ori_train.values, y_predtrain.values)        
        rsq_train = r2_score(y_ori_train.values, y_predtrain.values)

        y_predtrain = pd.DataFrame(data=y_predtrain,index=y_ori_train.index,columns=[output2predict+' - Model'])
        y_predtrainCV = pd.DataFrame(data=y_predtrainCV,index=y_ori_train.index,columns=[output2predict+' - Model'])
        
        r2train_list.append(rsq_train)
        q2_list.append(q2)
        rmseT_list.append(rmseT)
        rmseCV_list.append(rmseCV)

        
    return r2train_list,q2_list,rmseT_list,rmseCV_list