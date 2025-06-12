# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:40:23 2024

@author: temp.admin
"""

from accuracy import plotAccScatters_justtest
from matplotlib import rcParams
from spectrum_plotter import plot_partialspectrum2
from sklearn.model_selection import cross_val_predict
from accuracy import rmse
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_modelling(grid_search_results,pipeline,X_ori_train,y_ori_train,group_kfold,groups,
                   EEM4image,output2predict,sampledata,spectra,tag):

    rcParams['figure.figsize'] = (12, 5)
    rcParams['figure.subplot.wspace'] = 0.25  
    plt.rcParams['figure.dpi'] = 150
    
    for params, mean_score, scores in zip(grid_search_results['params'],
                                          grid_search_results['mean_test_score'],
                                          grid_search_results['split0_test_score']):
        # Assuming you have a classifier clf that you are using in GridSearchCV
        pipeline.set_params(**params)  # Set parameters for the classifier
        # model_ID = "PLS-R with "+str(pipeline.steps[-1][1].n_components)+" LVs"
        
        # model_ID = "Linear-R with m = "+str(np.round(pipeline.steps[-1][1].coef_[0][0],2))+\
        #     " and b = "+str(np.round(pipeline.steps[-1][1].intercept_[0],2))
        
        model_ID = "Linear-R"
        # Get predictions for the current parameter setting
        y_predtrainCV = cross_val_predict(pipeline,  X_ori_train, y_ori_train, cv=group_kfold, groups=groups)
        
        rmseCV = rmse(y_ori_train, y_predtrainCV)
        q2 = r2_score(y_ori_train, y_predtrainCV)
        y_predtrainCV = pd.DataFrame(data=y_predtrainCV,index=y_ori_train.index,columns=[output2predict+' - Model'])
         
        fig,(ax1,ax2) = plt.subplots(1,2)
        
        X_train = pipeline.named_steps['Moving-Window'].fit_transform(X_ori_train,y_ori_train)
        # X_train = pipeline.steps[1][1].fit_transform(X_ori_train,y_ori_train)
        
        if spectra == "2DF":
            plot_partialspectrum2(ax1,X_ori_train,X_train, spectrum_index=False)
            
        else:
            waves = [int(wave[7:10]) for wave in list(X_ori_train.columns)] 
            waves_rest = [int(wave[7:10]) for wave in list(X_train.columns)]   
            ax1.plot(waves,X_ori_train.loc[EEM4image].T)
            ax1.scatter(waves_rest,X_train.loc[EEM4image].T,c='C1')
            ax1.set_ylabel("Absorbance")
            ax1.set_xlabel("Wavelength (nm)")
        plotAccScatters_justtest(ax2,y_ori_train,y_predtrainCV,output2predict,model_ID,
                        'C1',q2,rmseCV,sampledata,tag,False)
        
        fig.tight_layout(pad=3.5)
        plt.show()

def plot_bestmodel(pipeline,X_ori_train,y_ori_train,group_kfold,groups,
                   EEM4image,output2predict,sampledata,spectra,model_ID,tag):
    
    # rcParams['figure.figsize'] = (6.25, 2.5)
    # rcParams['figure.figsize'] = (8.5, 3.5)
    rcParams['figure.figsize'] = (12, 5)
    rcParams['figure.subplot.wspace'] = 0.25
    plt.rcParams['figure.dpi'] = 150
    
    
    

    # Assuming you have a classifier clf that you are using in GridSearchCV
    # model_ID = "PLS-R with "+str(pipeline.steps[-1][1].n_components)+" LVs"
    
    # Get predictions for the current parameter setting
    y_predtrainCV = cross_val_predict(pipeline,  X_ori_train, y_ori_train, cv=group_kfold, groups=groups)
    
    rmseCV = rmse(y_ori_train.values, y_predtrainCV.values)
    q2 = r2_score(y_ori_train.values, y_predtrainCV.values)
    y_predtrainCV = pd.DataFrame(data=y_predtrainCV,index=y_ori_train.index,columns=[output2predict+' - Model'])
     
    fig,(ax1,ax2) = plt.subplots(1,2)
    
    X_train = pipeline.named_steps['Moving-Window'].fit_transform(X_ori_train,y_ori_train)
    # X_train = pipeline.steps[:2].fit_transform(X_ori_train,y_ori_train)

    if spectra == "2DF":
        plot_partialspectrum2(ax1,X_ori_train,X_train, spectrum_index=False)
        
    else:
        waves = [int(wave[7:10]) for wave in list(X_ori_train.columns)] 
        waves_rest = [int(wave[7:10]) for wave in list(X_train.columns)]   
        ax1.plot(waves,X_ori_train.loc[EEM4image].T)
        ax1.scatter(waves_rest,X_train.loc[EEM4image].T,c='C1')
        ax1.set_ylabel("Absorbance")
        ax1.set_xlabel("Wavelength (nm)")
    
    plotAccScatters_justtest(ax2,y_ori_train,y_predtrainCV,output2predict,model_ID,
                    'C1',q2,rmseCV,sampledata,tag,True)


    plt.show()
    
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, classification_report,recall_score

def plot_modelling2(grid_search_results,pipeline,X_ori_train,y_ori_train,group_kfold,groups,
                   EEM4image,output2predict,sampledata,spectra,tag,classification=False,
                   colors = ['C0','C1','C2','C3','C4','C5','C6','C7'],
                   markers = ['o','s','^','d','o','s','^','d','o','s'],
                   rgb_colors = False):


    rcParams['figure.subplot.wspace'] = 0.2
    rcParams['figure.dpi'] = 100
    
    rcParams['figure.figsize'] = (10, 4)
    
    for params, mean_score, scores in zip(grid_search_results['params'],
                                          grid_search_results['mean_test_score'],
                                          grid_search_results['split0_test_score']):
        # Assuming you have a classifier clf that you are using in GridSearchCV
        pipeline.set_params(**params)  # Set parameters for the classifier
        import re
        
        def clean_param_string(param_string, precision=2):
            def convert_np_number(match):
                value = match.group(2)
                num_type = match.group(1)
                if 'float' in num_type:
                    return str(round(float(value), precision))
                else:
                    return str(int(float(value)))  # handles int64, int32, etc.
        
            # Replace np.float64(...) and np.int64(...) etc. with rounded numbers
            cleaned = re.sub(r'np\.(float\d*|int\d*)\(([^)]+)\)', convert_np_number, str(param_string))
        
            # Optional: also round plain floats like gamma=0.2133654785111
            cleaned = re.sub(r'(?<=\=)(\d+\.\d+)', lambda m: str(round(float(m.group(1)), precision)), cleaned)
        
            return cleaned
        
        model_ID = clean_param_string(str(params['Algorithm']))
        
        
        
        # Get predictions for the current parameter setting
        

        
        if not np.isnan(mean_score):
        
            y_predtrainCV = cross_val_predict(pipeline,  X_ori_train, y_ori_train.values.ravel(), cv=group_kfold, groups=groups)
            
            model = pipeline.fit(X_ori_train,y_ori_train.values.ravel())  
            y_predtrain = model.predict(X_ori_train)
            
    
    
            y_predtrain = pd.DataFrame(data=y_predtrain,index=y_ori_train.index,columns=[output2predict+' - Model'])
            y_predtrainCV = pd.DataFrame(data=y_predtrainCV,index=y_ori_train.index,columns=[output2predict+' - Model'])
             
            fig,(ax1,ax2,ax3) = plt.subplots(1,3)
            
            X_train = pipeline.named_steps['Wave-selection'].fit_transform(X_ori_train,y_ori_train)
            # X_train = pipeline.steps[1][1].fit_transform(X_ori_train,y_ori_train)
            
            if spectra == "2DF":
                plot_partialspectrum2(ax1,X_ori_train,X_train, spectrum_index=False)
            
            elif spectra == "MiniSpec":
                plot_partialspectrum2(ax1,X_ori_train, X_train, spectrum_index=False, lims = ([460,800],[460,680]),
                                      contours=False, log = False, vmax = 1000)
                
            elif spectra == "Abs":
                waves = [int(wave[7:10]) for wave in list(X_ori_train.columns)] 
                waves_rest = [int(wave[7:10]) for wave in list(X_train.columns)] 
                ax1.plot(waves,X_ori_train.loc[EEM4image].T)
                ax1.scatter(waves_rest,X_train.loc[EEM4image].T,c='C1')
                ax1.set_ylabel("Absorbance")
                ax1.set_xlabel("Wavelength (nm)")
    
            else:
                waves = list(X_ori_train.columns)
                waves_rest = list(X_train.columns) 
                ax1.plot(waves,X_ori_train.mean().T)
                ax1.scatter(waves_rest,X_train.mean().T,c='C1')
                ax1.set_ylabel("Absorbance")
                ax1.set_xlabel("Wavelength (nm)")
                ax1.set_xlim([4000,500])      
            
            if not classification:
                rmseCV = rmse(y_ori_train.values.ravel(), y_predtrainCV.values.ravel())
                q2 = r2_score(y_ori_train.values.ravel(), y_predtrainCV.values.ravel())
                
                rmseT = rmse(y_ori_train.values.ravel(), y_predtrain.values.ravel())        
                rsq_train = r2_score(y_ori_train.values.ravel(), y_predtrain.values.ravel())
                
                plotAccScatters_justtest(ax2,y_ori_train,y_predtrain,output2predict,model_ID,
                                rsq_train,rmseT,sampledata,tag,False,"Train",tag,valcolor=colors,
                                valmarkers=markers,rgb_colors=rgb_colors)
                
                plotAccScatters_justtest(ax3,y_ori_train,y_predtrainCV,output2predict,model_ID,
                                q2,rmseCV,sampledata,tag,False,"CV",tag,valcolor=colors,
                                valmarkers=markers,rgb_colors=rgb_colors)    
            else:
                classes = y_ori_train[output2predict].unique()
                classes.sort()
                
                cm = confusion_matrix(y_ori_train, y_predtrain)
                report = classification_report(y_ori_train, y_predtrain,output_dict=True)
                cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                # disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=classes)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                # disp.plot(cmap='Blues', values_format='.1f',ax=ax2)
                im = disp.plot(cmap='Blues', values_format='.1f', colorbar=False, ax=ax2)
                ax2.set_title('Training Accuracy = ' + str(np.round(report['accuracy']*100,1))+'%')
                im.im_.set_clim(0, 100)
    
    
    
                cm = confusion_matrix(y_ori_train, y_predtrainCV)
                report = classification_report(y_ori_train, y_predtrainCV,output_dict=True)
                cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                # disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=classes)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                # disp.plot(cmap='Oranges', values_format='.1f',ax=ax3)
                im = disp.plot(cmap='Oranges', values_format='.1f', colorbar=False, ax=ax3)
                ax3.set_title('CV Accuracy = ' + str(np.round(report['accuracy']*100,1))+'%')
                im.im_.set_clim(0, 100)
                
            fig.tight_layout(pad=0.25)
            plt.show()
        
    
def plot_bestmodel2(pipeline,X_ori_train,y_ori_train,group_kfold,groups,
                   EEM4image,output2predict,sampledata,spectra,tag,classification=False,
                   colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'],
                   markers = ['o','s','^','d','o','s','^','d','o','s','o','^'],
                   randomCVfold = False,rgb_colors=False):
    

    rcParams['figure.subplot.wspace'] = 0.2
    rcParams['figure.dpi'] = 200
    
    rcParams['figure.figsize'] = (11, 4)
    

    
    # Get predictions for the current parameter setting
    
    model = pipeline.fit(X_ori_train,y_ori_train)  
    y_predtrain = model.predict(X_ori_train)
    
    if randomCVfold:
        y_predtrainCV = cross_val_predict(pipeline,  X_ori_train, y_ori_train, cv=randomCVfold, groups=groups)
    else:
        y_predtrainCV = cross_val_predict(pipeline,  X_ori_train, y_ori_train, cv=group_kfold, groups=groups)
    y_predtrainCV = pd.DataFrame(data=y_predtrainCV,index=y_ori_train.index,columns=[output2predict+' - Model'])
    

    y_predtrain = pd.DataFrame(data=y_predtrain,index=y_ori_train.index,columns=[output2predict+' - Model'])
     
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    
    X_train = pipeline.named_steps['Wave-selection'].fit_transform(X_ori_train,y_ori_train)
    # X_train = pipeline.steps[:2].fit_transform(X_ori_train,y_ori_train)

    if spectra == "2DF":
        plot_partialspectrum2(ax1,X_ori_train ,X_train, spectrum_index=False)
    
    elif spectra == "MiniSpec":
        plot_partialspectrum2(ax1,X_ori_train, X_train, spectrum_index=False, lims = ([460,800],[460,680]),
                              contours=False, log = False, vmax = 1000)
        
    elif spectra == "Abs":
        waves = [int(wave[7:10]) for wave in list(X_ori_train.columns)] 
        waves_rest = [int(wave[7:10]) for wave in list(X_train.columns)]   
        ax1.plot(waves,X_ori_train.loc[EEM4image].T)
        ax1.scatter(waves_rest,X_train.loc[EEM4image].T,c='C1')
        ax1.set_ylabel("Absorbance")
        ax1.set_xlabel("Wavelength (nm)")
        
    elif spectra == "FTIR":
        waves = list(X_ori_train.columns)
        waves_rest = list(X_train.columns)
        ax1.plot(waves,X_ori_train.mean().T)
        ax1.scatter(waves_rest,X_train.mean().T,c='C1')
        ax1.set_ylabel("Absorbance")
        ax1.set_xlabel("Wavelength (nm)") 
        ax1.set_xlim([4000,500])
        
    if not classification:    
            
        import re
        
        def clean_param_string(param_string, precision=2):
            def convert_np_number(match):
                value = match.group(2)
                num_type = match.group(1)
                if 'float' in num_type:
                    return str(round(float(value), precision))
                else:
                    return str(int(float(value)))  # handles int64, int32, etc.
        
            # Replace np.float64(...) and np.int64(...) etc. with rounded numbers
            cleaned = re.sub(r'np\.(float\d*|int\d*)\(([^)]+)\)', convert_np_number, str(param_string))
        
            # Optional: also round plain floats like gamma=0.2133654785111
            cleaned = re.sub(r'(?<=\=)(\d+\.\d+)', lambda m: str(round(float(m.group(1)), precision)), cleaned)
        
            return cleaned
        
        model_ID = clean_param_string(str(model['Algorithm']))
        
        rmseT = rmse(y_ori_train.values, y_predtrain.values)        
        rsq_train = r2_score(y_ori_train.values, y_predtrain.values)
        rmseCV = rmse(y_ori_train.values, y_predtrainCV.values)
        q2 = r2_score(y_ori_train.values, y_predtrainCV.values)
        
        plotAccScatters_justtest(ax2,y_ori_train,y_predtrain,output2predict,model_ID,
                        rsq_train,rmseT,sampledata,tag,0,"Train",tag,valcolor=colors,valmarkers = markers,
                        rgb_colors=rgb_colors)
        
        plotAccScatters_justtest(ax3,y_ori_train,y_predtrainCV,output2predict,model_ID,
                        q2,rmseCV,sampledata,tag,0,"CV",tag,valcolor=colors,valmarkers = markers,
                        rgb_colors=rgb_colors)
    else:
        
        classes = y_ori_train[output2predict].unique()
        classes.sort()
        
        cm = confusion_matrix(y_ori_train, y_predtrain)
        report = classification_report(y_ori_train, y_predtrain,output_dict=True)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        # disp.plot(cmap='Blues', values_format='.1f',ax=ax2)
        im = disp.plot(cmap='Blues', values_format='.1f', colorbar=False, ax=ax2)
        # im.im_.set_clim(0, 100)
        ax2.set_title('Training Accuracy = ' + str(np.round(report['accuracy']*100,1))+'%')

        cm = confusion_matrix(y_ori_train, y_predtrainCV)
        report = classification_report(y_ori_train, y_predtrainCV,output_dict=True)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        # disp.plot(cmap='Oranges', values_format='.1f',ax=ax3)
        im = disp.plot(cmap='Oranges', values_format='.1f', colorbar=False, ax=ax3)
        # im.im_.set_clim(0, 100)
        ax3.set_title('CV Accuracy = ' + str(np.round(report['accuracy']*100,1))+'%')
        
    fig.tight_layout(pad=0.25)
    plt.show()
    

        
        
        