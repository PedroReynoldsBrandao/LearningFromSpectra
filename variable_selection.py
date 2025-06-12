# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:16:57 2024

@author: pedro.brandao
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd

def add_windows(windows, prefix, counter = {}, window_dict = {}):
    """
    Add a list of windows to window_dict with a prefix 
    and incremented index.
    
    """
    if prefix not in counter:
        counter[prefix] = 1
    for win in windows:
        key = f"{prefix}{counter[prefix]}"
        window_dict[key] = win
        counter[prefix] += 1
    
    return counter, window_dict


class VIPFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, vip=1.0):
        self.vip = vip
        self.selected_features_ = None
        self.feature_names_ = None

    def fit(self, X, y):
        Xvip = X.values if isinstance(X, pd.DataFrame) else X
        yvip = y.values if isinstance(y, pd.DataFrame) else y
        pls = PLSRegression(n_components=min(X.shape[1], 5))  # Start with a reasonable number of components
        pls.fit(Xvip, yvip)
        T = pls.x_scores_
        W = pls.x_weights_
        p, h = W.shape
        Q = pls.y_loadings_

        # Calculate VIP scores
        vips = np.zeros((p,))
        s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = (W[i, :] ** 2) * s.T
            vips[i] = np.sqrt(p * (weight.sum() / total_s))
        
        self.selected_features_ = np.where(vips >= self.vip)[0]
        self.feature_names_ = [X.columns[i] for i in self.selected_features_]
        return self

    def transform(self, X):
        return X[self.feature_names_]
    

    
class waveRestrictor(BaseEstimator, TransformerMixin):
    
    def __init__(self, waves = [], name = ''):
        self.waves = waves
        self.name = name
    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        return X.loc[:,self.waves]
    
    def __repr__(self):
        return self.name      

class waveRestrictor2(BaseEstimator, TransformerMixin):
    
    def __init__(self, window_dict = {}, window = '', name = ''):
        self.window_dict = window_dict
        self.window = window
        self.name = name
        
    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        waves = self.window_dict[self.window]
        return X.loc[:,waves]
    
    def __repr__(self):
        return self.name    
    
    
def getEEMpositions3(ex1,ex2,em1,em2,step):
    
    positions = [] # EEP coordinates in the EEM
    waves = [] 
    ids = []
    ex = np.arange(ex1,ex2+step,step)
    em = np.arange(em1,em2+step,step)
    
    iteration = 0
        
    for j in range(len(ex)):
        for i in range(len(em)):
            exc = ex[i]
            emi = em[j]   
            if ex[i] < em[j]:
                positions.append([i,j])
                waves.append([ex[i],em[j]])          
                ids.append(f'EEP {exc} / {emi} nm')
                iteration +=1
    
    translations = pd.DataFrame(columns = ['ex','em'],data=waves)
    translations['coords'] = positions
    translations['ids'] = ids
          
    return positions,translations

    
def windowgen(positions, dim, overlap, rectangular=False, rect_dim=(None, None)):
    """
    Generate windows over a 2D space of positions.

    Parameters:
    - positions: list of tuples, each tuple containing (x, y) coordinates.
    - dim: int, size of the square window (ignored if rectangular is True).
    - overlap: float, fraction of overlap between windows.
    - rectangular: bool, whether to create rectangular windows.
    - rect_dim: tuple (width, height), dimensions of the rectangular window if rectangular is True.

    Returns:
    - window_indices: list of lists, each containing indices of positions within a window.
    - num_positions_in_window: list of integers, each representing the number of positions in a window.
    """
    window_indices = []
    num_positions_in_window = []

    if rectangular:
        width, height = rect_dim
        if width is None or height is None:
            raise ValueError("Both width and height must be specified for rectangular windows.")
        step_x = round(width * (1 - overlap))
        step_y = round(height * (1 - overlap))
        max_x = positions[-1][0] + int(width / 2)
        max_y = positions[-1][1] + int(height / 2)

        # Slide the rectangular window over the 2D space
        for x_start in range(0, max_x - width + 1, step_x):
            for y_start in range(0, max_y - height + 1, step_y):
                x_end = x_start + width
                y_end = y_start + height

                # Find indices of positions within the current window
                indices = [
                    i for i, (x, y) in enumerate(positions) 
                    if x_start <= x < x_end and y_start <= y < y_end
                ]

                window_indices.append(indices)
                num_positions_in_window.append(len(indices))
    else:
        step = round(dim * (1 - overlap))
        max_x = positions[-1][0] + int(dim / 2)
        max_y = positions[-1][1] + int(dim / 2)

        # Slide the square window over the 2D space
        for x_start in range(0, max_x - dim + 1, step):
            for y_start in range(x_start, max_y - dim + 1, step):
                x_end = x_start + dim
                y_end = y_start + dim

                # Find indices of positions within the current window
                indices = [
                    i for i, (x, y) in enumerate(positions) 
                    if x_start <= x < x_end and y_start <= y < y_end
                ]

                window_indices.append(indices)
                num_positions_in_window.append(len(indices))

    return window_indices, num_positions_in_window


class MovingWindowFTIR:
    def __init__(self, spectra_df, window_size, step_size):
        """
        Initialize the MovingWindowFTIR class.

        Parameters:
        spectra_df (pd.DataFrame): DataFrame with columns as wavenumbers and rows as samples.
        window_size (int): Size of the moving window.
        step_size (int): Step size for moving the window. Allows for overlapping windows if step_size < window_size.
        """
        self.spectra_df = spectra_df
        self.window_size = window_size
        self.step_size = step_size
        self.selected_windows = []

    def _calculate_window_indices(self, start_index):
        """
        Calculate the start and end indices of the moving window.

        Parameters:
        start_index (int): The starting index for the moving window.

        Returns:
        tuple: (start_index, end_index)
        """
        end_index = start_index + self.window_size
        if end_index > self.spectra_df.shape[1]:
            end_index = self.spectra_df.shape[1]
        return start_index, end_index

    def select_windows(self):
        """
        Selects variables using a moving window approach with possible overlaps.
        """
        num_wavenumbers = self.spectra_df.shape[1]

        for start_index in range(0, num_wavenumbers, self.step_size):
            start, end = self._calculate_window_indices(start_index)
            window = self.spectra_df.iloc[:, start:end]
            self.selected_windows.append(window)
            if end == num_wavenumbers:
                break  # Stop if the end of the spectra is reached

    def get_selected_windows(self):
        """
        Get the selected windows.

        Returns:
        list of pd.DataFrame: List of DataFrames, each representing a window.
        """
        return self.selected_windows

    def mean_center_windows(self):
        """
        Mean center the variables within each window.
        """
        for i, window in enumerate(self.selected_windows):
            self.selected_windows[i] = window - window.mean()
            
            
def movingwindow_forAbs(X,window_size,step_size,window_no):
    
    moving_window = MovingWindowFTIR(X, window_size=window_size, step_size=step_size) 
    moving_window.select_windows()
    windows = moving_window.get_selected_windows()
    windows = [list(window.columns) for window in windows]
    
    windows_ori = windows.copy()
    
    if window_no > 1:
        
        iteration = 0
        while iteration < window_no-1:
        
            selected_windowsB = windows_ori.copy()
            selected_windowsB.reverse()
      
            windows = [list(dict.fromkeys(a + b)) for a,b in zip(windows,selected_windowsB)]
            random.shuffle(windows)
            
            iteration += 1
        
    return windows
    


def fluoro1Dgen(varxID, moving_excitation=True, wave_interval= range(250,790,5),
                window_size=1, overlap=0, selected_excitations=None, 
                random_selection = False, desired_length=10):
    

    """
    Generate 1D fluorescence spectra for specified excitation intervals with variable windows and overlaps.

    Parameters:
    varxID (list): List of column names containing the fluorescence data.
    moving_excitation (bool): If True, filters based on excitation; if False, filters based on emission.
    wave_interval (list): List of all available wavelengths.
    window (int): Size of the window (number of excitation wavelengths) for each interval.
    overlap (int): Overlap size (number of excitation wavelengths) between consecutive windows.
    selected_excitations (list, optional): Specific excitation wavelengths to include (e.g., [470, 530, 640]).

    Returns:
    list: List of lists, each containing the column names for the specified excitation intervals.
    """
    filtered_varxID_list = []

    # Filter wave_interval if specific excitations are provided
    if selected_excitations is not None:
        wave_interval = [wave for wave in wave_interval if wave in selected_excitations]

    start_idx = 0
    while start_idx < len(wave_interval):
        # Determine the end index of the current window
        end_idx = min(start_idx + window_size, len(wave_interval))

        # Get the excitation wavelengths for the current window
        current_window = wave_interval[start_idx:end_idx]

        # Filter the columns for the current window
        filtered_varxID = []
        for wave in current_window:
            if moving_excitation:
                excitation_str = f'EEP {wave} / '
                filtered_varxID.extend([var for var in varxID if var.startswith(excitation_str)])
            else:
                emission_str = f' / {wave} nm'
                filtered_varxID.extend([var for var in varxID if var.endswith(emission_str)])

        if len(filtered_varxID) > 10:
            filtered_varxID_list.append(filtered_varxID)

        # Move to the next window position, taking overlap into account
        start_idx += (window_size - overlap)
    
    if random_selection:
        filtered_varxID_list_ori = filtered_varxID_list
        filtered_varxID_list = []  
        while True:
            spectra = random.sample(filtered_varxID_list_ori, random_selection)
            spectra = [item for sublist in spectra for item in sublist]
            filtered_varxID_list.append(spectra)
            if len(filtered_varxID_list) == desired_length:
                break
                
        # filtered_varxID_list = [item for sublist in filtered_varxID_list for item in sublist]

    
    if selected_excitations is not None:
        filtered_varxID_list = [[item for sublist in filtered_varxID_list for item in sublist]]

    return filtered_varxID_list


import random

def EEM_squaredwindowgenerator(ex_em=(250, 790, 260, 800, 5), 
                                size=25, overlap=0.75, 
                                numberofwindows=1, desired_length=500):
    """
    Generates random windows of a spectrum from indexes.

    Parameters:
    - ex_em: tuple (ex1, ex2, em1, em2, step) specifying excitation and emission ranges.
    - size: int, size of each window.
    - overlap: float, fraction of overlap between windows.
    - numberofwindows: int, number of window sets to generate.
    - desired_length: int, optional, desired total number of unique windows.

    Returns:
    - selected_windows: list of lists, each containing IDs for the generated windows.
    """
    ex1, ex2, em1, em2, step = ex_em
    positions, translations = getEEMpositions3(ex1, ex2, em1, em2, step)
    selected_windows = []

    while len(selected_windows) < desired_length:
        
        selected_indexes, numberofeeps = windowgen(positions, size, overlap)


        # Generate one set of windows
        windows = [
            translations[translations.index.isin(window)]['ids'].to_list() 
            for window in selected_indexes
        ]

        # Shuffle and combine for multiple window sets if needed
        if numberofwindows > 1:
            all_windows = []
            for _ in range(numberofwindows):
                rng = random.Random()
                rng.shuffle(selected_indexes)

                new_windows = [
                    translations[translations.index.isin(window)]['ids'].to_list() 
                    for window in selected_indexes
                ]
                all_windows.append(new_windows)

            # Merge and randomize windows across sets
            windows = [
                list(dict.fromkeys(sum(window_set, [])))  # Deduplicate and merge
                for window_set in zip(*all_windows)
            ]
        else:
            selected_windows.extend(windows)
            break

        selected_windows.extend(windows)
        if numberofwindows > 1:
            # Deduplicate the entire selected_windows list
            selected_windows = list(map(list, {tuple(w) for w in selected_windows}))

    # Trim to desired length if specified
    if desired_length is not None:
        selected_windows = selected_windows[:desired_length]

    return selected_windows

