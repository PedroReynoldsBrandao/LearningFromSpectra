import pandas as pd

def filter_dataframe(df: pd.DataFrame, filter_dict: dict, filter_out: bool = False) -> pd.DataFrame:
    """
    Filters a pandas DataFrame by its multi-index levels based on a dictionary of filtering criteria.
    
    Parameters:
        df (pd.DataFrame): The DataFrame with a MultiIndex to filter.
        filter_dict (dict): A dictionary where keys are index level names and values are lists of allowed values.
        filter_out (bool): If True, filters out the specified values instead of selecting them.
        
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    index_filter = [df.index.get_level_values(level).isin(values) for level, values in filter_dict.items()]
    if filter_out:
        combined_filter = index_filter[0]
        for condition in index_filter[1:]:
            combined_filter |= condition
        combined_filter = ~combined_filter
    else:
        combined_filter = index_filter[0]
        for condition in index_filter[1:]:
            combined_filter &= condition
    
    return df[combined_filter]

def main():
    # Create a sample DataFrame with a MultiIndex
    index = pd.MultiIndex.from_tuples(
        [('A', 1), ('A', 2), ('B', 1), ('B', 2), ('C', 1), ('C', 2)],
        names=['letter', 'number']
    )
    data = {'value': [10, 20, 30, 40, 50, 60]}
    df = pd.DataFrame(data, index=index)
    
    print("Original DataFrame:")
    print(df)
    
    # Define filtering criteria
    filter_dict = {'letter': ['A', 'B'], 'number': [1]}
    
    # Filter the DataFrame
    filtered_df = filter_dataframe(df, filter_dict)
    
    print("\nFiltered DataFrame (selecting 'A' and 'B' letters and '1' number):")
    print(filtered_df)
    
    # Filter the DataFrame with filter_out=True
    filtered_out_df = filter_dataframe(df, filter_dict, filter_out=True)
    
    print("\nFiltered DataFrame (excluding 'A' and 'B' letters or '1' number):")
    print(filtered_out_df)

if __name__ == "__main__":
    main()