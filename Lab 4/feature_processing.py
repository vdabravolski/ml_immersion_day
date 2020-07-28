import argparse
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

NUMERICAL_COLUMNS = ['Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 
                     'Dropoff_latitude', 'Passenger_count', 'Total_amount', 'Trip_distance']

COLUMNS_TO_DROP =  ['Payment_type', 'Trip_type ', 'Fare_amount', 
                    'Extra', 'MTA_tax','Tip_amount', 'Tolls_amount', 
                    'Ehail_fee', 'Store_and_fwd_flag', 'RateCodeID', 
                    'lpep_pickup_datetime', 'Lpep_dropoff_datetime']      # columns in original dataset which will be dropped

CATEGORICAL_COLUMNS = ['VendorID']   # columns which will be replaced with "one-hot encoded" columns

INPUT_PATH = "/opt/ml/processing/input/data"
OUTPUT_PATH = "/opt/ml/processing/output/data" 


def _get_data_files(extension=".csv", input_path=INPUT_PATH):
    """
    Get individual files uploaded to processing nodes.
    Files are stored in INPUT_PATH by default.
    """
    
    files = []
    
    print(os.listdir(INPUT_PATH))
    
    for file in os.listdir(input_path):
        print(file)
        if file.endswith(extension):
            files.append(os.path.join(input_path, file))
    
    return files
    
    

def _process_file(fpath):
    """
    - read file into Pandas dataframe;
    - drop undesired columns;
    - perform one-hot encoding on categorical features;
    - standartize numerical features
    """
    print(fpath)
    
    # read input file
    dfcolumns = pd.read_csv(fpath, nrows=1)
    df = pd.read_csv(fpath,header = None, skiprows = 1, 
                     usecols = list(range(len(dfcolumns.columns))), 
                     names = dfcolumns.columns)
    
    # Process individual file
    df = df.drop(COLUMNS_TO_DROP, axis=1)
    
#     preprocess = make_column_transformer(
#         (OneHotEncoder(), CATEGORICAL_COLUMNS),
#         (StandardScaler(), NUMERICAL_COLUMNS)
#     )    
    preprocess = make_column_transformer(
        (CATEGORICAL_COLUMNS, OneHotEncoder()),
        (NUMERICAL_COLUMNS, StandardScaler())
#         (NUMERICAL_COLUMNS, MinMaxScaler(feature_range=(0, 1), copy=False))
    )    
    processed_np = preprocess.fit_transform(df)
    
    # Create a new DataFrame with processed values
    new_columns = ['Vendor_1', 'Vendor_2'] + NUMERICAL_COLUMNS
    processed_df = pd.DataFrame(processed_np, columns=new_columns)
        
    # Saving processed dataframe locally
    fname = os.path.basename(fpath)
    processed_fname = f"processed_{fname}" # adding prefix to identify processed files
    processed_fpath = os.path.join(OUTPUT_PATH, processed_fname)
    processed_df.to_csv(processed_fpath)
    print(f"File {fname} has been processed and saved.")
    

def main():
    """
    Main processing method
    """    
    input_files = _get_data_files()
    total_files = len(input_files)
    skipped_files = 0
    processed_files = 0
    print(f"{total_files} are queued for processing.")
    
    for counter, file in enumerate(input_files):
        try:
            print(f"Processing file {file}")
            _process_file(file)
            processed_files += 1
        except Exception as e:
            print(e)
            print(f"File {file} cannot be processed. Skipping it...")
            skipped_files += 1
    
    print(f"{processed_files} file(s) out of {total_files} total number are processed. {skipped_files} files were skipped due to processing errors.")    

if __name__=="__main__":
    print("Starting processing.")
    main()
    print("Processing completed.")
    
    
