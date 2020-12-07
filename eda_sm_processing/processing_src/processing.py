from argparse import ArgumentParser
import os
import json
import pandas as pd
import numpy as np

def main():
    
    # 1. read raw data, uploaded to local storage
    data = pd.read_csv(f"{os.environ['DATA_DESTINATION']}/{os.environ['DATA_FILE']}")
    
    # 2. apply data transformations
    data['no_previous_contact'] = np.where(data['pdays'] == 999, 1, 0)
    data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)
    model_data = pd.get_dummies(data)
    model_data = model_data.drop(['duration', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], axis=1)
    train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data)), int(0.9 * len(model_data))])
    
    # 3. Save datasets into output directories, which will be uploaded to S3
    pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv(f"{os.environ['TRAIN_DATA_OUTPUT']}/train.csv", index=False, header=True)
    pd.concat([validation_data['y_yes'], validation_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv(f"{os.environ['VALIDATION_DATA_OUTPUT']}/validation.csv", index=False, header=True)
    


if __name__ == "__main__":
    main()

