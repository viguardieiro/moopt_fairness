import numpy as np
import pickle
from pathlib import Path
import pandas as pd

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

def get_data(dataset_used="adult", protected_attribute="sex", train_size=0.7):
    if dataset_used == "adult":
        if protected_attribute == "sex":
            dataset_orig = load_preproc_data_adult(['sex'])
        else:
            dataset_orig = load_preproc_data_adult(['race'])
        
    elif dataset_used == "german":
        if protected_attribute == "sex":
            dataset_orig = load_preproc_data_german(['sex'])  
        else:
            dataset_orig = load_preproc_data_german(['age'])

    elif dataset_used == "compas":
        if protected_attribute == "sex":
            dataset_orig = load_preproc_data_compas(['sex'])
        else:
            dataset_orig = load_preproc_data_compas(['race'])

    #random seed
    np.random.seed(1)

    # Split into train, validation, and test
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([train_size], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    # Convert to dataframe
    df_all, _ = dataset_orig.convert_to_dataframe()
    df_all = df_all.reset_index(drop=True)
    df_train, _ = dataset_orig_train.convert_to_dataframe()
    df_train = df_train.reset_index(drop=True)
    df_valid, _ = dataset_orig_valid.convert_to_dataframe()
    df_valid = df_valid.reset_index(drop=True)
    df_test, _ = dataset_orig_test.convert_to_dataframe()
    df_test = df_test.reset_index(drop=True)

    X_all = df_all.drop(dataset_orig.label_names, axis=1)
    y_all = df_all[dataset_orig.label_names[0]]
    X_train = df_train.drop(dataset_orig.label_names, axis=1)
    y_train = df_train[dataset_orig.label_names[0]]
    X_valid = df_valid.drop(dataset_orig.label_names, axis=1)
    y_valid = df_valid[dataset_orig.label_names[0]]
    X_test = df_test.drop(dataset_orig.label_names, axis=1)
    y_test = df_test[dataset_orig.label_names[0]]

    # Mab labels to favorable=1 and unfavorable=-1
    favorable = dataset_orig.favorable_label
    unfavorable = dataset_orig.unfavorable_label
    label_map = {favorable: 1, unfavorable: -1}
    y_all = y_all.map(label_map)
    y_train = y_train.map(label_map)
    y_valid = y_valid.map(label_map)
    y_test = y_test.map(label_map)

    return X_all, y_all, X_train, y_train, X_valid, y_valid, X_test, y_test

if __name__ == "__main__":
    datasets = [["adult", "race"],
                ["german", "sex"], 
                ["compas", "race"]]

    for dataset, attribute in datasets:
        print("[INFO] Obtaining "+dataset+"_"+attribute+" data...")
        Path("data/"+dataset+"_"+attribute).mkdir(parents=True, exist_ok=True)

        # get clean and splitted data
        X_all, y_all, X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(dataset, attribute)

        # save features
        X_all.to_pickle("data/"+dataset+"_"+attribute+"/X.pickle")
        X_train.to_pickle("data/"+dataset+"_"+attribute+"/X_train.pickle")
        X_valid.to_pickle("data/"+dataset+"_"+attribute+"/X_valid.pickle")
        X_test.to_pickle("data/"+dataset+"_"+attribute+"/X_test.pickle")

        # save labels
        pickle.dump(y_all, open("data/"+dataset+"_"+attribute+"/y.pickle", "wb"))
        pickle.dump(y_train, open("data/"+dataset+"_"+attribute+"/y_train.pickle", "wb"))
        pickle.dump(y_valid, open("data/"+dataset+"_"+attribute+"/y_valid.pickle", "wb"))
        pickle.dump(y_test, open("data/"+dataset+"_"+attribute+"/y_test.pickle", "wb"))

        print("[INFO] "+dataset+"_"+attribute+" data obtained.")

    print("[INFO] All data obtained.")