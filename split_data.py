import numpy as np
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
    df_train, feature_names = dataset_orig_train.convert_to_dataframe()
    df_valid, feature_names = dataset_orig_valid.convert_to_dataframe()
    df_test, feature_names = dataset_orig_test.convert_to_dataframe()

    return df_train, df_valid, df_test

if __name__ == "__main__":
    # get clean and splitted data
    df_train, df_valid, df_test = get_data("adult")

    # save data
    df_train.to_csv("Data/adult_train.csv")
    df_valid.to_csv("Data/adult_validation.csv")
    df_test.to_csv("Data/adult_test.csv")