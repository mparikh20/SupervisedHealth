import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report

# 

def label_data(testids_heldout_path, median_path, ic50_path):
    df_median = pd.read_csv(median_path)
    df_test_id = pd.read_csv(testids_heldout_path)
    cell_id = list(df_test_id['id_testset'])
    df_ic50 = pd.read_csv(ic50_path)
    df_ic50 = df_ic50[df_ic50['cell line name'].isin(cell_id)]
    labels_test_df = pd.merge(df_ic50, df_median, on='drug name', how='inner')
    labels_test_df['sensitivity_label'] = np.where(labels_test_df['exp_ic50'] <= labels_test_df['median_ic50'], 1, 0)
    labels_test_df = labels_test_df[['cell line name', 'drug name', 'sensitivity_label']]
    return labels_test_df    


# rmatraining_perdrug
def create_dataperdoi(labels_test_df, drug, features_all_df): 
    doi_labels_df = labels_test_df[labels_test_df['drug name']==drug].reset_index(drop=True) # Filters out selected drug rows.    
    doi_df = pd.merge(doi_labels_df, features_all_df, left_on = 'cell line name', right_on='cell_line_name')
    return doi_df 

def read_column_order(col_order_path):
    with open(col_order_path) as f:
        columns_lst = f.read().splitlines()
    return columns_lst

# rmapreprocess_trainfile
# generate normalized features and assign X and y.
def assignxy_doi(doi_df, columns_lst):
    # Take only gene features
    X = doi_df[doi_df.columns.difference(['drug name', 'cell_line_name', 'cell line name', 
        'sensitivity_label', 'cell_line_id', 'cosmic sample id', 
        'aero_digestive_tract', 'blood', 'bone', 'breast',
       'digestive_system', 'kidney', 'lung', 'nervous_system', 'pancreas',
       'skin', 'soft_tissue', 'thyroid', 'urogenital_system'], sort=False)]
    X_norm_2 = pd.DataFrame()
    y = doi_df['sensitivity_label']

    for column in X.columns:
        X_norm_2[column] = X[column]/X['GAPDH']
    X_norm_2 = X_norm_2[columns_lst]
    return X_norm_2, y

def load_models(pickled_model_path, selected_drugs_lst):
    drug_models = []
    for drug in selected_drugs_lst:

        model_path = f'{pickled_model_path}/{drug}_model'
        loaded_model = joblib.load(model_path)
        drug_models.append((drug, loaded_model))
    return drug_models

# minmax_trainvalidate

def run_models(selected_drugs_lst, labels_test_df, drug_models, features_in, columns_lst):
    features_all_df = pd.read_csv(features_in)

    for drug, model in drug_models:
        doi_df = create_dataperdoi(labels_test_df, drug, features_all_df)
        X_norm_2, y = assignxy_doi(doi_df, columns_lst)
        prob_estimate = model.predict_proba(X_norm_2)
        auc = roc_auc_score(y, prob_estimate[:,1])
        report = classification_report(y, model.predict(X_norm_2), output_dict=True) # F1, precision, recall
        print(f'{drug}, AUC, {auc}')
        print(f'{drug}, Report, {report}')

def main():

    # File paths
    labels_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/labels.csv'
    testids_heldout_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/testids_heldout.csv'
    median_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/median_ic50.csv'
    ic50_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/all_ic50.csv'

    features_in = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/features_rma_2.csv'
    pickled_model_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_models_8'
    col_order_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/features_columns.txt'

    # Implementation 
    labels_test_df = label_data(testids_heldout_path, median_path, ic50_path)
    selected_drugs_lst = list(labels_test_df['drug name'].unique())
    columns_lst = read_column_order(col_order_path)
    drug_models = load_models(pickled_model_path, selected_drugs_lst)
    run_models(selected_drugs_lst, labels_test_df, drug_models, features_in, columns_lst)

if __name__ == '__main__':
    main()