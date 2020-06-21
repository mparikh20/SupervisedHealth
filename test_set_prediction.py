import pandas as pd
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
def labels_retaintestset(labels_path, testids_heldout_path):
    labels_all = pd.read_csv(labels_path)
    id = pd.read_csv(testids_heldout_path)
    cellname_idlist = list(id['id_testset']) # list of model ids held out
    labels_test_df = labels_all[labels_all['cell line name'].isin(cellname_idlist) == True]
    return labels_test_df

def select_drugsofinterest(min_proportion_sensitive, max_proportion_sensitive, sensitivity_proportion_path):
    sensitivity_proportion_df = pd.read_csv(sensitivity_proportion_path)
    selected_drugs_lst = list(sensitivity_proportion_df[(sensitivity_proportion_df['proportion_sensitive'] >= min_proportion_sensitive) & (sensitivity_proportion_df['proportion_sensitive'] <= max_proportion_sensitive)]['drug name'])
    return selected_drugs_lst

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
    sensitivity_proportion_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/sensitivity_proportion.csv'
    labels_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/labels.csv'
    testids_heldout_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/testids_heldout.csv'
    features_in = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/features_rma_2.csv'
    pickled_model_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_models_6'
    col_order_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/features_columns.txt'

    # Implementation 
    labels_test_df = labels_retaintestset(labels_path, testids_heldout_path)
    selected_drugs_lst = select_drugsofinterest(5, 95, sensitivity_proportion_path)
    columns_lst = read_column_order(col_order_path)
    drug_models = load_models(pickled_model_path, selected_drugs_lst)
    run_models(selected_drugs_lst, labels_test_df, drug_models, features_in, columns_lst)

if __name__ == '__main__':
    main()