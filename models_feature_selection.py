import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectFromModel


def label_data(testids_heldout_path, median_path, ic50_path):
    df_median = pd.read_csv(median_path)
    df_test_id = pd.read_csv(testids_heldout_path)
    cell_id = list(df_test_id['id_testset'])
    df_ic50 = pd.read_csv(ic50_path)
    df_ic50 = df_ic50[~df_ic50['cell line name'].isin(cell_id)]
    labels_train_df = pd.merge(df_ic50, df_median, on='drug name', how='inner')
    labels_train_df['sensitivity_label'] = np.where(labels_train_df['exp_ic50'] <= labels_train_df['median_ic50'], 1, 0)
    labels_train_df = labels_train_df[['cell line name', 'drug name', 'sensitivity_label']]
    return labels_train_df


# rmatraining_perdrug
def create_dataperdoi(labels_train_df, drug, features_all_df): 
    doi_labels_df = labels_train_df[labels_train_df['drug name']==drug].reset_index(drop=True) # Filters out selected drug rows.    
    doi_df = pd.merge(doi_labels_df, features_all_df, left_on = 'cell line name', right_on='cell_line_name')
    return doi_df 

# rmapreprocess_trainfile
# generate normalized features and assign X and y.
def assignxy_doi(doi_df):
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
    return X_norm_2, y

# Write column order into a file.
def write_feature_columns(X_norm_2, col_order_path): 
    column_order = list(X_norm_2.columns)
    with open(col_order_path, 'w') as col_file:
        for column in column_order:
            col_file.write(f'{column}\n')

# minmax_trainvalidate
def getauc_scaletrainvalidate(X_norm_2, y, scaler, feature_selector, estimator):
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_norm_2, y, test_size=0.2, stratify=y, random_state=0)
    pipe = Pipeline([('scaler', scaler), ('feature_selection', feature_selector), ('model', estimator)])
    pipe.fit(Xn_train, yn_train)
    yn_scores = pipe.predict_proba(Xn_test)
    yn_pred = pipe.predict(Xn_test)
    precision = precision_score(yn_test, yn_pred)
    recall = recall_score(yn_test, yn_pred)
    accuracy = pipe.score(Xn_test, yn_test)
    auc = roc_auc_score(yn_test, yn_scores[:,1])
    return auc, accuracy, precision, recall

def run_models(selected_drugs_lst, labels_train_df, features_all_df, scaler, feature_selector, estimator, pickled_model_path):
    for drug in selected_drugs_lst:
        doi_df = create_dataperdoi(labels_train_df, drug, features_all_df)
        X_norm_2, y = assignxy_doi(doi_df)
        auc, accuracy, precision, recall = getauc_scaletrainvalidate(X_norm_2, y, scaler, feature_selector, estimator)
        print(f'{drug}, AUC, {auc}, Accuracy, {accuracy}, Precision, {precision}, Recall, {recall}') 
        pipe = Pipeline([('scaler', MinMaxScaler()), ('feature_selection', feature_selector), ('model', estimator)])
        pipe.fit(X_norm_2, y)
        dump(pipe, f'{pickled_model_path}/{drug}_model')
        coef = feature_selector.estimator_.coef_
        coef_df = pd.DataFrame({'features': X_norm_2.columns, 'coefficients': coef[0]})
        coef_df['abs_coef'] = coef_df['coefficients'].abs()
        coef_df = coef_df.sort_values('abs_coef', ascending=False)
        nonzerocoefs_df = coef_df[coef_df['abs_coef']!=0]
        nonzerocoefs_df.to_csv(f'{pickled_model_path}/{drug}_coefs.csv', index=False)

def main():

    # File paths
    ic50_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/all_ic50.csv'

    testids_heldout_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/testids_heldout.csv'
    features_in = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/features_rma_2.csv'
    pickled_model_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_models_8'
    col_order_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/features_columns.txt'
    median_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_8/median_ic50.csv'

    # Implementation 
    # Remove test ids from the labels file.
    labels_train_df = label_data(testids_heldout_path, median_path, ic50_path)
    selected_drugs_lst = list(labels_train_df['drug name'].unique())
    # Load the main dataframe
    features_all_df = pd.read_csv(features_in)
    doi_df = create_dataperdoi(labels_train_df, 'Paclitaxel', features_all_df)
    X_norm_2, y = assignxy_doi(doi_df)
    write_feature_columns(X_norm_2, col_order_path)
    scaler = MinMaxScaler()
    feature_selector = SelectFromModel(LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=0))
    estimator = LogisticRegression(penalty='l2', C=0.01, class_weight = 'balanced', max_iter=1000, random_state=0)
    run_models(selected_drugs_lst, labels_train_df, features_all_df, scaler, feature_selector, estimator, pickled_model_path)

if __name__ == '__main__':
    main()