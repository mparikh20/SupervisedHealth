import pandas as pd
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
from sklearn.svm import LinearSVC


def labels_removetestset(labels_path, testids_heldout_path):
    labels_all = pd.read_csv(labels_path)
    id = pd.read_csv(testids_heldout_path)
    cellname_idlist = list(id['id_testset']) # list of model ids to be held out
    labels_train_df = labels_all[labels_all['cell line name'].isin(cellname_idlist) == False]
    return labels_train_df

def select_drugsofinterest(min_proportion_sensitive, max_proportion_sensitive, sensitivity_proportion_path):
    sensitivity_proportion_df = pd.read_csv(sensitivity_proportion_path)
    selected_drugs_lst = list(sensitivity_proportion_df[(sensitivity_proportion_df['proportion_sensitive'] >= min_proportion_sensitive) & (sensitivity_proportion_df['proportion_sensitive'] <= max_proportion_sensitive)]['drug name'])
    return selected_drugs_lst

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
    sensitivity_proportion_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/sensitivity_proportion.csv'
    labels_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/labels.csv'
    testids_heldout_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/testids_heldout.csv'
    features_in = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/features_rma_2.csv'
    pickled_model_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_models_6'
    col_order_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_6/features_columns.txt'

    # Implementation 
    # Remove test ids from the labels file.
    labels_train_df = labels_removetestset(labels_path, testids_heldout_path)
    # Select drugs of interest based on class proportion
    selected_drugs_lst = select_drugsofinterest(5, 95, sensitivity_proportion_path)
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