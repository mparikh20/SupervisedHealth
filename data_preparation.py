import pandas as pd
import numpy as np

def write_labels(gdsc2_path, labels_path):
    ic50_df = pd.read_csv(gdsc2_path)
    # Convert column labels to lowercase
    ic50_df.columns = map(str.lower, ic50_df.columns)
    # Drop irrelevant columns

    ic50_df2 = ic50_df.drop(['auc', 'rmse', 'z score', 'dataset version'], axis=1)
    # Add a new column that takes ln(ic50) column and takes exponent of it.
    ic50_df2['exp_ic50'] = ic50_df2['ic50'].apply(lambda x: np.exp(x))
    ic50_df2['sensitivity_label'] = ic50_df2.apply(lambda x: 0 if x['exp_ic50'] > x['max conc'] else 1, axis=1)
    ic50_df2.drop_duplicates(subset=['drug name', 'cell line name'], inplace=True)
    ic50_df2[['drug name', 'cell line name', 'sensitivity_label']].to_csv(labels_path, index=False)
    return ic50_df2

def write_sensitivity_proportion(ic50_df2, out_path):
    sensitivity_proportion_df = ic50_df2.groupby('drug name')['sensitivity_label'].value_counts().unstack().reset_index()
    sensitivity_proportion_df.rename(columns = {0 : 'resistant', 1: 'sensitive'}, inplace=True)
    sensitivity_proportion_df['total_data'] = sensitivity_proportion_df['sensitive'] + sensitivity_proportion_df['resistant']
    sensitivity_proportion_df['proportion_sensitive'] = (sensitivity_proportion_df['sensitive'] / sensitivity_proportion_df['total_data'])*100
    sensitivity_proportion_df[(sensitivity_proportion_df['total_data']>=600)].to_csv(out_path, index=False)

def prepare_rma_features(rma_path, ic50_df2, features_out):
    rma_df = pd.read_csv(rma_path, delimiter = "\t")
    # Drop missing gene_symbols rows.
    rma_df = rma_df.dropna(subset=['GENE_SYMBOLS'])
    rma_df.drop(columns=['GENE_title'], inplace=True)
    rma_df2 = rma_df.transpose()
    rma_df3 = rma_df2.rename(columns=rma_df2.iloc[0]).drop(['GENE_SYMBOLS'])
    rma_df3.reset_index(inplace=True)
    rma_df3['cell_line_id'] = rma_df3['index'].apply(lambda x: x.split('.')[1])
    rma_df3.drop(columns=['index'], inplace=True)
    rma_df3['cell_line_id'] = rma_df3['cell_line_id'].astype('int')
    cell_lines = ic50_df2[['cell line name', 'cosmic sample id']].copy()
    cell_lines.drop_duplicates(inplace=True)
    rma_df4 = pd.merge(rma_df3, cell_lines, right_on='cosmic sample id', left_on='cell_line_id', how='inner')
    rma_df5 = rma_df4.rename(columns={'cell line name': 'cell_line_name'}).copy()
    cancer_type_df = ic50_df2[['cell line name', 'tissue']].drop_duplicates()
    cancer_type_df['cancer_type'] = 1
    cancer_type_df.rename(columns={'cell line name': 'cell_line_name'}, inplace=True)
    cancer_type_df2 = cancer_type_df.pivot(index='cell_line_name', values='cancer_type', columns='tissue').fillna(0).reset_index()
    rma_df5 = pd.merge(rma_df5, cancer_type_df2, on='cell_line_name', how='left').fillna(0)
    rma_df5.to_csv(features_out, index=False)
    return rma_df5

def testset_id(features_df, column, fraction, testids_heldout_path):
    n = int(len(features_df)*fraction)
    test_id = np.random.choice(features_df[column], n, replace=False)    
    pd.Series(test_id).to_csv(testids_heldout_path,index=False, header=['id_testset'])


def main():
    gdsc2_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_raw_2/pancancer_ic50_gdsc2.csv'
    
    labels_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_3/labels.csv'
    sensitivity_proportion_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_3/sensitivity_proportion.csv'
    rma_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_raw_2/cell_line_rma.csv'
    features_out = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_3/features_rma_2.csv'
    testids_heldout_path = '/Users/mukti/Documents/10_Insight_Project/sh_data/sh_processed_3/testids_heldout.csv'

    ic50_df2 = write_labels(gdsc2_path, labels_path)
    write_sensitivity_proportion(ic50_df2, sensitivity_proportion_path)
    features_df = prepare_rma_features(rma_path, ic50_df2, features_out)
    testset_id(features_df, 'cell_line_name', 0.1, testids_heldout_path)

if __name__ == '__main__':
    main()

