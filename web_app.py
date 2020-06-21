import streamlit as st
import numpy as np
import pandas as pd
import joblib

test_lines_path = '/home/ubuntu/supervisedhealth/data/testids_heldout.csv'
features_in = '/home/ubuntu/supervisedhealth/data/features_rma_2.csv'
columns_path = '/home/ubuntu/supervisedhealth/data/features_columns.txt'
pickled_model_path = '/home/ubuntu/supervisedhealth/models'

drugs_lst = ['5-Fluorouracil', 'ABT737', 'Acetalax', 'Afatinib', 'Afuresertib', 'Alisertib', 'Alpelisib', 'AT13148', 'AZ960', 'AZ6102', 'AZD1332', 'AZD2014', 'AZD4547', 'AZD5153', 'AZD5363', 'AZD5438', 'AZD5582', 'AZD5991', 'AZD6738', 'AZD7762', 'AZD8186', 'BDP-00009066', 'BMS-345541', 'BMS-536924', 'Bortezomib', 'Buparlisib', 'Camptothecin', 'CDK9_5038', 'CDK9_5576', 'Cediranib', 'Cisplatin', 'Crizotinib', 'Cytarabine', 'Dabrafenib', 'Dactinomycin', 'Dactolisib', 'Dasatinib', 'Dihydrorotenone', 'Dinaciclib', 'Docetaxel', 'Eg5_9814', 'Elephantin', 'Entinostat', 'Entospletinib', 'Epirubicin', 'ERK_2440', 'ERK_6604', 'Erlotinib', 'Fludarabine', 'Foretinib', 'Gemcitabine', 'GNE-317', 'GSK2606414', 'I-BET-762', 'I-BRD9', 'IAP_5620', 'Ibrutinib', 'IGF1R_3801', 'Ipatasertib', 'Irinotecan', 'JAK_8517', 'JAK1_8709', 'KRAS (G12C) Inhibitor-12', 'Lapatinib', 'Leflunomide', 'Linsitinib', 'Luminespib', 'MIM1', 'Mirin', 'Mitoxantrone', 'MK-1775', 'MK-2206', 'MK-8776', 'ML323', 'Navitoclax', 'Nilotinib', 'Niraparib', 'Nutlin-3a (-)', 'NVP-ADW742', 'Obatoclax Mesylate', 'OF-1', 'Olaparib', 'Osimertinib', 'OTX015', 'Oxaliplatin', 'P22077', 'Paclitaxel', 'PAK_5339', 'Palbociclib', 'PCI-34051', 'PD173074', 'PD0325901', 'Pevonedistat', 'Pictilisib', 'PLX-4720', 'Podophyllotoxin bromide', 'PRIMA-1MET', 'PRT062607', 'Pyridostatin', 'Rapamycin', 'Sapitinib', 'SCH772984', 'Selumetinib', 'Sorafenib', 'TAF1_5496', 'Talazoparib', 'Tamoxifen', 'Taselisib', 'Telomerase Inhibitor IX', 'Teniposide', 'Topotecan', 'Trametinib', 'Ulixertinib', 'ULK1_4989', 'UMI-77', 'Uprosertib', 'VE-822', 'VE821', 'Venetoclax', 'Vinblastine', 'Vincristine', 'Vinorelbine', 'Vorinostat', 'VSP34_8731', 'VX-11e', 'Wee1 Inhibitor', 'WEHI-539', 'WIKI4', 'WZ4003', 'YK-4-279']

@st.cache
def read_cell_lines(test_lines_path):
    test_lines_df = pd.read_csv(test_lines_path)
    test_lines_names = list(test_lines_df['id_testset'])
    return test_lines_names

@st.cache  # This function will be cached
def load_features(features_in):
    features_df = pd.read_csv(features_in)
    return features_df

@st.cache # This function will be cached
def read_column_order(columns_path):
    with open(columns_path) as f:
        columns_lst = f.read().splitlines()
    return columns_lst

@st.cache
def load_models(pickled_model_path, drugs_lst):
    drug_models = []
    for drug in drugs_lst:

        model_path = f'{pickled_model_path}/{drug}_model'
        loaded_model = joblib.load(model_path)
        drug_models.append((drug, loaded_model))
    return drug_models

def select_test_features(user_line, features_df, columns_lst):
    test_features = features_df[features_df['cell_line_name']==user_line]
    selected_features = test_features[columns_lst]
    return selected_features

st.title('Supervised Health')
st.header('Enabling Oncologists to Predict The Path of Least Resistance')

# A box for selecting cancer type.

user_cancer = st.selectbox("Please select patient's cancer type", ['Breast cancer', 'Lung cancer'])
st.write('You selected:', user_cancer)

# Add a box with a list of cell lines

test_lines_names = read_cell_lines(test_lines_path)

user_line = st.selectbox("Please select test gene profile", test_lines_names)
st.write('You selected:', user_line)

# Pull gene expression data for the selected line (var user_line above)

features_df = load_features(features_in)
    
columns_lst = read_column_order(columns_path)

selected_features = select_test_features(user_line, features_df, columns_lst)

# Load models

probabilities_lst = []
predictions_lst = []

drug_models = load_models(pickled_model_path, drugs_lst)

for drug, model in drug_models:

    prob_estimate = model.predict_proba(selected_features)
    probabilities_lst.append(prob_estimate[0,1]) 
    if prob_estimate[0,1] >= 0.5:
        prediction = 'Sensitive'
    else:
        prediction = 'Resistant'
    predictions_lst.append(prediction)

results_dct = {'Drug': drugs_lst, 'Probability estimate for sensitivity': probabilities_lst, 'Prediction': predictions_lst}

results_df = pd.DataFrame(results_dct)
st.write('Supervised Health predicts the following response profile for', user_line)
st.write(results_df)
    
