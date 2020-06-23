import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib

test_lines_path = '/home/ubuntu/supervisedhealth/data/testids_heldout.csv'
features_in = '/home/ubuntu/supervisedhealth/data/features_rma_2.csv'
columns_path = '/home/ubuntu/supervisedhealth/data/features_columns.txt'
pickled_model_path = '/home/ubuntu/supervisedhealth/models'

drugs_lst = ["5-Fluorouracil", "ABT737", "AGI-5198", "AGI-6780", "AMG-319", "AT13148", "AZ6102", "AZ960", "AZD1208", "AZD1332", "AZD2014", "AZD3759", "AZD4547", "AZD5153", "AZD5363", "AZD5438", "AZD5582", "AZD5991", "AZD6738", "AZD7762", "AZD8186", "Acetalax", "Afatinib", "Afuresertib", "Alisertib", "Alpelisib", "BDP-00009066", "BIBR-1532", "BMS-345541", "BMS-536924", "BPD-00008900", "Bortezomib", "Buparlisib", "CDK9_5038", "CDK9_5576", "CZC24832", "Camptothecin", "Carmustine", "Cediranib", "Cisplatin", "Crizotinib", "Cyclophosphamide", "Cytarabine", "Dabrafenib", "Dactinomycin", "Dactolisib", "Dasatinib", "Dihydrorotenone", "Dinaciclib", "Docetaxel", "EPZ004777", "EPZ5676", "ERK_2440", "ERK_6604", "Eg5_9814", "Elephantin", "Entinostat", "Entospletinib", "Epirubicin", "Erlotinib", "Fludarabine", "Foretinib", "Fulvestrant", "GDC0810", "GNE-317", "GSK1904529A", "GSK2578215A", "GSK2606414", "GSK343", "GSK591", "Gallibiscoquinazole", "Gefitinib", "Gemcitabine", "I-BET-762", "I-BRD9", "IAP_5620", "IGF1R_3801", "IRAK4_4710", "IWP-2", "Ibrutinib", "Ipatasertib", "Irinotecan", "JAK1_8709", "JAK_8517", "KRAS (G12C) Inhibitor-12", "LCL161", "LGK974", "LJI308", "LY2109761", "Lapatinib", "Leflunomide", "Linsitinib", "Luminespib", "MG-132", "MIM1", "MIRA-1", "MK-1775", "MK-2206", "MK-8776", "ML323", "MN-64", "Mirin", "Mitoxantrone", "NVP-ADW742", "Navitoclax", "Nelarabine", "Nilotinib", "Niraparib", "Nutlin-3a (-)", "OF-1", "OTX015", "Obatoclax Mesylate", "Olaparib", "Osimertinib", "Oxaliplatin", "P22077", "PAK_5339", "PCI-34051", "PD0325901", "PD173074", "PFI3", "PLX-4720", "PRIMA-1MET", "PRT062607", "Paclitaxel", "Palbociclib", "Pevonedistat", "Picolinici-acid", "Pictilisib", "Podophyllotoxin bromide", "Pyridostatin", "RVX-208", "Rapamycin", "Ruxolitinib", "SCH772984", "Sabutoclax", "Sapitinib", "Savolitinib", "Selumetinib", "Sepantronium bromide", "Sinularin", "Sorafenib", "Staurosporine", "TAF1_5496", "Talazoparib", "Tamoxifen", "Taselisib", "Telomerase Inhibitor IX", "Temozolomide", "Teniposide", "Topotecan", "Trametinib", "ULK1_4989", "UMI-77", "Ulixertinib", "Uprosertib", "VE-822", "VE821", "VSP34_8731", "VX-11e", "Venetoclax", "Vinblastine", "Vincristine", "Vinorelbine", "Vorinostat", "WEHI-539", "WIKI4", "WZ4003", "Wee1 Inhibitor", "Wnt-C59", "XAV939", "YK-4-279", "Zoledronate"]

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

# A checkbox showing a note about this work.
if st.checkbox('Show important note about this application'):
        st.write('This data scientist project is built by cancer biologist and entrepreneur Mukti Parikh, while being an Insight Health Data Science Fellow.')
        st.write('Although conceptualized as a tool that can be used by oncologists, this is not a clinical tool.')

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

def highlight_frame(row):
    if row.values[-1] == 'Sensitive':
        color = 'lightblue'
    else:
        color = 'yellow'
    return ['background-color: %s' % color]*len(row.values)

st.dataframe(results_df.style.apply(highlight_frame, axis=1))

# Show the histogram of values
plt.style.use('ggplot')
plt.hist(probabilities_lst, edgecolor='black')
plt.title('Histogram of probabilities')
plt.xlabel('Probability')
plt.ylabel('Number of drugs')
st.pyplot()
