import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction #, ordinal_encoder


model = joblib.load(r'gbm_model.joblib')    
#model = joblib.load(r'best_lgbm_model.pkl')
#model = joblib.load(r'NEW_lgbm_model.joblib')
#model = joblib.load(r'best_catboost_model.joblib')
#df_train = pd.read_pickle("./df_train.pkl") 
#print(df_train)

st.set_page_config(page_title="Loan Approval Prediction App", page_icon= " ", layout="wide")


#creating option list for dropdown menu
options_loan_grade = ['A', 'B', 'C', 'D', 'E']
options_person_home_ownership = ['RENT', 'MORTGAGE', 'OWN', 'OTHERS']
options_loan_intent = ['EDUCATION', 'MEDICAL', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']
options_cb_person_default_on_file = ['Y', 'N']

features = ['loan_grade', 'loan_percent_income', 'person_home_ownership', 'loan_int_rate', 'loan_intent', 'person_income', 'person_emp_length', 'loan_amount', 'cb_person_default_on_file', 'person_age', 'cb_person_cred_hist_length']

st.markdown("<h1 style = 'text-align: center;'>Loan Approval Prediction Application <h1>", unsafe_allow_html = True)
def main():
    with st.form("prediction_form"):

        st.subheader("Enter the input for the following features:") 

        loan_grade = st.selectbox("Select Loan Grade : ", options=options_loan_grade)
        loan_percent_income = st.slider("loan_percent_income : ", 0.1, 0.83, value = .3, format = "%f")
        person_home_ownership = st.selectbox("Select Person Home Ownership : ", options=options_person_home_ownership)
        loan_int_rate = st.slider("Loan Interest Rate : ", 5.42, 23.22, value = 23.22, format = "%f")
        loan_intent = st.selectbox("Select Loan Intent : ", options=options_loan_intent)
        person_income = st.slider("Person Income : ", 4200, 1900000, value = 4200, format = "%d")  #int
        person_emp_length = st.slider("Person Employment Length : ", 1.0 , 60.0, value = 9.0, format = "%f")
        loan_amnt = st.slider("Loan Amount : ", 500, 35000, value = 500, format = "%d")  #int
        cb_person_default_on_file= st.selectbox("Selects CB Person Default On File: ", options=options_cb_person_default_on_file)
        person_age = st.slider("Person Age : ", 20, 123, value = 20, format = "%d")     #int
        cb_person_cred_hist_length = st.slider("CB Person Cred Hist Length : ", 2, 30, value = 5, format = "%d") #int
        
        #Filling mode values for other features
        
        submit = st.form_submit_button("Predict")

    if submit:
        # Create the DataFrame with input data
        import pandas as pd
        data = pd.DataFrame({
            'person_age': [person_age],
            'person_income': [person_income],
            'person_home_ownership': [person_home_ownership],
            'person_emp_length': [person_emp_length],
            'loan_intent': [loan_intent],
            'loan_grade': [loan_grade],
            'loan_amnt': [loan_amnt],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_default_on_file': [cb_person_default_on_file],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length]
        })
        
        # Add derived features
        data['income_loan_ratio'] = data['person_income'] / data['loan_amnt']
        data['person_emp_length_to_person_age'] = data['person_emp_length'] / data['person_age']
        data['int_rate_income_ratio'] = data['loan_int_rate'] / data['person_income']
        data['cred_hist_age_ratio'] = data['cb_person_cred_hist_length'] / data['person_age']
        data["loan_percent_income_to_income"] = data["loan_percent_income"] / data["person_income"]
        data['person_age_to_person_income'] = data['person_age'] / data['person_income']
        data['loan_int_rate_to_loan_amnt'] = data['loan_int_rate'] / data['loan_amnt']


        # Print initial state
        print("\n")
        print("\nData before encoding and scaling :\n")
        print(data)

        enc_sc_data = data.copy()

        # Load the trained TargetEncoder object
        with open('target_encoder.pkl', 'rb') as file:
            loaded_encoder = pickle.load(file)

        print("\nTarget Encoder successfully loaded!\n")

        #print("encoded value=", loaded_encoder.transform(loan_grade))

        # Define categorical features
        categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

        # Apply the encoder to new data (e.g., validation or test data)
        enc_sc_data[categorical_features] = loaded_encoder.transform(enc_sc_data[categorical_features])

        #data_encoded = loaded_encoder.transform(data[cat_features])

        print("\nType of Data after encoding:\n", type(enc_sc_data))
        print("\nData after encoding:\n", enc_sc_data)

        # Load the scaler
        with open('scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)

        print("\nScaler successfully loaded!\n")

        # Define numerical features
        numerical_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'income_loan_ratio', 'person_emp_length_to_person_age',
                        'int_rate_income_ratio', 'cred_hist_age_ratio', 'loan_percent_income_to_income', 'person_age_to_person_income', 'loan_int_rate_to_loan_amnt']

        # Apply the scaler to new data (e.g., validation or test data)
        enc_sc_data[numerical_features] = loaded_scaler.transform(enc_sc_data[numerical_features])
        #data_scaled = loaded_scaler.transform(data[num_features])
        print("\nType of Data after scaling:\n", type(enc_sc_data))
        print("\nData after scaling:\n", enc_sc_data)

        # Display scaled data
        #import pandas as pd
        #X_valid_scaled_df = pd.DataFrame(data_scaled, columns=num_features)
        print(enc_sc_data.head())

        print("\nData after encoding and scaling\n", enc_sc_data)

        # Make predictions
        prediction = model.predict(enc_sc_data)

        # Display results
        st.write(f"The predicted severity is: {prediction}")

        if prediction[0] < 0.5:
            st.success("Loan Approved ✅")
        else:
            st.error("Loan Not Approved ❌")

if __name__ == '__main__':
            main()