import joblib
import pandas as pd
import streamlit as st

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmer penguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
"""
)

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    st.sidebar.write('or fill the features form bellow')
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)

        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])

        return features

    input_df = user_input_features()


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)


# Loading the trained model
model_artefacts = joblib.load('models/model.pkl')
features = model_artefacts['features']
model = model_artefacts['model']

X = input_df.filter(features)

# Apply model to make predictions
prediction_label = model.predict(X)
prediction_proba = model.predict_proba(X)

st.subheader('Prediction')
st.success(prediction_label[0])

st.subheader('Prediction Probability')
prediction_proba_df = pd.DataFrame(prediction_proba)
prediction_proba_df.columns = ['Adelie', 'Chinstrap', 'Gentoo']
st.write(prediction_proba_df)
