import streamlit as st
from PIL import Image
import pandas as pd
import time

import joblib
import numpy as np
import plotly.express as px

#------------------------------------------------------------------------------------------------
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer, AddMissingIndicator
from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.pipeline import Pipeline

import input.preprocessors as pp
from configurations import config

#------------------------------------------------------------------------------------------------

def prediccion_o_inferencia(pipeline_de_test, datos_de_test):
    
    datos_de_test.drop('Id', axis=1, inplace=True)
    datos_de_test['MSSubClass'] = datos_de_test['MSSubClass'].astype('O')
    datos_de_test = datos_de_test[config.FEATURES]

    new_vars_with_na = [
        var for var in config.FEATURES
        if var not in config.CATEGORICAL_VARS_WITH_NA_FREQUENT +
        config.CATEGORICAL_VARS_WITH_NA_MISSING +
        config.NUMERICAL_VARS_WITH_NA
        and datos_de_test[var].isnull().sum() > 0]
    
    datos_de_test.dropna(subset=new_vars_with_na, inplace=True)

    predicciones = np.round(pipeline_de_test.predict(datos_de_test), 2)
    predicciones_sin_escalar = np.round(np.exp(predicciones), 2)

    return predicciones, predicciones_sin_escalar, datos_de_test

# ----------------------- Interface Design ----------------------- #
st.title("ML Project - House Price Prediction")

image = Image.open('src/images/image.png')
st.image(image, use_container_width=True)

st.sidebar.write("Upload the CSV file with the features to use to make the prediction")

#------------------------------------------------------------------------------------------

# ----------------------- Upload CSV ----------------------- #
uploaded_file = st.sidebar.file_uploader(" ", type=['csv'])

if uploaded_file is not None:
    df_de_los_datos_subidos = pd.read_csv(uploaded_file)

    st.write('CSV file content as a DataFrame:')
    st.dataframe(df_de_los_datos_subidos)
#-------------------------------------------------------------------------------------------

# ----------------------- Upload Pipeline ----------------------- #
pipeline_de_produccion = joblib.load('src/house_prices_pipeline.joblib')

if st.sidebar.button("Click here to send the CSV to the Pipeline"):
    if uploaded_file is None:
        st.sidebar.write("The file was not uploades correctly. Upload it again...")
    else:
        with st.spinner('Pipeline and Model processing...'):

            prediccion, prediccion_sin_escalar, datos_procesados = prediccion_o_inferencia(pipeline_de_produccion, df_de_los_datos_subidos)
            time.sleep(5)
            st.success('Listo!', icon="✅")
            st.balloons()

            st.write('Prediction results:')
            st.write(prediccion)
            st.write(prediccion_sin_escalar)
            
            
            fig = px.histogram(
                np.exp(prediccion),
                nbins=10,
                labels={"value":"House price"},
                text_auto=True,
                title="House Prices Predictions - Histogram"
                )
            
            fig.update_layout(
                showlegend = False
            )

            st.plotly_chart(fig)

            # ----------------------- Download CSV and predictions ----------------------- #
            df_resultado = datos_procesados.copy()
            df_resultado['Scaled prediction'] = prediccion
            df_resultado['Unscaled prediction'] = prediccion_sin_escalar

            st.write('Original data and predictions:')
            st.dataframe(df_resultado)

            csv = df_resultado.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download CSV file with predictions",
                data=csv,
                file_name='house_prices_predictions.csv',
                mime='text/csv',
            )
            #-------------------------------------------------------------------
