import streamlit as st
import pandas as pd
import pickle
import os

# Usar rutas absolutas basadas en la ubicación del archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(BASE_DIR, 'models', 'model.pkl')
mean_std_filename = os.path.join(BASE_DIR, 'models', 'mean_std_values.pkl')

# Función para cargar el modelo de forma segura
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(model_filename):
            st.error(f"Archivo del modelo no encontrado en: {model_filename}")
            st.info("Archivos disponibles en el directorio:")
            st.write(os.listdir(BASE_DIR))
            if os.path.exists(os.path.join(BASE_DIR, 'models')):
                st.write("Archivos en models/:", os.listdir(os.path.join(BASE_DIR, 'models')))
            st.stop()
            
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

# Función para cargar los valores de normalización
@st.cache_resource
def load_normalization_values():
    try:
        if not os.path.exists(mean_std_filename):
            st.error(f"Archivo de normalización no encontrado en: {mean_std_filename}")
            st.stop()
            
        with open(mean_std_filename, 'rb') as f:
            mean_std_values = pickle.load(f)
        return mean_std_values
    except Exception as e:
        st.error(f"Error al cargar valores de normalización: {str(e)}")
        st.stop()

# Cargar modelo y valores de normalización al inicio
model = load_model()
mean_std_values = load_normalization_values()

def main():
    st.title('Predicción de Enfermedad Cardiaca')
    age = st.slider('Edad', 18, 100, 50)
    sex_options = ['Masculino', 'Femenino']
    sex = st.selectbox('Sexo', sex_options)
    sex_num = 1 if sex == 'Masculino' else 0
    cp_options = ['Angina Típica', 'Angina Atypical', 'Dolor No Anginal', 'Asintomático']
    cp = st.selectbox('Tipo de Dolor en el Pecho', cp_options)
    cp_num = cp_options.index(cp)
    trestbps = st.slider('Presión Arterial en Reposo', 90, 200, 120)
    chol = st.slider('Colesterol', 100, 600, 250)
    fbs_options = ['Falso', 'Verdadero']
    fbs = st.selectbox('Azúcar en sangre en ayunas > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)
    restecg_options = ['Normal', 'Anormalidad ST-T', 'Hipertrofia Ventricular Izquierda']
    restecg = st.selectbox('Resultados Electrocardiográficos en Reposo', restecg_options)
    restecg_num = restecg_options.index(restecg)
    thalach = st.slider('Frecuencia Cardíaca Máxima Alcanzada', 70, 220, 150)
    exang_options = ['No', 'Sí']
    exang = st.selectbox('Angina Inducida por Ejercicio', exang_options)
    exang_num = exang_options.index(exang)
    oldpeak = st.slider('Depresión ST Inducida por Ejercicio en Relación al Reposo', 0.0, 6.2, 1.0)
    slope_options = ['Pendiente ascendente', 'Plana', 'Pendiente descendente']
    slope = st.selectbox('Pendiente del Segmento ST en Ejercicio Pico', slope_options)
    slope_num = slope_options.index(slope)
    ca = st.slider('Número de Vasos Principales Coloreados por Fluoroscopia', 0, 4, 1)
    thal_options = ['Normal', 'Defecto Fijo', 'Defecto Reversible']
    thal = st.selectbox('Talasemia', thal_options)
    thal_num = thal_options.index(thal)

    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        # Apply saved transformation to new data
        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Negativo'
        else:
            bg_color = 'green'
            prediction_result = 'Positivo'

        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Predicción: {prediction_result}<br>Confianza: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()