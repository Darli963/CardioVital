import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Directorio base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN_STD_PATH = os.path.join(BASE_DIR, 'models', 'mean_std_values.pkl')

@st.cache_resource
def load_normalization_values():
    """
    Carga los valores de media y desviación estándar de entrenamiento desde el archivo pickle.
    Retorna un diccionario con las claves 'mean' y 'std' o None si no existe el archivo.
    """
    if not os.path.exists(MEAN_STD_PATH):
        return None
    try:
        with open(MEAN_STD_PATH, 'rb') as f:
            mean_std = pickle.load(f)
        return mean_std
    except Exception as e:
        st.warning(f"Advertencia al cargar los valores de normalización: {e}")
        return None

def preprocess_input(data_dict):
    """
    Toma un diccionario de entradas de usuario, las formatea en un DataFrame
    con el orden de columnas exacto y aplica estandarización ( StandardScaler )
    utilizando la media y escala guardadas de entrenamiento si existen.
    
    Argumentos:
        data_dict (dict): Diccionario con los nombres de características como clave y valores como entrada.
        
    Retorna:
        pd.DataFrame: DataFrame de 1 fila procesado (estandarizado si aplica).
    """
    # Lista ordenada de características requeridas por el modelo KNN
    feature_order = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Crear DataFrame con el orden de columnas estricto
    df = pd.DataFrame([data_dict])
    df = df[feature_order]
    
    # Asegurar conversión a tipo numérico correcto (flotantes para operaciones matemáticas)
    df = df.astype(float)
    
    # Intentar aplicar escalado
    mean_std = load_normalization_values()
    if mean_std is not None and 'mean' in mean_std and 'std' in mean_std:
        try:
            mean = np.array(mean_std['mean'], dtype=float)
            std = np.array(mean_std['std'], dtype=float)
            
            # Estandarización: (X - mean) / std
            df_scaled = (df.values - mean) / std
            # Reconstruir el DataFrame con las columnas originales
            df = pd.DataFrame(df_scaled, columns=feature_order)
        except Exception as e:
            st.error(f"Error al aplicar la normalización de los datos: {e}. Se utilizarán los datos crudos.")
            
    return df
