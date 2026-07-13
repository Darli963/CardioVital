import os
import pickle
import numpy as np
import streamlit as st

# Directorio base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

@st.cache_resource
def load_model():
    """
    Carga el modelo clasificador .pkl entrenado de forma segura.
    Retorna el modelo cargado o lanza un error si no se encuentra.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Archivo del modelo no encontrado en: {MODEL_PATH}")
    
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise RuntimeError(f"Error al deserializar el modelo: {e}")

def predict_heart_disease(X_processed):
    """
    Realiza la predicción de enfermedad cardíaca utilizando el modelo KNN cargado.
    
    Argumentos:
        X_processed (pd.DataFrame): DataFrame preprocesado y normalizado con las variables.
        
    Retorna:
        dict: Diccionario con la predicción oficial (0 o 1), confianza oficial del modelo,
              y métricas enriquecidas de vecinos cercanos (índice de riesgo de vecinos).
    """
    # Cargar el modelo
    model = load_model()
    
    # Convertir a arreglo numpy para evitar advertencias de nombres de características
    X_values = X_processed.values if hasattr(X_processed, 'values') else X_processed
    
    # Realizar predicción estándar
    prediction = int(model.predict(X_values)[0])
    predict_proba = model.predict_proba(X_values)[0]
    
    # La confianza oficial del modelo entrenado (K=1)
    official_confidence = float(predict_proba[prediction])
    
    # Propuesta Premium: Análisis de los 5 vecinos más cercanos para el "Índice de similitud de riesgo"
    neighbor_risk_index = 0.0
    distances = []
    neighbors_labels = []
    
    if hasattr(model, 'kneighbors') and hasattr(model, '_y'):
        try:
            # Obtener distancias e índices de los 5 vecinos más cercanos
            dist, indices = model.kneighbors(X_values, n_neighbors=5)
            distances = dist[0].tolist()
            
            # Obtener las etiquetas de la base de entrenamiento para esos vecinos
            # model._y contiene las clases de los datos de entrenamiento
            train_labels = np.array(model._y)
            neighbors_labels = [int(train_labels[idx]) for idx in indices[0]]
            
            # Calcular la tasa de vecinos que tienen enfermedad cardíaca (clase 1)
            # Esto provee un riesgo continuo entre 0.0 y 1.0 (en pasos de 0.2)
            neighbor_risk_index = float(np.mean(np.array(neighbors_labels) == 1))
        except Exception as e:
            # En caso de error, el índice de riesgo de vecinos se iguala al resultado de predicción
            neighbor_risk_index = float(prediction)
            
    else:
        # En caso de que no tenga los atributos necesarios
        neighbor_risk_index = float(prediction)
        
    return {
        'prediction': prediction,
        'official_confidence': official_confidence,
        'neighbor_risk_index': neighbor_risk_index,
        'neighbors_labels': neighbors_labels,
        'neighbors_distances': distances
    }
