import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Configuración de página de Streamlit
st.set_page_config(
    page_title="CardioVital - Predicción de Riesgo Cardíaco",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar utilidades locales
from utils.preprocess import preprocess_input
from utils.predictor import predict_heart_disease, load_model
from utils.helpers import (
    get_chest_pain_label,
    get_restecg_label,
    get_slope_label,
    get_thal_label,
    get_health_explanation,
    create_risk_gauge,
    create_comparison_chart
)

# Directorio base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar fuentes de Google e inyectar CSS de alta calidad
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
        
        /* Aplicar tipografías */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }
        
        /* Modificar colores primarios */
        .stButton>button {
            background: linear-gradient(135deg, #1D3557 0%, #457B9D 100%);
            color: white;
            border-radius: 8px;
            padding: 12px 28px;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px rgba(29, 53, 87, 0.15);
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #E63946 0%, #1D3557 100%);
            box-shadow: 0 6px 12px rgba(230, 57, 70, 0.25);
            transform: translateY(-2px);
            color: white;
        }
        
        /* Estilizar contenedores y tarjetas */
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-left: 5px solid #457B9D;
            margin-bottom: 15px;
        }
        
        /* Estilizar alertas de riesgo */
        .risk-high {
            background-color: #FFEAEB;
            border-left: 6px solid #E63946;
            color: #4A0E17;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .risk-low {
            background-color: #E8F5E9;
            border-left: 6px solid #2E7D32;
            color: #1B5E20;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        /* Pie de página */
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 50px;
            color: #8D99AE;
            font-size: 0.9em;
            border-top: 1px solid #E2E8F0;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- BARRA LATERAL (SIDEBAR) -----------------
st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

# Intentar cargar y mostrar el logo
logo_path = os.path.join(BASE_DIR, 'assets', 'logo.png')
if os.path.exists(logo_path):
    logo_image = Image.open(logo_path)
    st.sidebar.image(logo_image, use_container_width=True)
else:
    st.sidebar.title("❤️ CardioVital")

st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("Acerca del Proyecto")
st.sidebar.write(
    "**CardioVital** es una plataforma interactiva que permite estimar "
    "la probabilidad de enfermedades cardíacas basándose en perfiles médicos."
)

st.sidebar.info(
    "**Modelo de Inteligencia Artificial:**\n"
    "- Algoritmo: K-Nearest Neighbors (KNN)\n"
    "- Vecinos analizados: 1 vecino directo para el diagnóstico y 5 vecinos más similares "
    "para calcular el índice de riesgo de proximidad.\n"
    "- Precisión histórica: ~98% en conjunto de prueba."
)

st.sidebar.warning(
    "⚠️ **Nota importante:** Este sistema es una herramienta de apoyo académico "
    "y de concienciación. No sustituye un diagnóstico médico profesional ni una prueba clínica formal."
)

# ----------------- CUERPO PRINCIPAL -----------------
# Encabezado principal
st.title("CardioVital: Predicción Inteligente de Riesgo Cardíaco")
st.write(
    "Ingrese los datos del perfil médico a continuación. El modelo procesará, "
    "normalizará las variables en tiempo real y evaluará la presencia de anomalías cardíacas."
)

# Separación en Pestañas para organizar el formulario
tab_datos, tab_instrucciones = st.tabs(["📝 Formulario de Entrada", "📖 Guía de Variables Médicas"])

with tab_datos:
    # Formulario dividido en 3 columnas lógicas
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Datos Básicos")
        
        age = st.slider("Edad (Años)", min_value=18, max_value=100, value=45, step=1, help="Edad biológica del paciente.")
        
        sex_lbl = st.selectbox("Sexo", ["Masculino", "Femenino"], help="Sexo de nacimiento del paciente.")
        sex = 1 if sex_lbl == "Masculino" else 0
        
        cp_lbl = st.selectbox(
            "Tipo de Dolor de Pecho",
            [
                "Angina Típica",
                "Angina Atípica",
                "Dolor No Anginoso",
                "Asintomático"
            ],
            help="Dolor torácico descrito clínicamente."
        )
        # Mapear a códigos del dataset
        cp_map = {
            "Angina Típica": 0,
            "Angina Atípica": 1,
            "Dolor No Anginoso": 2,
            "Asintomático": 3
        }
        cp = cp_map[cp_lbl]

        exang_lbl = st.selectbox("Angina inducida por Ejercicio", ["No", "Sí"], help="¿El dolor en el pecho aparece al realizar esfuerzo físico?")
        exang = 1 if exang_lbl == "Sí" else 0

    with col2:
        st.subheader("🩺 Parámetros Fisiológicos")
        
        trestbps = st.slider(
            "Presión Arterial en Reposo (mmHg)", 
            min_value=90, 
            max_value=200, 
            value=120, 
            step=1,
            help="Presión arterial tomada al ingresar en reposo."
        )
        
        chol = st.slider(
            "Colesterol Sérico (mg/dl)", 
            min_value=100, 
            max_value=600, 
            value=200, 
            step=1,
            help="Colesterol total en sangre."
        )
        
        fbs_lbl = st.selectbox("Azúcar en Sangre en Ayunas (> 120 mg/dl)", ["Falso (Normal)", "Verdadero (Elevado)"], help="Glucosa tras ayuno.")
        fbs = 1 if fbs_lbl == "Verdadero (Elevado)" else 0

    with col3:
        st.subheader("⚡ Electrocardiograma y Esfuerzo")
        
        restecg_lbl = st.selectbox(
            "Resultados ECG en Reposo",
            [
                "Normal",
                "Anormalidad de onda ST-T",
                "Hipertrofia Ventricular Izquierda"
            ],
            help="Electrocardiograma de reposo."
        )
        restecg_map = {
            "Normal": 0,
            "Anormalidad de onda ST-T": 1,
            "Hipertrofia Ventricular Izquierda": 2
        }
        restecg = restecg_map[restecg_lbl]

        thalach = st.slider(
            "Frecuencia Cardíaca Máxima (bpm)", 
            min_value=60, 
            max_value=220, 
            value=150, 
            step=1,
            help="Frecuencia cardíaca más alta lograda durante la prueba de esfuerzo."
        )
        
        oldpeak = st.slider(
            "Depresión del ST por Ejercicio (mm)", 
            min_value=0.0, 
            max_value=6.2, 
            value=0.0, 
            step=0.1,
            help="Depresión del ST inducida por esfuerzo relativo a reposo."
        )

    # Variables clínicas avanzadas
    st.markdown("---")
    st.subheader("🔬 Detalles Clínicos Avanzados")
    col4, col5, col6 = st.columns(3)

    with col4:
        slope_lbl = st.selectbox(
            "Pendiente del Segmento ST",
            [
                "Pendiente Ascendente",
                "Plana",
                "Pendiente Descendente"
            ],
            help="Pendiente en ejercicio máximo."
        )
        slope_map = {
            "Pendiente Ascendente": 0,
            "Plana": 1,
            "Pendiente Descendente": 2
        }
        slope = slope_map[slope_lbl]

    with col5:
        ca = st.slider("Vasos Principales Coloreados (Fluoroscopia)", min_value=0, max_value=4, value=0, step=1, help="Número de arterias principales obstruidas u observadas.")

    with col6:
        thal_lbl = st.selectbox(
            "Talasemia / Perfusión Sanguínea",
            [
                "Normal",
                "Defecto Fijo",
                "Defecto Reversible",
                "No especificado / Nulo"
            ],
            help="Evaluación del flujo sanguíneo cardíaco."
        )
        thal_map = {
            "No especificado / Nulo": 0,
            "Defecto Fijo": 1,
            "Normal": 2,
            "Defecto Reversible": 3
        }
        thal = thal_map[thal_lbl]

    # Recopilar datos en un diccionario
    user_inputs = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # Botón para activar predicción
    st.write("")
    btn_predict = st.button("📊 Realizar Predicción")

    if btn_predict:
        # Cargar el modelo para validar que esté en orden antes de correr el preprocesamiento
        try:
            load_model()
        except Exception as e:
            st.error(f"Error crítico al cargar el modelo predictivo: {e}")
            st.stop()

        with st.spinner("Procesando y normalizando los datos médicos..."):
            # Preprocesar datos
            X_processed = preprocess_input(user_inputs)
            
            # Ejecutar modelo
            results = predict_heart_disease(X_processed)
            
            # Generar explicaciones médicas
            medical_eval = get_health_explanation(results, user_inputs)
            
        # ----------------- MOSTRAR RESULTADOS -----------------
        st.markdown("---")
        st.header("⚡ Resultados del Diagnóstico Inteligente")
        
        # Panel de visualización de predicción
        res_col1, res_col2 = st.columns([1.5, 2])
        
        with res_col1:
            st.subheader("Estado Clínico Estimado")
            if results['prediction'] == 1:
                st.markdown(f"""
                    <div class="risk-high">
                        <h3 style="margin-top:0; color:#4A0E17;">⚠️ {medical_eval['status_title']}</h3>
                        <p>{medical_eval['status_desc']}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="risk-low">
                        <h3 style="margin-top:0; color:#1B5E20;">✅ {medical_eval['status_title']}</h3>
                        <p>{medical_eval['status_desc']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            # Métricas rápidas
            met_col1, met_col2 = st.columns(2)
            with met_col1:
                pred_label = "Positivo (Riesgo)" if results['prediction'] == 1 else "Negativo (Saludable)"
                st.metric("Predicción del Modelo", pred_label)
            with met_col2:
                # El índice de riesgo de vecinos (proporción de los 5 vecinos más cercanos)
                risk_pct = f"{results['neighbor_risk_index'] * 100:.0f}%"
                st.metric("Similitud de Riesgo", risk_pct)

        with res_col2:
            st.subheader("Probabilidad por Similitud de Perfiles")
            gauge_chart = create_risk_gauge(results['neighbor_risk_index'])
            st.plotly_chart(gauge_chart, use_container_width=True)
            st.caption(
                "El medidor ilustra el riesgo estimado basándose en la coincidencia con los 5 vecinos médicos más similares "
                "en la base de entrenamiento del modelo (algoritmo KNN)."
            )

        # Gráfico comparativo e indicadores adicionales
        st.markdown("---")
        ind_col1, ind_col2 = st.columns(2)
        
        with ind_col1:
            st.subheader("Factores de Alerta y Análisis")
            if medical_eval['warnings']:
                for warn in medical_eval['warnings']:
                    st.warning(f"• {warn}")
            else:
                st.success("🎉 Todos los indicadores fisiológicos principales se encuentran en rangos saludables de referencia.")
                
            st.subheader("🩺 Recomendaciones de Prevención")
            for rec in medical_eval['recommendations']:
                st.write(f"👉 {rec}")
                
        with ind_col2:
            st.subheader("Comparativa de Factores Clave vs Referencia")
            comp_chart = create_comparison_chart(user_inputs)
            st.plotly_chart(comp_chart, use_container_width=True)
            st.caption("Los indicadores ideales de salud son: Presión arterial < 120, Colesterol < 200 y Frecuencia Cardíaca Máxima (220 - edad).")

with tab_instrucciones:
    st.subheader("📖 Guía de Interpretación de Variables")
    st.write(
        "Para ingresar datos precisos, revise las siguientes equivalencias médicas "
        "de las características requeridas por el modelo:"
    )
    
    st.markdown("""
    * **cp (Tipo de Dolor de Pecho):**
        * *Typical Angina (0):* Dolor de pecho opresivo causado por una baja de flujo sanguíneo al corazón.
        * *Atypical Angina (1):* Dolor de pecho agudo pero no relacionado con el corazón.
        * *Non-anginal pain (2):* Dolor punzante o quemazón típicamente esofágico.
        * *Asymptomatic (3):* Paciente sin molestias físicas, pero con signos de riesgo subclínico.
    * **trestbps (Presión Arterial):** Presión medida en el brazo en mm Hg. Valores >= 140 indican hipertensión.
    * **chol (Colesterol):** Nivel de colesterol total en sangre en mg/dl. Lo ideal es < 200 mg/dl.
    * **fbs (Azúcar en sangre):** Clasifica si el azúcar en sangre al despertar en ayunas supera los 120 mg/dl.
    * **restecg (Resultados del ECG):**
        * *Normal (0):* Ondas eléctricas correctas.
        * *ST-T Abnormality (1):* Alteraciones de onda ST-T que sugieren isquemia.
        * *Hipertrofia (2):* Crecimiento anormal del músculo cardíaco.
    * **thalach (Frecuencia Máxima):** Mayor número de pulsaciones del corazón alcanzadas durante ejercicio máximo.
    * **exang (Angina inducida):** Dolor torácico que aparece directamente con el esfuerzo físico.
    * **oldpeak & slope (Depresión del ST):** Mide la falta de oxígeno en el músculo cardíaco al realizar esfuerzo.
    * **ca (Vasos principales):** Número de arterias principales obstruidas (de 0 a 4). A mayor número, mayor riesgo.
    * **thal (Talasemia):** Trastorno de la hemoglobina evaluado en pruebas de esfuerzo.
    """)

# Pie de página (Footer)
st.markdown("---")
st.markdown("""
    <div class="footer">
        Desarrollado con ❤️ para la salud y la prevención cardiovascular<br>
        © 2026 | CardioVital App | <a href="https://github.com" target="_blank">Visitar repositorio en GitHub</a>
    </div>
""", unsafe_allow_html=True)