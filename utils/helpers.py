import plotly.graph_objects as go
import pandas as pd
import numpy as np

def get_chest_pain_label(code):
    """Retorna la etiqueta médica en español para el tipo de dolor de pecho."""
    labels = {
        0: "Angina Típica (Dolor opresivo por falta de flujo sanguíneo)",
        1: "Angina Atípica (Dolor torácico no opresivo)",
        2: "Dolor No Anginoso (Dolor de pecho no relacionado al corazón, ej. esofágico)",
        3: "Asintomático (Sin dolor de pecho pero con otros signos clínicos)"
    }
    return labels.get(int(code), "Desconocido")

def get_restecg_label(code):
    """Retorna la etiqueta médica en español para los resultados del electrocardiograma."""
    labels = {
        0: "Normal",
        1: "Anormalidad de onda ST-T (Inversión de onda T y/o alteración leve)",
        2: "Hipertrofia Ventricular Izquierda (Engrosamiento de las paredes del corazón)"
    }
    return labels.get(int(code), "Desconocido")

def get_slope_label(code):
    """Retorna la etiqueta para la pendiente del segmento ST durante ejercicio máximo."""
    labels = {
        0: "Pendiente Ascendente (Upsloping - Respuesta fisiológica normal)",
        1: "Plana (Flat - Posible signo de isquemia leve)",
        2: "Pendiente Descendente (Downsloping - Signo clásico de isquemia cardíaca)"
    }
    return labels.get(int(code), "Desconocido")

def get_thal_label(code):
    """Retorna la etiqueta para la prueba de talasemia / perfusión sanguínea."""
    labels = {
        0: "No especificado / Nulo",
        1: "Defecto Fijo (Sin flujo sanguíneo en alguna parte del corazón)",
        2: "Normal (Flujo sanguíneo correcto)",
        3: "Defecto Reversible (Flujo reducido durante el esfuerzo, pero normal en reposo)"
    }
    return labels.get(int(code), "Desconocido")

def get_health_explanation(pred_dict, inputs):
    """
    Genera explicaciones y recomendaciones médicas personalizadas basadas en la predicción 
    y en los valores de entrada ingresados por el usuario.
    """
    prediction = pred_dict['prediction']
    risk_index = pred_dict['neighbor_risk_index']
    
    warnings = []
    recommendations = []
    
    # Evaluar Presión Arterial (trestbps)
    bps = inputs['trestbps']
    if bps >= 140:
        warnings.append(f"Presión Arterial Alta: Tu presión en reposo es de {bps} mm Hg, lo cual se clasifica como hipertensión.")
        recommendations.append("Consulte con un médico sobre estrategias para controlar su presión arterial (dieta baja en sodio, ejercicio regular, o fármacos si es indicado).")
    elif bps >= 120:
        warnings.append(f"Presión Arterial Limítrofe: Tu presión es de {bps} mm Hg (prehipertensión).")
        recommendations.append("Monitoree su presión arterial periódicamente y limite el consumo de sal.")

    # Evaluar Colesterol (chol)
    chol = inputs['chol']
    if chol >= 240:
        warnings.append(f"Colesterol Alto: Tu nivel es de {chol} mg/dl, lo que aumenta significativamente el riesgo cardiovascular.")
        recommendations.append("Reduzca la ingesta de grasas saturadas y trans. Considere aumentar el consumo de fibra soluble y omega-3.")
    elif chol >= 200:
        warnings.append(f"Colesterol Elevado: Tu nivel es de {chol} mg/dl (rango moderado-alto).")
        recommendations.append("Mantenga una dieta equilibrada y realice actividad física cardiovascular de manera regular.")

    # Evaluar Electrocardiograma e Isquemia (oldpeak y cp)
    oldpeak = inputs['oldpeak']
    if oldpeak >= 1.5:
        warnings.append(f"Depresión ST Significativa: Se detectó una depresión del segmento ST de {oldpeak} mm durante el ejercicio, sugiriendo sobreesfuerzo o reducción de oxígeno cardíaco.")
        recommendations.append("Se recomienda una prueba de esfuerzo formal y consulta cardiológica para evaluar la salud coronaria.")

    # Evaluar vasos principales (ca)
    ca = inputs['ca']
    if ca > 0:
        warnings.append(f"Obstrucción Vascular: La fluoroscopia muestra {ca} vasos principales coloreados, lo cual es indicador de placas u obstrucción en arterias coronarias.")
        recommendations.append("Es crucial coordinar un seguimiento con un cardiólogo para evaluar el flujo sanguíneo de las arterias coronarias.")

    # Mensaje principal del diagnóstico
    if prediction == 1:
        status_title = "Riesgo de Enfermedad Cardíaca Detectado"
        status_color = "#E63946" # Rojo de alerta
        if risk_index >= 0.8:
            status_desc = "Atención: Tu perfil médico coincide fuertemente con el grupo de pacientes con diagnóstico confirmado de enfermedad cardíaca. Se sugiere una consulta médica preventiva urgente."
        else:
            status_desc = "El modelo detectó coincidencia con perfiles de riesgo de enfermedad cardíaca. Por favor, consulte con su especialista y considere exámenes preventivos."
    else:
        status_title = "Sin Indicios de Enfermedad Cardíaca Directa"
        status_color = "#2A9D8F" # Verde saludable
        if risk_index <= 0.2:
            status_desc = "Excelente: Tu perfil es muy similar al grupo de pacientes saludables. Continúa manteniendo hábitos de vida saludables."
        else:
            status_desc = "El modelo indica bajo riesgo de enfermedad cardíaca directa. Sin embargo, se observan algunos factores de riesgo aislados que valdría la pena vigilar."
            
    # Recomendaciones por defecto si la lista está vacía
    if not recommendations:
        recommendations.append("Mantenga una dieta rica en frutas, verduras y granos enteros.")
        recommendations.append("Realice al menos 150 minutos de ejercicio moderado a la semana.")
        
    return {
        'status_title': status_title,
        'status_color': status_color,
        'status_desc': status_desc,
        'warnings': warnings,
        'recommendations': recommendations
    }

def create_risk_gauge(risk_value):
    """
    Crea un gráfico de medidor (Gauge) en Plotly para representar visualmente 
    el Índice de Similitud de Riesgo.
    """
    percentage = risk_value * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%", 'font': {'size': 48, 'color': '#2B2D42', 'family': 'Outfit, sans-serif'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2B2D42", 'tickfont': {'size': 14}},
            'bar': {'color': "#1D3557", 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "#A8DADC",
            'steps': [
                {'range': [0, 35], 'color': '#D8F3DC'},   # Verde pastel
                {'range': [35, 70], 'color': '#FFE5EC'},  # Amarillo/naranja suave
                {'range': [70, 100], 'color': '#FAD2E1'}  # Rojo/rosa suave
            ],
            'threshold': {
                'line': {'color': "#E63946", 'width': 4},
                'thickness': 0.75,
                'value': percentage
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=50, b=30),
        height=250,
        font={'family': "Outfit, Inter, sans-serif"}
    )
    
    return fig

def create_comparison_chart(inputs):
    """
    Crea un gráfico de barras comparativo de indicadores clave
    del usuario (presión arterial, colesterol, frecuencia cardíaca máxima)
    con respecto a los umbrales clínicos saludables de referencia.
    """
    categories = ['Presión Arterial<br>(mmHg)', 'Colesterol<br>(mg/dl)', 'Frecuencia Máx.<br>(bpm)']
    user_values = [inputs['trestbps'], inputs['chol'], inputs['thalach']]
    
    # Límites superiores normales de referencia clínica
    reference_values = [120, 200, 220 - inputs['age']] 
    
    fig = go.Figure()
    
    # Barras del usuario
    fig.add_trace(go.Bar(
        x=categories,
        y=user_values,
        name='Tus Indicadores',
        marker_color='#E63946', # Rojo
        opacity=0.85,
        text=user_values,
        textposition='auto',
        hoverinfo='y'
    ))
    
    # Barras de referencia clínica
    fig.add_trace(go.Bar(
        x=categories,
        y=reference_values,
        name='Límite de Referencia',
        marker_color='#457B9D', # Azul
        opacity=0.5,
        text=reference_values,
        textposition='auto',
        hoverinfo='y'
    ))
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(gridcolor='rgba(0,0,0,0)'),
        yaxis=dict(
            title='Valores Clínicos',
            gridcolor='rgba(0,0,0,0.1)'
        ),
        font={'family': "Outfit, Inter, sans-serif"}
    )
    
    return fig
