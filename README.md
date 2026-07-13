# 🩺 CardioVital - Plataforma de Predicción de Riesgo Cardíaco

CardioVital es una aplicación web médica profesional e interactiva diseñada para predecir la probabilidad de presencia de enfermedades cardíacas en pacientes a partir de 13 parámetros clínicos clave. Construida sobre un modelo predictivo previamente entrenado y optimizado utilizando el algoritmo K-Nearest Neighbors (KNN), esta aplicación normaliza las entradas de usuario en tiempo real y ofrece análisis avanzados de riesgo por proximidad y recomendaciones preventivas.

---

## 🚀 Características Clave

* **Interfaz de Usuario de Alta Calidad:** Diseño moderno, limpio y responsivo con una paleta de colores corporativos enfocada en la salud (azul marino profundo, rojo de alerta y verde saludable).
* **Organización Modular de Datos:** Formulario estructurado en pestañas y columnas lógicas para evitar fatiga de scroll y facilitar el ingreso de parámetros.
* **Normalización en Tiempo Real:** Detección y aplicación automática del StandardScaler de entrenamiento (`mean_std_values.pkl`). Si no está presente, la aplicación funciona de forma adaptativa con los datos crudos.
* **Índice de Riesgo por Coincidencia (Propuesta Premium):** Dado que el modelo oficial utiliza $K=1$ vecinos para la clasificación binaria (100% de confianza), CardioVital consulta adicionalmente los **5 vecinos más cercanos** de la base de datos de entrenamiento para generar una probabilidad continua ("Índice de Similitud de Riesgo").
* **Visualizaciones Dinámicas con Plotly:**
  * Indicador de aguja (Gauge) interactivo que ilustra el nivel de riesgo en porcentaje.
  * Gráfico de barras agrupado que contrasta los indicadores clínicos del usuario (colesterol, presión y frecuencia cardíaca máxima) contra los límites normales clínicamente saludables de referencia.
* **Explicaciones Médicas Inteligentes:** Generación de mensajes y advertencias según los umbrales clínicos del paciente y recomendaciones basadas en pautas de prevención cardiovascular.
* **Lista para Producción:** Configuración completa y optimizada para despliegue inmediato en la plataforma Render.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **Interfaz de Usuario:** [Streamlit](https://streamlit.io/) (Framework de desarrollo ágil de aplicaciones de datos).
* **Visualización:** [Plotly](https://plotly.com/) (Gráficos interactivos de alta calidad).
* **Procesamiento de Datos:** [Pandas](https://pandas.pydata.org/) y [NumPy](https://numpy.org/).
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (KNN Classifier y StandardScaler).
* **Manipulación de Imágenes:** [Pillow](https://python-pillow.org/) (Carga de recursos visuales y logos).

---

## 📁 Estructura del Proyecto

La estructura del repositorio sigue las mejores prácticas de desarrollo de software modular:

```text
Prediccion_de_enfermedades-_cardiacas/
│
├── app.py                  # Punto de entrada de la aplicación Streamlit
├── requirements.txt        # Dependencias de Python para producción
├── runtime.txt             # Versión de Python especificada para Render
├── render.yaml             # Configuración de infraestructura como código para Render
├── README.md               # Este documento explicativo
├── .gitignore              # Archivos y cachés ignorados por Git
│
├── models/                 # Directorio de modelos de Machine Learning
│   ├── model.pkl           # Modelo clasificador KNN entrenado (K=1)
│   └── mean_std_values.pkl # Diccionario con medias y desviaciones del StandardScaler
│
├── utils/                  # Módulos de lógica auxiliar y de negocio
│   ├── predictor.py        # Carga del modelo y cálculo del índice de riesgo
│   ├── preprocess.py       # Lectura de medias/std y escalado de entradas del usuario
│   └── helpers.py          # Diccionarios de etiquetas clínicas, alertas y gráficos Plotly
│
├── assets/                 # Recursos gráficos de la aplicación
│   └── logo.png            # Logotipo oficial de CardioVital
│
└── notebooks/              # Cuadernos Jupyter históricos de entrenamiento y análisis
    ├── data-exploration.ipynb
    └── model-train.ipynb
```

---

## 💻 Instalación y Ejecución Local

Siga los siguientes pasos para clonar y ejecutar el proyecto en su entorno local:

### 1. Requisitos Previos
Asegúrese de tener instalado Python (versión 3.9 o superior) y Git en su computadora.

### 2. Clonar el Repositorio y Crear un Entorno Virtual
```bash
# Clonar el proyecto
git clone <url_de_tu_repositorio>
cd Prediccion_de_enfermedades-_cardiacas

# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 3. Instalar Dependencias
Instale las librerías necesarias con el archivo `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación Localmente
Inicie el servidor de desarrollo de Streamlit:
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en su navegador web predeterminado en `http://localhost:8501`.

---

## ☁️ Despliegue en Render

Este proyecto está configurado para desplegarse automáticamente en [Render](https://render.com/) mediante **Infraestructura como Código** usando el archivo `render.yaml`.

### Pasos para el Despliegue:

1. **Subir el código a GitHub:**
   Cree un repositorio en GitHub, inicialice Git localmente, haga commit de sus cambios y suba el código:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: CardioVital App ready for deploy"
   git branch -M main
   git remote add origin <url_de_tu_repositorio_en_github>
   git push -u origin main
   ```
2. **Crear una cuenta en Render:**
   Regístrese de manera gratuita en [render.com](https://render.com/) y conecte su cuenta de GitHub.
3. **Desplegar usando Blueprint:**
   * En el panel de Render, haga clic en el botón **New +** y elija **Blueprint**.
   * Seleccione su repositorio de GitHub que contiene este proyecto.
   * Render leerá automáticamente el archivo `render.yaml` y configurará el servicio web con la versión correcta de Python (`3.10.12`), el comando de construcción (`pip install -r requirements.txt`) y el puerto dinámico de Streamlit.
   * Haga clic en **Approve** y espere a que finalice la construcción.
4. **¡Listo!**
   Una vez completada la construcción, Render le proporcionará una URL pública segura del tipo `https://prediccion-enfermedad-cardiaca.onrender.com`.

---

## 📸 Capturas de Pantalla de la Aplicación

A continuación se muestran ejemplos visuales de la interfaz de usuario de CardioVital:

### 1. Panel de Ingreso de Datos (Formulario Médico)
<!-- [Placeholder: Inserte captura del formulario de entrada con sliders y tabs] -->

### 2. Resultados: Sin Riesgo Cardíaco Directo (Caso Saludable)
<!-- [Placeholder: Inserte captura del medidor en verde y gráfico de barras comparativo de bajo riesgo] -->

### 3. Resultados: Riesgo de Enfermedad Cardíaca Detectado (Caso de Alerta)
<!-- [Placeholder: Inserte captura del medidor en rojo, alertas rojas y recomendaciones médicas activadas] -->

---

## 👥 Autores y Colaboraciones
* **Autor:** [Tu Nombre / Desarrollador]
* **Contacto:** [Tu Email / Redes]
* **Año:** 2026

*Proyecto de carácter educativo y preventivo desarrollado para incentivar el uso de Machine Learning en salud.*
