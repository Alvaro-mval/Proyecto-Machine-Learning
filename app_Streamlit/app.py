# %%
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import random
import base64

# %%
st.set_page_config(
    page_title="Towards The Glory",
    page_icon="🏅",
    layout="wide"
)

# %%
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        div[data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Llamada:
set_background("C:/Users/Alvar/Documents/GitHub/Towars_the_glory_ML_project/app_Streamlit/imagen.jpg")


# %%
#Música de fondo
# -----------------------
MUSIC_PATH = r"C:\Users\Alvar\Documents\GitHub\Towars_the_glory_ML_project\app_Streamlit\musica_jjoo.mp3" 
st.sidebar.header("🎵 Música de fondo")
try:
    with open(MUSIC_PATH, "rb") as f:
        audio_bytes = f.read()
    st.sidebar.audio(audio_bytes, format="audio/mp3")
except FileNotFoundError:
    st.sidebar.info("Archivo de música no encontrado. Cambia MUSIC_PATH o sube un archivo.")

# %%
st.title("🏆 Towards The Glory")
st.subheader("🔍 Predicción de Medallas Olímpicas mediante Machine Learning")

st.markdown("""
Este sistema permite:
- Predecir si un atleta ganará medalla (modelo **binario**)
- Predecir qué medalla ganará (modelo **multiclase**)
- Comparar resultados reales de Tokio 2020 o Paris 2024 con nuestras predicciones
""")

# %%
st.sidebar.header("⚙️ Selección de Modelo")

tipo_modelo = st.sidebar.radio(
    "Elige el tipo de modelo:",
    ["Multiclase", "Binario"]
)

# Nombre del archivo según el modelo
modelo_archivo = {
    "Multiclase": r"C:\Users\Alvar\Documents\GitHub\Towars_the_glory_ML_project\Models\final_model_ML_multiclase.pkl",
    "Binario": r"C:\Users\Alvar\Documents\GitHub\Towars_the_glory_ML_project\Models\final_model_ML_binario.pkl"
}

# %%
@st.cache_resource
def cargar_modelo(path):
    with open(path, "rb") as f:
        return pickle.load(f)

modelo = cargar_modelo(modelo_archivo[tipo_modelo])

st.success(f"Modelo cargado: **{modelo_archivo[tipo_modelo]}**")

# %%
# OPCIÓN 1 — INPUT MANUAL
# -------------------------------------------------------------
st.header("📝 Predicción Individual")
st.markdown("Introduce los datos del atleta:")

# Input del usuario
name = st.text_input("Nombre del atleta")
sex = st.selectbox("Sexo", ["M", "F"])
age = st.number_input("Edad", min_value=12, max_value=65, value=25)
height = st.number_input("Altura (cm)", min_value=100, max_value=230, value=175)
weight = st.number_input("Peso (kg)", min_value=25, max_value=230, value=70)

# Selección de país
team_list = [
    "Germany","United States","Russia","Canada","Italy","France","United Kingdom",
    "Japan","Australia","China","Spain","Poland","Sweden","South Korea","Czech Republic",
    "Hungary","Switzerland","Netherlands","Romania","Brazil","Norway","Austria","Ukraine",
    "Bulgaria","Finland","Argentina","New Zealand","Serbia","Cuba","Belarus","Greece",
    "Mexico","Denmark","Kazakhstan","Belgium","Slovenia","Slovakia","South Africa","Egypt",
    "Portugal","Ireland","Croatia","Colombia","Turkey","India","Latvia","North Korea",
    "Nigeria","Taiwan","Estonia","Jamaica","Venezuela","Puerto Rico","Lithuania","Kenya",
    "Hong Kong","Thailand","Iran","Algeria","Israel","Morocco","Uzbekistan","Tunisia",
    "Chile","Iceland","Mongolia","Malaysia","Senegal","Indonesia","Virgin Islands","Philippines",
    "Guatemala","Peru","Bahamas","Trinidad and Tobago","Cameroon","Ethiopia","Georgia",
    "Azerbaijan","Kuwait","Singapore","Pakistan","Liechtenstein","Angola","Zimbabwe",
    "Ghana","Moldova","Costa Rica","Ecuador","Dominican Republic","Saudi Arabia","Lebanon",
    "Kyrgyzstan","Cyprus","Fiji","Armenia","Barbados","Uganda","Uruguay","Qatar","Congo",
    "Honduras","Ivory Coast","Andorra","Iraq","San Marino","Zambia","Vietnam","Syria",
    "Tanzania","Mauritius","United Arab Emirates","Bolivia","Luxembourg","Bermuda",
    "Bosnia and Herzegovina","Paraguay","Antigua and Barbuda","Monaco","Bahrain","Guam",
    "Seychelles","Sri Lanka","El Salvador","Sierra Leone","Madagascar","Papua New Guinea",
    "Samoa","Panama","Montenegro","Nicaragua","Botswana","Mali","Malawi","Nepal","Sudan",
    "North Macedonia","Malta","Libya","Cayman Islands","Jordan","Haiti","Mozambique",
    "Namibia","Guyana","Benin","Albania","Tajikistan","Belize","Suriname","Eswatini",
    "Lesotho","Central African Republic","Oman","Gabon","Myanmar","Togo","Guinea","Rwanda",
    "Gambia","Liberia","Bangladesh","Grenada","Laos","Turkmenistan","Curaçao","Maldives",
    "Yemen","Cambodia","Tonga","Afghanistan","Niger","Burkina Faso","Eritrea","Aruba",
    "Saint Vincent and the Grenadines","Saint Kitts and Nevis","Equatorial Guinea","Cook Islands",
    "Burundi","Djibouti","Bhutan","Chad","Vanuatu","Somalia","Mauritania","Saint Lucia",
    "Solomon Islands","Micronesia","Palau","Palestine","Guinea-Bissau","Dominica","Comoros",
    "Cape Verde","Sao Tome and Principe","Marshall Islands","Nauru","Refugees","Kiribati",
    "Brunei","East Timor","Kosovo","Tuvalu","South Sudan","USA"
]
team = st.selectbox("Equipo / País", team_list)

sport_list = [
    "Athletics","Swimming","Gymnastics","Cross Country Skiing","Rowing","Cycling",
    "Alpine Skiing","Shooting","Canoeing","Fencing","Biathlon","Sailing","Wrestling",
    "Football","Ice Hockey","Speed Skating","Boxing","Equestrianism","Judo","Hockey",
    "Handball","Volleyball","Basketball","Weightlifting","Water Polo","Bobsleigh","Archery",
    "Tennis","Table Tennis","Ski Jumping","Diving","Figure Skating","Short Track Speed Skating",
    "Badminton","Luge","Nordic Combined","Modern Pentathlon","Freestyle Skiing","Snowboarding",
    "Synchronized Swimming","Rhythmic Gymnastics","Taekwondo","Beach Volleyball","Triathlon",
    "Curling","Rugby Sevens","Skeleton","Trampolining","Golf"
]
sport = st.selectbox("Deporte", sport_list)

city = st.text_input("Ciudad sede (opcional)", value="Desconocida")
year = st.number_input("Año de la Olimpiada", max_value=2060, value=2024, step=2)
Games = f"Olympics {year}"

# Generar automáticamente columnas derivadas
ID = 0  # ID ficticio, no afecta el modelo

# Crear dataframe de entrada solo con las 9 columnas que espera el modelo
input_df = pd.DataFrame([{
    "ID": ID,
    "Sex": sex,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "Team": team,
    "Games": Games,
    "City": city,
    "Sport": sport
}])


# %%
 # Predicción
st.write("✅ Datos listos para predicción:")
st.dataframe(input_df)

# Botón de predicción
if st.button("Calcular Predicción"):
    # Predecir con el modelo cargado
    pred = modelo.predict(input_df)
    prob = modelo.predict_proba(input_df)
    
    st.subheader("📌 Resultado de la predicción")
    st.write(f"**Predicción:** {pred[0]}")
    
    st.subheader("📊 Probabilidades por clase")
    # Mostrar un DataFrame con columnas como las clases y probabilidades
    prob_df = pd.DataFrame(prob, columns=modelo.classes_)
    st.dataframe(prob_df)

# %%
# OPCIÓN 2 — CARGAR CSV Y COMPARAR VS TOKYO 2020
# -------------------------------------------------------------
st.header("📊 Comparativa vs próximos JJOO (CSV)")

features_modelo = ["ID", "Sex", "Age", "Height", "Weight", "Team", "Games", "City", "Sport"]

archivo = st.file_uploader(
    "Sube el archivo CSV con datos de Tokio 2020, Paris 2024 o JJOO futuros"
)

if archivo is not None:

    # -------------------------
    # Cargar CSV original
    # -------------------------
    df_original = pd.read_csv(archivo)

    # Copia completa para mostrar y para JSON
    df_visual = df_original.copy()

    # -------------------------
    # Validar columnas requeridas
    # -------------------------
    columnas_requeridas = features_modelo + ["Medal"]

    if not all(col in df_original.columns for col in columnas_requeridas):
        st.error(f"El CSV debe contener estas columnas: {columnas_requeridas}")

    else:
        # -------------------------
        # Mapeo de nombres → números (solo interno)
        # -------------------------
        unique_ids = df_original["ID"].unique()
        mapping = {name: i + 1 for i, name in enumerate(unique_ids)}

        df_model = df_original.copy()
        df_model["ID"] = df_model["ID"].map(mapping)

        # -------------------------
        # Preparar datos para el modelo
        # -------------------------
        X = df_model[features_modelo]

        # -------------------------
        # Predicciones del modelo
        # -------------------------
        preds = modelo.predict(X)

        # -------------------------
        # Añadir predicción y coincidencia al df visual (con nombres)
        # -------------------------
        df_visual["Predicted_Medal"] = preds
        df_visual["Coincide"] = df_visual["Predicted_Medal"] == df_visual["Medal"]

        # -------------------------
        # Mostrar muestra de 20 atletas (con nombres reales)
        # -------------------------
        st.write("### 🧪 Muestras de 20 atletas")
        st.dataframe(df_visual.sample(20))

        # -------------------------
        # Convertir a JSON usando el df visual (no el del modelo)
        # -------------------------
        comparativa_json = df_visual.to_dict(orient="records")
        json_str = json.dumps(comparativa_json, indent=4)

        # -------------------------
        # Botón para descargar JSON final
        # -------------------------
        st.download_button(
            label="📥 Descargar Comparativa en JSON",
            data=json_str,
            file_name="comparativa_tokyo2020.json",
            mime="application/json"
        )


# %%



