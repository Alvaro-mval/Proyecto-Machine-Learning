# 🏅 **TOWARDS THE GLORY**
## 🔍 Predicción de Medallas Olímpicas mediante Machine Learning  

---

# 📌 **Descripción del Proyecto**

**Towards the Glory** es un proyecto de *Machine Learning* cuyo objetivo es **predecir el rendimiento olímpico de atletas y países**.  
A partir de datos históricos de **www.sports-reference.com**, se desarrollan modelos capaces de:

- **Predecir si un atleta ganará una medalla** (clasificación binaria).  
- **Predecir qué tipo de medalla ganará**: Oro, Plata, Bronce o Ninguna (multiclase).  
- **Predecir cuántas medallas puede obtener un país** en futuras olimpiadas.

El proyecto incluye **limpieza de datos**, **EDA univariante y bivariante**, **modelos baseline**, **optimización de modelos**, uso de **pipelines avanzados**, y una aplicación final en **Streamlit**.

---

# 📂 **Dataset**

El dataset contiene variables demográficas, físicas y deportivas de atletas olímpicos.

## **Columnas principales**

| Columna  | Descripción |
|----------|-------------|
| `ID` | Identificador del atleta |
| `Name` | Nombre |
| `Sex` | Género |
| `Age` | Edad |
| `Height` | Altura |
| `Weight` | Peso |
| `Team` | Equipo |
| `NOC` | Comité Olímpico Nacional |
| `Games` | Edición |
| `Year` | Año |
| `Season` | Summer/Winter |
| `City` | Sede |
| `Sport` | Deporte |
| `Event` | Evento |
| `Medal` | Oro / Plata / Bronce / NA |

---

# 🧹 **Limpieza de Datos (Data Cleaning)**

Incluye:

- **Eliminación de duplicados**  
- **Corrección e imputación de valores faltantes**  
- **Normalización de categorías**  
- **Reformulación del target** en:
  - **Multiclase:** Oro / Plata / Bronce / Ninguna  
  - **Binario:** Medalla / No Medalla  
- **Ingeniería de características** cuando es relevante  

---

# 📊 **Exploratory Data Analysis (EDA)**

## **EDA Univariante**
- Distribuciones de edad, altura y peso  
- Número de atletas por país  
- Frecuencia de medallistas  

## **EDA Bivariante**
- Relación edad ↔ medalla  
- Correlación entre características físicas  
- Comparación de países y disciplinas  
- Evolución histórica del rendimiento  

---

# 🤖 **Modelos de Machine Learning**

Todos los modelos usan un **Pipeline** con:

- **PCA** (reducción de dimensionalidad)  
- **Scaler** (StandardScaler / MinMaxScaler)  
- **SelectKBest** (selección de características)  
- **Clasificador**

## **Modelos evaluados**
1. **Logistic Regression**  
2. **Random Forest**  
3. **XGBoost**  
4. **Gradient Boosting (GBC)**  
5. **Decision Tree**  
6. **AdaBoost**

---

# ⚖️ **Balanceo de Clases**

Para combatir el desbalance del dataset se probaron combinaciones de:

- **SMOTE**  
- **Random Undersampling**  
- **SMOTE + Undersampling**  
- **SMOTETomek**  
- Modelos sin reequilibrio  

Objetivo: **identificar la mejor estrategia para binario y multiclase**.

---

# 🔍 **Optimización de Hiperparámetros**

Se realizaron **tres fases de optimización**:

## **1️⃣ GridSearch inicial**
- **219 combinaciones**  
- Prueba de los 6 modelos  
- Métricas: **Accuracy**, **F1-macro**, **F1-weighted**

## **2️⃣ Optimización fina**
- Enfocada solo en el mejor modelo anterior  
- Ajuste de hiperparámetros específicos

## **3️⃣ Optimización final**
- Rango reducido y especializado  
- Búsqueda del *score* máximo posible

---

# 🏆 **Modelo Final**

Se obtiene un modelo con alto rendimiento para:

- **Clasificación multiclase** (tipo de medalla)  
- **Clasificación binaria** (medalla/no medalla)  

Además, se realiza un **benchmark con los datos reales de Tokio 2020** para medir la capacidad predictiva del modelo.

---

# 🌐 **Aplicación Web – Streamlit**

La app desarrollada permite:

### 🔹 **Predicciones individuales**
El usuario introduce datos como:
- Edad  
- Altura  
- Peso  
- País  
- Evento  
- Deporte  

### 🔹 **Comparación con Tokio 2020**
- Predicciones vs. resultados reales  
- Análisis detallado por país  

### 🔹 **Visualizaciones interactivas**
- Importancia de variables  
- Matriz de confusión  
- Distribuciones  
- PCA en 2D y 3D  

---

# 📈 **Resultados Destacados**

- **Mejoras significativas en F1-macro** respecto al baseline  
- Buen rendimiento para **predicción multiclase**  
- Predicciones razonablemente precisas cuando se comparan con **Tokio 2020**  
- Identificación de países con sobre/infraestimación de medallas  

---

# ⚙️ **Tecnologías Utilizadas**

- **Python 3**  
- **Pandas / NumPy**  
- **Scikit-Learn**  
- **Imbalanced-Learn**  
- **XGBoost**  
- **Streamlit**  
- **Matplotlib / Seaborn / Plotly**

---

# ▶ **Uso del Proyecto**

## **Instalar dependencias**
pip install -r requirements.txt

## **Ejecutar la aplicación Streamlit**
streamlit run streamlit_app/app.py

# 🙌 **Autores**

Proyecto realizado como iniciativa de análisis predictivo aplicado al rendimiento olímpico por Álvaro Martinez en el marco de un proyecto de Machine Learning del bootcamp Data Science & IA impartido por The Bridge. 