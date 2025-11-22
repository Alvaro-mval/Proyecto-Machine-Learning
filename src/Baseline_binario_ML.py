# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, r2_score
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

# %%
# Cargar CSV en un DataFrame
df = pd.read_csv("processed_data/athlete_events_clean_final.csv")

# %%
df

# %%
#En este modelo base vamos a intentar ver como seria nuestro proyecto si unicamente necesitamos predecir clases binarias (medalla/no medalla)

X = df.drop(['Medal', 'Medal_num','Medal_bin'], axis=1)
y = df['Medal_bin']

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
num_cols1 = ["Age","Height","Weight"]

# %%
#dividimos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Guardar train y test en CSV 
train_set = X_train.copy()
train_set['Medal_bin'] = y_train
train_set.to_csv('train/train_set_bin.csv', index=False)

test_set = X_test.copy()
test_set['Medal_bin'] = y_test
test_set.to_csv('test/test_set_bin.csv', index=False)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
#Creamos la matriz de correlación y para ello hacemos un preprocesamiento de datos para convertir variables categoricas con TargetEncoder

corr_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", TargetEncoder(), cat_cols),
        ("num", StandardScaler(), num_cols)
    ],
    remainder='drop'
)

corr_preprocessor.fit(X_train, y_train)
X_train_corr = corr_preprocessor.transform(X_train)

df_corr = pd.DataFrame(
    X_train_corr,
    columns=list(cat_cols) + list(num_cols)
)
df_corr["Medal_bin"] = y_train.values

# %%
corr = df_corr.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matriz de correlación tras TargetEncoder + StandardScaler")
plt.show()

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", TargetEncoder(), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder='drop'
)

# %%
#Como usamos la tecnica de SMOTE despues del preprocessor, necesitamos el modulo ImbPipeline ya que el Pipeline no trata bien el SMOTE.

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf = 10,
        class_weight="balanced",  #para que ponga cariño a las clases minoritarias
        random_state=42,
        n_jobs=-1
    ))
])

# %%
pipeline.fit(X_train, y_train)

# %%
y_pred_train = pipeline.predict(X_train)

y_proba_train = pipeline.predict_proba(X_train)


# %%
# Métricas de evaluación train con SMOTE
accuracy = round(accuracy_score(y_train, y_pred_train),2)
f1 = round(f1_score(y_train, y_pred_train, average='weighted'),2)
precision = round(precision_score(y_train, y_pred_train, average='weighted'),2)
recall = round(recall_score(y_train, y_pred_train, average='weighted'),2)
conf_matrix = confusion_matrix(y_train, y_pred_train)


print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Matriz de Confusión:\n", conf_matrix)
print("\nReporte de clasificación:\n", classification_report(y_train, y_pred_train))
print("\nMatriz de probabilidad:\n",y_proba_train)

#Accuracy y Recall: aumentan significativamente en el binario → más fácil predecir solo medalla/no medalla que diferenciar entre tipos de medalla.
#F1 Score y Precision: también mejoran → menos confusión entre clases minoritarias.
#Interpretación: Reducir la complejidad de la tarea (de 4 clases a 2) ayuda a que SMOTE y el modelo manejen mejor el desbalance y aumenten la efectividad.

# %%
#vamos a intentar comparar con un SMOTE avanzado.

pipeline_1 = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTETomek(random_state=42)), #Aqui remplazamos Smote por SmoteTomek
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf = 10,
        class_weight="balanced",  #para que ponga cariño a las clases minoritarias
        random_state=42,
        n_jobs=-1
    ))
])

# %%
pipeline_1.fit(X_train, y_train)

# %%
y_pred_train_balanced = pipeline_1.predict(X_train)

y_proba_train_balanced = pipeline_1.predict_proba(X_train)

# %%
# Métricas de evaluación train utilizando SMOTETomek
accuracy = round(accuracy_score(y_train, y_pred_train_balanced),2)
f1 = round(f1_score(y_train, y_pred_train_balanced, average='weighted'),2)
precision = round(precision_score(y_train, y_pred_train_balanced, average='weighted'),2)
recall = round(recall_score(y_train, y_pred_train_balanced, average='weighted'),2)
conf_matrix = confusion_matrix(y_train, y_pred_train_balanced)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Matriz de Confusión:\n", conf_matrix)
print("\nReporte de clasificación:\n", classification_report(y_train, y_pred_train_balanced))
print("\nMatriz de probabilidad:\n",y_proba_train_balanced)

#Accuracy y F1 Score se mantienen prácticamente iguales.
#Recall de la clase 1 (medalla) disminuye notablemente con SMOTETomek (0.57 vs 0.74 con SMOTE simple).
#Precision permanece igual → el modelo sigue siendo bueno identificando verdaderos positivos, pero ahora detecta menos atletas con medalla.
#matriz de confusion:
#Clase 0 (sin medalla) → 77% correctos, similar a SMOTE simple.
#Clase 1 (medalla) → 57% correctos, notable caída respecto al 74% de SMOTE simple.
#Error mayor en la clase minoritaria (medalla) → SMOTETomek parece más agresivo con el undersampling de la mayoría, afectando la sensibilidad a positivos.

# %%
#Vamos a implementar tecnicas de undersampling a ver que tal

pipeline_2 = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('undersample', RandomUnderSampler(random_state=42) ),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

# %%
pipeline_2.fit(X_train, y_train)

# %%
y_pred_train_balanced_under = pipeline_2.predict(X_train)

y_proba_train_balanced_under = pipeline_2.predict_proba(X_train)

# %%
# Métricas de evaluación train utilizando UnderSampling
accuracy = round(accuracy_score(y_train, y_pred_train_balanced_under),2)
f1 = round(f1_score(y_train, y_pred_train_balanced_under, average='weighted'),2)
precision = round(precision_score(y_train, y_pred_train_balanced_under, average='weighted'),2)
recall = round(recall_score(y_train, y_pred_train_balanced_under, average='weighted'),2)
conf_matrix = confusion_matrix(y_train, y_pred_train_balanced_under)
r2 = round(r2_score(y_train, y_pred_train_balanced_under),2)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Matriz de Confusión:\n", conf_matrix)
print("\nReporte de clasificación:\n", classification_report(y_train, y_pred_train_balanced_under))
print("\nMatriz de probabilidad:\n",y_proba_train_balanced_under)

#Accuracy es mejor que SMOTE y SMOTETomek  (0.80 vs 0.74).
#F1 Score también mejora, reflejando un mejor balance entre precisión y recall.
#Precision muy alta (0.92) → el modelo identifica correctamente la mayoría de los verdaderos negativos (sin medalla) y evita falsos positivos.
#Recall de 0.80 → captura bien la clase minoritaria (medalla) y mejora respecto a SMOTETomek  (0.57).
#Esto sugiere que undersampling favorece un modelo más balanceado entre clases, reduciendo la dominancia de la clase mayoritaria.
#El modelo sacrifica algunos ejemplos de la clase mayoritaria para mejorar la detección de medallas.

# %%
#Por último, vamos a intentar hacer una combinación de Undersampling y Smote a ver si mejoramos aÚn mas. 

from collections import Counter
counter = Counter(y_train)
n_major = counter[0]

# Definimos proporciones
smote = SMOTE(sampling_strategy={1: int(n_major*0.5)}, random_state=42)


pipeline_3 = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote), 
    ('undersample', RandomUnderSampler(sampling_strategy={0: int(n_major*0.8)}, random_state=42)),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

# %%
pipeline_3.fit(X_train, y_train)

# %%
y_pred_train_mix_balanced = pipeline_3.predict(X_train)

y_proba_train_mix_balanced = pipeline_3.predict_proba(X_train)

# %%
# Métricas de evaluación train utilizando UnderSampling + SMOTE
accuracy = round(accuracy_score(y_train, y_pred_train_mix_balanced),2)
f1 = round(f1_score(y_train, y_pred_train_mix_balanced, average='weighted'),2)
precision = round(precision_score(y_train, y_pred_train_mix_balanced, average='weighted'),2)
recall = round(recall_score(y_train, y_pred_train_mix_balanced, average='weighted'),2)
conf_matrix = confusion_matrix(y_train, y_pred_train_mix_balanced)
r2 = round(r2_score(y_train, y_pred_train_mix_balanced),2)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Matriz de Confusión:\n", conf_matrix)
print("\nReporte de clasificación:\n", classification_report(y_train, y_pred_train_mix_balanced))
print("\nMatriz de probabilidad:\n",y_proba_train_mix_balanced)

#Métricas extremadamente altas en entrenamiento, cercanas a 1. Overfitting??
#F1, Precision y Recall balanceados, indicando que el modelo aprendió a distinguir muy bien ambas clases en el train.
# El dataset se balancea casi perfectamente, por lo que el modelo tiene ejemplos suficientes de la clase minoritaria y no es dominado por la mayoritaria.

#Matriz de confusión:
#Clase 0: 120105 de 123213 correctamente identificados → recall ~0.97.
#Clase 1: 18045 de 19471 correctamente identificados → recall ~0.93.
#La clase minoritaria (medalla) tiene muy buen soporte y apenas se pierden positivos.
#La clase mayoritaria se ve ligeramente reducida por undersampling, pero sigue bien representada.

# %%
y_pred_test_mix_balanced = pipeline_3.predict(X_test)

y_proba_test_mix_balanced = pipeline_3.predict_proba(X_test)

# %%
# Métricas de evaluación test utilizando UnderSampling + SMOTE
accuracy = round(accuracy_score(y_test, y_pred_test_mix_balanced),2)
f1 = round(f1_score(y_test, y_pred_test_mix_balanced, average='weighted'),2)
precision = round(precision_score(y_test, y_pred_test_mix_balanced, average='weighted'),2)
recall = round(recall_score(y_test, y_pred_test_mix_balanced, average='weighted'),2)
conf_matrix = confusion_matrix(y_test, y_pred_test_mix_balanced)
r2 = round(r2_score(y_test, y_pred_test_mix_balanced),2)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Matriz de Confusión:\n", conf_matrix)
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred_test_mix_balanced))
print("\nMatriz de probabilidad:\n",y_proba_test_mix_balanced)

#Muy buen desempeño general en test.
#Accuracy y F1 cercanos, lo que indica balance razonable entre precisión y recall.
#Mejor que los modelos binarios con SMOTE o undersampling por separado.

#Matriz de confusión
#Se observa que el modelo todavía tiende a favorecer la clase mayoritaria en test, aunque es mucho mejor que SMOTE solo o undersampling solo.
# mejora representación de la clase minoritaria en train, pero el test sigue teniendo distribución real desequilibrada.

#Se puede observar un overfitting parcial, probablemente debido a SMOTE y undersampling que generaron ejemplos sintéticos muy similares al train. 

#Esto podria optimizarse dentro de un pipeline probando diferentes hiperparametros y modelos.


