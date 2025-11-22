# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, r2_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from collections import Counter
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA

# %%
# Cargar CSV en un DataFrame
train = pd.read_csv("train/train_set_bin.csv")
test = pd.read_csv("test/test_set_bin.csv")

# %%
train

# %%
X_train = train.drop(columns="Medal_bin")
y_train = train.Medal_bin
cat_cols = X_train.select_dtypes(include=["object"]).columns
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

# %%
# Encoder para columnas categóricas
encoder = TargetEncoder()
X_cat_encoded = encoder.fit_transform(X_train[cat_cols], y_train)
X_num = X_train[num_cols]  # columnas numéricas sin tocar
X_train_encoded = pd.concat([X_cat_encoded, X_num], axis=1)

# Balanceo de clases
counter = Counter(y_train)
n_major = counter[0]

smote = SMOTE(random_state=42, sampling_strategy={
    1: int(n_major*0.5)
})
undersample = RandomUnderSampler(sampling_strategy={0: int(n_major*0.8)}, random_state=42)

X_res, y_res = smote.fit_resample(X_train_encoded, y_train)
X_res, y_res = undersample.fit_resample(X_res, y_res)

# %%
#Con los datos obtenidos en los analisis anteriores, sabemos que para nuestro dataset mal balanceado, 
# la tecnica que mejor funciona es un SMOTE + undersampling. Además, necesitamos hacer un TargetEncoder a las columnas categoricas como preprocesado.

#<<<<Pipeline>>>>
pipe = ImbPipeline(steps=[
    ('pca', PCA()),                   
    ('scaler', StandardScaler()),      
    ("feature_selection", SelectKBest(k=5, score_func=f_classif)),
    ('classifier', SVC(probability=True, class_weight='balanced'))
])

#<<<<Logistic classifier>>>>
logistic_params = {
    'pca__n_components': [3, 5, 7,], 
    'scaler': [StandardScaler(), MinMaxScaler()],
    'classifier': [
        LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')],
    "feature_selection__k": [3,5],
    'classifier__C':[0.1,1,10], 
    'classifier__penalty': ['l2']
}

#<<<<Random Forest>>>>
random_forest_params = {
    'pca__n_components': [3, 5],
    'scaler': [None],
    'classifier': [RandomForestClassifier(class_weight='balanced', n_jobs=-1)],
    'classifier__max_depth': [3, 5, None], 
    'classifier__n_estimators': [200],
    'classifier__max_features':['sqrt','log2']
}

#<<<Vector Soporte>>>>
svm_param = {
    'pca__n_components': [3, 5, 7,],
    'scaler': [StandardScaler(), MinMaxScaler(), None],
    'classifier': [SVC(class_weight='balanced')],
    'classifier__C': [0.1, 1, 5],
}


#<<<<<<XG Boost>>>>>>>
xgb_params = {
    'pca__n_components': [3, 5, 7,],
    'classifier': [XGBClassifier(random_state=42,n_jobs=-1)],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.1, 0.2],
}

#<<<<< Gradient Boosting Classifier>>>>>
gbct_params = {
    'pca__n_components': [3, 5, 7,],
    'scaler': [None],
    'classifier': [GradientBoostingClassifier(random_state=42)],
    'classifier__n_estimators': [100],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.1, 0.2],
}

#<<<<<Single Tree>>>>>

tree_params = {
    'pca__n_components': [3, 5, 7,],
    'classifier': [DecisionTreeClassifier(random_state=42)],
    'classifier__max_depth': [None, 3, 5, 10],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 5]
}

#<<<<<ADA Boost>>>>>
ada_params = {
    'pca__n_components': [3, 5, 7,],
    'classifier': [AdaBoostClassifier(random_state=42)],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 1]
}

results_modelos = []

search_space = [
    logistic_params,
    random_forest_params,
    #svm_param,
    xgb_params,
    gbct_params,
    tree_params,
    ada_params
]

#hacemos un bucle for para sacar una tabla con los mejores parametros de cada modelo y compararlos entre si
for params in search_space:
    # Extraer el nombre del modelo
    model_name = type(params['classifier'][0]).__name__

    clf_CV = GridSearchCV(estimator = pipe,
                    param_grid = [params],
                    cv = 3,
                    verbose=2,
                    n_jobs=-1)

    clf_CV.fit(X_res, y_res)

    results_modelos.append({
        'model': model_name,
        'best_score': clf_CV.best_score_,
        'best_params': clf_CV.best_params_,
        'best_estimator': clf_CV.best_estimator_

    })

df_results = pd.DataFrame(results_modelos)
#display(df_results)


# %%
#Ahora que tenemos que el mejor modelo es el RandomForestClassifier y XGBClassifier, vamos a intentar optimizarlo aún mas utilizando una primera vez RandomizedSearchCV

#<<<<Pipeline>>>>
pipe_opt = ImbPipeline(steps=[
('classifier', SVC(probability=True))
])

#<<<<Random Forest>>>>
random_forest_params = {
    'classifier': [RandomForestClassifier(class_weight='balanced', n_jobs=-1)],
    'classifier__max_depth': [5, None], 
    'classifier__n_estimators': [200,300],
    'classifier__max_features':['log2'],
    
}

#<<<<<<XG Boost>>>>>>>
xgb_params = {
    'classifier': [XGBClassifier(random_state=42,tree_method="hist",n_jobs=-1)],
    'classifier__n_estimators': [200, 300],
    'classifier__max_depth': [3,5,7],
    'classifier__learning_rate': [0.2],
}

search_space1 = [
    random_forest_params,
    xgb_params
]

clf_random = RandomizedSearchCV(
    estimator=pipe_opt,
    param_distributions=search_space1,
    n_iter=100,
    cv=4,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# %%
clf_random.fit(X_res, y_res)

# %%
print(clf_random.best_estimator_)
print(clf_random.best_score_)
print(clf_random.best_params_)

# %%
#Vemos que no hemos afinado mucho, volvemos a intentarlo con otros parámetros

#Hagamos una última optimización para ver si podemos llegar a 0.9. Esta vez con GridSearchCV y quedandonos ya solo con Random Forest

#<<<<Pipeline>>>>
pipeRF = ImbPipeline(steps=[
    ('classifier', RandomForestClassifier(class_weight='balanced', n_jobs=-1))
])

#<<<<Random Forest>>>>
random_forest_params_final = {
    'classifier__n_estimators': [300,350,400],
    'classifier__criterion': ['gini'],
    'classifier__max_depth': [None],
    'classifier__max_features': ['log2'],
    'classifier__min_samples_split': [2,5],
    'classifier__min_samples_leaf': [2,1]   
}

clf_CV_final = GridSearchCV(estimator = pipeRF,
                    param_grid = random_forest_params_final,
                    cv = 5,
                    verbose=2,
                    n_jobs=-1)

# %%
clf_CV_final.fit(X_res, y_res)
print(clf_CV_final.best_estimator_)
print(clf_CV_final.best_score_)
print(clf_CV_final.best_params_)

# %%
#Entrenamos con el modelo final incluyendo dentro del pipe el preproces para poder aplicar al test
# Encoder para columnas categóricas

# Pipeline FINAL sin CV
final_model_pipe = ImbPipeline(steps=[
    ('encoder', TargetEncoder()),
    ('smote', SMOTE(
        random_state=42,
        sampling_strategy={
            1: int(n_major * 0.5)
        }
    )),
    ('undersample', RandomUnderSampler(
        sampling_strategy={0: int(n_major * 0.8)},
        random_state=42
    )),

    ('classifier', RandomForestClassifier(
        n_estimators=300,
        criterion='gini',
        max_depth=None,
        max_features='log2',
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ))
])

# %%
# ENTRENAR MODELO FINAL
final_model_pipe.fit(X_train, y_train)

# %%
# PREDICCIONES en train
y_pred_train = final_model_pipe.predict(X_train)
y_proba_train = final_model_pipe.predict_proba(X_train)

# %%
# Métricas de evaluación train 
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

# %%
X_test = test.drop(columns="Medal_bin")
y_test = test.Medal_bin

# %%
y_pred = final_model_pipe.predict(X_test)
y_proba_test = final_model_pipe.predict_proba(X_test)

# %%
# Métricas de evaluación test 
accuracy = round(accuracy_score(y_test, y_pred),2)
f1 = round(f1_score(y_test, y_pred, average='weighted'),2)
precision = round(precision_score(y_test, y_pred, average='weighted'),2)
recall = round(recall_score(y_test, y_pred, average='weighted'),2)
conf_matrix = confusion_matrix(y_test, y_pred)
r2 = round(r2_score(y_test, y_pred),2)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Matriz de Confusión:\n", conf_matrix)
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("\nMatriz de probabilidad:\n",y_proba_test)

# %%
y_proba_roc = final_model_pipe.predict_proba(X_test)[:, 1] 
fpr, tpr, thresholds = roc_curve(y_test, y_proba_roc)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - RandomForest Binaria')
plt.legend(loc="lower right")
plt.show()

# %%
pickle.dump(final_model_pipe, open('final_model_ML_binario.pkl', 'wb'))

# %%
modelo_importado = pickle.load(open("final_model_ML_binario.pkl", 'rb'))
modelo_importado

# %%
modelo_importado.predict(X_test)


