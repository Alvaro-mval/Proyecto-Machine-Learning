# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, r2_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
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
from sklearn.metrics import roc_curve, auc

# %%
# Cargar CSV en un DataFrame
train = pd.read_csv("train/train_set.csv")
test = pd.read_csv("test/test_set.csv")

# %%
train

# %%
X_train = train.drop(columns="Medal_num")
y_train = train.Medal_num
cat_cols = X_train.select_dtypes(include=["object"]).columns
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

# %%
#Con los datos obtenidos en los analisis anteriores, sabemos que para nuestro dataset mal balanceado, 
# # la tecnica que mejor funciona es un SMOTE + undersampling. 
# Además, necesitamos hacer un TargetEncoder a las columnas categoricas como preprocesado.
# Encoder para columnas categóricas 
encoder = TargetEncoder() 
X_cat_encoded = encoder.fit_transform(X_train[cat_cols], y_train) 
X_num = X_train[num_cols]  
X_train_encoded = pd.concat([X_cat_encoded, X_num], axis=1)


# Balanceo de clases
counter = Counter(y_train)
n_major = counter[0]

smote = SMOTE(random_state=42, sampling_strategy={
    1: int(n_major*0.5),
    2: int(n_major*0.5),
    3: int(n_major*0.5)
})

undersample = RandomUnderSampler(sampling_strategy={0: int(n_major*0.8)}, random_state=42)

X_res, y_res = smote.fit_resample(X_train_encoded, y_train)
X_res, y_res = undersample.fit_resample(X_res, y_res)

# %%
#Con los datos obtenidos en los analisis anteriores, sabemos que para nuestro dataset mal balanceado, 
# la tecnica que mejor funciona es un SMOTE + undersampling. Además, necesitamos hacer un TargetEncoder a las columnas categoricas como preprocesado.

#<<<<Pipeline>>>>
pipe_linear = ImbPipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("smote", smote),
    ('undersample',undersample),
    ("classifier", LogisticRegression())
])

#<<<<Logistic classifier>>>>
logistic_params = {
    "scaler": [StandardScaler(), MinMaxScaler()],
    "pca__n_components": [3, 5,7],
    "classifier": [
        LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    ],
    "classifier__C": [1, 10],
    "classifier__penalty": ["l2"]
}

pipe_trees = ImbPipeline(steps=[
    ("pca", PCA()),
    ("smote", smote),
    ('undersample',undersample),
    ("classifier", RandomForestClassifier())
])


#<<<<Random Forest>>>>
random_forest_params = {
    "pca__n_components": [3, 5,7],
    "classifier": [
        RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)
    ],
    "classifier__max_depth": [5, 10, None],
    "classifier__n_estimators": [100,200],
    "classifier__max_features": ["sqrt","log2"]
}

#<<<<<<XG Boost>>>>>>>
xgb_params = {
    "pca__n_components": [3, 5,7],
    "classifier": [
        XGBClassifier(
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss"
        )
    ],
    "classifier__n_estimators": [100,200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.1,0.2]
}

#<<<<< Gradient Boosting Classifier>>>>>
gbct_params = {
    "pca__n_components": [3, 5,7],
    "classifier": [GradientBoostingClassifier(random_state=42)],
    "classifier__n_estimators": [100,200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.1,0.2]
}

#<<<<<Single Tree>>>>>

tree_params = {
    "pca__n_components": [3, 5,7],
    "classifier": [DecisionTreeClassifier(random_state=42)],
    "classifier__max_depth": [None, 3, 5, 10],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 5]
}


pipe_ada = ImbPipeline(steps=[
    ("pca", PCA()),
    ("smote", smote),
    ('undersample',undersample),
    ("classifier", AdaBoostClassifier())
])

#<<<<<ADA Boost>>>>>
ada_params = {
    "pca__n_components": [3, 5,7],
    "classifier": [AdaBoostClassifier(random_state=42)],
    "classifier__n_estimators": [50, 100, 200],
    "classifier__learning_rate": [0.1, 1]
}

# %%
#hacemos un bucle for para sacar una tabla con los mejores parametros de cada modelo y compararlos entre si

pipelines = [
    (pipe_linear, logistic_params),
    (pipe_trees, random_forest_params),
    (pipe_trees, xgb_params),
    (pipe_trees, gbct_params),
    (pipe_trees, tree_params),
    (pipe_ada, ada_params)
]

results = []

for pipe, params in pipelines:
    model_name = type(params["classifier"][0]).__name__

    clf = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    clf.fit(X_train_encoded, y_train)

    results.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_
    })

df = pd.DataFrame(results)
#display(df)

# %%
#Ahora que tenemos que el mejor modelo es el RandomForestClassifier, vamos a intentar optimizarlo aún mas utilizando una primera vez RandomizedSearchCV

#<<<<Pipeline>>>>
pipeRF = ImbPipeline(steps=[
    ('classifier', RandomForestClassifier(class_weight='balanced', n_jobs=-1))
])

#<<<<Random Forest>>>>
random_forest_params = {
    'classifier__max_depth': [3,4, 5, None], 
    'classifier__n_estimators': [100,200,300],
    'classifier__max_features':['sqrt','log2']
}

clf_random = RandomizedSearchCV(
    estimator=pipeRF,
    param_distributions=random_forest_params,
    n_iter=50,
    cv=5,
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

#Hagamos una última optimización para ver si podemos llegar a 0.8. Esta vez con GridSearchCV

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

# %%
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
            1: int(n_major * 0.5),
            2: int(n_major * 0.5),
            3: int(n_major * 0.5)
        }
    )),
    ('undersample', RandomUnderSampler(
        sampling_strategy={0: int(n_major * 0.8)},
        random_state=42
    )),

    ('classifier', RandomForestClassifier(
        n_estimators=350,
        criterion='gini',
        max_depth=None,
        max_features='log2',
        min_samples_split=2,
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
X_test = test.drop(columns="Medal_num")
y_test = test.Medal_num

# %%
# PREDICCIONES
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
from sklearn.preprocessing import label_binarize

classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes) # convierte a binario para cada clase
y_proba_roc = final_model_pipe.predict_proba(X_test) # shape (n_samples, n_classes)

plt.figure(figsize=(8,6))

for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_roc[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Clase {c} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - RandomForest Multiclase (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()

# %%
pickle.dump(final_model_pipe, open('final_model_ML_multiclase.pkl', 'wb'))

# %%
modelo_importado = pickle.load(open("final_model_ML_multiclase.pkl", 'rb'))
modelo_importado

# %%
modelo_importado.predict(X_test)


