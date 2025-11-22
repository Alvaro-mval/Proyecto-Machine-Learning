# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
import pandas as pd

# Cargar CSV en un DataFrame
df = pd.read_csv("processed_data/athlete_events_clean.csv")

# %%
df

# %%
# Primero dividir las variables por tipo:

variables_num = ['Age', 'Height', 'Weight']

variables_cat = ['Team','NOC','Games','City','Sport','Event','Name','Sex','Season','ID','Year']
variables_cat1 = ['Team','NOC','Games','City','Sport','Event','Name','Sex','Season']

target = ['Medal']

# %%
peso_alto = df[df['Weight'] > 180].shape[0]
peso_bajo = df[df['Weight'] < 30].shape[0]
edad_mayor = df[df['Age'] > 60].shape[0]
altura_alta = df[df['Height'] > 220].shape[0]
altura_baja = df[df['Height'] < 135].shape[0]

print(f"Número de atletas con peso > 180 kg: {peso_alto}")
print(f"Número de atletas con peso < 30 kg: {peso_bajo}")
print(f"Número de atletas con mas de 60 años: {edad_mayor}")
print(f"Número de atletas con altura > 220cm : {altura_alta}")
print(f"Número de atletas con altura < 120cm : {altura_baja}")

#Vemos que hay muy pocos valores con estas caracteristicas, los trataeremos como outliers y los eliminaremos.

# %%
df = df[df['Age'] <= 60]
df = df[df['Weight'] < 180]
df = df[df['Weight'] > 30]
df = df[df['Height'] < 220]
df = df[df['Height'] > 135]

# %%
sns.boxplot(x='Medal', y='Age', data=df)

#Vemos es algo mas dificil conseguir una medalla cuando se es algo mas joven. Y que la edad perfecta es alrededor de los 23-24 años.

# %%
sns.boxplot(x='Medal', y='Height', data=df)

#La media de altura es mas alta para los atletas con medalla que los que no tienen medalla.

# %%
sns.boxplot(x='Medal', y='Weight', data=df)

#De igual forma, el peso parece tener una correlación positiva con el Hecho de ganar una medalla.

# %%
#Violin plot

for col in variables_num:
    plt.figure(figsize=(10,5))
    sns.violinplot(x='Medal', y=col, data=df, palette="muted", inner="quartile")
    plt.title(f"Distribución de {col} por Medalla")
    plt.xlabel("Medalla")
    plt.ylabel(col)
    plt.show()

# %%
#Conteo de medallas por país

pd.crosstab(df['Team'], df['Medal'])

# %%
#Conteo de medallas por deporte

pd.crosstab(df['Sport'], df['Medal'])

# %%
#Conteo de medallas por genero

pd.crosstab(df['Sex'], df['Medal'])

# %%
# Filtrar solo filas con medalla válida
df1 = df[df['Medal'].isin(['Gold', 'Silver', 'Bronze'])]

# Top 20 equipos por número de medallas
df1.groupby('Team')['Medal'].count().sort_values(ascending=False).head(20).plot(
    kind='bar', figsize=(12,5), color='skyblue'
)
plt.title("Top 20 equipos por número de medallas")
plt.ylabel("Número de medallas")
plt.xlabel("Equipo")
plt.xticks(rotation=45)
plt.show()

# Medallas por deporte
plt.figure(figsize=(15,6))
sns.countplot(x='Sport', hue='Medal', data=df1)
plt.title("Medallas por deporte")
plt.xticks(rotation=90)
plt.xlabel("Deporte")
plt.ylabel("Número de medallas")
plt.show()

# Medallas por pais
plt.figure(figsize=(15,6))
sns.countplot(x='Team', hue='Medal', data=df1)
plt.title("Medallas por deporte")
plt.xticks(rotation=90)
plt.xlabel("País")
plt.ylabel("Número de medallas")
plt.show()

# %%
#Edades promedio por deporte (mas altas y mas bajas):

top10_age_high = round(df.groupby('Sport')['Age'].mean().sort_values(ascending=False).head(10),1)
print(top10_age_high)

top10_age_low = round(df.groupby('Sport')['Age'].mean().sort_values(ascending=True).head(10),1)
print(top10_age_low)

# %%
#Pesos y alturas promedio por Pais (mas altas y mas bajas):

top10_team_Height_high = round(df.groupby('Team')['Height'].mean().sort_values(ascending=False).head(10),1)
print(top10_team_Height_high)

top10_team_Height_low = round(df.groupby('Team')['Height'].mean().sort_values(ascending=True).head(10),1)
print(top10_team_Height_low)

top10_team_Weight_high = round(df.groupby('Team')['Weight'].mean().sort_values(ascending=False).head(10),1)
print(top10_team_Weight_high)

top10_team_Weight_low = round(df.groupby('Team')['Weight'].mean().sort_values(ascending=True).head(10),1)
print(top10_team_Weight_low)


# %%
#Boxplots de peso por sexo
sns.boxplot(x='Sex', y='Weight', data=df)

# %%
#Boxplots de altura por sexo
sns.boxplot(x='Sex', y='Height', data=df)

# %%
sns.pairplot(df1[['Age','Height','Weight','Medal']], hue='Medal')

# %%
medal_map = {'No medal': 0, 'Bronze': 1, 'Silver': 2, 'Gold': 3}
df['Medal_num'] = df['Medal'].map(medal_map)

# %%
X = df.drop(['Medal', 'Medal_num'], axis=1)
y = df['Medal_num']

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
num_cols1 = ["Age","Height","Weight"]

# %%
# Visualización de histogramas por grupos

target = ['Medal_num']  # columna para agrupar

import bootcampviztools as bt


for col in num_cols1:
    bt.plot_grouped_histograms(
        df,
        cat_col=target[0], 
        num_col=col,        
        group_size=4
    )

# %%
# Visualización de histogramas por grupos excluyendo las no medallas.

df_medallas = df[df['Medal_num'] != 0]

target = ['Medal_num']  # columna para agrupar


for col in num_cols1:
    bt.plot_grouped_histograms(
        df_medallas,
        cat_col=target[0], 
        num_col=col,        
        group_size=4
    )

# %%
medal_map = {'No medal': 0, 'Bronze': 1, 'Silver': 2, 'Gold': 3}
df['Medal_num'] = df['Medal'].map(medal_map)

# Calcular correlación usando la nueva columna
corr = df[['Age','Height','Weight','Medal_num']].corr()
print(corr)
sns.heatmap(corr, annot=True)
plt.show()

# %%
#Repetimos creando una columna de medalla binaria, quizas mas facil para la correlacion. 

medal_map_bin = {'No medal': 0, 'Bronze': 1, 'Silver': 1, 'Gold': 1}
df['Medal_bin'] = df['Medal'].map(medal_map_bin)

# Calcular correlación usando la nueva columna
corr = df[['Age','Height','Weight','Medal_bin']].corr()
print(corr)
sns.heatmap(corr, annot=True)
plt.show()

# %%
#Vamos a eliminar las columnas que nos parecen que no aportan ninguna información. El caso de name es equivalente que ID, NOC equivalente a Team,
#Year y Season se puede unir en una columna ya existente llamada Games y Event es parecida a Sport.

df = df.drop(['Name', 'NOC', 'Year', 'Season', 'Event'], axis=1)

df

# %%
df.to_csv("processed_data/athlete_events_clean_final.csv", index=False)


