# %%
import pandas as pd

# Cargar CSV en un DataFrame
df = pd.read_csv("raw_data/athlete_events.csv")

# %%
df.info()

#Vemos que existen muchos nulos en medal (los que no ganaron medalla), pero tambien hay nulos para edad, height y weight

# %%
df.describe()

#vemos que Year es una columna numérica cuando debería ser categórica. 


