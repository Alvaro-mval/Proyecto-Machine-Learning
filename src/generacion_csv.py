# %%
import pandas as pd

# Lista de atletas de ejemplo, datos realistas de Tokio 2020
data = [
    # 5 medallistas de oro
    {"ID":"Neeraj Chopra", "Sex":"M", "Age":23, "Height":182, "Weight":86, "Team":"India", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Athletics", "Medal":1},
    {"ID":"Lasha Bekauri", "Sex":"M", "Age":20, "Height":192, "Weight":90, "Team":"Georgia", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Judo", "Medal":1},
    {"ID":"Uta Abe", "Sex":"F", "Age":24, "Height":156, "Weight":57, "Team":"Japan", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Judo", "Medal":1},
    {"ID":"Caeleb Dressel", "Sex":"M", "Age":27, "Height":185, "Weight":88, "Team":"USA", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Swimming", "Medal":1},
    {"ID":"Guan Chenchen", "Sex":"F", "Age":22, "Height":170, "Weight":60, "Team":"China", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Gymnastics", "Medal":1},

    # 5 medallistas de plata
    {"ID":"Mirko Zanni", "Sex":"M", "Age":25, "Height":177, "Weight":75, "Team":"Italy", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Weightlifting", "Medal":2},
    {"ID":"Marie-Josée Ta Lou", "Sex":"F", "Age":23, "Height":165, "Weight":55, "Team":"France", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Athletics", "Medal":2},
    {"ID":"Muslim Evloev", "Sex":"M", "Age":26, "Height":180, "Weight":78, "Team":"Russia", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Wrestling", "Medal":2},
    {"ID":"Femke Heemskerk", "Sex":"F", "Age":21, "Height":168, "Weight":58, "Team":"Netherlands", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Swimming", "Medal":2},
    {"ID":"Simon Geschke", "Sex":"M", "Age":29, "Height":183, "Weight":82, "Team":"Germany", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Cycling", "Medal":2},

    # 5 medallistas de bronce
    {"ID":"Alison dos Santos", "Sex":"M", "Age":22, "Height":176, "Weight":70, "Team":"Brazil", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Athletics", "Medal":3},
    {"ID":"Ellie Black", "Sex":"F", "Age":24, "Height":160, "Weight":54, "Team":"Canada", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Gymnastics", "Medal":3},
    {"ID":"Igor Son", "Sex":"M", "Age":28, "Height":181, "Weight":77, "Team":"Kazakhstan", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Weightlifting", "Medal":3},
    {"ID":"An Ba-ul", "Sex":"F", "Age":25, "Height":167, "Weight":56, "Team":"South Korea", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Judo", "Medal":3},
    {"ID":"Kjetil Borch", "Sex":"M", "Age":26, "Height":179, "Weight":79, "Team":"Norway", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Rowing", "Medal":3},

    # 5 sin medalla
    {"ID":"Kyle Chalmers", "Sex":"M", "Age":24, "Height":175, "Weight":72, "Team":"Australia", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Swimming", "Medal":0},
    {"ID":"Dina Asher-Smith", "Sex":"F", "Age":22, "Height":162, "Weight":53, "Team":"UK", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Athletics", "Medal":0},
    {"ID":"Alejandro Valverde", "Sex":"M", "Age":27, "Height":180, "Weight":80, "Team":"Spain", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Cycling", "Medal":0},
    {"ID":"Vanessa Ferrari", "Sex":"F", "Age":23, "Height":165, "Weight":57, "Team":"Italy", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Gymnastics", "Medal":0},
    {"ID":"Tadeusz Michalik", "Sex":"M", "Age":28, "Height":178, "Weight":75, "Team":"Poland", "Games":"Olympics 2020", "City":"Tokyo", "Sport":"Wrestling", "Medal":0},
]

# %%
# Crear dataframe
df = pd.DataFrame(data)

# %%
# Guardar CSV
df.to_csv("tokyo2020_test.csv", index=False)

# %%
df["Medal"] = df["Medal"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

print(df.head(20))


# %%
df.to_csv("tokyo2020_test_binario.csv", index=False)

# %%



