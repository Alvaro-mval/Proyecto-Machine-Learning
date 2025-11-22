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

# %%
#Vamos a cambiar los NaN por la etiqueta 'No medalla' 

df['Medal'] = df['Medal'].fillna('No medal')
df

# %%
df.Medal.value_counts(normalize=True)

#Vemos que las clases del target estan muy desbalanceadas, habra que aplicar algÚn metodo de balanceo para equilibrarlas. 

# %%


# %%
#Para eliminar los nulos de las columnas Age, Height y Weight, vamos a crear una media agrupando por pais y por deporte.
columnas_nulos = ['Age', 'Height', 'Weight']

# Imputar usando la media por genero, Team (Mejor NOC) y Sport
for col in columnas_nulos:
    df[col] = df.groupby(['Sex', 'NOC', 'Sport'])[col].transform(
        lambda x: x.fillna(x.mean())
    )
#Para los valores donde no se puede agrupar por sex, NOC y Sport, eliminamos el pais
for col in columnas_nulos:
    df[col] = df.groupby(['Sex', 'Sport'])[col].transform(
        lambda x: x.fillna(x.mean())
    )
#Para los valores donde no se puede agrupar por sex y Sport, eliminamos el deporte
for col in columnas_nulos:
    df[col] = df.groupby(['Sex'])[col].transform(
        lambda x: x.fillna(x.mean())
    )


# %%
df.info()

# %%
#Vemos que a veces hay diferentes formas de llamar a un pais, y puede ser complicado con los paises que ya no existen o se han fusionado.

paises_unicos = df['Team'].unique()
df_paises = pd.DataFrame(sorted(df['Team'].unique()), columns=['Pais'])
print(df_paises)

# %%
#Vemos que hay muchos paises con nombres raros, equipos que no se sabe bien a donde pertenecen... como los datos que tenemos son de TODOS
#los JJOO de la historia, el primero data de 1896 hasta 2016. Vamos a intentar reducir los años y eliminar los primeros juegos olimpicos
#que no son nada representativos en cuanto a deporte, equipos etc para hacer predicciones en la actualidad.
#Vamos a quedarnos con los JJOO de mas alla de 1970 para evitar esto. 

df = df[df['Year'] >= 1970]

print(df['Year'].min())

# %%
df.info()

# %%
#Una vez que hemos utlizado el año para filtrar, cambiamos el año y el ID a categorica
df['Year'] = df['Year'].astype(str)
df['ID'] = df['ID'].astype(str)

# %%
df.Medal.value_counts(normalize=True) #vemos que la proporcion no ha cambiado mucho eliminando JJOO de antes de 1970

# %%
df.Medal.value_counts() #y que aún tenemos muchos valores absolutos de medallas. 

# %%
#Vemos que pasamos de 1183 equipos o paises a 386

paises_unicos = df['Team'].unique()
df_paises = pd.DataFrame(sorted(df['Team'].unique()), columns=['Pais'])
print(df_paises)

# %%
#Vamos a intentar asignar los paises de la época a paises actuales. Para ello lo primero eliminamos de nuestro df_paises los paises que existen actualmente.

paises_actuales = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso",
    "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic",
    "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Brazzaville)", "Congo (Kinshasa)",
    "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti",
    "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea",
    "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea",
    "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia",
    "Iran", "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
    "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania",
    "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro",
    "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",
    "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan",
    "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Puerto Rico", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe",
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia",
    "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan", "Spain",
    "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania",
    "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey",
    "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen",
    "Zambia", "Zimbabwe"
]

# Suponiendo que la columna con los países se llama 'Team'
df_not_current = df_paises[~df_paises['Pais'].isin(paises_actuales)]

# Eliminar duplicados si quieres solo la lista única de países
df_not_current = df_not_current.drop_duplicates(subset=['Pais'])


df_not_current

# %%
#vamos ahora a crear un diccionario y asignar los paises actuales a los nombres de paises antiguos. 

asignacion_paises = {
    "American Samoa": "Samoa",
    "Argentina-1": "Argentina",
    "Argentina-2": "Argentina",
    "Aruba": "Aruba",
    "Australia-1": "Australia",
    "Australia-2": "Australia",
    "Australia-3": "Australia",
    "Austria-1": "Austria",
    "Austria-2": "Austria",
    "Belarus-1": "Belarus",
    "Belarus-2": "Belarus",
    "Bermuda": "Bermuda",
    "Bonaparte": "France",
    "Brazil-1": "Brazil",
    "Brazil-2": "Brazil",
    "Brentina": "Italy",
    "Briar": "United Kingdom",
    "British Virgin Islands": "Virgin Islands",
    "Bulgaria-1": "Bulgaria",
    "Bulgaria-2": "Bulgaria",
    "Calimucho": "Spain",
    "Canada-1": "Canada",
    "Canada-2": "Canada",
    "Canada-3": "Canada",
    "Cape Verde": "Cape Verde",
    "Cayman Islands": "Cayman Islands",
    "China-1": "China",
    "China-2": "China",
    "China-3": "China",
    "Chinese Taipei": "Taiwan",
    "Chinese Taipei-1": "Taiwan",
    "Chinese Taipei-2": "Taiwan",
    "Clearwater": "United States",
    "Cook Islands": "Cook Islands",
    "Congo (Brazzaville)": "Congo",
    "Congo (Kinshasa)": "Congo",
    "Cote d'Ivoire": "Ivory Coast",
    "Cuba-1": "Cuba",
    "Cuba-2": "Cuba",
    "Czech Republic-1": "Czech Republic",
    "Czech Republic-2": "Czech Republic",
    "Czechoslovakia": "Czech Republic",
    "Czechoslovakia-1": "Czech Republic",
    "Czechoslovakia-2": "Czech Republic",
    "Czechoslovakia-3": "Czech Republic",
    "Denmark-1": "Denmark",
    "Denmark-2": "Denmark",
    "Diabolo St Maurice": "Switzerland",
    "Digby": "United Kingdom",
    "Don Schufro": "Germany",
    "Dow Jones": "United States",
    "East Germany": "Germany",
    "East Germany-1": "Germany",
    "East Germany-2": "Germany",
    "East Germany-3": "Germany",
    "Elvis Va": "United States",
    "Federated States of Micronesia": "Micronesia",
    "Floresco": "France",
    "France-1": "France",
    "France-2": "France",
    "France-3": "France",
    "Galopin De La Font": "France",
    "Germany-1": "Germany",
    "Germany-2": "Germany",
    "Germany-3": "Germany",
    "Gran Gesto": "Spain",
    "Great Britain": "United Kingdom",
    "Great Britain-1": "United Kingdom",
    "Great Britain-2": "United Kingdom",
    "Great Britain-3": "United Kingdom",
    "Greece-1": "Greece",
    "Greece-2": "Greece",
    "Greenoaks Dundee": "United Kingdom",
    "Guam": "Guam",
    "Guinea Bissau": "Guinea-Bissau",
    "Hong Kong": "Hong Kong",
    "Hong Kong-1": "Hong Kong",
    "Hong Kong-2": "Hong Kong",
    "Hungary-1": "Hungary",
    "Hungary-2": "Hungary",
    "India-1": "India",
    "India-2": "India",
    "Individual Olympic Athletes": "Russia",
    "Indonesia-1": "Indonesia",
    "Indonesia-2": "Indonesia",
    "Ireland-1": "Ireland",
    "Ireland-2": "Ireland",
    "Israel-1": "Israel",
    "Israel-2": "Israel",
    "Italy-1": "Italy",
    "Italy-2": "Italy",
    "Italy-3": "Italy",
    "Jamaica-1": "Jamaica",
    "Jamaica-2": "Jamaica",
    "Japan-1": "Japan",
    "Japan-2": "Japan",
    "Japan-3": "Japan",
    "Kosovo": "Kosovo",
    "Lancet": "United Kingdom",
    "Latvia-1": "Latvia",
    "Latvia-2": "Latvia",
    "Lucky Star": "United Kingdom",
    "Macedonia": "North Macedonia",
    "Malaysia-1": "Malaysia",
    "Malaysia-2": "Malaysia",
    "Mauritius-1": "Mauritius",
    "Mauritius-2": "Mauritius",
    "Mexico-1": "Mexico",
    "Mexico-2": "Mexico",
    "Mistral Hojris": "Denmark",
    "Monaco-1": "Monaco",
    "Monaco-2": "Monaco",
    "Mythilus": "Greece",
    "Nadine": "United Kingdom",
    "Netherlands Antilles": "Curaçao",
    "Netherlands-1": "Netherlands",
    "Netherlands-2": "Netherlands",
    "New Zealand-1": "New Zealand",
    "New Zealand-2": "New Zealand",
    "Nigeria-1": "Nigeria",
    "Nigeria-2": "Nigeria",
    "North Korea-1": "North Korea",
    "North Korea-2": "North Korea",
    "North Yemen": "Yemen",
    "Norway-1": "Norway",
    "Norway-2": "Norway",
    "Notavel": "Portugal",
    "Orion": "United Kingdom",
    "Oxalis": "United Kingdom",
    "Poland-1": "Poland",
    "Poland-2": "Poland",
    "Poland-3": "Poland",
    "Pop Art": "United States",
    "Portugal-1": "Portugal",
    "Portugal-2": "Portugal",
    "Puerto Rico-1": "Puerto Rico",
    "Puerto Rico-2": "Puerto Rico",
    "Quando Quando": "Italy",
    "Rambo": "United States",
    "Ravel": "France",
    "Refugee Olympic Athletes": "Refugees",
    "Relampago": "Spain",
    "Romania-1": "Romania",
    "Romania-2": "Romania",
    "Russia-1": "Russia",
    "Russia-2": "Russia",
    "Russia-3": "Russia",
    "Salinero": "Spain",
    "Satchmo": "United States",
    "Serbia and Montenegro": "Serbia",
    "Serbia-1": "Serbia",
    "Serbia-2": "Serbia",
    "Singapore-1": "Singapore",
    "Singapore-2": "Singapore",
    "Slovakia-1": "Slovakia",
    "Slovakia-2": "Slovakia",
    "Slovenia-1": "Slovenia",
    "Slovenia-2": "Slovenia",
    "Solos Carex": "France",
    "South Korea-1": "South Korea",
    "South Korea-2": "South Korea",
    "South Vietnam": "Vietnam",
    "South Yemen": "Yemen",
    "Soviet Union": "Russia",
    "Soviet Union-1": "Russia",
    "Soviet Union-2": "Russia",
    "Soviet Union-3": "Russia",
    "Spain-1": "Spain",
    "Spain-2": "Spain",
    "Sunrise": "USA",
    "Swaziland": "Eswatini",
    "Sweden-1": "Sweden",
    "Sweden-2": "Sweden",
    "Switzerland-1": "Switzerland",
    "Switzerland-2": "Switzerland",
    "Thailand-1": "Thailand",
    "Thailand-2": "Thailand",
    "Timor Leste": "East Timor",
    "Ukraine-1": "Ukraine",
    "Ukraine-2": "Ukraine",
    "Unified Team": "Russia",
    "Unified Team-1": "Russia",
    "Unified Team-2": "Russia",
    "Unified Team-3": "Russia",
    "United States Virgin Islands": "Virgin Islands",
    "United States Virgin Islands-1": "Virgin Islands",
    "United States Virgin Islands-2": "Virgin Islands",
    "United States-1": "United States",
    "United States-2": "United States",
    "United States-3": "United States",
    "Uzbekistan-1": "Uzbekistan",
    "Uzbekistan-2": "Uzbekistan",
    "West Germany": "Germany",
    "West Germany-1": "Germany",
    "West Germany-2": "Germany",
    "West Germany-3": "Germany",
    "Whisper": "United Kingdom",
    "Whitini Star": "United Kingdom",
    "Yugoslavia": "Serbia",
    "Yugoslavia-1": "Serbia",
    "Yugoslavia-2": "Serbia"
}

# Aplicamos el diccionario 
df['Team'] = df['Team'].replace(asignacion_paises)

#Vemos que ahora tenemos 206 paises que son los que compiten actualmente en los JJOO
df['Team'].nunique()

# %%
df.head(20)

# %%
df.Sport.value_counts()

# %%
#Vemos que existen 51 disciplinas de deportes
df.Sport.nunique()

# %%
#Tambien podemos intentar reducir los datos eliminando las disciplinar que actualmente no estan en los JJOO y por tanto no necesitamos predecir. 

sports_actual = [
    "Athletics",
    "Swimming",
    "Gymnastics",
    "Rowing",
    "Cycling",
    "Sailing",
    "Shooting",
    "Wrestling",
    "Football", 
    "Equestrianism",
    "Judo",
    "Handball",
    "Volleyball",
    "Basketball",
    "Weightlifting",
    "Water Polo",
    "Archery",
    "Tennis",
    "Table Tennis",
    "Taekwondo",
    "Triathlon",
    "Badminton",
    "Cross Country Skiing",
    "Alpine Skiing",
    "Biathlon",
    "Bobsleigh",
    "Ice Hockey",
    "Speed Skating",
    "Ski Jumping",
    "Figure Skating",
    "Short Track Speed Skating",
    "Nordic Combined",
    "Freestyle Skiing",
    "Snowboarding",
    "Luge",
    "Modern Pentathlon",
    "Rugby Sevens",
    "Curling",
    "Golf",
    "Synchronized Swimming",
    "Boxing",
    "Rhythmic Gymnastics",
    "Canoeing",
    "Skeleton",
    "Beach Volleyball",
    "Trampolining",
    "Fencing",
    "Hockey",
    "Diving"
]

deportes_originales = set(df["Sport"].unique())

df = df[df["Sport"].isin(sports_actual)]
deportes_eliminados = deportes_originales - set(df["Sport"].unique())

print("Deportes eliminados:")
print(deportes_eliminados)
df.Sport.nunique()

#Despues del filtrado, eliminamos Softball y Baseball que ya no son olímpicos.

# %%
df.to_csv("processed_data/athlete_events_clean.csv", index=False)


