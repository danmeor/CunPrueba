import seaborn as sns
import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
import pandas as pd
from prophet import Prophet

# Cargar los datos desde URLs
url ="https://raw.githubusercontent.com/danmeor/CunPrueba/main/data1.csv"
url1 ="https://raw.githubusercontent.com/danmeor/CunPrueba/main/sales_predictions_simulado.csv"
df = pd.read_csv(url)      # Datos principales
df1 = pd.read_csv(url1)    # Predicciones simuladas de ventas

print(df1)  # Mostrar el DataFrame df1

# Unir los dos DataFrames por 'store_id'
df2= df.merge(df1,how="left",on='store_id')
df2

# Eliminar la columna 'name' que no es necesaria
df2.drop(columns=['name'],inplace=True)
df2

# Mostrar información del DataFrame resultante
df2.info()

# Verificar valores nulos en el DataFrame
print(df2.isnull().sum())

# Obtener datos meteorológicos de Open-Meteo para cada tienda
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

df3 = pd.DataFrame([])  # DataFrame vacío para almacenar los datos meteorológicos

url = "https://api.open-meteo.com/v1/forecast"
for i in df2['id'].unique():
    # Configurar sesión de caché y reintentos para cada tienda
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Definir los parámetros de la consulta
    params = {
        "latitude": df2.loc[df2['id']==i,'latitude'].values[0],
        "longitude": df2.loc[df2['id']==i,'longitude'].values[0],
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": "auto",
        "start_date": df2.loc[df2['id']==i,'date'].values[0],
        "end_date": df2.loc[df2['id']==i,'date'].values[0]
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Procesar los datos diarios de temperatura
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["average_temperature"] =(daily_data["temperature_2m_max"] + daily_data["temperature_2m_min"])/ 2
    
    daily_dataframe = pd.DataFrame(data = daily_data)
    daily_dataframe['id'] = i
    
    df3 = pd.concat([df3,daily_dataframe])  # Agregar los datos de cada tienda

# Ajustar el formato de la columna 'date'
df3['date']=df3['date'].dt.date
df3.reset_index(drop=True, inplace=True)

df3  # Mostrar el DataFrame con datos meteorológicos

# Unir los datos meteorológicos con los datos de ventas
salesp  =  df2.merge(df3[['id','average_temperature']],how='left',on=['id'])
salesp 

# Convertir la columna 'date' a formato datetime y luego a int64
salesp['date'] = pd.to_datetime(salesp['date'])
salesp['date'] = salesp['date'] .astype('int64')

# Calcular y mostrar la matriz de correlación
matriz_correlacion = salesp[['date','product_id','store_id','sales','average_temperature']].corr()
sns.heatmap(matriz_correlacion,annot=True,cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# Preparar los datos para la regresión lineal simple
x=salesp['average_temperature'].values.reshape(-1,1)
y=salesp["sales"].values.reshape(-1,1)
sc_x = StandardScaler()
sc_y = StandardScaler()
x_std = sc_x.fit_transform(x)
y_std = sc_y.fit_transform(y)
slr = LinearRegression()
slr.fit(x_std,y_std)  # Ajustar el modelo de regresión lineal estandarizado

# Graficar la regresión lineal estandarizada
plt.scatter(x_std,y_std)
plt.plot(x_std,slr.predict(x_std),color="Red")
plt.title("Regresión Lineal")
plt.xlabel("Temperatura")

# Regresión lineal sin estandarizar y evaluación
reg_L = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
reg_L.fit(X_train, y_train)
y_pred = reg_L.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Error cuadrático medio (MSE): {mse}")
print(f"Coeficiente de determinación (R^2):{r2}")

# Restaurar la columna 'date' a formato datetime
salesp['date'] = pd.to_datetime(salesp['date'], unit='ns')

# Agregar columna con predicciones de ventas usando la temperatura promedio
salesp["sales_prediction"] = reg_L.predict(salesp["average_temperature"].values.reshape(-1, 1))
salesp

# Guardar el DataFrame con predicciones en una base de datos PostgreSQL
usuario = 'postgres'
contrasena = "Danmeor2010"
host = 'localhost'
puerto = '5432'
base_datos = 'postgres'
conexion = create_engine(f'postgresql+psycopg2://{usuario}:{contrasena}@{host}:{puerto}/{base_datos}')
salesp.to_sql('sales_prediction', conexion, if_exists='replace', index=False)

# Preparar los datos para Prophet (cambiar nombres de columnas)
salesp.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

# Inicializar y entrenar el modelo Prophet
model = Prophet()       
model.add_regressor('store_id')              # Agregar regresores externos
model.add_regressor('average_temperature')
model.fit(salesp)                            # Ajustar el modelo

# Crear DataFrame para predicciones futuras (30 días)
future = model.make_future_dataframe(periods=30)
future['store_id'] = salesp['store_id'].iloc[-1]  # Usar último valor de store_id
future['average_temperature'] = salesp['average_temperature'].iloc[-1]  # Usar última temperatura

# Obtener predicciones futuras
forecast = model.predict(future)

# Graficar los resultados de Prophet
model.plot(forecast)

# Evaluar el modelo Prophet con los últimos 30 días reales vs predichos
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_true = salesp['y'][-30:]         # Últimos 30 días reales
y_pred = forecast['yhat'][-30:]    # Últimos 30 días predichos

mae = mean_absolute_error(y_true, y_pred)  # Error absoluto medio
rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Raíz del error cuadrático medio

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")