import streamlit as st
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
from plotly.offline import iplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 



prognoz = pd.read_csv('Clean.csv')


def show_prognozneir(): 
    st.title('Прогнозирование (модель нейронной сети)') 
    global prognoz
    buffer = io.StringIO()
    # Сохраняем текущий поток stdout
    old_stdout = sys.stdout
# Перенаправляем stdout в наш буфер
    sys.stdout = buffer
# Возвращаем stdout в его обычное состояние
    sys.stdout = old_stdout
# Получаем вывод из буфера
    info_text = buffer.getvalue()



# Удаление выбранных столбцов
    columns_to_drop = ['Age', 'Gender']
    prognoz = prognoz.drop(columns=columns_to_drop, axis=1)

# Вычисление межквартильного размаха
    Q1 = prognoz['Salary'].quantile(0.25)
    Q3 = prognoz['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    prognoz = prognoz[(prognoz['Salary'] > lower_bound) & (prognoz['Salary'] < upper_bound)]

# Преобразование категориальных признаков в числовые
    label_encoder = LabelEncoder()
    prognoz['Education Level'] = label_encoder.fit_transform(prognoz['Education Level'])
    label_encoder = LabelEncoder()
    prognoz['Job Title'] = label_encoder.fit_transform(prognoz['Job Title'])

# Подготовка данных
    X = prognoz[['Education Level', 'Job Title', 'Years of Experience']].values
    y = prognoz['Salary'].values



    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=500, random_state=11)
    rf.fit(X_train, y_train)
    score = rf.score(X_train, y_train)*100
    predicted_salary = np.round(rf.predict(X_test))
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Выходной слой без активации для регрессии
])
# Компиляция модели
    model.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


    # Оценка модели
    y_pred_nn = model.predict(X_test)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    st.write('Среднеквадратичная ошибка нейронной сети:', mse_nn)
   
    # Оценка точности и правильности модели нейронной сети
    loss_nn = model.evaluate(X_test, y_test, verbose=0)



# Оценка точности модели нейронной сети
    accuracy_nn = (1 - loss_nn) * 100
    st.write('Точность нейронной сети:', accuracy_nn)

# Вычисление R2 Score для модели нейронной сети
    r2_nn = r2_score(y_test, y_pred_nn)
    st.write("Коэффициент детерминации нейронной сети:", r2_nn) 

# Вычисление MAE для модели нейронной сети
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    st.write("Средняя абсолютная ошибка нейронной сети:", mae_nn)



    # Обратное масштабирование прогнозируемых значений
    predicted_values = scaler_y.inverse_transform(y_pred_nn).flatten()
    y_test_inverse = scaler_y.inverse_transform(y_test).flatten()

# Создание DataFrame с реальными и прогнозируемыми значениями
    comparison_df = pd.DataFrame({'Actual': y_test_inverse, 'Predicted': predicted_values})

# Визуализация с помощью Plotly Express
    fig = px.scatter(comparison_df, x='Actual', y='Predicted', title='Сравнение реальных и прогнозируемых зарплат',
                 labels={'Actual': 'Реальная зарплата', 'Predicted': 'Прогнозируемая зарплата'})
    st.plotly_chart(fig)


    