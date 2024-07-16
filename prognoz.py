import streamlit as st
import pandas as pd
import seaborn as sns
import io
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



prognoz = pd.read_csv('Clean.csv')


def show_prognoz(): 
    st.title('Прогнозирование') 
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


    columns_to_drop = ['Age', 'Gender']  # Замените 'column1' и 'column2' на фактические названия столбцов, которые вы хотите удалить
    prognoz = prognoz.drop(columns=columns_to_drop, axis=1)

    # Вычисление межквартильного размаха
    Q1 = prognoz['Salary'].quantile(0.25)
    Q3 = prognoz['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    prognoz = prognoz[(prognoz['Salary'] > lower_bound) & (prognoz['Salary'] < upper_bound)]
    label_encoder = LabelEncoder()
    prognoz['Education Level'] = label_encoder.fit_transform(prognoz['Education Level'])
    label_encoder = LabelEncoder()
    prognoz['Job Title'] = label_encoder.fit_transform(prognoz['Job Title'])


    # Подготовка данных
    X = prognoz[['Education Level', 'Job Title', 'Years of Experience']].values
    y = prognoz['Salary'].values

    # Разделение на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Создаем экземпляр модели Gradient Boosting
    gradient_boosting = GradientBoostingRegressor()
    
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
}

# Поиск оптимальных параметров модели Gradient Boosting с помощью GridSearchCV
    grid_search_gb = GridSearchCV(gradient_boosting, param_grid, cv=4, scoring='r2')
    grid_result_gb = grid_search_gb.fit(X_train, y_train)

# Вывод лучших параметров модели Gradient Boosting
    st.write("Лучшие параметры Gradient Boosting: ", grid_result_gb.best_params_)

# Обучение модели Gradient Boosting
    best_model_gb = GradientBoostingRegressor(**grid_result_gb.best_params_)
    best_model_gb.fit(X_train, y_train)

# Получение прогнозов для модели Gradient Boosting
    predictions_gb = best_model_gb.predict(X_test)

# Оценка точности с помощью R2 Score
    r2_gb = r2_score(y_test, predictions_gb)
    st.write(f"Точность прогнозирования (R2 Score) Gradient Boosting: {r2_gb}")

    # Вычисление средней абсолютной ошибки (MAE)
    mae_gb = mean_absolute_error(y_test, predictions_gb)
    st.write(f"Средняя абсолютная ошибка (MAE) Gradient Boosting: {mae_gb}")


    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape_gb = mean_absolute_percentage_error(y_test, predictions_gb)
    st.write(f"Среднее абсолютное процентное отклонение (MAPE) Gradient Boosting: {mape_gb}")


# Создание DataFrame с реальными и прогнозируемыми значениями
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions_gb})

# Визуализация с помощью Plotly Express
    fig = px.scatter(comparison_df, x='Actual', y='Predicted', title='Сравнение реальных и прогнозируемых зарплат',
                 labels={'Actual': 'Реальная зарплата', 'Predicted': 'Прогнозируемая зарплата'})
    st.plotly_chart(fig)
   