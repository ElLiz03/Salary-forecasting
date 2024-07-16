import streamlit as st
import pandas as pd
import seaborn as sns
import io
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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



data = pd.read_csv('Clean.csv')
data_cod = pd.read_csv('data_cod.csv')


def show_cluster(): 
    st.title('Кластеризация') 
    global data
    global data_cod
    global data_cluster
    buffer = io.StringIO()
    # Сохраняем текущий поток stdout
    old_stdout = sys.stdout
# Перенаправляем stdout в наш буфер
    sys.stdout = buffer
# Возвращаем stdout в его обычное состояние
    sys.stdout = old_stdout
# Получаем вывод из буфера
    info_text = buffer.getvalue()
   

    scaler = MinMaxScaler() 
    normalized_data = scaler.fit_transform(data_cod) 
    data_norm = pd.DataFrame(normalized_data, columns=data.columns)


    data_cluster = pd.read_csv('data_cluster.csv')



# Выбираем столбцы для кластеризации
    X = data_cluster[['Age', 'Education Level', 'Salary']]
# Создаем модель KMeans
    kmeans = KMeans(n_clusters=3)
# Применяем модель
    kmeans.fit(X)
# Получаем результаты кластеризации
    data_cluster['Cluster'] = kmeans.predict(X)


    kbest_scores = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data_cluster)

    # Используем inertia_ в качестве оценки
        score = kmeans.inertia_
        labels = kmeans.predict(data_cluster)

    # Расчитываем размеры кластеров
        sizes = pd.Series(labels).value_counts().to_dict()

        st.write(f'KMeans имеет {i} кластеры размеров {sizes} с оценкой {score:.2f}')
        kbest_scores.append(score)


    
    k_values = range(1, 11)
    inertias = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_norm)
        inertias.append(kmeans.inertia_)
        
    fig = plt.figure()
    plt.plot(k_values, inertias, 'bx-')
    plt.xlabel('Значения k')
    plt.ylabel('Расстояние')
    plt.title('Метод "локтя" для определения k')
    
    st.pyplot(fig)





    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(data_cluster['Age'],
                         data_cluster['Education Level'],
                         data_cluster['Salary'],
                         c=data_cluster['Cluster'],
                         cmap='viridis',
                         s=40)

    ax.set_title('Кластеризация данных')
    ax.set_xlabel('Возраст')
    ax.set_ylabel('Уровень образования')
    ax.set_zlabel('Зарплата')
    plt.colorbar(scatter)
    st.pyplot(fig)


    st.write('Количество наблюдений в каждом кластере',)
    st.bar_chart(data_cluster['Cluster'].value_counts().sort_index())




    # Обратное преобразование для столбца Age
    age_min = data['Age'].min()
    age_max = data['Age'].max()
    salary_min = data['Salary'].min()
    salary_max = data['Salary'].max()
    year_min = data['Years of Experience'].min()
    year_max = data['Years of Experience'].max()


    # Пример кода для присвоения текстового обозначения уровню образования в каждом кластере
    def categorize_education_level(education_level):
        if 0.00 < education_level < 0.05:
            return 'Магистр'
        elif 0.05 < education_level < 0.07:
            return 'Школа'
        elif 0.07 <= education_level < 0.73:
             return 'Бакалавр'
        elif  0.73 <= education_level <= 1:
            return 'Кандидат наук'
        else:
            return 'Некорректное значение'

# Вычисляем средние значения для каждого кластера
    cluster_means = data_cluster.groupby('Cluster').mean() ### может df.groupby('Cluster')

# Выводим средние значения для каждого кластера
    for i in range(len(cluster_means)):
        print('--------------------------------------')
        print(f'Средние значения для кластера {i+1}:')
        print(f'Средний возраст: {cluster_means.iloc[i]["Age"]:.2f}')
        print(f'Средняя зарплата: {cluster_means.iloc[i]["Salary"]:.2f}')
    # Добавляем интерпретацию уровня образования
        education_level = cluster_means.iloc[i]["Education Level"]
        print(f'Уровень образования: {categorize_education_level(education_level)}')
        print(f'Средний опыт работы: {cluster_means.iloc[i]["Years of Experience"]:.2f}')


    # Обратное масштабирование для возраста
    backscaled_age = 0.35 * (age_max - age_min) + age_min
    backscaled_age_int = int(backscaled_age)
    print("Обратно масштабированный возраст:", backscaled_age_int)

# Обратное масштабирование для возраста
    backscaled_age = 0.27 * (age_max - age_min) + age_min
    backscaled_age_int = int(backscaled_age)
    print("Обратно масштабированный возраст:", backscaled_age_int)

# Обратное масштабирование для возраста
    backscaled_age = 0.69 * (age_max - age_min) + age_min
    backscaled_age_int = int(backscaled_age)
    print("Обратно масштабированный возраст:", backscaled_age_int)

    # Обратное масштабирование для зарплаты
    backscaled_salary = 0.41 * (salary_max - salary_min) + salary_min
    print("Обратно масштабированная зарплата:", backscaled_salary)

# Обратное масштабирование для зарплаты
    backscaled_salary = 0.28 * (salary_max - salary_min) + salary_min
    print("Обратно масштабированная зарплата:", backscaled_salary)

# Обратное масштабирование для зарплаты
    backscaled_salary = 0.66 * (salary_max - salary_min) + salary_min
    print("Обратно масштабированная зарплата:", backscaled_salary)


    # Обратное масштабирование для опыта
    backscaled_year = 0.29 * (year_max - year_min) + year_min
    backscaled_year_int = int(backscaled_year)
    print("Обратно масштабированный опыт:", backscaled_year_int)

# Обратное масштабирование для опыта
    backscaled_year = 0.19 * (year_max - year_min) + year_min
    backscaled_year_int = int(backscaled_year)
    print("Обратно масштабированный опыт:", backscaled_year_int)

# Обратное масштабирование для опыта
    backscaled_year = 0.62 * (year_max - year_min) + year_min
    backscaled_year_int = int(backscaled_year)
    print("Обратно масштабированный опыт:", backscaled_year_int)


    #import math
# Обратное масштабирование для опыта
    #backscaled_year = 0.29 * (year_max - year_min) + year_min
    #backscaled_year_ceiled = math.ceil(backscaled_year)
    #print("Обратно масштабированный возраст:", backscaled_year_ceiled)



    # Функция для вывода среднего значения кластера
    def print_cluster_average(cluster, age, salary, education, experience):
        st.write(f"--------------------------------------")
        st.write(f"Средние значения для кластера {cluster}:")
        st.write(f"Средний возраст: {age} года")
        st.write(f"Средняя годовая зарплата: {salary}$")
        st.write(f"Уровень образования: {education}")
        st.write(f"Средний опыт работы: {experience} лет")


    # Данные для каждого кластера
    #cluster_data = {
        #0: {"age": 42, "salary": 173500, "education": "Кандидат наук", "experience": 15},
        #1: {"age": 34, "salary": 142000, "education": "Магистр", "experience": 9},
        #2: {"age": 27, "salary": 61000, "education": "Без высшего образования", "experience": 2},
        #3:  {"age": 31, "salary": 115000, "education": "Бакалавриат", "experience": 8},
    #}
    # Вывод средних значений для каждого кластера
    #for cluster, data in cluster_data.items():
        #print_cluster_average(cluster, data["age"], data["salary"], data["education"], data["experience"])





 

     # Данные для каждого кластера
    cluster_data = {
        0: {"age": 32, "salary": 117250, "education": "Магистр", "experience": 7},
        1: {"age": 29, "salary": 88000, "education": "Бакалавриат", "experience": 4},
        2: {"age": 43, "salary": 173500, "education": "Кандидат наук", "experience": 15},
        
    }
    # Вывод средних значений для каждого кластера
    for cluster, data in cluster_data.items():
        print_cluster_average(cluster, data["age"], data["salary"], data["education"], data["experience"])
