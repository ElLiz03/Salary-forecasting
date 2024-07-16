import pandas as pd
import numpy as np
import plotly.express as px
from plotly.offline import iplot
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import io
import sys
import joblib
import streamlit as st
import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv('Salary_Data.csv')


# Explore Page 
def show_data():    
    st.title('Обработка и визуализация датасета') 
    global df
    buffer = io.StringIO()
    # Сохраняем текущий поток stdout
    old_stdout = sys.stdout
# Перенаправляем stdout в наш буфер
    sys.stdout = buffer
# Вызываем info(), вывод пойдет в буфер
    df.info()
# Возвращаем stdout в его обычное состояние
    sys.stdout = old_stdout
# Получаем вывод из буфера
    info_text = buffer.getvalue()
    st.subheader('Информация о датасете')
    st.text(info_text)
    st.subheader('Описание датасета') 
    st.write(df.describe())


    # Указание индексов строк, которые нужно удалить
    rows_to_delete = [259, 4633, 1890, 2654]
    # Удаление указанных строк
    df = df.drop(rows_to_delete)

    
    st.subheader('Проверка наличия пропущенных значений') 
    st.write(df.isnull().sum())
    df[df['Salary'].isna()]
    df[df['Education Level'].isna()]
    df.dropna(inplace=True)
    df['Gender'] = df['Gender'].replace('Other', 'Male')
    # Замена эквивалентных значений в столбце 'Education Level'
    education_mapping = {
    "Bachelor's": "Bachelor's Degree",
    "Master's": "Master's Degree",
    "phD": "PhD"
}

    df['Education Level'] = df['Education Level'].replace(education_mapping) 
    #Гистгорамма зарплат
    st.subheader('Гистограмма зарплат') 
    fig, ax = plt.subplots()
    ax.hist(df['Salary'], bins=30, edgecolor='black')
    ax.set_xlabel('Зарплата')
    ax.set_ylabel('Частота')
    st.pyplot(fig)

    # Первая визуализация - уникальные значения переменной "Job Title" 
    st.subheader('Уникальные значения переменной "Job Title"') 
    unique_job = df['Job Title'].unique() 
    st.write(unique_job) 
    # Количество значений в переменной "Education Level" 
    st.subheader('Количество значений в переменной "Education Level"') 
    education_counts = df['Education Level'].value_counts() 
    st.bar_chart(education_counts) 
 
# Топ 10 должностей 
    st.subheader('Топ 10 должностей') 
    st.bar_chart(df['Job Title'].value_counts()[:10]) 
 
# Количество значений в переменной "Years of Experience" 
    st.subheader('Количество значений в переменной "Years of Experience"') 
    year_counts = df['Years of Experience'].value_counts() 
    st.bar_chart(year_counts) 

 
# Процентное соотношение мужчин и женщин 
    st.subheader('Процентное соотношение мужчин и женщин') 
    gender_counts = df['Gender'].value_counts(normalize=True) * 100 
    st.bar_chart(gender_counts)

    # Столбчатые диаграммы для анализа связи факторов и зарплаты 
    st.subheader('Связь между отдельным фактором и Зарплатой') 
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7)) 
    xfactor = "Gender" 
    g = sns.boxplot(x=xfactor, y="Salary", data=df, ax=axes[0, 0], order=df.groupby(xfactor)['Salary'].median().sort_values().index) 
    g.set(title='Gender', xlabel=None) 
    xfactor = "Education Level" 
    g = sns.boxplot(x=xfactor, y="Salary", data=df, ax=axes[0, 1], order=df.groupby(xfactor)['Salary'].median().sort_values().index) 
    g.set(title='Education Level', xlabel=None) 
    xfactor = "Age" 
    plt.xticks(rotation=90) 
    g = sns.boxplot(x=xfactor, y="Salary", ax=axes[1, 0], data=df) 
    g.set(title='Age', xlabel=None) 
    xfactor = "Years of Experience" 
    plt.xticks(rotation=90) 
    g = sns.boxplot(x=xfactor, y="Salary", ax=axes[1, 1], data=df) 
    g.set(title='Years of Experience', xlabel=None) 
    st.pyplot(plt)

    # Вычисление квартилей
    Q1 = df['Salary'].quantile(0.25)
    Q3 = df['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Salary'] >= lower_bound) & (df['Salary'] <= upper_bound)]
    Q1 = df['Age'].quantile(0.25)
    Q3 = df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]
    Q1 = df['Years of Experience'].quantile(0.25)
    Q3 = df['Years of Experience'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Years of Experience'] >= lower_bound) & (df['Years of Experience'] <= upper_bound)]





    def get_average_salaries_by_position(df):
        average_salaries = df.groupby("Job Title")["Salary"].mean()
        formatted_salaries = average_salaries.apply(lambda x: f"${x:,.2f}")
        for position, formatted_salary in formatted_salaries.items():
            st.write(f"Средняя зарплата для {position}: {formatted_salary}")
    st.write(get_average_salaries_by_position(df))


df.to_csv('Clean_Data.csv', index=False)  # Сохранение в формате CSV без сохранения индексов