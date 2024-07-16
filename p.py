import streamlit as st
import pandas as pd
import seaborn as sns
import io
import sys
import matplotlib.pyplot as plt
import pickle
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



prognoz = pd.read_csv('Prog.csv')


def show_p(): 
    st.title('Прогнозирование ввод') 
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
    print("Лучшие параметры Gradient Boosting: ", grid_result_gb.best_params_)

# Обучение модели Gradient Boosting
    best_model_gb = GradientBoostingRegressor(**grid_result_gb.best_params_)
    best_model_gb.fit(X_train, y_train)

# Сохранение модели в файл
    with open('best_model_gb.pkl', 'wb') as file:
        pickle.dump(best_model_gb, file)

# Получение прогнозов для модели Gradient Boosting
    predictions_gb = best_model_gb.predict(X_test)

# Оценка точности с помощью R2 Score
    r2_gb = r2_score(y_test, predictions_gb)
    print(f"Точность прогнозирования (R2 Score) Gradient Boosting: {r2_gb}")

# Создание DataFrame с реальными и прогнозируемыми значениями
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions_gb})


    def predict_salary(education_level, job_title, years_experience):
        with open('best_model_gb.pkl', 'rb') as file:
            best_model = pickle.load(file)
        return best_model.predict([[education_level, job_title, years_experience]])
    
    st.title('Прогноз зарплаты')
    st.write('Введите значения для прогнозирования зарплаты:')





    job_titles = {
    "0":"Account Manager",
"1":"Accountant",
"2":"Administrative Assistant",
"3":"Back end Developer",
"4":"Business Analyst",
"5":"Business Development Manager",
"6":"Business Intelligence Analyst",
"7":"CEO",
"8":"Chief Data Officer",
"9":"Chief Technology Officer",
"10":"Content Marketing Manager",
"11":"Copywriter",
"12":"Creative Director",
"13":"Customer Service Manager",
"14":"Customer Service Rep",
"15":"Customer Service Representative",
"16":"Customer Success Manager",
"17":"Customer Success Rep",
"18":"Data Analyst",
"19":"Data Entry Clerk",
"20":"Data Scientist",
"21":"Delivery Driver",
"22":"Digital Content Producer",
"23":"Digital Marketing Manager",
"24":"Digital Marketing Specialist",
"25":"Director",
"26":"Director of Business Development",
"27":"Director of Data Science",
"28":"Director of Engineering",
"29":"Director of Finance",
"30":"Director of HR",
"31":"Director of Human Capital",
"32":"Director of Human Resources",
"33":"Director of Marketing",
"34":"Director of Operations",
"35":"Director of Product Management",
"36":"Director of Sales",
"37":"Director of Sales and Marketing",
"38":"Event Coordinator",
"39":"Financial Advisor",
"40":"Financial Analyst",
"41":"Financial Manager",
"42":"Front End Developer",
"43":"Front end Developer",
"44":"Full Stack Engineer",
"45":"Graphic Designer",
"46":"HR Generalist",
"47":"HR Manager",
"48":"Help Desk Analyst",
"49":"Human Resources Coordinator",
"50":"Human Resources Director",
"51":"Human Resources Manager",
"52":"IT Manager",
"53":"IT Support",
"54":"IT Support Specialist",
"55":"Junior Account Manager",
"56":"Junior Accountant",
"57":"Junior Advertising Coordinator",
"58":"Junior Business Analyst",
"59":"Junior Business Development Associate",
"60":"Junior Business Operations Analyst",
"61":"Junior Copywriter",
"62":"Junior Customer Support Specialist",
"63":"Junior Data Analyst",
"64":"Junior Data Scientist",
"65":"Junior Designer",
"66":"Junior Developer",
"67":"Junior Financial Advisor",
"68":"Junior Financial Analyst",
"69":"Junior HR Coordinator",
"70":"Junior HR Generalist",
"71":"Junior Marketing Analyst",
"72":"Junior Marketing Coordinator",
"73":"Junior Marketing Manager",
"74":"Junior Marketing Specialist",
"75":"Junior Operations Analyst",
"76":"Junior Operations Coordinator",
"77":"Junior Operations Manager",
"78":"Junior Product Manager",
"79":"Junior Project Manager",
"80":"Junior Recruiter",
"81":"Junior Research Scientist",
"82":"Junior Sales Associate",
"83":"Junior Sales Representative",
"84":"Junior Social Media Manager",
"85":"Junior Social Media Specialist",
"86":"Junior Software Developer",
"87":"Junior Software Engineer",
"88":"Junior UX Designer",
"89":"Junior Web Designer",
"90":"Junior Web Developer",
"91":"Juniour HR Coordinator",
"92":"Juniour HR Generalist",
"93":"Marketing Analyst",
"94":"Marketing Coordinator",
"95":"Marketing Director",
"96":"Marketing Manager",
"97":"Marketing Specialist",
"98":"Network Engineer",
"99":"Office Manager",
"100":"Operations Analyst",
"101":"Operations Director",
"102":"Operations Manager",
"103":"Principal Engineer",
"104":"Principal Scientist",
"105":"Product Designer",
"106":"Product Manager",
"107":"Product Marketing Manager",
"108":"Project Engineer",
"109":"Project Manager",
"110":"Public Relations Manager",
"111":"Receptionist",
"112":"Recruiter",
"113":"Research Director",
"114":"Research Scientist",
"115":"Sales Associate",
"116":"Sales Director",
"117":"Sales Executive",
"118":"Sales Manager",
"119":"Sales Operations Manager",
"120":"Sales Representative",
"121":"Senior Account Executive",
"122":"Senior Account Manager",
"123":"Senior Accountant",
"124":"Senior Business Analyst",
"125":"Senior Business Development Manager",
"126":"Senior Consultant",
"127":"Senior Data Analyst",
"128":"Senior Data Engineer",
"129":"Senior Data Scientist",
"130":"Senior Engineer",
"131":"Senior Financial Advisor",
"132":"Senior Financial Analyst",
"133":"Senior Financial Manager",
"134":"Senior Graphic Designer",
"135":"Senior HR Generalist",
"136":"Senior HR Manager",
"137":"Senior HR Specialist",
"138":"Senior Human Resources Coordinator",
"139":"Senior Human Resources Manager",
"140":"Senior Human Resources Specialist",
"141":"Senior IT Consultant",
"142":"Senior IT Project Manager",
"143":"Senior IT Support Specialist",
"144":"Senior Manager",
"145":"Senior Marketing Analyst",
"146":"Senior Marketing Coordinator",
"147":"Senior Marketing Director",
"148":"Senior Marketing Manager",
"149":"Senior Marketing Specialist",
"150":"Senior Operations Analyst",
"151":"Senior Operations Coordinator",
"152":"Senior Operations Manager",
"153":"Senior Product Designer",
"154":"Senior Product Development Manager",
"155":"Senior Product Manager",
"156":"Senior Product Marketing Manager",
"157":"Senior Project Coordinator",
"158":"Senior Project Engineer",
"159":"Senior Project Manager",
"160":"Senior Quality Assurance Analyst",
"161":"Senior Research Scientist",
"162":"Senior Researcher",
"163":"Senior Sales Manager",
"164":"Senior Sales Representative",
"165":"Senior Scientist",
"166":"Senior Software Architect",
"167":"Senior Software Developer",
"168":"Senior Software Engineer",
"169":"Senior Training Specialist",
"170":"Senior UX Designer",
"171":"Social Media Man",
"172":"Social Media Manager",
"173":"Social Media Specialist",
"174":"Software Developer",
"175":"Software Engineer",
"176":"Software Engineer Manager",
"177":"Software Manager",
"178":"Software Project Manager",
"179":"Strategy Consultant",
"180":"Supply Chain Analyst",
"181":"Supply Chain Manager",
"182":"Technical Recruiter",
"183":"Technical Support Specialist",
"184":"Technical Writer",
"185":"Training Specialist",
"186":"UX Designer",
"187":"UX Researcher",
"188":"VP of Finance",
"189":"VP of Operations",
"190":"Web Developer"
}

   

    education_levels = {
    0: "Bachelor's Degree",
    1: "High School",
    2: "Master's Degree",
    3: "PhD"
    # Добавьте остальные значения здесь
}

    selected_education_level = next(key for key, value in education_levels.items() if value == education_level)
    selected_job_title = next(key for key, value in job_titles.items() if value == job_title)
    years_experience = st.slider('Количество лет опыта', min_value=0, max_value=34, value=1)

    if st.button('Прогнозировать'):
        prediction = predict_salary(selected_education_level, selected_job_title, years_experience)
        st.write(f'Прогнозируемая зарплата: {prediction[0]}')