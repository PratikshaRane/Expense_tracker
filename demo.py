import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly import graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

st.title("Welcome to Expense Tracker!!!")

rad =st.sidebar.radio("Navigation",["Home","Visualization","Prediction"])
if rad=="Home":
    st.image("expense.jpg",width=900)

    if st.checkbox("See the Dataset"):
        df = pd.read_csv("Expenses_new.csv")
        print(df)
        st.dataframe(df,width=700,height= 500)
        
if rad=="Visualization":
    vis=st.selectbox("",["Monthly","Overall"])
    if vis=="Monthly":
    
        
        data = pd.read_csv("Expenses.csv")
        v2 = st.selectbox("Select the month to see expenditure visualization",["January","February","March","April","May","June","July","August","September","October","November","December"])
        if v2=="January":
            specified=data.loc[data['Month'] == 'Jan']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of January Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="February":
            specified=data.loc[data['Month'] == 'Feb']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of february Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="March":
            specified=data.loc[data['Month'] == 'Mar']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of March Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="April":
            specified=data.loc[data['Month'] == 'April']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of April Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="May":
            specified=data.loc[data['Month'] == 'May']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of may Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="June":
            specified=data.loc[data['Month'] == 'June']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of June Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="July":
            specified=data.loc[data['Month'] == 'July']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of July Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="August":
            specified=data.loc[data['Month'] == 'Aug']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of August Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="September":
            specified=data.loc[data['Month'] == 'Sep']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of Sepetember Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="October":
            specified=data.loc[data['Month'] == 'Oct']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of October Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="November":
            specified=data.loc[data['Month'] == 'Nov']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of November Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
        if v2=="December":
            specified=data.loc[data['Month'] == 'Dec']
            x = specified['Category']
            y = specified['Expenses']
            fig1, ax1 = plt.subplots()
            ax1.pie(y, labels=x, radius=1.2,autopct='%0.01f%%', shadow=True)
            st.pyplot(fig1)
            st.write("\n\n")
            st.markdown("## Line Plot of December Month: ")
            line=plt.figure(figsize=(20,10))
            plt.xlabel("Category")
            plt.ylabel("Expense")
            plt.title("Line plot")
            plt.plot(x, y, color = 'green',
            linestyle = 'solid', marker = 'o',
            markerfacecolor = 'red', markersize = 12)
            st.pyplot(line)
    if vis=="Overall":
        st.write("""\n\n\n""")
        data = pd.read_csv("Expenses.csv")
        st.markdown(" ## Bar-Plot Visualization of all Expenses: ")
        fig=plt.figure(figsize=(20, 10))
        sns.barplot(x="Month",y="Expenses",hue="Category",data=data)
        st.pyplot(fig)
        st.write("""\n\n\n""")
        st.markdown(" ## Stacked Bar Plot Visualization of all Expenses: ")
        df = pd.read_csv("Expenses_new.csv")
        month = df['Month']
        expenses = df['Total_Ex']
        food = df['Food']
        travel = df['Travel']
        health = df['Health']
        maintenance = df['Maintenance']
        rent = df['Rent']
        other = df['Other']
        fig=plt.figure(figsize=(20,10))
        plt.bar(month, food, 0.5, label="Food")
        plt.bar(month, travel, 0.5, bottom=food, label="Travel")
        plt.bar(month, health, 0.5, bottom=(food+travel), label="Health")
        btm3 = (food+travel+health)
        plt.bar(month, maintenance, 0.5, bottom=btm3, label="Maintenance")
        btm4 = (food+travel+health+maintenance)
        plt.bar(month, rent, 0.5, bottom=btm4, label="Rent")
        btm5 = (food+travel+health+maintenance+rent)
        plt.bar(month, other, 0.5, bottom=btm5, label="Other")
        plt.xlabel("Month")
        plt.ylabel("Total_Ex")
        plt.legend()
        st.pyplot(fig)
    
   
if rad=="Prediction":
    st.image("block.webp",width=900)
    df = pd.read_csv("Expenses_new.csv")
    st.markdown(" ## Predict Your  Month Total Expense..")
    
    reg = linear_model.LinearRegression()
    X =df.drop(columns = ['Month','Total_Ex'],axis=1)
    Y = df['Total_Ex']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    training_data_prediction =regressor.predict(X_train)
    test_data_prediction =regressor.predict(X_test)
    no=st.number_input("Enter the sequence no. of month:",1,12)
    food=st.number_input("Enter expenses on food",1,15000)
    travel=st.number_input("Enter expenses on travel",1,15000)
    health=st.number_input("Enter expenses on health",1,15000)
    main=st.number_input("Enter expenses on maintainence",1,15000)
    rent=st.number_input("Enter expenses on rent",1,15000)
    other=st.number_input("Enter expenses on other",1,15000)
    input=(no,food,travel,health,rent,main,other)
    input_data_as_numpy_array = np.asarray(input)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = regressor.predict(input_data_reshaped)
    if st.button("Predict"):
        st.success(f"Predicted expense of given month is {round(prediction[0])}")
    if no==1:
        specified=df.loc[df['No'] == 1]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
        
        
    if no==2:
        specified=df.loc[df['No'] == 2]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==3:
        specified=df.loc[df['No'] == 3]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==4:
        specified=df.loc[df['No'] == 4]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==5:
        specified=df.loc[df['No'] == 5]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==6:
        specified=df.loc[df['No'] == 6]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==7:
        specified=df.loc[df['No'] == 7]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==8:
        specified=df.loc[df['No'] == 8]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==9:
        specified=df.loc[df['No'] == 9]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==10:
        specified=df.loc[df['No'] == 10]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==11:
        specified=df.loc[df['No'] == 11]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    if no==12:
        specified=df.loc[df['No'] == 12]
        x = specified['Total_Ex']
        st.write("Actual Expense of this month:",x)
    st.markdown(" ## Scatter-Plot Visualization of all Expenses: ")
    fig=plt.figure(figsize=(16,10))
    plt.scatter(Y_train, training_data_prediction, c='blue',marker='o',label='Training data')
    plt.scatter(Y_test, test_data_prediction, c='red',marker='o',label='Test data')
    plt.xlabel("Actual Expenses")
    plt.ylabel("Predicted Expenses")
    plt.title(" Actual Expenses vs Predicted Expenses")
    st.pyplot(fig)

    
    
    
    
    

   
    

