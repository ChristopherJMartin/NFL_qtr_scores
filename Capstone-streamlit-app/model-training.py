import streamlit as st
from sklearn.model_selection import train_test_split
#import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
#import load_data

#load data
qwins = pd.read_csv('data/Quarterly_Wins-for_Modeling.csv')

# Clump 14
Clump_all_qs=qwins[['x0_H', 'x0_HH', 'x0_HHH', 'x0_T', 'x0_V', 'x0_VV', 'x0_VVV',
        'x1_H','x1_HH', 'x1_HHH', 'x1_T', 'x1_V', 'x1_VV', 'x1_VVV', 'x2_H', 'x2_HH',
       'x2_HHH', 'x2_T', 'x2_V', 'x2_VV', 'x2_VVV', 'x3_H', 'x3_HH', 'x3_HHH',
       'x3_T', 'x3_V', 'x3_VV', 'x3_VVV']]


#X = Clump_all_qs
X = Clump_all_qs
y = qwins['Winner']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 32221)
modelname = 'RandomForestClassifier'
model = RandomForestClassifier()
model.fit(X_train, y_train)
    #pred = model.predict(X_test)
    #score = accuracy_score(y_test, pred)
    #print(score)



# save the model
with open("rfc-model.pkl", "wb") as file:
    pickle.dump(model, file)
