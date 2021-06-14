# model-training-Q1.py
import streamlit as st
from sklearn.model_selection import train_test_split
#import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
#import load_data

#load data
qwins = pd.read_csv('data/Quarterly_Wins-for_Modeling.csv')

# Clump 0
Clump_all_qs=qwins[['x0_H', 'x0_HH', 'x0_HHH', 'x0_T', 'x0_V', 'x0_VV', 'x0_VVV',
        ]]


#X = Clump_all_qs
X = Clump_all_qs
y = qwins['Winner']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 32221)
modelname = 'RandomForestClassifier'
modelQ1 = RandomForestClassifier()
modelQ1.fit(X_train, y_train)
    #pred = model.predict(X_test)
    #score = accuracy_score(y_test, pred)
    #print(score)



# save the model
with open("rfc-model-Q1.pkl", "wb") as fileQ1:
    pickle.dump(modelQ1, fileQ1)
