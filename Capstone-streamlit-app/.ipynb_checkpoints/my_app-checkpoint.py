from numpy.core.fromnumeric import reshape
import streamlit as st
import pickle
import numpy as np
import pandas as pd



st.title("Predicting NFL Game outcomes")
st.header("Can you predict a game winner just by looking at the quarterly outcomes?")
#st.markdown("----------------------------------------")

# load model

with open('rfc-model.pkl', 'rb') as file:
    model = pickle.load(file)

#def loaddata(x):
data = pd.read_csv('data/Quarterly_Wins-for_Modeling.csv')
data_raw = pd.read_csv('data/QuarterlyWins.csv')

# random row
st.markdown("Pick an NFL game from the dataset:")
random_row = st.slider('', min_value = 0, max_value = 5323, step = 1)

# button click prediction
st.markdown("### Will the model predict the outcome? ")
click = st.button("Predict")

if click:

    random_obs = data.iloc[random_row,:28]
    winner_code = data_raw.iloc[random_row,24]
    st.subheader(f'Game #: {random_row}')
    #st.subheader(f' The quarter codes are: {data_raw.iloc[random_row,19:23]}')
    st.subheader(f'This game took place in Week {data_raw.iloc[random_row,3]} of the {data_raw.iloc[random_row,11]} season.')
    st.subheader(f'It featured the {data_raw.iloc[random_row,6]} playing at the {data_raw.iloc[random_row,12]}.')

    if winner_code == 'HW':
        st.header(f'This game was won by the home team by {-(data_raw.iloc[random_row,23])} points.')
    elif winner_code == "VW":
        st.header(f'This  game was won by the visiting team by {data_raw.iloc[random_row,23]} points.')
    else:
        st.header(f'This actual ended deadlocked after four quarters of play.')
    #st.header(f'The winner code is: {data_raw.iloc[random_row,24]}')
    #st.subheader(f'It is a: {type(random_obs)}')
    #st.subheader(f'Its length is: {len(random_obs)}')


# button click prediction
#st.markdown("### Let's see what we got")
#click = st.button("Let's see what we got")

#if click:

    #st.header(f'Here we go!')
    # prints code - good for testing
    #st.subheader(f'X has: {len(X)}')
    all_features = np.array(random_obs)
    #all_features = pd.DataFrame(random_obs)
    #st.subheader(f'allfeatures is a: {type(all_features)}')
    prediction = model.predict(all_features.reshape(1, -1))
    #prediction

    if prediction == 1:
        st.header(f'The model predicts that the visiting team will win.')
    elif prediction == -1:
        st.header(f'The model predicts that the home team will win.')
    else:
        st.header(f'The model predicts a tie.')


    if ((prediction == -1) and (winner_code == 'HW')):
        st.title(f'The model was RIGHT!')
    elif ((prediction == 1) and (winner_code == 'VW')):
        st.title(f'The model was RIGHT!')
    elif ((prediction == 0) and (winner_code == 'T')):
        st.title(f'The model was RIGHT!')
    else:
        st.title(f'The model Was WRONG! Darn it.')
    st.markdown("### Move the slider bar to choose another game. ")
#.reshape(1, -1))
#    random_row = np.random.randint(0,data.shape[0],1)
#    random_obs = data.iloc[random_row,:]

# Q1 = st.number_input('Input Q1 value:')
# Q2 = st.number_input('Input Q2 value:')
# Q3 = st.number_input('Input Q3 value:')
# Q4 = st.number_input('Input Q4 value:')
# user_values = np.array([Q1,Q2,Q3,Q4])
#
# RQlist = []
# for val in user_values:
#     if val == -3:
#         RQ = 'HHH'
#     elif val == -2:
#         RQ = 'HH'
#     elif val == -1:
#         RQ = 'H'
#     elif val == 0:
#         RQ = 'T'
#     elif val == 1:
#         RQ = 'V'
#     elif val == 2:
#         RQ = 'VV'
#     elif val == 3:
#         RQ = 'VVV'
#     RQlist.append(RQ)
# #inputs = np.array(RQ[0],RQ[1],RQ[2],RQ[3])



#with st.echo():
    # prints code - good for testing
#    prediction = model.predict(RQlist)#.reshape(1, -1))
#    prediction

#st.write(type(prediction))
#st.write('##')

#st.header(f'The model predicts: {prediction[0]}')

#st.balloons()

# to make columns
#col1, col2, col3 = st.beta_columns(3)
#with col1:
#    'I am printing things'
#with col2:
#    df_iris
#with col3:
#    st.subheader("cool stuff")
