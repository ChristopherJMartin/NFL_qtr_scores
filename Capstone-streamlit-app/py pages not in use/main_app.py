#main_app.py
import appQ1
import my_app
import streamlit as st
PAGES = {
    "All Quarters": my_app,
    "First quarter only": appQ1
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
