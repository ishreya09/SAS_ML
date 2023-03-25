import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import datetime
import os

st.header("SAS ML")

with st.sidebar:
    st.title("Navigation")
    option = st.radio("Select an option", ["Insert images", "Flood prediction", "Damage percentage","Areas analogy"])

if option == "Insert images":
    st.write("Insert images of the area")

    label_name1 = st.text_input("Enter label name",key='1')
    if label_name1 != "":
        st.write("Label:", label_name1)
        
    label_name2 = st.text_input("Enter label name",key='2')
    if label_name2 != "":
        st.write("Label:", label_name2)

    label_name3 = st.text_input("Enter label name",key='3')
    if label_name3 != "":
        st.write("Label:", label_name3)

    label_name4 = st.text_input("Enter label name",key='4')
    if label_name4 != "":
        st.write("Label:", label_name4)
    
    label_name5 = st.text_input("Enter label name", key='5')
    if label_name5 != "":
        st.write("Label:", label_name5)

if option == "Flood prediction":
    st.write("Predicting whether a flood occurred or not")
    data=pd.read_csv("flood_s1.csv")
    if st.button("Predict"):
        st.success(f"rain prediction")
        fig = px.imshow(data.corr())
        fig.update_layout(title="Correlation Heatmap")
        fig.update_xaxes(title="Variables")
        fig.update_yaxes(title="Variables")
        st.plotly_chart(fig)

if option == "Damage percentage":
    st.write("Computing damage percentage of an area")

if option == "Areas analogy":
    st.write("Comparing different regions")
