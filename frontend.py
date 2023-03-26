import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import tensorflow as tf
from predict import model_prediction,set_data
from damage import damage_percentage


st.header("SAS ML")

with st.sidebar:
    st.title("Navigation")
    option = st.radio("Select an option", ["Insert images", "Flood prediction", "Damage percentage", "Areas analogy"])

if option == "Insert images":
    st.write("Insert images of the area")
    folder = 'TimewiseCSV'
    file_list = os.listdir(folder)
    selected_files = st.multiselect('Select file(s)', file_list)
    if selected_files:
        for filename in selected_files:
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r') as file:
                contents = file.read()

if option == "Flood prediction":
    st.write("Predicting whether a flood occurred or not")    
    folder = 'TimewiseCSV\\'
    file_list = os.listdir(folder)
    selected_files = st.multiselect('Select file(s)', file_list)
    a=[]
    if selected_files:
        for filename in selected_files:
            file_path = os.path.join(folder, filename)
            data=set_data(file_path)
            for images,labels in data:
                for k in range(len(images)):
                    try:
                        p=model_prediction(images[k])
                        a.append(p)
                        if p==1:
                            base="sen12flood\\sen12floods_s1_labels\\sen12floods_s1_labels\\"
                            damage=damage_percentage()
                    except:
                        pass

if st.button("Predict"):
    st.success("rain prediction")
if option == "Damage percentage":
    st.write("Computing damage percentage of an area")
    
if option == "Areas analogy":
    st.write("Comparing different regions")
  