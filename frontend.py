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
    option = st.radio("Select an option", ["Flood prediction", "Damage percentage", "Areas analogy"])

a=[]
damage=[]
label=[]
if option == "Flood prediction":
    st.write("Predicting whether a flood occurred or not")    
    folder = 'TimewiseCSV\\'
    file_list = os.listdir(folder)
    selected_files = st.multiselect('Select file(s)', file_list)
    
    if selected_files:
        for filename in selected_files:
            file_path = os.path.join(folder, filename)
            df=pd.read_csv(file_path)
            data=set_data(df)
            for images,labels in data:
                label=labels
                for k in range(len(images)):
                    try:
                        p=model_prediction(images[k])
                        a.append(p) 
                        # print(df.iloc(k)['image_dir'])
                        # if int(p)==1:
                        base=df['image_dir'][k]
                        print("df",df['image_dir'].iloc(k))
                        d=damage_percentage(base)
                        print("h",d)
                        damage.append(d)
                        st.write(d)

                    except:
                        pass

if st.button("Predict"):
    st.success(a)
    st.success(label)
if option == "Damage percentage":
    st.write(damage)
    
if option == "Areas analogy":
    st.write("Comparing different regions")
  