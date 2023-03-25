import streamlit as st
import os

# Use the file_browser widget to choose a directory
directory = st.file_browser()

# Display the selected directory
if directory is not None:
    st.write(f"Selected directory: {directory}")
