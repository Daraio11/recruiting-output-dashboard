import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Recruiting Output Dashboard", layout="wide")

st.write("✅ App booted. If you see this, Streamlit is running.")

DATA_PATH = "data/PFF_Recruiting_Performance_Delta.csv"

st.write("Looking for data at:", DATA_PATH)
st.write("File exists:", Path(DATA_PATH).exists())

if not Path(DATA_PATH).exists():
    st.error(f"Data file not found: {DATA_PATH}. Check the repo has it at /data/ with exact name.")
    st.stop()
