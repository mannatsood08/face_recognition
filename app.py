import streamlit as st
import pandas as pd
import time
from datetime import datetime 

ts=time.time()
timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

df=pd.read_csv("Attendance/Attendance_"+ date +".csv")
st.dataframe(df.style.highlight_max(axis=0))