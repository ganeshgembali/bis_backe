import streamlit as st

st.title("Backend Server Status")
st.write("Your FastAPI application server is up and running successfully!")
st.info("You can point your API requests directly to this URL.")

# This imports and wraps your existing FastAPI instance invisibly into Streamlit
from backend.main import app as asgi_app

# Streamlit hooks up the web server routing under the hood here
