import streamlit as st
from src.card_agent_graph import run_card_agent

st.title("Autonomous Card LLM Agent (Bedrock + LangGraph)")
user_request = st.text_input("Enter your card request:")

if st.button("Run Agent"):
    if user_request:
        result = run_card_agent(user_request)
        st.success(result)
    else:
        st.warning("Please enter a request.")
