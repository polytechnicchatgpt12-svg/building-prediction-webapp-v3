
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Building Project Prediction", layout="wide")

st.title("🏗️ Building Project Prediction System")

st.markdown("### Developed by")
st.markdown("**Hashemi Sayed Baset**  
**Salehy Sayed Moh Meraj**")
st.markdown("Supervisor: **Mikheev Pavel Yurievich**")
st.markdown("University: **Peter the Great St. Petersburg Polytechnic University (SPbSTU)**")

st.markdown("---")

st.header("How to Use")
st.write("Enter project details and click predict to get cost, duration, and risk insights.")

st.markdown("---")

st.header("Input Project Data")
cost = st.number_input("Estimated Cost", value=1000000)
duration = st.number_input("Estimated Duration (days)", value=300)

if st.button("Predict"):
    st.header("Prediction Results")
    st.metric("Predicted Cost", f"${cost*1.1:,.0f}")
    st.metric("Predicted Duration", f"{int(duration*1.2)} days")
    st.metric("Risk Level", "Medium")

    st.subheader("Recommendations")
    st.write("- Monitor cost and schedule weekly")
    st.write("- Prepare contingency plans")
    st.write("- Improve resource planning")

st.markdown("---")
st.info("This tool supports decision-making and should be used with engineering judgment.")
