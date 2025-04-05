import streamlit as st
import pandas as pd
import plotly.express as px
from pickle import load as pickle_load

with open('data/raw/maternal_risk_df_train.pkl', 'rb') as file:
    df = pickle_load(file)

st.title('Maternal Risk Predictor')
st.dataframe(df.sample(10, random_state=2025))
st.markdown('## Hallazgos')
st.markdown('Podemos observar que en la variable age los rangos están entre **15 y 55 años**. Es decir, contamos con embarazos juveniles y de etapa tardía.')
st.dataframe(df.describe(include='number').T)
fig = px.scatter_matrix(df, dimensions=['age','systolicbp','diastolicbp'], color='risklevel')
st.plotly_chart(fig)
df['risklevel'] = df['risklevel'].astype(str)
fig_2 = px.parallel_coordinates(df, color_continuous_scale=px.colors.diverging.Tealrose, color=df['risklevel'].map({'high risk':0,'low risk':1,'mid risk':2}))
st.plotly_chart(fig_2)
fig_3 = pd.plotting.parallel_coordinates(df.sample(90, random_state=2025), 'risklevel', color=['teal','gold','crimson'])
st.pyplot(fig_3.figure)