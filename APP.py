import streamlit as st
import pickle
from PIL import Image
import xgboost as xgb
def main():
    st.title('Flame Extinction - Status Prediction')
    image = Image.open('flame_extinguish.jpeg')
    st.image(image, width=700)
    st.markdown("""
        This application predicts the status of flame extinction based on sound wave fire-extinguishing system experiments.
        """, unsafe_allow_html=True)

    st.markdown("""
    **How to use this app:**

    1. Enter the fuel container size (in cm) in the "SIZE" field.
    2. Select the fuel type from the "FUEL" dropdown menu.
    3. Enter the distance (in cm) in the "DISTANCE" field.
    4. Enter the decibel level in the "DESIBEL" field.
    5. Enter the airflow (in m/s) in the "AIRFLOW" field.
    6. Enter the frequency (in Hz) in the "FREQUENCY" field.
    7. Click the "STATUS" button to predict the flame extinction status.
    """, unsafe_allow_html=True)



    model = pickle.load(open('Flame_Extinction_model.sav','rb'))
    scaler = pickle.load(open('MinMaxScaler.sav','rb'))
    size=st.text_input("SIZE",'')
    fuel_type=st.radio("FUEL",["***gasoline:0***","***kerosene:1***","***lpg:2***","***thinner:3***"])
    if fuel_type=='gasoline':
        fuel=0
    elif fuel_type == 'kerosene':
        fuel=1
    elif fuel_type=='lpg':
        fuel=2
    else:
        fuel=3
    
    dist=st.text_input("DISTANCE",'')
    des=st.text_input("DESIBEL",'')
    air=st.text_input("AIRFLOW",'')
    fre=st.text_input("FREQUENCY",'')
    pred = st.button('STATUS')
    if pred:
        if size and fuel and dist and des and air and fre:
            prediction = model.predict(
                scaler.transform([[float(size), fuel, float(dist), float(des), float(air), float(fre)]]))
            if prediction == 0:
                st.write("The flame is not extinguished")
            else:
                st.write("The flame is extinguished")
        else:
            st.write("Please enter all values before making a prediction")
    


main()
