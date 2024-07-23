import streamlit as st
import pickle
from PIL import Image
import xgboost as xgb
def main():
    st.title('Flame Extinction Status Prediction')
    image = Image.open('flame_extinguish.jpeg')
    st.image(image, width=800)



    model = pickle.load(open('Flame_Extinction_model.sav','rb'))
    scaler = pickle.load(open('MinMaxScaler.sav','rb'))
    #SIZE  FUEL  DISTANCE  DESIBEL   AIRFLOW   FREQUENCY  STATUS
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
    # fuel=st.text_input("FUEL",'')
    dist=st.text_input("DISTANCE",'')
    des=st.text_input("DESIBEL",'')
    air=st.text_input("AIRFLOW",'')
    fre=st.text_input("FREQUENCY",'')
    pred = st.button('STATUS')
    if pred:
        prediction=model.predict(scaler.transform([[float(size),fuel,float(dist),float(des),float(air),float(fre)]]))
        if prediction==0:
            st.write("The flame is not extinguished")
        else:
            st.write("The flame is extinguished")


main()