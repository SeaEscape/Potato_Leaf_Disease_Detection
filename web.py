import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model=tf.keras.models.load_model(r"C:\Users\PALLAVI\AICTE-Internship-files-main\Potato_Leaf_Disease_Detection\dataset\train_plant_disease_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title('Potato Disease System for Sustainable Agriculture')
app_mode=st.sidebar.selectbox('Select Page',['Home','Disease Recognition'])

from PIL import Image
import streamlit as st

img = Image.open(r'C:\Users\PALLAVI\AICTE-Internship-files-main\Potato_Leaf_Disease_Detection\dataset\d3.png')
resized_img = img.resize((150, 150))  # Width: 300px, Height: 300px

st.image(resized_img)




if(app_mode=='Home'):
    st.markdown("<h1 style='text-align:center';>Potato Disease System for Sustainable Agriculture",unsafe_allow_html=True)

elif(app_mode=='Home'):
    st.header('Plant Disease Detection System for Sustainable Agriculture')

test_image=st.file_uploader('chose an image')
if(st.button('Show Image')):
    st.image(test_image,width=300)

if(st.button('Predict')):
    st.snow()    
    st.write('Our Prediction')
    result_index=model_prediction(test_image)

    class_names=['Potato___Early_blight','Potato__Late_blight','Potato__healthy']
    st.success('Model is Predicting its a {} '.format(class_names[result_index]))