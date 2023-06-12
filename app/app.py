import time
import streamlit as st
import torch
from torchvision import transforms
import requests
from utils import *


html_temp = '''
    <div style =  padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Medical Masks Detection</h1></center>
    </div>
    '''

st.markdown(html_temp, unsafe_allow_html=True)
html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Please upload an image of people to detect medical masks on their faces</h3></center>
    </div>
    '''

st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(html_temp, unsafe_allow_html=True)
opt = st.selectbox('How do you want to upload the image?\n',
                   ('Please Select', 'Upload image via link', 'Upload image from device'))

if opt == 'Upload image from device':
    img_path = st.file_uploader('Select', type=['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if img_path is not None:
        image = Test_images(transforms.Compose([transforms.ToTensor()]), img_path)

elif opt == 'Upload image via link':
    try:
        url = st.text_input('Enter the Image Address')
        image = Test_images(transforms.Compose([transforms.ToTensor()]), io.BytesIO(requests.get(url).content))
    except:
        show = st.error("Please Enter a valid Image Address!")
        time.sleep(3)
        show.empty()

try:
    if st.button('Predict'):
        model = model_detection(4)
        model.load_state_dict(torch.load('weights/model.pt', map_location=torch.device('cpu')))
        model.eval()

        prediction = model([image[0]])  

        buf_plot = plot_image(image[0], clean_boxes(prediction[0], area_boxes))
        st.image(buf_plot, use_column_width=True)
except Exception as e:
    st.info(e)
    pass
