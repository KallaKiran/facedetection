from retinaface import RetinaFace
import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.header('AI&DS::RCEE')
st.title('Faces Detector')

image = st.file_uploader('Take any image')

if image:
    img = Image.open(image)
    st.image(img)
    img = np.array(img)
    obj = RetinaFace.detect_faces(img)
    c = len(obj) 
    st.write(' The number of faces detected are ' + str(c)) 

    #facial_area = {}
    for key in obj.keys():
        identity = obj[key]
        facial_area = identity['facial_area']
        cv2.rectangle(img,(facial_area[2],facial_area[3]),(facial_area[0],facial_area[1]),(255,255,0),5)
    st.image(img)
    
