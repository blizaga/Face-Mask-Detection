import streamlit as st
from utils.mask_detection import MaskDetection


st.title("Mask Detection")

img_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


for n, img_file_buffer in enumerate(img_files):
  if img_file_buffer is not None:
    detection = MaskDetection(img_file_buffer)
    detect_mask = detection.detect_mask()
    
    if detect_mask is not None:
      st.image(detect_mask, channels="BGR", \
      caption=f'Detection Results ({n+1}/{len(img_files)})')
        
