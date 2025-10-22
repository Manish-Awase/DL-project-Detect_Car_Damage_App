import streamlit as st
from torchvision import transforms
from PIL import Image
from predict_damage import predict
st.title('Detect Car Damage')
uploaded_image=st.file_uploader("Upload an image",type=["jpg", "png"])
image_path="./temp_image.jpg"

resize=transforms.Compose([transforms.Resize(256)])

if uploaded_image :
    st.info('image uploaded')
    # st.image(uploaded_image,use_container_width=True)
    with open(image_path,'wb') as f:
        f.write(uploaded_image.getbuffer())
    img=Image.open(image_path).resize((600,350))
    st.image(img)
    # st.image(uploaded_image , caption="Uploaded File", use_container_width=True)
    # prediction
    pct,prediction_class= predict(image_path)
    st.write("damage type")
    st.success(f'class: {prediction_class}    confidence: {pct*100:0.1f}%')
