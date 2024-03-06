import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

def colorize(Y_original, Y_channel,gen,w,h):
    generated_UV = gen(tf.reshape(Y_channel, (1,256,256,1)))
    generated_UV = tf.image.resize(generated_UV,(h,w))
    # st.text(generated_UV.shape)
    # st.text(Y_original.shape)
    colorized_image = create_rgb_image(Y_original, tf.reshape(generated_UV,(h,w,2)))
    colorized_image = colorized_image*255
    colorized_image = Image.fromarray(np.uint8(colorized_image))
    return colorized_image
# Function to create YUV image from Y, U, and V channels and then convert to rgb
def create_rgb_image(y_channel, uv_channel):
    # Assuming YUV format where Y, U, and V channels are separate
    # y_channel = np.array(y_channel)
    # uv_channel = np.array(uv_channel)
    u_channel, v_channel = tf.split(uv_channel, num_or_size_splits=2, axis=-1)

    # Combine Y, U, and V channels into a YUV image
    rgb_image = tf.image.yuv_to_rgb(tf.concat([y_channel, u_channel, v_channel], axis=-1))
    return rgb_image



# Main function to run the Streamlit app
def main():
    st.title('Image Colorization App')

    # File uploader for users to upload grayscale images
    uploaded_file = st.file_uploader("Upload grayscale image", type=["jpg", "png", "jpeg","webp"])

    if uploaded_file is not None:
        # Display the grayscale image
        grayscale_image = Image.open(uploaded_file)
        w, h = grayscale_image.size
        st.image(grayscale_image, caption='Uploaded Grayscale Image', use_column_width='auto')
        grayscale_image = Image.open(uploaded_file)
        img_to_tensor = tf.convert_to_tensor(grayscale_image)/255
        yuv_image = tf.image.rgb_to_yuv(img_to_tensor)
        # Separate LAB into L and AB components
        Y_channel = tf.expand_dims(yuv_image[:, :, 0], axis=-1)  # L channel
        # UV_channels = yuv_image[:, :, 1:]  # AB channels
        gen = load_model('./gen.h5')
        colorized_image = colorize(Y_channel,tf.image.resize(Y_channel,(256,256)),gen,w,h)
        # Colorize the grayscale image
        if st.button('Colorize'):
            # for col in st.columns(4):
            #     col.image(colorized_images, width=150, use_column_width='auto')
            # st.image(colorized_images,  use_column_width=True)
            st.image(colorized_image, caption='Colorized Image', use_column_width='auto')
if __name__ == "__main__":
    main()
