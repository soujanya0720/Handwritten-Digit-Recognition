
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# ---------------------------------
# Load Trained CNN Model
# ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error("Model not found. Please run train.py first.")
    st.stop()

# ---------------------------------
# Application Title
# ---------------------------------
st.title("🔢 Handwritten Digit Recognition")
st.write(
    "Upload an image of a handwritten digit. "
    "Our CNN model will predict the number from 0 to 9."
)

st.info("Supported formats: JPG, JPEG, PNG")

# ---------------------------------
# Upload Image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload your handwritten digit image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# Prediction Function
# ---------------------------------
def predict_digit(uploaded_image):

    # Convert to grayscale
    image = uploaded_image.convert("L")

    # Resize while maintaining the image content
    image = ImageOps.fit(image, (28, 28))

    # Improve contrast
    image = ImageEnhance.Contrast(image).enhance(2.0)

    # Convert to NumPy array
    image_array = np.array(image).astype("float32")

    # MNIST uses white digits on a black background.
    # Invert images when the background is mostly white.
    if np.mean(image_array) > 127:
        image_array = 255 - image_array

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch and channel dimensions
    image_array = image_array.reshape(1, 28, 28, 1)

    # Predict
    predictions = model.predict(image_array, verbose=0)

    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    return predicted_digit, confidence

# ---------------------------------
# Display Image and Prediction
# ---------------------------------
if uploaded_file is not None:

    try:
        uploaded_image = Image.open(uploaded_file)

        st.subheader("Uploaded Image")
        st.image(
            uploaded_image,
            caption="Your Handwritten Digit",
            width=250
        )

        if st.button("Predict Digit", type="primary"):

            with st.spinner("Recognizing your digit..."):

                digit, confidence = predict_digit(uploaded_image)

            st.success(f"Predicted Digit: {digit}")

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(int(confidence))

            st.write("### Prediction Details")

            st.write(f"**Recognized Number:** {digit}")
            st.write(f"**Confidence Score:** {confidence:.2f}%")

    except Exception as e:
        st.error(f"Unable to process image: {e}")

# ---------------------------------
# Footer
# ---------------------------------
st.divider()

st.caption(
    "Developed using Python, TensorFlow, CNN and Streamlit."
)

st.caption(
    "Machine Learning Internship Project - Handwritten Digit Recognition"
)
