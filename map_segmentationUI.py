import streamlit as st
from PIL import Image
from mapSegmentationV2 import ImageSegmentation


# Apply custom CSS styles
def apply_custom_css():
    st.markdown(
        """
        <style>
        .custom-header {
            color: #000000;  /* Change this to your desired color */
            font-size: 2em;  /* Adjust the font size if needed */
        }
        .main {
            background-color: #eca050; /* Background color for the main area */
            color: #000000; /* Text color for main area */
        }

        .stButton>button {
            background-color: #ffffff; /* Button background color */
            color: #000000; /* Button text color */
            width: 100%; /* Make all buttons the same width */
            padding: 0.5rem; /* Add padding to buttons */
            margin-top: 10px; /* Space between buttons */
            border-radius: 5px; /* Rounded button corners */
            border: none; /* Remove button border */
        }
        .stFileUploader > label {
            color: #000000; /* Text color for file uploader label */
        }
        .st-upload {
            color: #000000; /* Text color for upload button */
        }

        /* Custom Progress Bar Styling */
        .stProgress > div {
            background-color: #06f2d7; /* Background color of the progress bar itself */
            border-radius: 5px; /* Rounded corners for the progress bar itself */
        }
        .stProgress {
            background-color: #000000; /* Background color of the progress bar track */
            border-radius: 5px; /* Rounded corners for the progress bar track */
        }
        </style>
        """,
        unsafe_allow_html=True
    )


apply_custom_css()  # Apply the CSS styles

st.title("Marvel Area Segmentation", anchor='custom-header')

st.write("\n")

# Column 1: Input image upload
st.header("Upload Input Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

st.write("\n")

# Display the uploaded image
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.header("Input Image")

    st.image(input_image, width=512)  # Fixed size 512x512

st.write("\n\n")

# Process button with spinner
if st.button("Process"):
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Simulate processing time
            sam = ImageSegmentation(model_path="Models/sam_l.pt", model_type='', image_path="input/1.png")
            seg_image = sam.segment_SAM()
            st.header("Output Image")
            # Placeholder for the output image
            output_placeholder = st.empty()
            # Display output image with fixed size
            output_image = seg_image
            output_placeholder.image(output_image, width=512)  # Fixed size 512x512
    else:
        st.warning("Please upload an input image first.")