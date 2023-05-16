from contextlib import contextmanager
from io import StringIO
#from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME

from threading import current_thread

import streamlit as st

from fer import Video
from fer import FER

import matplotlib.pyplot as plt
import os
import sys
import tempfile

# Menambahkan judul dan deskripsi
st.title("TUGAS BESAR PENGOLAHAN CITRA DIGITAL")
st.title("Kelompok 5 - The Saham")
st.write("Mochamad Hafidh Dwyanto - 211511043")
st.write("Muchamad Diaz Adhari    - 211511044")
st.write("Uqyanzie Bintang KFF    - 211511062")
st.write("Challenges in Representation Learning: A report on three machine learning contests")

uploaded_file = st.file_uploader(
    "Silahkan Upload Video disini", type=["mp4", "mkv", "mov"], accept_multiple_files=False)

st.video(uploaded_file)

if uploaded_file is None:
    st.warning("Please upload a video file.")
else:
    with st.spinner('Wait for it...'):

        detector = FER(mtcnn=True)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
            video = Video(video_path)
        if not os.path.exists(video_path):
            st.warning("Video file not found.")
        else:
            raw_data = video.analyze(detector, display=False)
            # continue with analysis
        os.unlink(video_path)  # delete temporary file
    st.success('Done!')
    df = video.to_pandas(raw_data)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    # st.image(fig)
    st.write(df)
    st.line_chart(df)