import cv2
import mediapipe as mp
import numpy as np
import streamlit as st


st.set_page_config(page_title="Hand Tracking", page_icon="🖐️")
st.title("Hand Tracking")
st.write("Take a photo with your camera to detect hand landmarks.")


@st.cache_resource
def load_hand_tracker():
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )


camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image_bytes = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("The camera image could not be read.")
    else:
        tracker = load_hand_tracker()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = tracker.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                )
        else:
            st.info("No hands were detected. Try another photo.")

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Result")