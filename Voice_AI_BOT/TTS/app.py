import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from voxcpm import VoxCPM
import soundfile as sf
import tempfile

st.title("🎤 AI Voice Generator (VoxCPM)")

# load model (first time slow untundi)
@st.cache_resource
def load_model():
    return VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

model = load_model()

# user input
text = st.text_area("Enter text", "Hello, how are you?")

if st.button("Generate Voice"):
    with st.spinner("Generating..."):
        wav = model.generate(
            text=text,
            cfg_value=2.0,
            inference_timesteps=10
        )

        # save temp file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_file.name, wav, model.tts_model.sample_rate)

        st.audio(tmp_file.name)
        st.success("Done!")