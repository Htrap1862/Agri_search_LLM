import streamlit as st
import os
import io
import base64
from gtts import gTTS
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from src.ingest import process_and_add_to_db
from src.agent import get_agri_agent
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriBridge: Multilingual AI", layout="wide", page_icon="🌾")

# Ensure directories exist
os.makedirs("data", exist_ok=True)

# Session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# --- AUDIO FUNCTIONS ---

def speech_to_text(audio_bytes, lang_code):
    recognizer = sr.Recognizer()

    try:
        if not audio_bytes or len(audio_bytes) < 100:
            return None

        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)

        # Try selected language first
        try:
            text = recognizer.recognize_google(audio_data, language=lang_code)
        except:
            # Fallback to English if it fails
            text = recognizer.recognize_google(audio_data, language="en-IN")

        return text

    except sr.UnknownValueError:
        st.warning("⚠️ Couldn't understand audio. Try again.")
        return None
    except Exception as e:
        st.error(f"❌ Audio Error: {str(e)}")
        return None


def clean_text_for_tts(text):
    # Remove markdown symbols
    text = re.sub(r'\*+', '', text)  # remove *
    text = re.sub(r'- ', '', text)  # remove bullet dashes
    text = re.sub(r'\n+', '. ', text)  # replace newlines with pauses
    text = re.sub(r'\s+', ' ', text)  # clean extra spaces

    return text.strip()

def text_to_speech(text, lang_code):
    try:
        clean_text = clean_text_for_tts(text)
        tts = gTTS(text=clean_text, lang=lang_code)
        tts.save("response.mp3")

        with open("response.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            audio_html = f"""
                <audio src="data:audio/mp3;base64,{b64}" controls autoplay style="width: 100%;">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

        os.remove("response.mp3")

    except Exception as e:
        st.error(f"TTS Error: {e}")


# --- SIDEBAR ---

with st.sidebar:
    st.header("🌍 Language & Settings")

    lang_display = st.selectbox(
        "Preferred Language",
        ["English", "Hindi", "Tamil", "Telugu", "Gujarati", "Bengali", "Malayalam", "Marathi"]
    )

    lang_map = {
        "English": "en-IN",
        "Hindi": "hi-IN",
        "Tamil": "ta-IN",
        "Telugu": "te-IN",
        "Gujarati": "gu-IN",
        "Bengali": "bn-IN",
        "Malayalam": "ml-IN",
        "Marathi": "mr-IN",
    }

    selected_lang = lang_map[lang_display]

    st.divider()
    st.header("📂 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload Aadhar or Scheme Rules",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )

    if st.button("🚀 Process & Index Files"):
        if uploaded_files:
            with st.spinner("Indexing your documents..."):
                for uploaded_file in uploaded_files:
                    with open(os.path.join("data", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                process_and_add_to_db()
                st.success("Successfully Indexed!")
        else:
            st.warning("Please upload a file first.")


# --- MAIN UI ---

st.title("🌾 AgriBridge: Your Multilingual Farm Assistant")
st.write(f"Currently assisting in: **{lang_display}**")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- VOICE INPUT SECTION ---

st.write("---")
col1, col2 = st.columns([1, 4])

with col1:
    st.write("🎤 Speak")

    voice_data = mic_recorder(
        start_prompt="Start",
        stop_prompt="Stop",
        key="recorder"
    )

# --- HANDLE VOICE INPUT ---

if voice_data:
    with st.spinner("Transcribing..."):
        transcript = speech_to_text(voice_data["bytes"], selected_lang)

        if transcript:
            st.session_state.voice_text = transcript
            st.success(f"🗣️ {transcript}")


# --- TEXT INPUT (auto-filled with voice) ---

text_prompt = st.chat_input(
    "Type or use voice...",
    key="chat_input"
)

# If voice input exists, override text input ONCE
if st.session_state.voice_text:
    user_input = st.session_state.voice_text
    st.session_state.voice_text = ""  # reset after use
else:
    user_input = text_prompt


# --- AGENT EXECUTION ---

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        agent = get_agri_agent()

        if not agent:
            st.error("Please upload documents first!")
        else:
            with st.spinner("Analyzing..."):
                instruction = f"""
                User query:
                {user_input}

                IMPORTANT:
                - Respond ONLY in {lang_display}
                - Do NOT switch language based on location
                - Even if user is from Gujarat, respond in {lang_display}

                Follow system rules strictly.
                """

                result = agent.invoke({"messages": [("user", instruction)]})

                raw_content = result["messages"][-1].content

                if isinstance(raw_content, list):
                    response_text = raw_content[0].get("text", "No text found")
                else:
                    response_text = raw_content

                st.markdown(response_text)

                # 🔊 Voice output
                st.write(f"🔊 Speaking in {lang_display}...")



                text_to_speech(response_text, selected_lang.split("-")[0])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })