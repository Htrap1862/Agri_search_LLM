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

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgriBridge: Multilingual AI", layout="wide", page_icon="🌾")

# Ensure directories exist
if not os.path.exists("data"):
    os.makedirs("data")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 1. AUDIO PROCESSING (The Fix for your Error) ---

def speech_to_text(audio_bytes, lang_code):
    recognizer = sr.Recognizer()
    try:
        # Check if audio_bytes is empty
        if not audio_bytes or len(audio_bytes) < 100:
            return "⚠️ Recording was too short. Please try again."

        # Convert bytes to standardized WAV
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)
            # The 'timeout' helps prevent the 'Line 1 Column 1' error
            text = recognizer.recognize_google(audio_data, language=lang_code)
            return text
    except sr.UnknownValueError:
        return "⚠️ I couldn't understand the audio. Speak clearly!"
    except Exception as e:
        return f"❌ Audio Error: {str(e)}"
def text_to_speech(text, lang_code):
    """Generates a native voice and plays it in the browser."""
    try:
        tts = gTTS(text=text, lang=lang_code)
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
        st.error(f"Text-to-Speech Error: {e}")

# --- 2. SIDEBAR: CONTROL PANEL ---

with st.sidebar:
    st.header("🌍 Language & Settings")
    lang_display = st.selectbox(
        "Preferred Language", 
        ["English", "Hindi", "Tamil", "Telugu", "Gujarati"]
    )
    
    # Map for SpeechRecognition and gTTS
    lang_map = {
        "English": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN", 
        "Telugu": "te-IN", "Gujarati": "gu-IN"
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

# --- 3. MAIN CHAT INTERFACE ---

st.title("🌾 AgriBridge: Your Multilingual Farm Assistant")
st.write(f"Currently assisting in: **{lang_display}**")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# VOICE INPUT BAR
st.write("---")
cols = st.columns([1, 4])
with cols[0]:
    st.write("Click to Speak:")
    voice_data = mic_recorder(
        start_prompt="🎤 Start", 
        stop_prompt="🛑 Stop", 
        key='recorder'
    )

# Logic to handle Voice or Text
user_input = None

if voice_data:
    with st.spinner("Transcribing..."):
        # We pass the 'gu-IN' or 'hi-IN' code here
        transcript = speech_to_text(voice_data['bytes'], selected_lang)
        if "Error" not in transcript and "⚠️" not in transcript:
            user_input = transcript
        else:
            st.error(transcript)

if text_prompt := st.chat_input("Or type your question here..."):
    user_input = text_prompt

# --- 4. AGENT EXECUTION ---

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process via Agent
    with st.chat_message("assistant"):
        agent = get_agri_agent()
        if not agent:
            st.error("Please upload documents first!")
        else:
            with st.spinner("Analyzing rules and identity..."):
                # Force Gemini to respond in the selected language
                instruction = f"Answer only in {lang_display}. User query: {user_input}"
                result = agent.invoke({"messages": [("user", instruction)]})
                
                # Handle Gemini's list/text format
                raw_content = result["messages"][-1].content
                if isinstance(raw_content, list):
                    response_text = raw_content[0].get('text', 'No text found')
                else:
                    response_text = raw_content

                st.markdown(response_text)
                
                # Auto-play the Audio Response
                st.write(f"🔊 Playing audio in {lang_display}...")
                # Use only the first two letters for gTTS (e.g., 'gu' from 'gu-IN')
                text_to_speech(response_text, lang_code=selected_lang.split('-')[0])
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})