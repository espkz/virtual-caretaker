import streamlit as st
from streamlit_mic_recorder import speech_to_text
from datetime import datetime, timedelta
import os, io
from anthropic import Anthropic
from gtts import gTTS

# setup prompt template and default
prompt_template_file = 'prompts/prompt_template.md'
default_role_file = 'prompts/default_role.md'
with open(prompt_template_file) as f:
    prompt_template = f.read()
with open(default_role_file) as f:
    default_role = f.read()

# --- Constants ---
INACTIVITY_TIMEOUT_MINUTES = 15

st.set_page_config(page_title="Virtual Caretaker", layout="wide")

st.markdown(
    """
    <style>
    /* Make chat container wider */
    .stChat {
        max-width: 900px;
        margin: auto;
    }

    /* Style chat bubbles */
    .chat-message {
        display: flex;
        margin-bottom: 12px;
    }
    .chat-message.user {
        justify-content: flex-end;
    }
    .chat-bubble {
        max-width: 75%;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 4px;
        font-size: 16px;
        line-height: 1.5;
    }
    .chat-bubble.user {
        background-color: #DCF8C6;
        text-align: right;
    }
    .chat-bubble.assistant {
        background-color: #E0E0E0;
        text-align: left;
    }

    /* Add spacing below chat input */
    .stChatInput {
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Type or speak to start chatting."}]
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()
if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = datetime.now()
if "log_filename" not in st.session_state:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("history", exist_ok=True)
    st.session_state.log_filename = f"history/chat_{timestamp}.txt"
    with open(st.session_state.log_filename, "w", encoding="utf-8") as f:
        f.write("")

# --- Sidebar ---
with st.sidebar.expander("⚙️ Settings / Prompt Editor", expanded=True):
    api_key = st.text_input("Claude API key", type="password")
    system_prompt = st.text_area("System prompt:", value=default_role, height=200)


# --- Inactivity Reset ---
def reset_session_if_needed():
    now = datetime.now()
    if (now - st.session_state.last_interaction) > timedelta(minutes=INACTIVITY_TIMEOUT_MINUTES):
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! Type or speak to start chatting."}]
        st.session_state.session_start = now
        st.session_state.last_interaction = now
        st.session_state.log_filename = f"history/chat_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

reset_session_if_needed()

# --- Save Conversation ---
def save_to_file(role, content):
    with open(st.session_state.log_filename, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] {role.capitalize()}: {content.strip()}\n\n")
    return open(st.session_state.log_filename, "r", encoding="utf-8").read()


# --- Claude Chat Function ---
def get_claude_response(api_key, prompt):
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text if response.content else "⚠️ Claude returned an empty response."


# --- TTS Function ---
def text_to_speech_gtts(text: str):
    tts = gTTS(text=text, lang="en")
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


# --- Title ---
st.title("💬 Virtual Caretaker")

# --- Display Chat ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio") and msg["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            st.audio(msg["audio"], format="audio/mp3", autoplay=True)

# --- Bottom Input Row: Text + Voice ---
col1, col2 = st.columns([8, 1])

with col1:
    typed_input = st.chat_input("Type your message here...")

with col2:
    # Center the mic button vertically
    st.markdown(
        """
        <div style="display:flex; align-items:center; height:100%;">
            <div id="mic-container"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    voice_input = speech_to_text(
        language='en',
        just_once=True,
        use_container_width=True,
        key='STT'
    )


user_text = typed_input or voice_input

# --- Handle User Input ---
if user_text:
    st.session_state.last_interaction = datetime.now()

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    save_to_file("user", user_text)

    with st.chat_message("user"):
        st.markdown(user_text)

    # Get assistant response
    if not api_key:
        st.error("Please enter your Claude API key in the sidebar.")
    else:
        current_role = system_prompt if system_prompt.strip() else default_role

        full_prompt = prompt_template.format(role=current_role) + "\n" + save_to_file("user", user_text)
        try:
            placeholder = st.empty()
            placeholder.markdown("...")
            response = get_claude_response(api_key, full_prompt)

            # Generate TTS
            audio_bytes = text_to_speech_gtts(response)

            # Append assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "audio": audio_bytes
            })
            save_to_file("assistant", response)

            placeholder.markdown(response)
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        except Exception as e:
            error_msg = f"❌ Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            save_to_file("assistant", error_msg)
            placeholder.markdown(error_msg)

# --- Download Conversation ---
st.markdown("---")
with open(st.session_state.log_filename, "r", encoding="utf-8") as f:
    file_contents = f.read()

st.download_button(
    label="💾 Download Conversation",
    data=file_contents,
    file_name=os.path.basename(st.session_state.log_filename),
    mime="text/plain"
)
