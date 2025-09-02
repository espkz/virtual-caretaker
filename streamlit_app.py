import streamlit as st
from streamlit_mic_recorder import speech_to_text
from datetime import datetime, timedelta
import os, io, re
from anthropic import Anthropic
from openai import OpenAI

prompt_template_file = 'prompts/prompt_template.md'
default_role_file = 'prompts/default_role.md'

with open(prompt_template_file) as f:
    prompt_template = f.read()
with open(default_role_file) as f:
    default_role = f.read()

INACTIVITY_TIMEOUT_MINUTES = 15

st.set_page_config(page_title="Virtual Caretaker", layout="wide")

st.markdown("""
<style>
/* Chat container */
#chat-container {
    max-height: 70vh;
    overflow-y: auto;
    margin-bottom: 90px;
}

/* Chat bubbles */
.stChat .chat-message {
    display: flex;
    margin-bottom: 12px;
}
.stChat .chat-message.user {
    justify-content: flex-end;
}
.stChat .chat-bubble {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 4px;
    font-size: 16px;
    line-height: 1.5;
}
.stChat .chat-bubble.user {
    background-color: #DCF8C6;
    text-align: right;
}
.stChat .chat-bubble.assistant {
    background-color: #E0E0E0;
    text-align: left;
}

/* Fixed input row */
.chat-input-row {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: white;
    border-top: 1px solid #ccc;
    padding: 8px 12px;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 Virtual Caretaker")

# sidebar for prompt editing and API key
with st.sidebar.expander("Settings", expanded=True):
    claude_api_key = st.text_input("Claude API key", type="password")
    gpt_api_key = st.text_input("GPT API key", type="password")
    system_prompt = st.text_area("System prompt:", value=default_role, height=200)

# initialize
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

# inactivity check
def reset_session_if_needed():
    now = datetime.now()
    if (now - st.session_state.last_interaction) > timedelta(minutes=INACTIVITY_TIMEOUT_MINUTES):
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! Type or speak to start chatting."}]
        st.session_state.session_start = now
        st.session_state.last_interaction = now
        st.session_state.log_filename = f"history/chat_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

reset_session_if_needed()

# save conversation
def save_to_file(role, content):
    with open(st.session_state.log_filename, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] {role.capitalize()}: {content.strip()}\n\n")
    return open(st.session_state.log_filename, "r", encoding="utf-8").read()

# claude response
def get_claude_response(api_key, prompt):
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text if response.content else "⚠️ Claude returned an empty response."

# def text_to_speech_gtts(text: str):
#     tts = gTTS(text=text, lang="en")
#     audio_buffer = io.BytesIO()
#     tts.write_to_fp(audio_buffer)
#     audio_buffer.seek(0)
#     return audio_buffer

def parse_response(text):
    # match "anything" then "{...}" at the end
    match = re.match(r"^(.*?)(\[.*\])?$", text.strip(), re.DOTALL)
    if match:
        dialogue = match.group(1).strip()
        emotion = match.group(2).strip("[]") if match.group(2) else None
        return dialogue, emotion
    return text, None

def text_to_speech_openai(dialogue: str, emotion_instructions : str, voice="coral", model="gpt-4o-mini-tts"):
    tts_client = OpenAI(api_key=gpt_api_key)
    response = tts_client.audio.speech.create(
        model=model,
        voice=voice,
        input=dialogue,
        instructions=emotion_instructions,
    )
    return response.read()

# chat history
st.markdown('<div id="chat-container">', unsafe_allow_html=True)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("content") == "...":
            st.markdown("...")  # thinking placeholder
        else:
            st.markdown(msg["content"])
        if msg.get("audio") and msg["role"] == "assistant":
            st.audio(msg["audio"], format="audio/mp3", autoplay=True)

st.markdown('</div>', unsafe_allow_html=True)

# chat input
st.markdown('<div class="chat-input-row">', unsafe_allow_html=True)
col1, col2 = st.columns([9, 1])
with col1:
    typed_input = st.chat_input("Type your message here...")
with col2:
    voice_input = speech_to_text(
        start_prompt="🎙️",
        stop_prompt="⏹️",
        language='en',
        just_once=True,
        use_container_width=True,
        key='STT'
    )
st.markdown('</div>', unsafe_allow_html=True)

user_text = typed_input or voice_input

# user input processing
if user_text:
    st.session_state.last_interaction = datetime.now()
    
    st.session_state.messages.append({"role": "user", "content": user_text})
    save_to_file("user", user_text)

    if not claude_api_key:
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ Please enter your Claude API key."})
    else:
        current_role = system_prompt if system_prompt.strip() else default_role
        full_prompt = prompt_template.format(role=current_role) + "\n" + save_to_file("user", user_text)

        try:
            with st.spinner("🤖 The AI is thinking..."):
                response = get_claude_response(claude_api_key, full_prompt)
                dialogue, emotion = parse_response(response)
                audio_bytes = text_to_speech_openai(dialogue, emotion)

            st.session_state.messages.append({
                "role": "assistant",
                "content": dialogue,
                "audio": audio_bytes
            })
            save_to_file("assistant", response)

        except Exception as e:
            error_msg = f"❌ Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            save_to_file("assistant", error_msg)

    st.rerun()


# download
st.markdown("---")
with open(st.session_state.log_filename, "r", encoding="utf-8") as f:
    file_contents = f.read()

st.download_button(
    label="💾 Download Conversation",
    data=file_contents,
    file_name=os.path.basename(st.session_state.log_filename),
    mime="text/plain"
)
