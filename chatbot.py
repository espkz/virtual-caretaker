import streamlit as st
from anthropic import Anthropic
from datetime import datetime, timedelta
import base64, requests

def push_to_github(filename, local_path):
    with open(local_path, "rb") as f:
        content = f.read()
    encoded = base64.b64encode(content).decode("utf-8")

    repo = "espkz/virtual-caretaker"
    path_in_repo = f"history/{filename}"
    api_url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    token = st.secrets["github_token"]

    # Check if the file already exists (to get the SHA for updates)
    get_response = requests.get(api_url, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    })
    sha = get_response.json().get("sha")

    payload = {
        "message": f"Save chat log {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "content": encoded,
        "branch": "main",  # or 'master' depending on your repo
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(api_url, json=payload, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    })

    if response.status_code in [200, 201]:
        st.success("Conversation pushed to GitHub")
    else:
        st.error(f"GitHub push failed: {response.json().get('message')}")

# setup prompt and repository
prompt_file = 'prompt.md'
with open(prompt_file) as f:
    base_prompt = f.read()

# --- Sidebar: API Key ---
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Claude API key", type="password")

# --- Chat Title ---
st.title("💬 Virtual Caretaker")

# --- Constants ---
INACTIVITY_TIMEOUT_MINUTES = 15

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please type a greeting to begin speaking with the chatbot."}]
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()
if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = datetime.now()
if "log_filename" not in st.session_state:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.log_filename = f"history/chat_{timestamp}.txt"

# --- Inactivity Check ---
def reset_session_if_needed():
    now = datetime.now()
    if (now - st.session_state.last_interaction) > timedelta(minutes=INACTIVITY_TIMEOUT_MINUTES):
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please type a greeting to begin speaking with the chatbot."}]
        st.session_state.session_start = now
        st.session_state.last_interaction = now
        # reset
        st.session_state.log_filename = f"history/chat_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

reset_session_if_needed()

# --- Save to File with Timestamp+Read the File to Converge with Prompt---
def save_to_file(role: str, content: str):
    with open(st.session_state.log_filename, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {role.capitalize()}: {content.strip()}\n\n")
    with open(st.session_state.log_filename, "r", encoding="utf-8") as f:
        conversation = f.read()
    return conversation


# --- Claude Chat Function ---
def get_claude_response(api_key, prompt, history):
    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text if response.content else "⚠️ Claude returned an empty response."

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if user_turn := st.chat_input("Talk to the chatbot..."):
    st.session_state.last_interaction = datetime.now()

    st.session_state.messages.append({"role": "user", "content": user_turn})
    conversation = save_to_file("user", user_turn)

    # full prompt
    prompt = base_prompt + '\n' + conversation

    with st.chat_message("user"):
        st.markdown(user_turn)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please enter your Claude API key in the sidebar.")
        else:
            placeholder = st.empty()
            placeholder.markdown("...")
            try:
                response = get_claude_response(api_key, prompt, st.session_state.messages)
                placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_to_file("assistant", response)
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                conversation = save_to_file("assistant", error_msg)
# --- Save and Push to Git ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    if st.button("💾 Save Conversation"):
        filename = os.path.basename(st.session_state.log_filename)
        push_to_github(filename, st.session_state.log_filename)
