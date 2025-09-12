# Virtual Caretaker Chatbot
Simple LLM-integrated chatbot for student nurses practicing speaking to patients and their caretakers

`for any inquiries, please contact me through Github`

## How to Use
1. Input GPT API key to the left.
 - Refreshing will result in the key being erased as well. If you're resetting, make sure to put in the key first.
2. Begin speaking with the chatbot.
   - You may adjust the role prompt (to the left).
   - Chat input can be done with voice or text
3. Save conversation if you feel like you're done. The chatbot may continue speaking (see Known Issues).

## Known Issues
- Conversation may not properly end
- Streamlit latency
- Emotion module sounds excited when it's supposed to sound sad

## TODO
- Minimize latency (AWS requires security review, possible non-AWS option?)
- More mid-dialogue emotion actions (it's possible, but GPT has occasions where it just reads the stage directions)
  - Prompt: If there is any emotion to express in the dialogue, such as bursting into tears or choking on your voice, output them in parentheses, such as (voice cracking) or (pause) or (sigh). Do NOT put any actions, such as (glancing at patient), they should be voice or emotion related. Any additional voice-based instructions such as male/female voice and voice tone should be output in brackets after the dialogue.

## How to pull repository into SON/MathCS server
1) Log into SON/MathCS server (caretaker VM or vip VM)
   1) Ideally Docker/Podman is configured but if not configure Docker/Podman
   2) The SON user may need a password that I've created, please contact me if you require it.
2) Create directory `app/`
3) Create Dockerfile necessary to pull repository (see Streamlit to Docker guide)
   1) The repository must be made public for the Dockerfile to be able to pull, otherwise it requires additional authentication
4) Build image with Dockerfile
5) Remake/restart container if there already exists a container

## Resources/Docs
- [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech)
- [Streamlit to Docker](https://docs.streamlit.io/deploy/tutorials/docker)