# Virtual Caretaker Chatbot
Simple LLM-integrated chatbot for practicing student nurses


## How to Use
1. Input GPT API key to the left.
 - Refreshing will result in the key being erased as well. If you're resetting, make sure to put in the key first.
2. Begin speaking with the chatbot.
   - You may adjust the role prompt (to the left).
   - Chat input can be done with voice or text
3. Save conversation if you feel like you're done. The chatbot may continue speaking (see Known Issues).

## Known Issues
- Conversation may not properly end?
- Streamlit latency

## TODO
- Minimize latency (AWS requires security review, possible non-AWS option?)
- More mid-dialogue emotion actions (it's possible, but GPT has occasions where it just reads the stage directions)
  - Prompt: If there is any emotion to express in the dialogue, such as bursting into tears or choking on your voice, output them in parentheses, such as (voice cracking) or (pause) or (sigh). Do NOT put any actions, such as (glancing at patient), they should be voice or emotion related. Any additional voice-based instructions such as male/female voice and voice tone should be output in brackets after the dialogue.

## Resources
- [OpenAI TTS](https://ttsopenai.com/)
