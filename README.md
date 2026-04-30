# Virtual Caretaker

Virtual patient/caregiver roleplay platform for nursing education.

Main app stack is now **Django**.

## How It Works

There are two main account roles:

- `Instructor`:
  - manages prompts (create, edit, activate/deactivate, delete, upload/download)
  - creates and manages student accounts and class group assignments
  - reviews student chat logs
  - can use a test-chat interface to validate prompt behavior

- `Student`:
  - can only access active prompts assigned through the app flow
  - chats with the roleplay assistant using text or speech input
  - can enable/disable emotion voice behavior during TTS playback
  - can download conversation logs

## Tech Stack

- Python3
- Django
- SQLite (default app database)
- OpenAI API (chat generation + TTS)
- gTTS (non-emotion TTS fallback)
- Browser Web Speech API (STT for microphone input)
- Podman (containerized deployment runtime)

## Branches

- `main`: current Django application (active branch)
- `streamlit-v.20260430`: legacy Streamlit implementation

If you need the old Streamlit code, check out the `streamlit-v.20260430` branch.

## Django App Location

- Project root: `vipdjango/`
- App: `vipdjango/vip/`

## Local Development (Django)

1. Create/activate a virtual environment.
2. Install requirements:

```bash
cd vipdjango
pip install -r requirements.txt
```

3. Export env vars (example uses `env.txt` at repo root):

```bash
cd ..
set -a
source env.txt
set +a
```

4. Run migrations and start dev server:

```bash
cd vipdjango
python manage.py migrate
python manage.py runserver 127.0.0.1:8001
```

## Notes

- Prompt templates/files are under `prompts/`.
- OpenAI key and Django settings are environment-variable based.

## Resources

- [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech)
- [Django Docs](https://docs.djangoproject.com/)
