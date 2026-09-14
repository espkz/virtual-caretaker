# Virtual Caretaker

Virtual Caretaker is a Django platform for nursing-education roleplay. A learner talks with an LLM-driven simulated patient or caregiver, while instructors manage scenarios, student accounts, and conversation logs.

The Django application is responsible for authentication, role-prompt management, chat sessions, transcript persistence, browser text/speech interaction, and text-to-speech playback. The conversation engine parses a scenario, builds the model context, enforces character ownership and bounded progression, and returns dialogue plus voice metadata. The active pipeline is documented in [conversation_pipeline.md](conversation_pipeline.md).

**Start here for the handoff:** [DEPLOYMENT.md](DEPLOYMENT.md) explains Django, Windows setup, the instructor/student workflow, and deploying to the existing school server. [TESTING.md](TESTING.md) records the usability findings, fixes, verification, and remaining classroom checks.

**Reversed roles:** Scenario 2 also has a [clinician demonstration mode](CLINICIAN_DEMO.md), with the AI playing the hospice nurse and the human playing Rachel. Import that separate draft with `python vipdjango/manage.py load_scenarios --clinician-demo`, then select **Scenario 2: AI hospice nurse (you play Rachel)** in instructor Test Chat. Margaret remains the noncommunicating patient.

The two Rachel scenarios now use bounded **core-question practice**: roughly two instructor-authored concerns per theme, at most one clarification per theme, and an explicit closing after the final answer. Spoken content is constrained to those questions and short application-owned reactions. A model selects question IDs and assesses whether an answer needs clarification; it cannot invent dialogue or control the turn budget. These scenarios normally finish in 7–10 learner submissions. Older unstructured prompts retain their generative conversation path, with a 20-submission hard stop for all scenarios. Use the imported core-question drafts for the short classroom exercise.

## Project structure

```text
prompts/
  global_prompt.md                 Shared conversation behavior
  role_prompt.md                   Scenario-authoring template
  role_*.md                        Checked-in scenario examples

testing/
  manual_conversation.py           Terminal text-chat harness

conversation_pipeline.md           Detailed engine and request-flow guide
test_conversations/                Historical/manual JSON and text transcripts
archived/                           Legacy backend and superseded prompts

vipdjango/
  manage.py                         Django command entry point
  requirements.txt                  Python dependencies
  vip/
    conversation_engine.py          Prompt assembly, model call, response guards
    core_questions.py               Bounded question selection and closing
    clinician_demo.py               AI clinician dialogue and readiness handling
    speech.py                       Streaming TTS with disconnect cleanup
    conversation_graph.py           LangGraph lifecycle and request state
    conversation_scenario.py        Scenario Markdown parser and data model
    prompt_utils.py                 Heading and section parsing helpers
    models.py                       RolePrompt, ChatSession, ChatMessage
    views.py                        HTTP chat flow, persistence, and TTS
```

`archived/prompts/Core Questions.docx` is supporting scenario-authoring material. It is not loaded by the runtime. The old standalone backend and old prompt variants remain under `archived/` for reference only.

## Setup

Use Python 3.13 or a compatible supported Python version, then create an environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r vipdjango/requirements.txt
```

The project loads an ignored root `.env` automatically; existing shell variables take precedence. Create local configuration with `python testing/setup_local.py` and set `OPENAI_API_KEY` in `.env`, or export variables in your shell:

```bash
export DJANGO_SECRET_KEY='replace-with-a-local-secret'
export OPENAI_API_KEY='your-openai-api-key'
export DJANGO_DEBUG=true
export DJANGO_SECURE_SSL_REDIRECT=false
export DJANGO_SESSION_COOKIE_SECURE=false
export DJANGO_CSRF_COOKIE_SECURE=false
export DJANGO_SECURE_PROXY_SSL_HEADER=false
```

`DJANGO_SECRET_KEY` is required by Django. `OPENAI_API_KEY` is required for live conversation generation and TTS. The default database is SQLite at `vipdjango/db.sqlite3`; set `SQLITE_PATH` to override it. PostgreSQL is available only when `DJANGO_DB_ENGINE=postgres`, with `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, and optional `POSTGRES_PORT` configured.

Other optional Django settings include `DJANGO_ALLOWED_HOSTS`, `DJANGO_CSRF_TRUSTED_ORIGINS`, `DJANGO_DEBUG`, and the `DJANGO_SECURE_*` flags in `vipdjango/vipson_manager/settings.py`. For a local HTTP `runserver`, the four security flags shown above avoid HTTPS-only redirects and cookies intended for a TLS deployment.

Run the database migrations from the repository root:

```bash
python vipdjango/manage.py migrate
python vipdjango/manage.py load_scenarios
python vipdjango/manage.py createsuperuser
```

The ignored `env.txt` convention is equivalent to:

```bash
set -a
source env.txt
set +a
```

## Manual text chatbot testing

This path exercises only the text conversation engine. It does not start Django, use the database, run the frontend, or generate audio.

From the repository root, with the virtual environment active and `OPENAI_API_KEY` exported:

```bash
python testing/manual_conversation.py
```

The script lists the checked-in `prompts/role_*.md` scenarios. Choose one, then type learner messages at the `YOU:` prompt. The engine prints the simulated character response and any normalized voice style metadata. A live run calls the same `ConversationEngine` used by the Django chat flow.

To select a scenario directly or inspect the structured debug state:

```bash
python testing/manual_conversation.py --scenario prompts/role_rachel_ellison_1.md
python testing/manual_conversation.py --scenario prompts/role_rachel_ellison_2.md --verbose
```

Available commands during a chat are `/help`, `/status`, `/save`, and `/stop`. EOF or Ctrl-C also stops the session and saves a transcript when one exists. Completed or explicitly saved transcripts go to `test_conversations/` by default; use `--output-dir PATH` to choose another directory. Use `--api-key KEY` only when an environment variable is not convenient.

The terminal harness keeps its transcript in memory. It is ideal for manually checking prompting and response behavior; use the Django path and automated view tests when you need to inspect database-backed session-state persistence.

## Running the local Django server

`manage.py` is at `vipdjango/manage.py`. After setup, environment configuration, and migrations, start the development server from the repository root:

```bash
python vipdjango/manage.py runserver
```

The default local URL is [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The application redirects unauthenticated users to the login flow under `/accounts/login/`. An OpenAI key is needed when a chat turn or TTS request reaches the model; the server can start without one, but live conversation responses will report that it is missing.

Log in using the local superuser account, test the imported draft scenarios, then activate them for students. Create a named class and add students through Student Accounts; accounts in Unassigned cannot chat. The browser uses its Speech Recognition API for speech input, lets students review the transcription, and streams optional OpenAI TTS after saving each assistant response. See [DEPLOYMENT.md](DEPLOYMENT.md) for the full workflow. Your school website account is separate from a newly created local database.

## Conversation-engine development

Start with [conversation_pipeline.md](conversation_pipeline.md) for the end-to-end flow.

- Change shared prompting rules in `prompts/global_prompt.md`
- Change scenario identity, background, stages, topic guidance, or closing in `prompts/role_*.md` or the corresponding database `RolePrompt.content`
- Inspect prompt parsing in `vipdjango/vip/conversation_scenario.py` and `vipdjango/vip/prompt_utils.py`
- Inspect model context, structured output handling, and response guards in `vipdjango/vip/conversation_engine.py`
- Inspect lifecycle, turn budgets, and graph state in `vipdjango/vip/conversation_graph.py`
- Use `testing/manual_conversation.py` for a quick terminal conversation
- Run automated tests with `python vipdjango/manage.py test vip.tests --settings=vipson_manager.test_settings`

Keep the manual harness focused on terminal I/O. Conversation policy belongs in the engine and prompt files so the browser and terminal paths continue to exercise the same behavior.
