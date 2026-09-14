# Virtual Caretaker handoff

## What Django does

Django is the Python web application behind the website. A browser sends a request (log in, open a scenario, send a reply); Django checks the account, calls the conversation engine where needed, saves the transcript in a database, and returns HTML or audio. Templates produce the pages. Models describe the database tables; **migrations** update those tables as the code changes. `manage.py` runs administrative commands.

Locally, `runserver` handles HTTP on your computer. On a server, this repository uses **Gunicorn** to run Django and **WhiteNoise** to serve CSS/JavaScript. The school's HTTPS proxy forwards browser requests to Gunicorn. PostgreSQL stores accounts, prompts, sessions, and transcripts independently of application restarts. Django's [deployment checklist](https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/) explains why `runserver` is for development.

```text
Student browser --HTTPS--> school proxy --HTTP--> Gunicorn / Django
                                                     |       |
                                                PostgreSQL  OpenAI
```

A login associated with `nursing.emory.edu` does not establish the chatbot's exact URL or grant permission to upload or restart server code. It may be a website account, university authentication, or a hosting account. Ask the person managing the existing installation for the exact chatbot URL, deployment host/platform, deployment access, environment settings, database location/backups, and restart procedure. This repository does not contain that information or university SSO integration.

## Local use on this Windows computer

The repair session installed `.venv`, created an ignored `.env` using the existing local API key, migrated a new SQLite database, and imported two inactive scenario drafts. No remote server was modified. Secrets were not printed or embedded in application code.

From the repository root, create your local instructor login, then start the website:

```powershell
.\.venv\Scripts\python.exe vipdjango/manage.py createsuperuser
.\.venv\Scripts\python.exe vipdjango/manage.py runserver
```

Open **http://127.0.0.1:8000/** and use the account you just created. The school's existing website login is separate from this new local database.

For a fresh machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r vipdjango/requirements.txt
.\.venv\Scripts\python.exe testing/setup_local.py
```

Set `OPENAI_API_KEY` in `.env`. The setup script generates a local Django secret; it does not overwrite existing configuration. Then:

```powershell
.\.venv\Scripts\python.exe vipdjango/manage.py migrate
.\.venv\Scripts\python.exe vipdjango/manage.py load_scenarios
.\.venv\Scripts\python.exe vipdjango/manage.py createsuperuser
.\.venv\Scripts\python.exe vipdjango/manage.py runserver
```

On macOS/Linux use `python3 -m venv .venv`, then `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. Python 3.13 is the common development/container version. Requirements constrain Django to the 5.2 maintenance series so different Python versions do not silently select different Django major versions.

## Instructor and student workflow

For the reversed Scenario 2 role assignment (AI nurse, human Rachel), see [CLINICIAN_DEMO.md](CLINICIAN_DEMO.md). Import its separate draft with `load_scenarios --clinician-demo`; it does not replace the AI-Rachel scenarios.

1. In **Prompts**, choose **Test** beside each imported Rachel scenario. Instructors can test an inactive draft. Students cannot access drafts.
2. Test both a reasonable conversation and an evasive/incorrect one. The prompt table distinguishes **Core-question practice** from **Open-ended legacy scenario**.
3. Activate the reviewed core-question scenarios. Deactivate superseded versions to keep the student list clear. Importing files does not update or replace existing database prompts; `load_scenarios` deliberately leaves existing rows unchanged.
4. In **Student Accounts**, create the class and add/import students into that named class. An account left in **Unassigned** cannot enter the student chat. Existing initial-password behavior uses NetID; communicate credentials privately and have students change their password in Account Settings. These are Django accounts, not automatically university SSO accounts.
5. Students log in, select a scenario, and start a new chat. They can type, or use the microphone and review the transcription before sending. Voice playback is optional; the UI identifies it as AI voice. They can stop at any time and download their transcript. Faculty can review student logs.

For core-question practice, use a top-level theme bullet and indented numbered questions in the **Middle** section, as in the Rachel files. The opening raises the first theme's first concern; do not use an unrelated opening with this format. Questions, opening, and closing are instructor-authored spoken text. The model selects among remaining question IDs and four short reactions; it does not generate additional clinical statements. The mode trades free-form emotional dialogue for predictable, bounded practice. Review that tradeoff with the teaching team when approving the lesson.

Two concerns per theme (including the opening) produce six concerns in these scenarios. Each theme can get one clarification, so the normal length is 7–10 learner submissions including the greeting and final answer. There is a 20-submission hard stop for all scenarios. Unresolved answers lead to a support-seeking closing, not a readiness claim. A closure is the end of practice, **not a competency score**. Five minutes is a pacing aim: typing/speaking speed, pauses, and network conditions still affect duration.

## Updating the existing school installation

The following is a server-administrator handoff, not a claim that these commands have been run at Emory.

1. Identify the running application version and back up its database and server configuration. Rehearse against a staging copy. Preserve the production Django secret and credentials.
2. Deploy the source, including `prompts/`, migrations, and `vip/static/`. Exclude `.env`, `openai.json`, `django_key.txt`, `.venv`, local SQLite databases, and test transcripts. Never upload the local database over the school's database.
3. Install `vipdjango/requirements.txt` in the server environment, or build `Containerfile`. Set production environment variables through the hosting platform. The container's entrypoint runs migrations and `collectstatic` before Gunicorn starts; for a non-container installation run those commands yourself before restarting workers. For multiple replicas, run migrations once as a release step.
4. Apply migration **0011**. It adds core-question progress, an expiring request claim, and a scenario snapshot for each session. Existing transcripts/accounts remain. Scenario edits thereafter affect new sessions; saved sessions retain the content they began with.
5. Run `python manage.py load_scenarios` from `vipdjango/`. This adds inactive copies only. Test them with faculty, activate the reviewed copies, and have students start new sessions. The older database prompts do not inherit the updated Markdown just because code was uploaded.
6. Check login, student/class assignment, both scenarios, instructor draft access, transcript downloads, retry after a provider failure, and audio over the real HTTPS URL. Run a class-sized concurrent rehearsal to establish capacity and API limits before scheduling students. The local smoke tests did not establish class-scale capacity or clinical assessment validity.

Example **production** settings (replace placeholders; do not copy the local `.env`):

```dotenv
DJANGO_SECRET_KEY=<preserve-the-existing-production-secret>
OPENAI_API_KEY=<server-api-key>
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_TTS_MODEL=gpt-4o-mini-tts
DJANGO_DEBUG=false
DJANGO_ALLOWED_HOSTS=<exact-chatbot-hostname>
DJANGO_CSRF_TRUSTED_ORIGINS=https://<exact-chatbot-hostname>
DJANGO_SECURE_SSL_REDIRECT=true
DJANGO_SESSION_COOKIE_SECURE=true
DJANGO_CSRF_COOKIE_SECURE=true
DJANGO_SECURE_HSTS_SECONDS=31536000
DJANGO_SECURE_PROXY_SSL_HEADER=true
DJANGO_DB_ENGINE=postgres
POSTGRES_DB=<database>
POSTGRES_USER=<database-user>
POSTGRES_PASSWORD=<database-password>
POSTGRES_HOST=<database-host>
POSTGRES_PORT=5432
```

Enable `DJANGO_SECURE_PROXY_SSL_HEADER` only behind a trusted proxy that strips client-supplied forwarding headers and sets `X-Forwarded-Proto` itself. The proxy should pass the original `Host`, terminate TLS, and forward to port 8080 (or configured `PORT`). Disable proxy buffering for audio; Django emits `X-Accel-Buffering: no`. Proxy/Gunicorn request timeouts must allow up to 90 seconds for a bounded model request. Every synthesis stream occupies a synchronous worker; measure capacity before choosing worker/thread counts. OpenAI supports [streamed speech output](https://developers.openai.com/api/docs/guides/text-to-speech); the browser also buffers some audio before playback.

Use PostgreSQL for the concurrent class deployment: request claims rely on row locking, which SQLite does not implement in the same way. SQLite is sufficient for the local single-user workflow. Keep the database on persistent storage with backups. The container should reach OpenAI over outbound HTTPS. Its API key needs access to the configured chat/TTS models; a website login is not an API key.

For non-container deployment, from `vipdjango/`:

```bash
python manage.py check --deploy
python manage.py migrate --noinput
python manage.py collectstatic --noinput
gunicorn vipson_manager.wsgi:application --bind 0.0.0.0:8080 --workers 3 --timeout 120
```

The bind address belongs behind the school's proxy/firewall. Your administrator should adapt it to the existing service manager/platform. Copying Python files alone does not migrate the database, rebuild static files, or restart the running application.

## Verification performed

See [TESTING.md](TESTING.md) for reproducible commands, measured smoke-test results, and remaining deployment validation.
