# ElevenLabs Speech Engine voice practice

Live voice practice keeps `ConversationEngine` on the Django server. ElevenLabs
handles browser audio, speech recognition, and speech output; the authenticated
adapter turns only a finalized learner transcript into the existing Django turn
flow.

```text
Browser --WebRTC--> ElevenLabs Speech Engine --authenticated WSS--> Django adapter
   |                       |                                      |
   |                       +-- speech-to-text / text-to-speech     +-- ConversationEngine
   +-- narrator MP3 from Django (before WebRTC is opened)          +-- ChatSession / ChatMessage
```

## Configuration owned by ElevenLabs

Create and configure the base Speech Engine in the ElevenLabs platform (or in
your deployment's infrastructure configuration). Django does not create,
update, or synchronize that resource. Provide the externally issued resource
ID and the server API key as environment variables; for local development put
them in the ignored repository-root `env.txt` file:

```dotenv
ELEVENLABS_API_KEY=<server-only-elevenlabs-api-key>
ELEVENLABS_SPEECH_ENGINE_ID=seng_<external-speech-engine-id>
SPEECH_ENGINE_PORT=8081
# Optional; defaults to 600 seconds (10 minutes). This is the explicit
# provider session cap shown to the learner if it is reached.
ELEVENLABS_SPEECH_ENGINE_MAX_DURATION_SECONDS=600
# Optional; defaults to eleven_v3.
ELEVENLABS_NARRATOR_MODEL=eleven_v3
```

`vipson_manager.settings` loads `env.txt` for local development. In production,
provide the same values through the deployment secret store instead. The
`ELEVENLABS_SPEECH_ENGINE_ID` value is read at runtime by the token endpoint and
the adapter; it is never exposed to the browser. Django returns only a
short-lived, single-call token.

Set the external resource's public `wss://.../ws` upstream URL, authentication,
first-message behavior, silence policy, duration, and client events in the
ElevenLabs platform (or through your deployment's infrastructure tooling). No
Django `manage.py` command is needed for provider configuration. If that URL or
provider configuration changes, update the external resource there and restart
the adapter process if its local runtime settings changed.

The upstream URL must reach `run_speech_engine`; keep provider authentication
enabled and the engine's first message disabled. The application supplies no
ElevenLabs `firstMessage` override: the scenario introduction is narrated
separately and must never be an agent greeting.

ElevenLabs Speech Engine resources have a fixed TTS voice. To preserve the
prompt editor's Roleplay Voice choice without changing a shared resource during
another learner's call, Django creates one managed, provider-side resource per
roleplay `voice_id`. Each inherits the external engine's authenticated upstream
configuration and uses `eleven_v3_conversational`. This mapping is necessary;
the current Speech Engine initiation API does not provide a per-call TTS voice
override. Do not delete managed resources that are referenced by
`SpeechEngineVoiceResource` unless the corresponding prompts no longer need
them.

## Strict introduction-to-listening handoff

The introduction is intentionally outside the live Speech Engine connection:

1. The browser creates the Django chat/voice-call record and gets a short-lived
   token, but does not open WebRTC yet.
2. It plays the selected narrator's complete introduction audio.
3. Only after the `<audio>` element emits `ended` does it open Speech Engine.
4. It immediately mutes that new connection, binds its provider conversation ID,
   and POSTs the durable `/student/voice/ready/` marker.
5. The browser enables roleplay input, restores the learner's mute preference,
   and transitions to **Listening** once.

The adapter is a second boundary: it discards every finalized provider transcript
until the ready marker exists, and also discards transcripts while muted. Browser
interim/final callbacks are likewise ignored until roleplay input is enabled.
Consequently narrator audio, initial room noise, and Mute cannot create a
`ConversationEngine` turn.

## Local development

1. Start Django:

   ```bash
   python vipdjango/manage.py migrate
   python vipdjango/manage.py runserver
   ```

2. Start the separate authenticated adapter:

   ```bash
   python vipdjango/manage.py run_speech_engine --port 8081 --debug
   ```

3. Expose it with TLS, for example `ngrok http 8081`, then set the resulting
   `wss://.../ws` URL on the externally managed Speech Engine resource.

4. After a deploy that changes `vip/speech_engine/adapter.py`, restart the
   standalone `run_speech_engine` process. Django's development-server reload
   does not reload that process.

After applying migrations, restart any existing Django and adapter workers as
well. A worker loaded before the deployment can otherwise keep stale model code
in memory.

The adapter is reached by ElevenLabs, not by the browser. Do not expose an
unauthenticated public WebSocket endpoint or pass `disable_auth=True`.

## Runtime behavior

- The narrator uses the prompt's Introduction Voice and is presentation-only. It
  is shown as `SIMULATION`, is not a `ChatMessage`, and is never passed to
  `ConversationEngine`.
- The live role character uses the prompt's Roleplay Voice through its managed
  Speech Engine resource.
- Mute only changes the microphone/input gate. It does not submit text,
  generate a response, or alter conversation state. Pause/Resume is not part
  of the voice interface.
- A normal End Call or an upstream failure finalizes the Django session and opens
  its saved transcript rather than starting a new session.
- ElevenLabs' WebSocket protocol handles its own ping/pong keepalive. Silence
  does not create a synthetic learner turn and does not end a call while
  `silence_end_call_timeout` is disabled. The explicit provider maximum duration
  is the remaining normal limit; the browser identifies that limit, explains it,
  saves the transcript, and opens it.
- When `ConversationEngine` has already marked the session complete, the browser
  waits for the provider to return to **Listening**, which is the SDK's
  speaking-to-listening boundary after the final audio is drained. It then
  closes the live provider session, finalizes, and opens the saved transcript.
  The `agent_response_complete` event is retained for diagnostics; it is not
  treated as an audio-playback completion signal. Manual End Call remains a
  separate, immediate path.

## Verification

```bash
python vipdjango/manage.py test vip.tests --settings=vipson_manager.test_settings
python vipdjango/manage.py makemigrations --check --dry-run --settings=vipson_manager.test_settings
```

For a live non-production test, confirm that the narrator ends before the
ElevenLabs connection is opened, `/student/voice/ready/` succeeds once, speech
during narration produces no `ChatMessage`, and the first spoken learner turn
after Listening produces exactly one roleplay response. Also leave the session
silent past the previous silence interval, confirm it stays connected, and let a
normal closing line finish before confirming that the saved transcript opens.
