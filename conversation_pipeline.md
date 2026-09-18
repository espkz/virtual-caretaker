# Conversation pipeline

## Request flow

```text
Browser login / selected scenario
    -> ChatSession (immutable scenario_content snapshot)
    -> saved Introduction (text only)
    -> learner POST with retry ID
    -> locked session claim + saved learner message
    -> ConversationEngine.respond
         -> explicit stop / 25-turn boundary
         -> clinician demonstration for an explicit clinician_demo mode
         -> fixed Opening Line for legacy family/patient scenarios
         -> guided scenario practice for Middle themes with numbered examples
         -> legacy LangGraph generation otherwise
    -> assistant text + voice style + state, committed together
    -> redirect to transcript; optional streaming audio request
```

Both the browser and `testing/manual_conversation.py` call the same engine and carry state into subsequent turns. The terminal tester has no database or audio. Its debug snapshot becomes the next call's conversation state.

## Scenario parsing

`conversation_scenario.py` and `prompt_utils.py` parse Markdown aliases for character, learner, background, introduction, opening, stages, objectives, closing, and voice. A top-level `- Theme` bullet in Middle with indented numbered questions becomes a `ScenarioTopic` with `possible_expressions`. Both Rachel files use this format, expanded with the supplied full faculty scenarios.

Editing a RolePrompt affects new sessions. Migration 0011 snapshots existing sessions' current prompt content; it cannot reconstruct earlier edits made before migration. `load_scenarios` imports the examples as inactive drafts without overwriting existing prompts. Review drafts before activation; Instructor Test Chat lists active prompts.

## Guided scenario practice

`core_questions.py` owns the turn budget, concern ledger, and completion. A single structured model request produces Rachel’s answer/reaction, question selection, and assessment. Fresh questions are rendered from the selected faculty wording, so spoken concerns match their tracked IDs. Directed answers and the one permitted unclear/unsafe follow-up remain generated. The request includes all parsed scenario sections (including hidden conditional responses), the transcript, known/asked/answered concerns, and the remaining budget.

- Dialogue answers the nurse first; invitations to explain understanding or feelings have their own assessment category.
- A shuffled question order is saved per session. One fresh concern per eligible theme is offered, favoring themes with fewer questions; earlier themes remain eligible for unasked concerns. New sessions draw new orders, while reloads preserve the existing order.
- Approximately 10–12 concerns can be explored across the full question banks. Questions already answered ahead of time are omitted; a later repair can clear an earlier unresolved concern.
- The scheduler reserves roughly seven exchanges per theme and the final two for understanding/readiness. After two concerns have been raised, adjacent-theme choices let Rachel follow the nurse into a new topic earlier. Only unclear or unsafe answers allow one follow-up on a pending concern. Reasonable explanations advance to a fresh question; nurse-directed questions get direct answers. It does not require asking every example question.
- Known IDs, allowed next concerns, role/dose checks, and duplicate-output checks constrain generated output. Rejected dialogue gets one rewrite with a 15-second timeout and no provider retries; further invalid output uses the existing user retry workflow without committing dialogue or progress. At the final turn, invalid dialogue produces a deterministic pause instead. These checks are heuristics, not a guarantee against hallucination.
- Successful completion requires coverage of all themes, no recorded unresolved concern, and a model-reported readiness check quoted from the latest nurse message. The scenario supplies the final successful line. At turn 25 Rachel answers briefly, asks no new question, and pauses if readiness has not been established.
- `core_question_state` retains legacy fields and adds addressed IDs, per-concern follow-up counts, active theme, and current turn. No schema migration is required. Old state is upgraded in memory; fresh sessions are recommended for the revised scenarios.

The old six-question completion rule and fixed clarification/reaction vocabulary have been removed. Generated conversation must still be reviewed by faculty for scenario fidelity and educational quality.

## Clinician demonstration

`Scenario.simulation_mode` defaults to `roleplay` for existing prompts. An explicit `## Simulation Mode` value of `clinician_demo` dispatches to `clinician_demo.py` before the fixed family opening and core-question/legacy branches. Its clinician-specific system instructions replace the family global prompt. It uses the existing covered-topic and readiness fields to persist discussion progress, with separate clinician closing and output checks. The scenario editor and downloaded Markdown preserve the mode. See [CLINICIAN_DEMO.md](CLINICIAN_DEMO.md) for the Scenario 2 role mapping, limits, and verification.

## Legacy generative scenarios

Prompts without question examples retain the existing initialize -> generate -> finalize LangGraph. The engine assembles shared prompting, identities, background, active stage guidance, all Middle/Ending guidance, objectives, topics, and transcript memory. Its structured response supplies dialogue, voice, stage, completion, stop intent, and progress.

Normalization checks forward stages, role-drift markers, repeated questions, progress IDs, and voice style. Candidate endings may use a separate semantic verifier. These are heuristic safeguards around generated text. The prompt table labels these **Open-ended legacy scenario**. Use the revised guided scenarios for the classroom exercise.

`TARGET_TURNS` and `MAX_TURNS` are 25. Guided family scenarios answer the 25th learner message before the application ends the session. Legacy and clinician modes retain their deterministic boundary pause. Standalone stop commands and the Close Conversation button still end immediately; “stop the feeding pump” is scenario dialogue, not a session-stop command.

## Persistence and recovery

`views.py` accepts one learner turn at a time. A retry ID maps to one learner row and at most one assistant row through a database uniqueness constraint. The session claim has a unique worker ID and timestamp. After 90 seconds a retry can reclaim an abandoned request; a stale worker cannot commit or release a newer claim. API timeout/retry settings are bounded below that lease in the core-question path.

Provider failures release the claim without inserting an error as character dialogue. After failure or reload the page displays the saved learner text as read-only with **Retry response**, preserving its turn ID. Closing during generation prevents a late response from committing. PostgreSQL supplies production row locking; SQLite serves local single-user development and isolated tests, not concurrent-class verification.

## Browser speech

Student and instructor templates share `vip/static/vip/chat.js`. Text forms work without JavaScript. Supported browsers use Speech Recognition for microphone input; the student reviews the recognized text before sending. Controls handle unavailable recognition, denied access, blocked storage, duplicate sends, and autoplay restrictions.

The voice request authorizes ownership of the saved assistant message; introductions remain text-only. `SpeechStream` opens `audio.speech.with_streaming_response.create` and forwards MP3 chunks through Django `StreamingHttpResponse`. It closes the provider response and client on completion, failure, or disconnect. Upstream errors before headers become a generic 503; mid-stream failures terminate playback, and the browser offers text/retry. `X-Accel-Buffering: no` asks the proxy to avoid buffering. Audible latency also depends on browser buffering and the network.

Expressive and neutral styles use the configured OpenAI TTS model. Replay reuses the audio element's source where possible; no cross-user or durable audio cache is implemented. Stop cancels the current stream. Text generation still completes before TTS starts, and normal message submission still reloads the transcript page.

## Configuration and tests

Root `.env` loads without overriding exported variables. `OPENAI_CHAT_MODEL` defaults to `gpt-4.1-mini`; `OPENAI_TTS_MODEL` defaults to `gpt-4o-mini-tts`. The chat client has a 30-second timeout and one SDK retry; speech has a 30-second timeout with no SDK retry. The core classifier sets `store=False` and uses a bounded response schema.

See [README.md](README.md) for commands, [TESTING.md](TESTING.md) for results, and [DEPLOYMENT.md](DEPLOYMENT.md) for the Windows and school-server handoff.
