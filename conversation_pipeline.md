# Conversation pipeline

## Request flow

```text
Browser login / selected scenario
    -> ChatSession (immutable scenario_content snapshot)
    -> saved Introduction (text only)
    -> learner POST with retry ID
    -> locked session claim + saved learner message
    -> ConversationEngine.respond
         -> explicit stop / 20-turn boundary
         -> clinician demonstration for an explicit clinician_demo mode
         -> fixed Opening Line for family/patient scenarios
         -> core-question practice for Middle themes with numbered examples
         -> legacy LangGraph generation otherwise
    -> assistant text + voice style + state, committed together
    -> redirect to transcript; optional streaming audio request
```

Both the browser and `testing/manual_conversation.py` call the same engine and carry state into subsequent turns. The terminal tester has no database or audio. Its debug snapshot becomes the next call's conversation state.

## Scenario parsing

`conversation_scenario.py` and `prompt_utils.py` parse Markdown aliases for character, learner, background, introduction, opening, stages, objectives, closing, and voice. A top-level `- Theme` bullet in Middle with indented numbered questions becomes a `ScenarioTopic` with `possible_expressions`. Both Rachel files use this format and match the supplied Core Questions document.

Editing a RolePrompt affects new sessions. Migration 0011 snapshots existing sessions' current prompt content; it cannot reconstruct earlier edits made before migration. `load_scenarios` imports the examples as inactive drafts without overwriting existing prompts. Faculty can test drafts before activation.

## Core-question practice

`core_questions.py` owns progression. The model's schema has only `answer_status`, `reaction`, and `question_id`. It has no generated spoken dialogue, role field, or completion flag.

1. The opening counts as the first concern of the first theme. Its canonical first example is excluded from later selection to avoid immediately repeating that concern.
2. On each learner reply, `_assess_core_reply()` supplies all themes, background, goals, the pending concern, remaining candidates, and labeled history to the configured model. Native user/assistant roles remain intact; the application-owned introduction is excluded.
3. The model classifies the reply as addressed, unclear, or unsafe and selects an available question ID that fits the conversation and avoids already-explained material. This is conversational guidance, not a validated clinical score.
4. An unclear/unsafe reply gets at most one clarification opportunity per theme. A concern still unresolved after the available repair is recorded, and the conversation advances. Such sessions receive a support-seeking closing.
5. The application selects an unused question in the current theme, falling back to the first available candidate if the ID is invalid. It prefixes one of four fixed reactions. Untrusted free text is never spoken.
6. Two concerns per theme move practice forward. The final question must receive a learner reply before closing. The three-theme Rachel scenarios finish in 7-10 learner submissions, including greeting, final answer, and any clarifications.
7. Normal completion uses the scenario Closing only if no answered concern remains flagged unresolved. Otherwise it uses a fixed support-seeking closing. Prior unresolved flags are conservative: later replies are not automatically treated as repairing unrelated earlier concerns.

`core_question_state` records asked IDs, pending question, themes that used a clarification, and unresolved question IDs. Existing topic fields record discussion progress for compatibility. Counts indicate practice exposure, not competency. State is committed atomically with the assistant row. Older transcripts lacking this state are recovered conservatively from identifiable authored questions; start a fresh session when switching a class to the new workflow.

The question bank, opening, closing, and fixed reactions are the only spoken content in this mode. This bounds hallucination/role drift at the output level but reduces free-form dialogue. Model question selection and answer classification can still be imperfect. Faculty review remains necessary for educational validity.

## Clinician demonstration

`Scenario.simulation_mode` defaults to `roleplay` for existing prompts. An explicit `## Simulation Mode` value of `clinician_demo` dispatches to `clinician_demo.py` before the fixed family opening and core-question/legacy branches. Its clinician-specific system instructions replace the family global prompt. It uses the existing covered-topic and readiness fields to persist discussion progress, with separate clinician closing and output checks. The scenario editor and downloaded Markdown preserve the mode. See [CLINICIAN_DEMO.md](CLINICIAN_DEMO.md) for the Scenario 2 role mapping, limits, and verification.

## Legacy generative scenarios

Prompts without question examples retain the existing initialize -> generate -> finalize LangGraph. The engine assembles shared prompting, identities, background, active stage guidance, all Middle/Ending guidance, objectives, topics, and transcript memory. Its structured response supplies dialogue, voice, stage, completion, stop intent, and progress.

Normalization checks forward stages, role-drift markers, repeated questions, progress IDs, and voice style. Candidate endings may use a separate semantic verifier. These are heuristic safeguards around generated text; they do not have the closed-vocabulary guarantees of core-question practice. The prompt table labels these **Open-ended legacy scenario**. Use the imported core-question versions for the reported classroom exercise.

`TARGET_TURNS` is 10 and `MAX_TURNS` is 20. Before either path calls the model at the 20th learner submission, the engine emits a character-side pause and completes. Standalone stop commands work even before the opening. A learner saying "stop the feeding pump" is scenario dialogue, not a standalone session-stop command. The Close Conversation button works at any point.

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
