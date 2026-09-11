# Conversation pipeline

This document describes the active Django conversation engine. The older standalone implementation and legacy prompts are kept under `archived/` for reference; they are not part of the runtime described here.

## At a glance

```text
Role prompt (database or prompts/*.md)
        |
        v
parse_scenario_prompt -> Scenario and parsed topic/objective guidance
        |
        v
Browser POST or terminal input
        |
        v
ConversationEngine.respond
        |
        v
LangGraph: initialize -> generate -> finalize
        |
        v
System prompt + labeled conversation history
        |
        v
OpenAI structured response
        |
        v
Validate stage, progress, role ownership, ending, and voice metadata
        |
        +--> Django: ChatMessage / ChatSession -> browser -> optional TTS
        |
        +--> Terminal tester: in-memory transcript -> optional JSON save
```

The engine returns dialogue and voice metadata separately. The application owns fixed introduction/opening text and persistence; the model supplies natural in-character turns and structured progress signals.

## Main files

- `vipdjango/vip/views.py` contains the HTTP entry points, session/message persistence, prompt loading, and TTS response path
- `vipdjango/vip/conversation_engine.py` parses each scenario, builds model context, calls OpenAI, and normalizes the model result
- `vipdjango/vip/conversation_graph.py` defines the request-scoped LangGraph state and lifecycle nodes
- `vipdjango/vip/conversation_scenario.py` turns role-prompt Markdown into a structured `Scenario`
- `vipdjango/vip/prompt_utils.py` normalizes headings and resolves section aliases
- `vipdjango/vip/models.py` stores `RolePrompt`, `ChatSession`, and `ChatMessage`
- `prompts/global_prompt.md` contains shared character, role-ownership, style, progression, and ending rules
- `prompts/role_prompt.md` is the authoring template; `prompts/role_*.md` are checked-in scenario examples
- `testing/manual_conversation.py` is the terminal-only text-chat harness
- `vipdjango/vip/tests/` contains automated unit and Django tests, not the manual harness

## Where a conversation begins

There are two entry paths, and both use `ConversationEngine`.

### Browser path

1. `student_dashboard()` or `professor_test_chat()` delegates to `_chat_dashboard()` in `views.py`.
2. The selected `RolePrompt` is loaded from the database. Its `content` is the scenario prompt.
3. Creating a session calls `_create_chat_session()`, which calls `_seed_introduction()`. The parsed scenario Introduction is stored as the first assistant `ChatMessage` with no voice metadata, so it is application-owned text and is not sent to TTS.
4. A learner submits `action=send_message`. `_accept_learner_turn()` locks the `ChatSession`, creates or reuses the learner message, and prevents duplicate in-flight turns.
5. `_generate_assistant_response()` passes the ordered session messages and the persisted session state to `ConversationEngine.respond()`.

Text and microphone submissions use this same server path. Speech recognition happens before the POST; TTS happens after the completed assistant response is persisted.

### Terminal path

`testing/manual_conversation.py` reads a selected `prompts/role_*.md` file, loads `prompts/global_prompt.md`, and calls the same `ConversationEngine.respond()` used by the Django view. It keeps an in-memory list of `Message` objects and saves a JSON transcript under `test_conversations/` when requested or when a conversation completes. It does not start Django, use the database, or generate audio.

## How learner input enters the engine

`ConversationEngine.respond(role_text, messages, conversation_state=...)` first parses the role prompt and counts learner messages.

- With zero learner messages, it returns the scenario Introduction, or `DEFAULT_INTRODUCTION` when the prompt has none
- On the first learner turn, it returns the scenario Opening Line when one exists and has not already been emitted
- Later turns build a graph input from the parsed scenario, the labeled transcript, the persisted state, and the current learner-turn count

The browser supplies `conversation_state` from `ChatSession`. The terminal harness currently supplies no database state; it relies on its in-memory transcript and the engine's history-derived topic checks. Its debug state is displayed and saved, but is not fed back as a persisted session snapshot.

## Scenario parsing and state

`parse_scenario_prompt()` uses `split_markdown_sections()` and alias matching rather than requiring one exact heading spelling. It extracts:

- character identity and background context
- learner role
- Introduction and Opening Line
- Beginning, Middle, and Ending guidance
- beginning-to-middle and middle-to-ending cues
- optional conversation objectives
- concrete topic clusters parsed from Middle bullets
- Closing and voice settings

The immutable `Scenario` is converted to a dictionary with `Scenario.to_state()` before entering LangGraph.

For the browser, `ChatSession` persists the state that must survive requests:

- `conversation_stage`, `conversation_phase`, and `completion_status`
- active, covered, and unresolved objectives
- active, covered, and unresolved topics plus topic turn counts
- `recent_topics` and `ending_ready`
- the active learner-turn claim and last completed turn ID

`ChatMessage` stores the transcript. Assistant dialogue is in `content`; normalized voice instructions are in `voice_metadata`; and `turn_id` ties one learner message to its assistant response.

## LangGraph lifecycle

`build_conversation_graph()` creates three nodes:

1. `initialize` validates the turn budget, normalizes objective progress, selects defaults, and computes the pressure phase. `TARGET_TURNS` is 20 and `MAX_TURNS` is 24; the target is guidance, while the maximum is an operational safety limit.
2. `generate` calls the injected response function, currently `ConversationEngine._llm_turn()`. It accepts only a valid forward stage and carries response, completion, voice, progress, and debug fields into state.
3. `finalize` recomputes `turns_remaining` and preserves the semantic completion result. It does not manufacture a closing because a turn counter was reached.

Turn count selects pressure phases (`normal`, `resolution_guidance`, `closure_preference`, and `closure_flexibility`). It does not directly select Beginning, Middle, or Ending.

## Prompt construction and model call

`views._extract_template_prefix()` loads `prompts/global_prompt.md`. `ConversationEngine._system_prompt()` combines that shared contract with parsed scenario reference data and the current state. The raw role-prompt Markdown is not appended to the model request; it is parsed first, and only the relevant fields below are included. The system message includes:

- immutable simulated-character and learner identities
- relevant background context
- the active stage guidance and its transition cues
- objective and topic progress
- recent learner topics and derived answered-question memory
- voice instructions, lifecycle phase, and turn-budget pressure
- the JSON output contract

`_history()` converts persisted `ChatMessage` objects to native API roles: learner messages become `user`, and character messages become `assistant`. It removes only legacy trailing voice annotations from assistant dialogue, omits the application-owned Introduction, and prefixes serialized content with generic `LEARNER:` or `SIMULATED CHARACTER:` labels. Native API roles remain intact.

`_request_llm_turn()` sends one system message followed by that history to `OpenAI.responses.create()`. The latest learner turn is the final native `user` message and appears once. The request uses a strict JSON schema with dialogue, voice, stage, completion/stop flags, objective fields, topic fields, and ending readiness. `ConversationEngine` lazily creates and reuses one OpenAI client within the request object.

## Response processing

`_llm_turn()` processes the structured result in this order:

1. A structured `stop_requested` result can end the conversation immediately without adding character dialogue.
2. Proposed stages are validated against Beginning → Middle → Ending; regressions and unknown values are rejected.
3. “Other questions” requests are checked against the remaining topic ledger. If the model does not ask a real remaining question, the engine supplies a bounded topic fallback.
4. Obvious role drift is replaced with a character-side topic fallback. A question that lexically repeats an already answered character concern is also blocked and redirected.
5. Objective and topic IDs are validated. Covered progress is monotonic, topic turn counts are application-owned, and unresolved topics cannot be silently discarded by a malformed model result.
6. Candidate endings can receive a separate conservative semantic ending check. Phrase matching only decides whether to attempt that check; it does not decide completion by itself.
7. Dialogue is separated from any legacy embedded voice annotation. Voice metadata is normalized to `male voice, ...` or `female voice, ...` and stored separately.
8. Completion is accepted only when the model requests it at Ending with a forward transition and the configured progress is ready. A scenario Closing replaces the generated dialogue only for a valid completion.

The model can advance early, address multiple concerns in one turn, or continue past the soft target. Reaching 20 turns does not end a conversation. The browser refuses new non-retry turns at the 24-turn safety limit; an explicit user “close conversation” action can also end a session.

## Persistence, rendering, and termination

After `ConversationEngine.respond()` returns, `_persist_assistant_response()` writes the assistant row, updates `ChatSession` stage/phase/progress fields, clears the learner-turn claim, and sets `ended_at` when completion is true. The browser then redirects back to the chat page.

`_rendered_chat_messages()` removes legacy embedded voice annotations for display. `student_message_tts()` reads the stored dialogue and voice metadata and generates one complete audio response after the model response is complete. The Introduction remains text-only.

The terminal tester stops when the engine reports completion, or when the user enters `/stop`, sends EOF, or presses Ctrl-C. `/save` writes the current transcript; `/status` shows the current debug snapshot; `/help` lists commands.

## Where to make changes

- Prompt wording and shared role behavior: `prompts/global_prompt.md`
- Scenario identity, background, goals, stages, cues, and closing: database `RolePrompt.content` or the Markdown files in `prompts/`
- Markdown heading/field parsing: `vipdjango/vip/prompt_utils.py` and `conversation_scenario.py`
- Prompt assembly and model output schema: `ConversationEngine._system_prompt()` and `_request_llm_turn()` in `conversation_engine.py`
- Topic/objective progression and response guards: the progress helpers and `_llm_turn()` in `conversation_engine.py`
- Request lifecycle and stage/turn state: `conversation_graph.py`
- Database persistence and browser/TTS behavior: the chat helpers in `views.py` and `ChatSession`/`ChatMessage` in `models.py`
- Manual text testing: `testing/manual_conversation.py`

Keep prompt behavior changes separate from state/progression changes when possible. The automated tests under `vipdjango/vip/tests/` cover both layers and are the quickest regression check.

## Useful local commands

From the repository root, after activating the virtual environment and exporting the required environment variables:

```bash
python vipdjango/manage.py migrate
python vipdjango/manage.py test vip.tests
python testing/manual_conversation.py
python testing/manual_conversation.py --scenario prompts/role_rachel_ellison_1.md --verbose
```

The manual tester requires `OPENAI_API_KEY` (or its explicit `--api-key` option). The Django server additionally requires `DJANGO_SECRET_KEY`; see `README.md` for local setup and the HTTP security flags used by `runserver`.
