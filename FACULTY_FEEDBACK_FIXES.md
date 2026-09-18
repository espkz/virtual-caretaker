# September 17 follow-up — sessions 62 and 63

## Email-ready update

Thank you for testing again. I reviewed sessions 62 and 63 and made the following adjustments:

1. **Five more exchanges:** conversations now allow up to **25 nurse messages and 25 Rachel replies**, excluding the introductory screen. They can still finish earlier when Rachel is ready or you choose to stop.
2. **More varied questions:** each new conversation gets a randomized question plan. Rachel starts with a concern appropriate to the opening situation, then can move among relevant themes. This varies the sample on repeat attempts while keeping the conversation coherent. Refreshing an existing conversation preserves its plan; start a new conversation for a new sequence.
3. **Less repetition:** Rachel is instructed to accept reasonable explanations, brief confirmations, and honest offers to check information. After a reasonable explanation, she moves directly to the selected faculty question without a generated recap. When asked a question, she answers it without automatically repeating her previous question. A genuinely unclear or unsafe statement can receive one focused follow-up.
4. **Cleaner endings:** removed the repeated pause statement at the end of the transcripts.

The updated scenarios will appear as **“September 17, 2026 revision”** after importing and activating them. Please select those versions and start fresh conversations for the next test.

## Verification for this follow-up

- 64 automated tests pass, including the 25-exchange engine and database workflow, stable randomized plans, variation across new sessions, eligible-theme sampling, repeat-question prevention, direct answers to nurse questions, and duplicate closing prevention.
- Additional live checks of the final version were blocked by automatic approval review because they send the revised scenario prompts and synthetic dialogue to the configured external API (`api.openai.com`). These checks require approval before proceeding. No final-version live quality or latency claim is made.
- A preliminary replay of sessions 62 and 63 helped reveal that generated question wording could disagree with its selected ID. The final implementation now speaks the selected faculty question directly, retaining generated answers to questions from the nurse. The preliminary transcript is not validation of that final fix.
- Django reports no missing migrations; checks of the tracked changes found no whitespace errors.
- Dialogue and assessment still depend on the model. Randomization varies the sample; it does not guarantee that two attempts will never share a question. The clinical progression is retained rather than randomly jumping into unrelated topics.

## Deploy and import this follow-up

From the project directory on the server:

```bash
./manual_update.sh
podman exec virtual-caretaker python manage.py load_scenarios --faculty-feedback-v2
```

Review and activate the two **September 17, 2026 revision** drafts. This import preserves the previous revisions and any instructor edits. Existing sessions retain their scenario snapshots; use fresh sessions. The command runs inside the container where Django and the deployed database configuration are available. No additional database migration is required.

---

The September 15 work and its verification are retained below as historical context. The follow-up above replaces its 20-exchange limit and routine follow-up policy.

# Faculty feedback fixes — September 2026

## Reply email draft

Thank you for the detailed feedback and the four test transcripts. I reviewed both full scenarios and traced the repeated wording to the conversation code. The previous version selected from fixed questions and four canned reactions, rather than allowing Rachel to respond naturally. It also treated questions such as “What do you understand?” as unclear answers, and normally ended after six concerns plus up to three clarification attempts.

The following changes are implemented:

1. **Up to 20 exchanges.** Each session now allows 20 nurse messages and 20 Rachel replies, excluding the introductory screen. The six-question automatic ending has been removed. The last reply responds to the nurse before closing; the session can still finish earlier when the scenario’s readiness conditions are met or the user stops it.
2. **Direct answers to the nurse.** Rachel can explain what she understands, what Daniel told her, what she hopes for, and what worries her. This also works when the nurse asks a question in the initial greeting.
3. **Natural dialogue between core questions.** Generated responses use the preceding conversation and scenario guidance. Rachel no longer has to begin with one of four canned reactions. Emotional responses should change gradually as the nurse listens and explains.
4. **Specific clarification.** The repeated “Could you explain that concern in simpler terms?” fallback has been removed. Rachel is instructed to question the particular uncertainty or claim—for example, how the nurse knows Margaret can hear, or whether morphine is intended to relieve symptoms or cause death.
5. **Full scenario guidance.** Both revised prompts include the faculty’s detailed hidden role instructions and conditional reactions. Scenario 1 covers understanding, prognosis/awareness, potential treatments, and survival. Scenario 2 covers Rachel’s existing knowledge and caregiving experience, Margaret’s wishes, Daniel’s role, feeding and suffering, formal review of the plan, morphine fears, and what to expect near death. The routine request to speak to someone else has been replaced with a pause directed to the current nurse.
6. **Broader coverage with limits on repetition.** The question banks support approximately 10–12 distinct concerns rather than limiting the session to two questions per theme. The application reserves time for all three themes and a final understanding/readiness check. It tracks answers given ahead of time and later repairs to earlier misunderstandings.
7. **More appropriate endings.** Asking questions is no longer treated as proof that Rachel’s concerns were resolved. A successful ending requires coverage of each theme, no recorded unresolved concern, and a readiness check from the nurse. If those conditions are not established by the limit, Rachel pauses without falsely agreeing to proceed.
8. **Role and response safeguards.** Structured output restricts question IDs to valid choices. Checks reject obvious clinician-role leakage, numeric medication doses, the reported stock clarification, certain transfer requests, and exact repetition of the preceding reply. The scenarios retain their boundary before technical caregiver or medication teaching.

The changes retain one model request per normal exchange, with one bounded rewrite if dialogue is rejected; no model upgrade was made. Automated verification and live replay results are recorded below. A fresh interactive instructor test remains the next step for judging conversational quality.

## What changed in the code

| Area | Change |
| --- | --- |
| `vipdjango/vip/core_questions.py` | Replaced fixed reaction/question concatenation with validated generated dialogue; added addressed concerns, repair tracking, per-theme pacing, readiness gating, and final-turn closure. |
| `vipdjango/vip/conversation_engine.py` | Sends every parsed scenario section, full conversation history, and pacing state to the model. Added a learner-question assessment category and natural first-turn responses. Guided sessions process the 20th nurse message. |
| `vipdjango/vip/conversation_graph.py` | Raised the soft target from 10 to 20; retained the existing hard boundary of 20 learner messages. The old 7–10-turn behavior came primarily from question exhaustion, not that hard boundary. |
| `prompts/role_rachel_ellison_1.md` and `role_rachel_ellison_2.md` | Integrated the supplied full scenarios’ hidden instructions and successful endpoints. Expanded hospice questions to include formal review and restarting-feeding concerns. Replaced routine requests to speak to someone else. |
| Scenario editor and labels | Replaced obsolete two-question/one-clarification descriptions with guided scenario practice. |
| `load_scenarios --faculty-feedback` | Imports separate, inactive September 2026 drafts without overwriting existing prompts or instructor edits. Repeatable. |
| Regression tests and replay script | Tests conversation controls and persistence; `testing/replay_faculty_feedback.py` reruns the four original nurse transcripts against freshly generated Rachel replies. |

## Verification

- **55 automated tests passed** in an isolated SQLite test database. Coverage includes both 20-exchange scenarios, nurse questions, first-turn understanding, topic pacing, nurse-led transitions, recognition of earlier answers, repair of prior concerns, readiness gating, duplicate-output repair, final-turn fallback, database persistence, retries, access control, speech behavior, and the revision import/editor workflow.
- Django’s migration check reports **no changes detected**; its system check reports no issues. Whitespace checks pass for the tracked files changed in this revision. The full working tree also contains pre-existing CRLF whitespace findings in unrelated chat templates.
- **All four faculty transcripts replayed successfully:** 37 newly generated Rachel replies using the existing configured `gpt-5.6-luna` model. Median text-generation time was **2.91 seconds**, with a **1.94–5.84 second** range. No reply contained the reported stock clarification, care-team handoff, or “I'm still worried about this” prefix. All four encounters remained open after the original 9–10 nurse messages, rather than automatically ending at the old cutoff. [Replay results](sessions/faculty_feedback_replay_verified.jsonl).
- **Both full live encounters completed at 20 exchanges.** Scenario 1 raised 11 tracked concerns across its themes (4/4/3); Scenario 2 raised 12 (3/5/4). Both used a pause ending because the repeated synthetic explanations left concerns unresolved. Successful readiness is covered by automated tests; these live runs do not demonstrate successful learner performance. [Full encounter and audio results](sessions/smoke_ai_rachel_2026-09-15_20-20-10_UTC_6teuc3s9.jsonl).
- **Audio smoke test passed:** first chunk in 0.86 seconds, 64,128 bytes received. This is provider-stream timing, not a browser playback measurement.
- Replaying fixed nurse lines is a regression probe, not a fresh interactive assessment: later lines sometimes answer questions from the old transcript rather than Rachel’s new question. It therefore does not establish that every new response is natural or that all scenario objectives are met.

### Examples from the replay

When the nurse asked what Rachel understood in Scenario 1, she answered:

> I know her heart stopped and her brain went without oxygen. Daniel said the injury was severe and that nobody could promise she would wake up—but then I see her eyes open, and sometimes her hand moves, and it doesn’t feel like nobody is in there.

When the nurse asked what she understood in Scenario 2, she answered:

> Daniel told me the doctors don’t think Mom will regain awareness, that she signed an advance directive, and that he’s her health-care agent. He said hospice is stopping the tube feeding because she wouldn’t want to be kept alive this way, but I haven’t read the papers myself—and I’m afraid we’re ending her chance too soon.

## Rollout

The code and prompt files are changed locally; this work does not deploy the running service or change active database prompts.

For the Podman deployment in `manual_update.sh`, run the import inside the application container after deploying the updated code:

```bash
podman exec virtual-caretaker python manage.py load_scenarios --faculty-feedback
```

The container has Django installed and uses the deployed application's database. Running `python3 vipdjango/manage.py ...` directly on the host uses a separate Python environment and may fail with `No module named 'django'` or target a different database.

If the container reports that `--faculty-feedback` is unrecognized, it still has the older code. Deploy the revision with `./manual_update.sh`, then rerun the container import above.

For a non-container installation only, use that installation's Python virtual environment and configured application database to run `manage.py load_scenarios --faculty-feedback`.

In the instructor dashboard, review and activate:

- **Rachel Ellison 1: Caregiver training (September 2026 revision)**
- **Rachel Ellison 2: Hospice communication (September 2026 revision)**

Select those prompts and start fresh sessions. Existing sessions retain their original scenario snapshots; rebuilding the container alone does not replace those prompts. No additional database migration is needed for this revision’s JSON progress fields.

## Practical limits

Generated dialogue improves flexibility but cannot guarantee clinical/scenario fidelity. The output checks catch specific violations; they are not a complete semantic safety check. Readiness and answered-concern assessments still depend on the model. Faculty should specifically review direct answers, response to empathy, factual uncertainty, repair after misleading statements, movement among themes, and readiness versus pause endings. Exact response latency and total encounter duration depend on the provider and the nurse’s responses.
