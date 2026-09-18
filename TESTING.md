# September 2026 faculty-feedback revision

Current behavior and verification are documented in [FACULTY_FEEDBACK_FIXES.md](FACULTY_FEEDBACK_FIXES.md). The earlier 7–10 submission, fixed-dialogue design described below is historical and has been replaced.

Run offline regression tests:

```bash
python vipdjango/manage.py test vip.tests --settings=vipson_manager.test_settings --noinput
```

Replay the four faculty transcripts against the configured model (requires API access and incurs usage):

```bash
python testing/replay_faculty_feedback.py --live --output sessions/faculty_feedback_replay.jsonl
```

This replays original nurse lines with newly generated Rachel replies. Review the new dialogue manually: later nurse messages may refer to questions from the old transcript. It does not replace a fresh interactive faculty test.

---

# Verification and usability assessment

The subsequent [Scenario 2 clinician demonstration](CLINICIAN_DEMO.md) adds 17 regression tests (40 total), a second Chrome workflow test, and its own live conversation/role/dose checks. The original repair results below describe the AI-Rachel mode.

## Findings before changes

The basic student/instructor pages and transcript persistence were present, but this checkout was not ready for the requested short classroom exercise:

- Stage transitions were model-controlled, and only the current stage's guidance reached the model. A model stuck at Beginning did not see the actual Middle question bank.
- The soft target was 20 learner turns, with a 24-turn request ceiling that could leave a session open with an error instead of a closing.
- Topic counts could advance independently of the actual spoken concern. Regex repairs did not prevent arbitrary role drift or invented clinical facts.
- Successful closing text could replace a model's unsuccessful endpoint. The short practice path now chooses a support-seeking endpoint when answers remain unresolved.
- TTS read the full audio response before sending any bytes. The alternate voice setting used a separate Google TTS provider. Both voice styles now use the configured OpenAI streaming path.
- A provider error propagated as an error page and left a pending learner message with no visible way to retry after reloading. A dead worker's claim had no expiry.
- The instructor test page required a scenario to be active, so testing a draft exposed it to students.
- Student/instructor browser scripts were duplicated, and unavailable local storage could abort audio initialization.
- The terminal tester did not feed saved progress into the next turn.
- Dependencies were unbounded, PostgreSQL's driver was missing, Python versions differed across setups, and the devcontainer launched a nonexistent Streamlit application.
- README and pipeline documentation referred to a test suite absent from this checkout. There was no beginner-oriented production handoff.

The short core-question request in `known issues.txt` was used as the intended classroom behavior. Some original Word scenarios describe a longer encounter; the implementation does not attempt to fit every question in those documents into a five-minute exercise.

## Automated regression suite

```powershell
.\.venv\Scripts\python.exe vipdjango/manage.py test vip.tests --settings=vipson_manager.test_settings
.\.venv\Scripts\python.exe vipdjango/manage.py makemigrations --check --dry-run --settings=vipson_manager.test_settings
```

23 automated tests initially passed on September 13, 2026. Tests use synthetic replies and mocked providers in an isolated temporary SQLite database; they require no API key and do not alter the local/production database. Coverage includes both scenarios, normal and unsuccessful endings, at most one clarification per theme, invalid model data, arbitrary output text/role attempts, hard stops, missing keys, persisted progress, retries, duplicate submissions, stale-worker claims, closing during generation, immutable scenario snapshots, draft visibility, transcript/audio authorization, streaming, and provider resource cleanup. `makemigrations --check` found no missing migrations. Fresh migrations and static collection also succeeded locally.

The final repeat of all 23 tests passed. `pip check`, Python compilation, Django's normal checks, and `check --deploy --fail-level WARNING` with production security flags also passed. The deployment check was local configuration validation, not a test of the Emory server. A real container build and PostgreSQL concurrency test were not performed here.

## Real browser check

Optional dependency: `python -m pip install playwright`; install Chrome, or adapt the browser test to your available Playwright browser.

```powershell
.\.venv\Scripts\python.exe vipdjango/manage.py test vip.tests.browser_smoke --settings=vipson_manager.test_settings
```

The headless Chrome check passed a real login, scenario creation, form submission, simulated provider failure, retry, and complete seven-turn scenario. It also disables browser local storage to exercise graceful fallback. No JavaScript errors occurred. Providers are mocked in this browser check; it does not test a physical microphone or audible speaker output.

## Live API smoke test

This opt-in command uses the configured API key and incurs API usage:

```powershell
.\.venv\Scripts\python.exe testing/smoke_live.py --live
```

Both this command and `--live --clinician-demo` automatically save a new UTF-8 JSONL transcript in the repository's `sessions/` folder. The terminal prints the full path. Filenames include the mode, UTC timestamp, and a unique suffix, so earlier runs are preserved. Files contain the configured model names, scenario/role labels, every synthetic human message and AI reply, completion reasons, timings, and test results. Nurse-mode logs also include both boundary checks; family-mode logs include the speech sample text and audio statistics (not the audio file).

Each event is saved immediately, including human input before a provider request. If a run fails or you press Ctrl+C, the partial transcript remains with a failure/interruption record. Exception details and API keys are not written. Five offline transcript tests verify both commands, partial results, interruption, and unique filenames without API usage.

On September 13, 2026, using `gpt-4.1-mini` and `gpt-4o-mini-tts`:

| Check | Result |
| --- | --- |
| Scenario 1 | 7 learner turns; all three themes, two concerns each; successful closing |
| Scenario 2 | 7 learner turns; all three themes, two concerns each; successful closing |
| Model-backed turn latency | 0.679–4.607 seconds across 12 requests |
| First 4 KiB of MP3 audio | 2.011 seconds |
| Complete speech sample | 61,056 bytes received |

The fixture supplies reasonable synthetic nurse replies based on the provided scenario guidance. This is an integration test, not evidence of clinical validity. No before/after latency benchmark was available; these numbers describe this small post-change sample, not a guaranteed improvement or service-level promise.

## Remaining validation for classroom release

- Faculty review of question selection, the constrained reactions, and the classifier's interpretation of realistic student answers. No automated competency grading is implemented.
- Real microphone, permissions, playback, and network testing on the school's actual student devices and HTTPS endpoint.
- Concurrent load and API rate-limit testing at the intended class size, against PostgreSQL and the production proxy/server configuration.
- Confirmation of the exact Emory deployment endpoint/access and a staging rollout. No server deployment was attempted during this repair.

The software controls length and allowable spoken content in the core-question path. It cannot guarantee a five-minute wall-clock duration, perfect model assessment, or student learning outcomes. Open-ended legacy prompts retain generative dialogue; use the imported core-question versions for the reported short practice exercise.
