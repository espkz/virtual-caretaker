# Voice Pipeline Latency Report

## Scope

This report measures the six complete voice traces in
`test_conversations/chat_session_37.txt`, recorded on 2026-08-25 with the
Rachel Ellison scenario. The sample contains one fixed opening response and
five normal LLM turns. The trace IDs are retained in the source log for
cross-checking.

The other saved conversation transcripts do not contain comparable timing
events, so this is a six-turn real-browser sample from one session, not a
cross-session benchmark. Missing values are intentionally not estimated.

## Per-trace measurements

All values are milliseconds. “GPT → usable TTS” is the measured span from the
first LLM output to the first text accepted for TTS; it can include remaining
LLM output and post-generation work before the response is authoritative.

| Trace | GPT request → first output | GPT first output → usable TTS | TTS request → first audio | First audio → playback |
| --- | ---: | ---: | ---: | ---: |
| 914d6a1a (fixed opening) | n/a | n/a | 374.6 | 7.9 |
| 0335adad | 12,416.9 | 2,689.0 | 408.6 | 8.3 |
| e59dc9f9 | 18,287.5 | 1,089.0 | 1,550.7 | 6.0 |
| 388d248c | 12,325.7 | 18,246.2 | 689.2 | 524.2 |
| 4630acd9 | 7,880.9 | 21,309.0 | 1,098.4 | 5.4 |
| 43ae3d98 | 11,872.0 | 11,889.0 | 2,130.2 | 35.1 |

The five normal LLM turns have these summary values:

| Measured span | n | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT request → first output | 5 | 7,880.9 | 12,325.7 | 12,556.6 | 18,287.5 |
| GPT first output → usable TTS | 5 | 1,089.0 | 11,889.0 | 11,044.4 | 21,309.0 |

Across all six traces:

| Measured span | n | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTS request → first audio | 6 | 374.6 | 893.8 | 1,042.0 | 2,130.2 |
| First audio → playback | 6 | 5.4 | 8.1 | 97.8 | 524.2 |
| Response generation → client pipeline complete | 6 | 6,465.6 | 15,850.4 | 17,007.4 | 31,311.0 |

The final span includes the remaining browser audio playback, so it is not a
startup-latency measure.

## Findings

1. The largest measured startup contributors are the LLM request and the
   first-output-to-usable-TTS span. Together they typically add several
   seconds and reach about 30 seconds on the slowest normal trace.
2. The first-output-to-usable-TTS span is not attributable to one cause with
   the current marks. It includes completion of the streamed structured
   response and any post-generation repair or semantic endpoint verification.
   Traces 388d248c, 4630acd9, and 43ae3d98 show especially large unexplained
   portions: 18,246.2 ms, 21,309.0 ms, and 11,889.0 ms.
3. The interrupt classifier runs concurrently with the normal request, but
   its 7.4–15.1 second duration can delay visible streamed output when the
   stream gate is waiting for the stop decision. This sample does not show a
   separate user speech-end mark, so its effect on end-to-end audible latency
   cannot be calculated.
4. TTS first-audio time is smaller than the LLM-side spans in this sample
   (0.4–2.1 seconds). Browser audio handoff is normally 5–35 ms; the 524.2 ms
   trace is an outlier that should be monitored separately.
5. All six traces have null `recording_end_to_first_audible_response`,
   `stt_start_to_stt_completion`, and related STT durations. No conclusion
   about microphone or speech-recognition latency is supported by this log.

## Phase 2 decision

The supplied log provides several real voice traces and identifies the
largest measured latency spans. No optimization is justified from this sample
alone. A follow-up measurement set should add substage marks around repair
and endpoint verification, include `recording_end`/STT marks, and cover more
than one session before Phase 3 changes are evaluated.
