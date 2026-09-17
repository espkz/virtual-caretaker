/* Live ElevenLabs call UI. ConversationEngine remains server-side text in/text out. */
(() => {
    const root = document.getElementById("voice-practice");
    if (!root || root.dataset.enabled !== "1" || root.dataset.sessionEnded === "1") return;

    const modeSelection = document.getElementById("mode-selection");
    const choice = document.getElementById("voice-start-choice");
    const start = document.getElementById("voice-start");
    const controls = document.getElementById("voice-controls");
    const mute = document.getElementById("voice-mute");
    const end = document.getElementById("voice-end");
    const status = document.getElementById("voice-connection-status");
    const detail = document.getElementById("voice-state-detail");
    const transcript = document.getElementById("voice-live-transcript");
    const connectionLost = document.getElementById("voice-connection-lost");
    const openSavedTranscript = document.getElementById("voice-open-saved-transcript");
    const duration = document.getElementById("voice-duration");
    const narrator = document.getElementById("narrator-audio");
    const download = document.getElementById("voice-download");
    let conversation = null;
    let voiceCallId = "";
    let muted = false;
    let endedByUser = false;
    let endingAutomatically = false;
    let tentativeNode = null;
    let timer = null;
    let connectedAt = 0;
    let finalization = null;
    let connected = false;
    let introductionPlaying = false;
    let roleplayInputEnabled = false;
    let listeningTransitioned = false;
    let unexpectedConnectionLoss = false;
    let maxDurationSeconds = 0;
    let terminalResponsePending = false;
    let terminalResponseEventId = "";
    let terminalAudioCompleted = false;
    let completedAudioEventId = "";
    let latestAgentResponseEventId = "";
    const completedAgentResponseEventIds = new Set();
    const renderedEvents = new Set();
    const recentMessages = new Map();

    const stateDetails = {
        "Connecting...": "Setting up the secure voice session.",
        "Connecting listener...": "The simulation introduction is complete. Opening the live voice connection.",
        "Simulation introduction...": "The narrator is introducing the simulation. Listening begins automatically afterward.",
        "Listening...": `${root.dataset.characterLabel || "The character"} can hear you.`,
        "Thinking...": `${root.dataset.characterLabel || "The character"} is preparing a response.`,
        "Completing...": "The final response has finished. Saving the completed transcript.",
        "Connection lost": "The voice connection ended unexpectedly. Your transcript is being saved.",
        "Ended": "The voice conversation has ended and the transcript is saved.",
        "Error": "The voice connection needs attention. You can retry without changing modes.",
    };

    function setStatus(value, detailText = "") {
        status.textContent = value;
        detail.textContent = detailText || stateDetails[value] || "";
        root.dataset.state = value.toLowerCase().replace(/[^a-z]+/g, "-");
    }

    function speakingStatus() {
        const name = root.dataset.characterLabel || "Character";
        setStatus(`${name} speaking...`, "The role character is responding.");
    }

    function csrfToken() {
        const cookie = document.cookie.split(";").map(value => value.trim()).find(value => value.startsWith("csrftoken="));
        return cookie ? decodeURIComponent(cookie.slice("csrftoken=".length)) : "";
    }

    async function postJSON(url, payload) {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken(),
                "X-Requested-With": "XMLHttpRequest",
            },
            credentials: "same-origin",
            body: JSON.stringify(payload),
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.error || "Voice service request failed.");
        return body;
    }

    function textFrom(value) {
        if (typeof value === "string") return value.trim();
        if (value && typeof value === "object") return String(value.text || value.content || "").trim();
        return "";
    }

    function transcriptRole(role, source = "") {
        const value = String(role || source || "").toLowerCase();
        return ["agent", "assistant", "ai"].includes(value) ? "agent" : "user";
    }

    // Keep this aligned with the server parser. The model emits short,
    // lowercase bracketed delivery cues at response/phrase boundaries, so a
    // generic matcher removes every supported cue without enumerating only
    // the tags we happened to see in a particular conversation.
    const inlineAudioTagPattern = /(^|[.!?…,:;—]\s+|\]\s+)\[(?:[a-z][a-z'-]*(?:[ \t]+[a-z][a-z'-]*){0,3})\](?=\s|$)/g;

    function cleanProviderAudioTags(content) {
        return content.replace(inlineAudioTagPattern, "$1").replace(/[ \t]{2,}/g, " ").trim();
    }

    function appendMessage(role, content, eventId = "") {
        content = textFrom(content);
        if (!content) return;
        role = transcriptRole(role);
        if (role === "agent") content = cleanProviderAudioTags(content);
        if (!content) return;
        const eventKey = eventId ? `${role}:${eventId}` : "";
        if (eventKey && renderedEvents.has(eventKey)) return;
        const recent = recentMessages.get(role);
        const now = Date.now();
        // onUserTranscript and onMessage may report the same final result.
        // Limit content fallback dedupe to the callback burst so a learner can
        // legitimately repeat the same words in a later turn.
        if (recent && recent.content === content && now - recent.at < 2000) return;
        if (eventKey) renderedEvents.add(eventKey);
        recentMessages.set(role, {content, at: now});
        if (tentativeNode && role === "user") {
            tentativeNode.remove();
            tentativeNode = null;
        }
        const item = document.createElement("article");
        item.className = `voice-transcript-message ${role}`;
        item.dataset.role = role;
        const label = document.createElement("strong");
        label.textContent = role === "agent" ? (root.dataset.characterLabel || "Character").toUpperCase() : "YOU";
        const body = document.createElement("span");
        body.textContent = content;
        item.append(label, body);
        transcript.append(item);
        transcript.scrollTop = transcript.scrollHeight;
    }

    function clearTentative() {
        if (!tentativeNode) return;
        tentativeNode.remove();
        tentativeNode = null;
    }

    function canProcessUserSpeech() {
        return Boolean(
            conversation
            && connected
            && roleplayInputEnabled
            && !introductionPlaying
            && !muted
            && root.dataset.sessionEnded !== "1"
        );
    }

    function keepMicrophoneMuted() {
        return introductionPlaying || !roleplayInputEnabled || muted;
    }

    function canEnterListeningState() {
        return Boolean(
            conversation
            && connected
            && roleplayInputEnabled
            && !introductionPlaying
            && root.dataset.sessionEnded !== "1"
        );
    }

    function transitionToListeningOnce() {
        if (!canEnterListeningState() || listeningTransitioned) return;
        listeningTransitioned = true;
        setStatus(
            "Listening...",
            muted
                ? "The simulation introduction is complete. Your microphone remains muted."
                : "The simulation introduction is complete. The character can hear you."
        );
    }

    function finalizeUserTranscript(content, eventId = "") {
        clearTentative();
        if (!canProcessUserSpeech()) return;
        content = textFrom(content);
        if (!content) return;
        appendMessage("user", content, eventId);
        setStatus("Thinking...");
    }

    function showTentative(content) {
        content = textFrom(content);
        if (!content || !canProcessUserSpeech()) {
            if (!canProcessUserSpeech()) clearTentative();
            return;
        }
        if (!tentativeNode) {
            tentativeNode = document.createElement("article");
            tentativeNode.className = "voice-transcript-message user tentative";
            tentativeNode.dataset.role = "tentative";
            const label = document.createElement("strong");
            label.textContent = "YOU (LISTENING)";
            tentativeNode.append(label, document.createElement("span"));
            transcript.append(tentativeNode);
        }
        tentativeNode.querySelector("span").textContent = content;
        transcript.scrollTop = transcript.scrollHeight;
    }

    function setSessionData(data) {
        root.dataset.sessionId = String(data.chat_session_id);
        root.dataset.forceNew = "0";
        maxDurationSeconds = Number.isFinite(Number(data.max_duration_seconds))
            ? Number(data.max_duration_seconds)
            : 0;
        if (data.character_label) {
            root.dataset.characterLabel = data.character_label;
            document.getElementById("voice-heading").textContent = data.character_label;
        }
        if (data.download_url) {
            download.href = data.download_url;
            download.classList.remove("is-hidden");
        }
        const url = new URL(window.location.href);
        url.searchParams.delete("new");
        url.searchParams.set("prompt", root.dataset.promptId);
        url.searchParams.set("session", data.chat_session_id);
        window.history.replaceState({}, "", url);
    }

    async function bindConversation(providerConversationId) {
        if (!voiceCallId || !providerConversationId) return;
        await postJSON(root.dataset.bindUrl, {
            voice_call_id: voiceCallId,
            provider_conversation_id: providerConversationId,
        });
    }

    function updateDuration() {
        if (!connectedAt) return;
        const elapsed = Math.max(0, Math.floor((Date.now() - connectedAt) / 1000));
        const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
        const seconds = String(elapsed % 60).padStart(2, "0");
        duration.textContent = `${minutes}:${seconds}`;
    }

    function startTimer() {
        if (timer) return;
        connectedAt = Date.now();
        updateDuration();
        timer = window.setInterval(updateDuration, 1000);
    }

    function stopTimer() {
        if (timer) window.clearInterval(timer);
        timer = null;
    }

    function showCall() {
        modeSelection?.classList.add("is-hidden");
        root.classList.remove("is-hidden");
        root.scrollIntoView({block: "start", behavior: "smooth"});
    }

    function showConnectedControls() {
        start.hidden = true;
        controls.hidden = false;
        mute.disabled = false;
        end.disabled = false;
    }

    function showNarratorControls() {
        start.hidden = true;
        controls.hidden = false;
        // There is no live microphone to mute yet. End remains
        // available so narration can be cancelled without creating a call.
        mute.disabled = true;
        end.disabled = false;
    }

    function resetForRetry() {
        conversation = null;
        connected = false;
        introductionPlaying = false;
        roleplayInputEnabled = false;
        listeningTransitioned = false;
        unexpectedConnectionLoss = false;
        start.hidden = false;
        start.disabled = false;
        start.textContent = "Retry Call";
        controls.hidden = true;
        connectionLost.hidden = true;
        openSavedTranscript.disabled = true;
        muted = false;
        endingAutomatically = false;
        terminalResponsePending = false;
        terminalResponseEventId = "";
        terminalAudioCompleted = false;
        completedAudioEventId = "";
        latestAgentResponseEventId = "";
        completedAgentResponseEventIds.clear();
        stopTimer();
    }

    function savedTranscriptUrl() {
        const sessionId = root.dataset.sessionId;
        if (!sessionId) return "";
        const url = new URL(window.location.href);
        url.searchParams.delete("new");
        url.searchParams.set("prompt", root.dataset.promptId);
        url.searchParams.set("session", sessionId);
        return url.toString();
    }

    function openSavedTranscriptPage() {
        const transcriptUrl = savedTranscriptUrl();
        if (transcriptUrl) window.location.assign(transcriptUrl);
    }

    function connectionDiagnostics(eventType, event = {}) {
        let providerConversationId = "";
        try {
            providerConversationId = conversation?.getId?.() || "";
        } catch (error) {
            // A torn-down browser SDK instance may no longer expose its ID.
            // The durable server-side ID is still included in the diagnostic.
            console.warn("Could not read SpeechEngine conversation ID", error);
        }
        const closeCode = Number.isInteger(event?.code) ? event.code : null;
        const closeReason = String(event?.reason || event?.message || "").slice(0, 240);
        return {
            voice_call_id: voiceCallId,
            event: eventType,
            connection_state: root.dataset.state || "",
            close_code: closeCode,
            close_reason: closeReason,
            provider_conversation_id: providerConversationId,
        };
    }

    function reportConnectionDiagnostic(eventType, event = {}) {
        const diagnostic = connectionDiagnostics(eventType, event);
        console.warn("SpeechEngine connection event", diagnostic);
        if (!diagnostic.voice_call_id || !root.dataset.diagnosticUrl) return;
        postJSON(root.dataset.diagnosticUrl, diagnostic).catch(error => {
            console.warn("Could not record SpeechEngine connection event", error);
        });
    }

    function normalizedEventId(value) {
        return value === undefined || value === null || value === "" ? "" : String(value);
    }

    function isTerminalCompletionEvent(eventId) {
        const normalized = normalizedEventId(eventId);
        return !terminalResponseEventId || !normalized || normalized === terminalResponseEventId;
    }

    async function endCompletedConversationAfterAudio() {
        if (!terminalResponsePending || !terminalAudioCompleted || endingAutomatically || endedByUser) return;
        endingAutomatically = true;
        roleplayInputEnabled = false;
        clearTentative();
        mute.disabled = true;
        end.disabled = true;
        setStatus("Completing...");
        try {
            // ElevenLabs confirms this event only after playback is finished.
            // Ending now cannot truncate the closing response.
            if (conversation) await conversation.endSession();
        } catch (error) {
            console.warn("Could not close the completed SpeechEngine session", error);
            reportConnectionDiagnostic("completed_end_error", error);
        } finally {
            conversation = null;
            connected = false;
            await finalizeAndShowTranscript();
        }
    }

    function markAgentAudioComplete(eventId = "") {
        // Some browser SDK releases provide the completion event without an
        // ID.  The immediately preceding agent response supplies the same
        // provider event ID, so it is safe to use only as a correlation aid.
        const normalized = normalizedEventId(eventId) || latestAgentResponseEventId;
        if (!terminalResponsePending) {
            // Never let a non-terminal response's completion close a later
            // terminal turn. Keep only a concrete ID for the harmless race
            // where completion reaches the browser before our Django status
            // check completes.
            if (normalized) completedAgentResponseEventIds.add(normalized);
            return;
        }
        if (!isTerminalCompletionEvent(normalized)) return;
        terminalAudioCompleted = true;
        completedAudioEventId = normalized;
        endCompletedConversationAfterAudio();
    }

    async function inspectCompletedConversation(eventId = "") {
        if (!voiceCallId || !root.dataset.completionUrl || root.dataset.sessionEnded === "1") return;
        const completion = await postJSON(root.dataset.completionUrl, {voice_call_id: voiceCallId});
        if (!completion.conversation_complete) return;
        terminalResponsePending = true;
        terminalResponseEventId = normalizedEventId(eventId);
        // A terminal response must be allowed to play without a new learner
        // utterance interrupting it.  This only closes the input gate; it does
        // not end SpeechEngine or submit a turn.
        roleplayInputEnabled = false;
        clearTentative();
        if (conversation) conversation.setMicMuted(true).catch(error => {
            console.warn("Could not mute input while the closing response plays", error);
        });
        if (terminalResponseEventId && completedAgentResponseEventIds.has(terminalResponseEventId)) {
            markAgentAudioComplete(terminalResponseEventId);
        } else if (terminalAudioCompleted && isTerminalCompletionEvent(completedAudioEventId)) {
            await endCompletedConversationAfterAudio();
        }
    }

    function handleProviderDebugEvent(event = {}) {
        const completion = event?.agent_response_complete_event || event?.agentResponseCompleteEvent;
        if (event?.type !== "agent_response_complete" && !completion) return;
        // This provider event means that the agent finished producing its
        // response (and any tools), not that the WebRTC audio sink has drained
        // every sample.  Treating it as playback completion can cut off a
        // terminal closing line.  The mode transition to `listening` below is
        // the browser-side audio boundary used for shutdown.
        console.debug("SpeechEngine agent response produced", completion || event);
    }

    async function finalizeVoiceCall({failed = false} = {}) {
        if (finalization) return finalization;
        if (!voiceCallId || !root.dataset.sessionId) return false;

        finalization = (async () => {
            stopTimer();
            narrator.pause();
            controls.hidden = true;
            try {
                await postJSON(root.dataset.endUrl, {
                    voice_call_id: voiceCallId,
                    failed,
                });
                root.dataset.sessionEnded = "1";
                return true;
            } catch (error) {
                console.error("Could not finalize Django voice session", error);
                if (unexpectedConnectionLoss) {
                    setStatus("Connection lost", "The voice connection ended unexpectedly, and its transcript could not be finalized. Reload to recover it.");
                } else {
                    setStatus("Error", "The call ended, but the saved session could not be finalized. Reload and try again.");
                }
                finalization = null;
                return false;
            }
        })();
        return finalization;
    }

    async function finalizeAndShowTranscript({failed = false} = {}) {
        const finalized = await finalizeVoiceCall({failed});
        if (!finalized) return false;
        // Keep an unexpected-close indicator visible through finalization.
        // The saved-session page uses the same durable error state, so this
        // must not be briefly replaced with a successful-looking status.
        setStatus(failed ? "Error" : "Ended", failed
            ? "The voice connection ended unexpectedly. Opening the saved transcript."
            : "The voice conversation has ended. Opening the saved transcript.");
        openSavedTranscriptPage();
        return true;
    }

    async function handleUnexpectedConnectionLoss(eventType, event = {}) {
        if (endedByUser || endingAutomatically || unexpectedConnectionLoss) return;
        const reachedProviderLimit = Boolean(
            maxDurationSeconds
            && connectedAt
            && Date.now() - connectedAt >= Math.max(0, maxDurationSeconds - 2) * 1000
        );
        unexpectedConnectionLoss = true;
        reportConnectionDiagnostic(eventType, event);
        conversation = null;
        connected = false;
        clearTentative();
        stopTimer();
        controls.hidden = true;
        connectionLost.hidden = false;
        openSavedTranscript.disabled = true;
        setStatus(
            reachedProviderLimit ? "Ended" : "Connection lost",
            reachedProviderLimit
                ? `This voice session reached the provider's ${Math.ceil(maxDurationSeconds / 60)}-minute limit. Saving the transcript now.`
                : "The voice connection ended unexpectedly. Saving the transcript now."
        );
        if (await finalizeAndShowTranscript({failed: !reachedProviderLimit})) return;
        openSavedTranscript.disabled = false;
        setStatus("Connection lost", "The voice connection ended unexpectedly. Open the saved transcript to recover it.");
    }

    function playNarrator(url) {
        return new Promise((resolve, reject) => {
            narrator.src = url;
            narrator.onended = resolve;
            narrator.onerror = () => reject(new Error("The narrator introduction could not be played."));
            const playback = narrator.play();
            if (playback) playback.catch(reject);
        });
    }

    async function completeIntroduction() {
        if (roleplayInputEnabled || root.dataset.sessionEnded === "1") return;
        if (!conversation || !connected) {
            throw new Error("The live voice connection was not ready after the simulation introduction.");
        }
        await postJSON(root.dataset.readyUrl, {voice_call_id: voiceCallId});
        roleplayInputEnabled = true;
        introductionPlaying = false;
        await conversation.setMicMuted(keepMicrophoneMuted());
        transitionToListeningOnce();
    }

    function speechEngineSessionOptions(Conversation, token) {
        return {
            conversationToken: token,
            connectionType: "webrtc",
            onConnect: ({conversationId} = {}) => {
                connected = true;
                showConnectedControls();
                startTimer();
                setStatus("Connecting listener...");
                bindConversation(conversationId).catch(() => setStatus("Error", "The voice call connected but could not be matched to this transcript."));
            },
            onDisconnect: disconnectEvent => {
                if (endedByUser || endingAutomatically) {
                    if (endedByUser) setStatus("Ended");
                } else {
                    handleUnexpectedConnectionLoss("disconnect", disconnectEvent);
                }
            },
            onError: error => {
                console.error("ElevenLabs voice error", error);
                if (!endingAutomatically && !endedByUser) handleUnexpectedConnectionLoss("error", error);
            },
            onStatusChange: ({status: connectionStatus} = {}) => {
                if (connectionStatus === "connecting") setStatus("Connecting listener...");
                if (connectionStatus === "connected") transitionToListeningOnce();
            },
            onModeChange: ({mode} = {}) => {
                // Compatibility fallback: a voice-mode switch back to
                // listening is emitted after the role character's playback.
                // It must run independently of the learner microphone gate,
                // because a user may have muted themselves during the final
                // line.
                if (mode === "listening") markAgentAudioComplete();
                if (endingAutomatically) return;
                if (!canProcessUserSpeech()) return;
                if (mode === "speaking") speakingStatus();
                else {
                    // This is a compatibility fallback for browser SDKs that
                    // do not surface agent_response_complete through onDebug.
                    // In voice mode, listening resumes only after playback.
                    if (!listeningTransitioned) transitionToListeningOnce();
                    else setStatus("Listening...");
                }
            },
            onMessage: ({role, source, message, event_id: eventId, eventId: camelEventId} = {}) => {
                if (!roleplayInputEnabled || introductionPlaying) return;
                const normalizedRole = transcriptRole(role, source);
                if (normalizedRole === "agent") {
                    const responseEventId = eventId || camelEventId || "";
                    latestAgentResponseEventId = normalizedEventId(responseEventId);
                    appendMessage(normalizedRole, message, responseEventId);
                    inspectCompletedConversation(responseEventId).catch(error => {
                        console.warn("Could not check whether the roleplay conversation completed", error);
                    });
                } else {
                    // Some client versions also emit finalized user text as
                    // a generic message; the shared finalizer deduplicates it.
                    finalizeUserTranscript(message, eventId || camelEventId || "");
                }
            },
            onTentativeUserTranscript: ({transcript: liveTranscript} = {}) => showTentative(liveTranscript),
            onUserTranscript: ({transcript: finalTranscript, eventId, event_id: eventIdSnake} = {}) => {
                finalizeUserTranscript(finalTranscript, eventId || eventIdSnake || "");
            },
            // The current client forwards enabled raw provider events here.
            // Keep the handler defensive because older SDK builds expose the
            // same completion via onModeChange instead.
            onDebug: handleProviderDebugEvent,
        };
    }

    async function connectAfterIntroduction(Conversation, token) {
        setStatus("Connecting listener...");
        conversation = await Conversation.startSession(speechEngineSessionOptions(Conversation, token));
        const providerConversationId = conversation.getId?.();
        if (providerConversationId) await bindConversation(providerConversationId);
        // This connection begins only after the narrator has ended. Keep it
        // muted until the server has recorded the same durable handoff.
        await conversation.setMicMuted(true);
        await completeIntroduction();
    }

    async function startVoice() {
        if (conversation || start.disabled) return;
        showCall();
        const Conversation = window.ElevenLabsClient?.Conversation;
        if (!Conversation) {
            setStatus("Error", "The ElevenLabs browser client did not load. Check the network connection and retry.");
            resetForRetry();
            return;
        }
        start.disabled = true;
        start.textContent = "Connecting...";
        endedByUser = false;
        connected = false;
        setStatus("Connecting...");
        try {
            const probe = await navigator.mediaDevices.getUserMedia({audio: true});
            probe.getTracks().forEach(track => track.stop());
            const tokenData = await postJSON(root.dataset.tokenUrl, {
                prompt_id: root.dataset.promptId,
                session_id: root.dataset.sessionId || "",
                force_new: root.dataset.forceNew === "1",
            });
            voiceCallId = tokenData.voice_call_id;
            setSessionData(tokenData);
            // Do not connect the live microphone while narration is playing.
            // The narrator is local presentation audio, so it does not need a
            // Speech Engine connection. This removes the startup race where a
            // speaker or ambient sound could become the first learner turn.
            introductionPlaying = true;
            roleplayInputEnabled = false;
            listeningTransitioned = false;
            showNarratorControls();
            setStatus("Simulation introduction...");
            try {
                await playNarrator(tokenData.introduction_audio_url);
            } catch (narratorError) {
                console.error("Narrator introduction failed", narratorError);
                throw narratorError;
            }
            await connectAfterIntroduction(Conversation, tokenData.token);
        } catch (error) {
            narrator.pause();
            conversation = null;
            const denied = error?.name === "NotAllowedError";
            if (voiceCallId) {
                const finalized = await finalizeAndShowTranscript({failed: true});
                if (finalized) return;
            }
            setStatus("Error", denied
                ? "Microphone or audio permission was denied. Allow it in browser settings and retry."
                : (error?.message || "Voice practice could not start. Please retry."));
            resetForRetry();
        }
    }

    async function toggleMute() {
        if (!conversation) return;
        const nextMuted = !muted;
        muted = nextMuted;
        try {
            // Persist the input gate before changing the local microphone so
            // a late provider callback cannot cross the mute boundary.
            await postJSON(root.dataset.muteUrl, {voice_call_id: voiceCallId, muted});
            await conversation.setMicMuted(keepMicrophoneMuted());
        } catch (error) {
            muted = !nextMuted;
            throw error;
        }
        mute.setAttribute("aria-pressed", String(muted));
        mute.textContent = muted ? "🔇 Unmute" : "🎤 Mute";
        if (introductionPlaying || !roleplayInputEnabled) {
            setStatus(
                "Simulation introduction...",
                muted
                    ? "Your microphone will stay muted after the introduction."
                    : "Your microphone will open after the introduction finishes."
            );
        } else {
            setStatus(
                "Listening...",
                muted ? "Your microphone is muted; the call remains connected." : "Microphone restored. The character can hear you."
            );
        }
    }

    async function endVoice() {
        if (!conversation && !voiceCallId) return;
        endedByUser = true;
        [mute, end].forEach(button => { if (button) button.disabled = true; });
        setStatus("Ended");
        stopTimer();
        narrator.pause();
        try {
            // Mark the durable transcript as intentionally ended before
            // closing the provider. This removes the close-callback race in
            // which an expected disconnect could otherwise be recorded as a
            // failed call.
            await finalizeVoiceCall();
            if (conversation) await conversation.endSession();
        } finally {
            conversation = null;
            await finalizeAndShowTranscript();
        }
    }

    choice?.closest("form")?.addEventListener("submit", event => {
        event.preventDefault();
        startVoice();
    });
    start?.addEventListener("click", startVoice);
    mute?.addEventListener("click", () => toggleMute().catch(() => setStatus("Error")));
    end?.addEventListener("click", () => endVoice().catch(() => setStatus("Error")));
    openSavedTranscript?.addEventListener("click", openSavedTranscriptPage);
    window.addEventListener("pagehide", () => {
        narrator.pause();
        if (conversation) conversation.endSession().catch(() => {});
        stopTimer();
    });
})();
