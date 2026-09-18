/* Shared student/instructor chat controls. Text remains usable without JS. */
(() => {
    const form = document.querySelector("form.message-form");
    const input = document.getElementById("student-message-input");
    const send = document.getElementById("send-message-btn");
    const mic = document.getElementById("stt-toggle");
    const status = document.getElementById("stt-status");
    const turn = document.getElementById("voice-turn-id");
    const chat = document.querySelector(".chat-window");
    if (chat) chat.scrollTop = chat.scrollHeight;

    function preference(key, fallback) {
        try { return localStorage.getItem(key) ?? fallback; } catch (_) { return fallback; }
    }
    function toggle(id, key, fallback) {
        const control = document.getElementById(id);
        if (control) {
            control.checked = preference(key, fallback) === "1";
            control.addEventListener("change", () => {
                try { localStorage.setItem(key, control.checked ? "1" : "0"); } catch (_) { /* optional */ }
            });
        }
        return control;
    }
    const emotion = toggle("emotion-voice-toggle", "emotion_voice_enabled", "1");
    const autoplay = toggle("autoplay-voice-toggle", "autoplay_voice_enabled", "0");
    const players = document.querySelectorAll(".message-audio-controls audio");
    function stopOthers(except) {
        players.forEach(player => {
            if (player !== except) player.pause();
        });
    }
    async function playVoice(button) {
        const wrap = button.closest(".chat-message").querySelector(".message-audio-controls");
        const player = wrap.querySelector("audio");
        const label = wrap.querySelector(".audio-status");
        const url = button.dataset.ttsUrl + "?emotion=" + (emotion ? Number(emotion.checked) : preference("emotion_voice_enabled", "1"));
        stopOthers(player);
        label.textContent = "Loading voice… You can keep reading the response.";
        if (player.dataset.loadedUrl !== url || player.error) {
            player.src = url;
            player.dataset.loadedUrl = url;
        } else if (player.ended) {
            player.currentTime = 0;
        }
        try { await player.play(); }
        catch (error) {
            label.textContent = error.name === "NotAllowedError"
                ? "Press Play Voice to allow audio playback."
                : "Voice is unavailable. Press Play Voice to retry, or continue using text.";
        }
    }
    document.querySelectorAll(".tts-button").forEach(button => button.addEventListener("click", () => playVoice(button)));
    players.forEach(player => {
        const wrap = player.closest(".message-audio-controls");
        const label = wrap.querySelector(".audio-status");
        player.addEventListener("playing", () => { stopOthers(player); label.textContent = "Playing."; });
        player.addEventListener("waiting", () => { label.textContent = "Buffering voice…"; });
        player.addEventListener("ended", () => { label.textContent = "Finished."; });
        player.addEventListener("error", () => { label.textContent = "Voice is unavailable. Retry Play Voice or continue using text."; });
        wrap.querySelector(".tts-stop").addEventListener("click", () => {
            player.pause();
            player.removeAttribute("src");
            delete player.dataset.loadedUrl;
            player.load(); // Cancels the HTTP stream, including synthesis on the server.
            label.textContent = "Stopped.";
        });
    });

    let recognition;
    let listening = false;
    let sendAfterStop = false;
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (mic && input && !input.readOnly && Recognition && window.isSecureContext) {
        recognition = new Recognition();
        recognition.lang = "en-US";
        recognition.interimResults = true;
        recognition.continuous = true;
        let committed = "";
        recognition.onstart = () => {
            stopOthers(null);
            committed = input.value.trim() + " ";
            listening = true;
            mic.textContent = "Stop microphone";
            mic.setAttribute("aria-pressed", "true");
            status.textContent = "Listening. Stop the microphone to review your words, then press Send.";
        };
        recognition.onresult = event => {
            let interim = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                if (result.isFinal) committed += result[0].transcript.trim() + " ";
                else interim += result[0].transcript;
            }
            input.value = (committed + interim).trim();
        };
        recognition.onerror = event => {
            sendAfterStop = false;
            status.textContent = event.error === "not-allowed"
                ? "Microphone access was denied. Allow it in browser settings or type your response."
                : "Speech recognition stopped. Review your text or type your response.";
        };
        recognition.onend = () => {
            listening = false;
            mic.textContent = "Microphone";
            mic.setAttribute("aria-pressed", "false");
            if (sendAfterStop) {
                sendAfterStop = false;
                form.requestSubmit(send);
            }
        };
        mic.addEventListener("click", () => {
            if (listening) {
                recognition.stop();
                status.textContent = "Review your words, then press Send.";
            } else {
                try { recognition.start(); }
                catch (_) { status.textContent = "Microphone could not start. You can type your response."; }
            }
        });
    } else if (mic) {
        mic.disabled = true;
        if (status && input && !input.readOnly) status.textContent = "Voice input is unavailable here. Type your response, or try Chrome or Edge over HTTPS.";
    }
    if (form && input) {
        input.addEventListener("keydown", event => {
            if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
                event.preventDefault();
                form.requestSubmit(send);
            }
        });
        form.addEventListener("submit", event => {
            const action = event.submitter?.value || "send_message";
            if (action !== "send_message") {
                sendAfterStop = false;
                if (recognition && listening) recognition.abort();
                return;
            }
            if (listening) {
                event.preventDefault();
                sendAfterStop = true;
                recognition.stop();
                return;
            }
            if (!input.value.trim() || form.dataset.sending === "1") {
                event.preventDefault();
                return;
            }
            if (!turn.value) turn.value = window.crypto?.randomUUID?.() || Date.now() + "-" + Math.random().toString(16).slice(2);
            form.dataset.sending = "1";
            send.disabled = true;
            send.textContent = "Waiting for response…";
            if (mic) mic.disabled = true;
            stopOthers(null);
            status.textContent = "Your response is being processed. Please wait.";
        });
    }
    window.addEventListener("pageshow", event => { if (event.persisted) window.location.reload(); });
    const params = new URLSearchParams(window.location.search);
    // Scrolling the transcript alone does not move the surrounding page after
    // a form submission. Reveal the new reply and composer after navigation's
    // scroll restoration, including when the learner needs to retry a reply.
    if (params.get("autoplay") === "1" || input?.readOnly) {
        window.addEventListener("pageshow", () => requestAnimationFrame(() => {
            if (chat) {
                chat.scrollTop = chat.scrollHeight;
                chat.lastElementChild?.scrollIntoView({block: "center", behavior: "instant"});
            }
            const nextAction = input || document.querySelector(".closed-banner");
            nextAction?.scrollIntoView({block: "nearest", behavior: "instant"});
        }), {once: true});
    }
    if (params.get("autoplay") === "1") {
        const buttons = document.querySelectorAll(".tts-button");
        if ((autoplay ? autoplay.checked : preference("autoplay_voice_enabled", "0") === "1") && buttons.length) playVoice(buttons[buttons.length - 1]);
        params.delete("autoplay");
        window.history.replaceState({}, "", window.location.pathname + (params.size ? "?" + params : ""));
    }
})();
