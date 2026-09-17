/* Shared text-conversation behavior. Voice mode is isolated in voice_chat.js. */
(() => {
    const form = document.querySelector("form.message-form");
    const input = document.getElementById("student-message-input");
    const send = document.getElementById("send-message-btn");
    const turn = document.getElementById("conversation-turn-id") || document.getElementById("voice-turn-id");
    const status = document.getElementById("message-status") || document.getElementById("stt-status");
    const chat = document.querySelector(".chat-window");
    if (chat) chat.scrollTop = chat.scrollHeight;

    if (form && input && send) {
        input.addEventListener("keydown", event => {
            if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
                event.preventDefault();
                form.requestSubmit(send);
            }
        });
        form.addEventListener("submit", event => {
            const action = event.submitter?.value || "send_message";
            if (action !== "send_message") return;
            if (!input.value.trim() || form.dataset.sending === "1") {
                event.preventDefault();
                return;
            }
            if (turn && !turn.value) {
                turn.value = window.crypto?.randomUUID?.() || Date.now() + "-" + Math.random().toString(16).slice(2);
            }
            form.dataset.sending = "1";
            send.disabled = true;
            send.textContent = "Waiting for response…";
            if (status) status.textContent = "Your response is being processed. Please wait.";
        });
    }

    window.addEventListener("pageshow", event => {
        if (event.persisted) window.location.reload();
    });
    const params = new URLSearchParams(window.location.search);
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
        params.delete("autoplay");
        window.history.replaceState({}, "", window.location.pathname + (params.size ? "?" + params : ""));
    }
})();
