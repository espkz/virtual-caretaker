#!/bin/sh
set -e

cd /app/vipdjango

python manage.py migrate --noinput
python manage.py collectstatic --noinput

PORT="${PORT:-8080}"
WORKERS="${GUNICORN_WORKERS:-3}"
TIMEOUT="${GUNICORN_TIMEOUT:-120}"

# Both services are required.  Keep them as siblings so a Speech Engine
# startup/runtime failure is visible in the container logs and also takes the
# web process down instead of silently leaving a partly working deployment.
python manage.py run_speech_engine --port "${SPEECH_ENGINE_PORT:-8081}" &
speech_engine_pid=$!

gunicorn vipson_manager.wsgi:application --bind "0.0.0.0:${PORT}" --workers "${WORKERS}" --timeout "${TIMEOUT}" &
web_pid=$!

stop_children() {
    kill -TERM "${speech_engine_pid}" "${web_pid}" 2>/dev/null || true
    wait "${speech_engine_pid}" 2>/dev/null || true
    wait "${web_pid}" 2>/dev/null || true
    exit 0
}

trap stop_children INT TERM

process_running() {
    child_stat_path="/proc/$1/stat"
    [ -r "${child_stat_path}" ] || return 1
    IFS= read -r child_stat_line < "${child_stat_path}" || return 1
    # /proc/<pid>/stat has the process name in parentheses; the state is the
    # first field after that closing parenthesis.  Longest-match trimming
    # handles names containing spaces, such as Gunicorn's master process.
    child_state="${child_stat_line##*) }"
    child_state="${child_state%% *}"
    [ "${child_state}" != "Z" ] && [ "${child_state}" != "X" ]
}

while :; do
    if ! process_running "${speech_engine_pid}"; then
        if wait "${speech_engine_pid}"; then service_status=0; else service_status=$?; fi
        echo "SpeechEngine process exited with status ${service_status}; stopping web service." >&2
        kill -TERM "${web_pid}" 2>/dev/null || true
        wait "${web_pid}" 2>/dev/null || true
        exit "${service_status}"
    fi
    if ! process_running "${web_pid}"; then
        if wait "${web_pid}"; then service_status=0; else service_status=$?; fi
        echo "Web service exited with status ${service_status}; stopping SpeechEngine." >&2
        kill -TERM "${speech_engine_pid}" 2>/dev/null || true
        wait "${speech_engine_pid}" 2>/dev/null || true
        exit "${service_status}"
    fi
    sleep 1
done
