FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first for better build cache reuse.
COPY vipdjango/requirements.txt /app/vipdjango/requirements.txt
RUN pip install --no-cache-dir -r /app/vipdjango/requirements.txt

# Copy Django project.
COPY vipdjango /app/vipdjango
COPY prompts /app/prompts

# Entrypoint handles migrations + static collection before app startup.
COPY vipdjango/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app/vipdjango

EXPOSE 8080 8081

ENTRYPOINT ["/app/entrypoint.sh"]
