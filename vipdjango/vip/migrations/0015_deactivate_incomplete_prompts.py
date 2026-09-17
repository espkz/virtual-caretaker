import re

from django.db import migrations


REQUIRED_HEADINGS = (
    "simulation mode",
    "role",
    "background and context",
    "user role",
    "conversation goals",
    "introduction",
    "opening line",
    "beginning",
    "beginning to middle transition",
    "middle",
    "middle to ending transition",
    "ending",
    "closing",
    "introduction voice",
    "roleplay voice",
    "voice style",
)


def normalized_heading(value):
    value = value.strip().lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def populated_sections(content):
    sections = {}
    current = ""
    for line in (content or "").splitlines():
        match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if match:
            current = normalized_heading(match.group(1))
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line)
    return {
        heading
        for heading, lines in sections.items()
        if "\n".join(lines).strip()
    }


def deactivate_incomplete_prompts(apps, schema_editor):
    RolePrompt = apps.get_model("vip", "RolePrompt")
    incomplete_ids = []
    for prompt in RolePrompt.objects.filter(is_active=True).iterator():
        sections = populated_sections(prompt.content)
        if not (prompt.title or "").strip() or any(
            heading not in sections for heading in REQUIRED_HEADINGS
        ):
            incomplete_ids.append(prompt.pk)
    if incomplete_ids:
        RolePrompt.objects.filter(pk__in=incomplete_ids).update(is_active=False)


class Migration(migrations.Migration):
    dependencies = [("vip", "0014_voice_pause_and_resources")]

    operations = [
        migrations.RunPython(deactivate_incomplete_prompts, migrations.RunPython.noop),
    ]
