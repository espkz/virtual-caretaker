from django.db import migrations, models


def snapshot_scenarios(apps, schema_editor):
    Session = apps.get_model("vip", "ChatSession")
    for session in Session.objects.select_related("role_prompt").iterator():
        if session.role_prompt:
            Session.objects.filter(pk=session.pk).update(scenario_content=session.role_prompt.content)


class Migration(migrations.Migration):
    dependencies = [("vip", "0010_remove_chatmessage_pipeline_timing")]
    operations = [
        migrations.AddField("chatsession", "scenario_content", models.TextField(blank=True, default="")),
        migrations.AddField("chatsession", "core_question_state", models.JSONField(blank=True, default=dict)),
        migrations.AddField("chatsession", "active_claimed_at", models.DateTimeField(blank=True, null=True)),
        migrations.RunPython(snapshot_scenarios, migrations.RunPython.noop),
    ]
