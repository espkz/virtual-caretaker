from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vip", "0007_chatmessage_pipeline_timing"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="active_objective",
            field=models.CharField(blank=True, default="", max_length=160),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="covered_objectives",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="unresolved_objectives",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="recent_topics",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="ending_ready",
            field=models.BooleanField(default=False),
        ),
    ]
