from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("vip", "0008_chatsession_objective_progress"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="active_topic",
            field=models.CharField(blank=True, default="", max_length=160),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="covered_topics",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="unresolved_topics",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="topic_turn_counts",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
