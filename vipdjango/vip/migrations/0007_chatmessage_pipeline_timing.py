from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("vip", "0006_chatsession_active_claim"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatmessage",
            name="pipeline_timing",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
