from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vip", "0002_chatsession_chatmessage_roleprompt_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="conversation_phase",
            field=models.CharField(default="normal", max_length=32),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="conversation_stage",
            field=models.CharField(default="beginning", max_length=20),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="completion_status",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="chatmessage",
            name="voice_metadata",
            field=models.TextField(blank=True, default=""),
        ),
    ]
