from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("vip", "0013_chatsession_interaction_mode")]

    operations = [
        migrations.AddField(
            model_name="voiceconversation",
            name="is_paused",
            field=models.BooleanField(default=False),
        ),
        migrations.CreateModel(
            name="SpeechEngineVoiceResource",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("voice_id", models.CharField(max_length=128, unique=True)),
                ("speech_engine_id", models.CharField(max_length=128, unique=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
