import uuid

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vip", "0011_bounded_practice"),
    ]

    operations = [
        migrations.CreateModel(
            name="VoiceConversation",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("provider_conversation_id", models.CharField(blank=True, default="", max_length=128)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("token_issued", "Token issued"),
                            ("connected", "Connected"),
                            ("ended", "Ended"),
                            ("error", "Error"),
                        ],
                        default="token_issued",
                        max_length=20,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("connected_at", models.DateTimeField(blank=True, null=True)),
                ("ended_at", models.DateTimeField(blank=True, null=True)),
                ("failure_reason", models.CharField(blank=True, default="", max_length=240)),
                (
                    "chat_session",
                    models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="voice_conversations", to="vip.chatsession"),
                ),
            ],
        ),
        migrations.AddConstraint(
            model_name="voiceconversation",
            constraint=models.UniqueConstraint(
                condition=~models.Q(("provider_conversation_id", "")),
                fields=("provider_conversation_id",),
                name="unique_nonempty_voice_provider_conversation",
            ),
        ),
    ]
