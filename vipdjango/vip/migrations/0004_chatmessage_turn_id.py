from django.db import migrations, models
from django.db.models import Q


class Migration(migrations.Migration):
    dependencies = [
        ("vip", "0003_conversation_state_and_voice_metadata"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatmessage",
            name="turn_id",
            field=models.CharField(blank=True, default="", db_index=True, max_length=64),
        ),
        migrations.AddConstraint(
            model_name="chatmessage",
            constraint=models.UniqueConstraint(
                condition=~Q(turn_id=""),
                fields=("session", "sender", "turn_id"),
                name="unique_nonempty_message_turn_per_sender",
            ),
        ),
    ]
