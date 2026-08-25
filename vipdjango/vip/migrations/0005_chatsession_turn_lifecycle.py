from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vip", "0004_chatmessage_turn_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="active_turn_id",
            field=models.CharField(blank=True, db_index=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="last_completed_turn_id",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
    ]
