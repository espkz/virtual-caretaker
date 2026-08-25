from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vip", "0005_chatsession_turn_lifecycle"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="active_claim_id",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
    ]
