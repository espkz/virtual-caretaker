from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("vip", "0016_voiceconversation_introduction_completed_at")]

    operations = [
        migrations.AddField(
            model_name="voiceconversation",
            name="is_muted",
            field=models.BooleanField(default=False),
        ),
    ]
