from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("vip", "0015_deactivate_incomplete_prompts")]

    operations = [
        migrations.AddField(
            model_name="voiceconversation",
            name="introduction_completed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
