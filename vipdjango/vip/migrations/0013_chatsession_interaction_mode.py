from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vip", "0012_voiceconversation"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="interaction_mode",
            field=models.CharField(
                choices=[("text", "Text"), ("voice", "Voice")],
                default="text",
                max_length=10,
            ),
        ),
    ]
