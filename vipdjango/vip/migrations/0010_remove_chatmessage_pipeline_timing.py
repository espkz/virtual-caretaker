from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("vip", "0009_chatsession_topic_progress"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="chatmessage",
            name="pipeline_timing",
        ),
    ]
