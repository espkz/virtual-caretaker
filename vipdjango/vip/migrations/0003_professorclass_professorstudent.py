# Generated for professor-scoped student and class ownership.

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vip", "0002_chatsession_chatmessage_roleprompt_and_more"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="ProfessorClass",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=150)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "professor",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="owned_classes",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["name"],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("professor", "name"),
                        name="unique_professor_class_name",
                    )
                ],
            },
        ),
        migrations.CreateModel(
            name="ProfessorStudent",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("student_number", models.CharField(blank=True, max_length=30)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "class_group",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="student_assignments",
                        to="vip.professorclass",
                    ),
                ),
                (
                    "professor",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="owned_students",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "student",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="professor_assignment",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["student__username"],
            },
        ),
    ]
