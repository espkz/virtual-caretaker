from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from vip.models import RolePrompt


class Command(BaseCommand):
    help = "Import the two checked-in Rachel scenarios as inactive drafts without overwriting existing prompts."

    def add_arguments(self, parser):
        parser.add_argument("--faculty-feedback-v2", action="store_true", help="Import separate September 17 drafts with 25 exchanges and varied question choices.")
        parser.add_argument("--faculty-feedback", action="store_true", help="Import the September 2026 full-scenario revision as new inactive drafts, preserving existing prompts.")
        parser.add_argument("--clinician-demo", action="store_true", help="Also import the Scenario 2 AI hospice nurse demonstration as a separate draft.")

    def handle(self, *args, **options):
        suffix = "September 2026 revision" if options["faculty_feedback"] else "core questions"
        if options["faculty_feedback_v2"]:
            suffix = "September 17, 2026 revision"
        for number, label in [(1, "Caregiver training"), (2, "Hospice communication")]:
            path = Path(settings.BASE_DIR).parent / "prompts" / f"role_rachel_ellison_{number}.md"
            prompt, created = RolePrompt.objects.get_or_create(
                title=f"Rachel Ellison {number}: {label} ({suffix})",
                defaults={"content": path.read_text(encoding="utf-8"), "is_active": False},
            )
            self.stdout.write(f"{'Imported draft' if created else 'Already exists, unchanged'}: {prompt.title}")
        if options["clinician_demo"]:
            path = Path(settings.BASE_DIR).parent / "prompts" / "role_hospice_nurse_2.md"
            prompt, created = RolePrompt.objects.get_or_create(
                title="Scenario 2: AI hospice nurse (you play Rachel)",
                defaults={"content": path.read_text(encoding="utf-8"), "is_active": False},
            )
            self.stdout.write(f"{'Imported draft' if created else 'Already exists, unchanged'}: {prompt.title}")
