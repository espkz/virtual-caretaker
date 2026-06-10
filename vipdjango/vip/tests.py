from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase, override_settings
from django.urls import reverse

from .models import ProfessorClass, ProfessorStudent
from .views import _student_queryset


@override_settings(SECURE_SSL_REDIRECT=False)
class ProfessorStudentIsolationTests(TestCase):
    def setUp(self):
        User = get_user_model()
        professor_group = Group.objects.create(name="Professor")
        student_group = Group.objects.create(name="Student")

        self.professor_a = User.objects.create_user(username="prof_a", password="pass")
        self.professor_b = User.objects.create_user(username="prof_b", password="pass")
        self.professor_a.groups.add(professor_group)
        self.professor_b.groups.add(professor_group)

        self.student_a = User.objects.create_user(username="alice", password="pass", first_name="Alice")
        self.student_b = User.objects.create_user(username="bob", password="pass", first_name="Bob")
        self.student_a.groups.add(student_group)
        self.student_b.groups.add(student_group)

        self.class_a = ProfessorClass.objects.create(professor=self.professor_a, name="Nursing A")
        self.class_b = ProfessorClass.objects.create(professor=self.professor_b, name="Nursing B")
        ProfessorStudent.objects.create(
            professor=self.professor_a,
            student=self.student_a,
            class_group=self.class_a,
            student_number="1001",
        )
        ProfessorStudent.objects.create(
            professor=self.professor_b,
            student=self.student_b,
            class_group=self.class_b,
            student_number="2002",
        )

    def test_student_queryset_is_scoped_to_professor(self):
        self.assertEqual(list(_student_queryset(self.professor_a)), [self.student_a])
        self.assertEqual(list(_student_queryset(self.professor_b)), [self.student_b])

    def test_account_page_only_lists_current_professor_roster(self):
        self.client.force_login(self.professor_a)

        response = self.client.get(reverse("vip:professor_manage_accounts"))

        self.assertContains(response, "alice")
        self.assertContains(response, "Nursing A")
        self.assertNotContains(response, "bob")
        self.assertNotContains(response, "Nursing B")

    def test_professor_cannot_open_another_professors_student_logs(self):
        self.client.force_login(self.professor_a)

        response = self.client.get(reverse("vip:professor_student_logs", args=[self.student_b.id]))

        self.assertEqual(response.status_code, 404)
