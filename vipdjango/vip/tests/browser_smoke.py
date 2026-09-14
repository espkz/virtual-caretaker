"""Opt-in real Chrome check: manage.py test vip.tests.browser_smoke --settings=vipson_manager.test_settings"""
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from playwright.sync_api import sync_playwright

from vip.conversation_engine import ConversationEngine
from vip.models import ChatSession, RolePrompt
from .test_practice import scenario_text, selection
from .test_clinician_demo import nurse_text, reply
from vip.conversation_scenario import parse_scenario_prompt


class BrowserSmokeTests(StaticLiveServerTestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    def test_instructor_can_select_nurse_draft_and_play_rachel(self):
        user = get_user_model().objects.create_user("nurse-demo-instructor")
        user.groups.add(Group.objects.create(name="Professor"))
        prompt = RolePrompt.objects.create(title="AI nurse demo", content=nurse_text(), is_active=False)
        self.client.force_login(user)
        cookie = self.client.cookies["sessionid"].value
        topics = [objective.id for objective in parse_scenario_prompt(nurse_text()).objectives]
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
            try:
                page = browser.new_page()
                page.context.add_cookies([{"name": "sessionid", "value": cookie, "url": self.live_server_url}])
                page.goto(f"{self.live_server_url}/professor/test-chat/?prompt={prompt.id}")
                self.assertIn("AI nurse demo", page.locator('.prompt-select option:checked').inner_text())
                with page.expect_navigation():
                    page.get_by_role("button", name="New Chat", exact=True).click()
                self.assertIn("you play Rachel", page.locator('.chat-window').inner_text())
                page.locator('#student-message-input').fill("Won't she feel hungry and thirsty?")
                nurse_answer = "We will continue comfort care. Do these explanations make sense, and do you feel ready?"
                with patch.object(ConversationEngine, "_request_clinician_reply", return_value=reply(nurse_answer, topics=topics, readiness=True)):
                    with page.expect_navigation():
                        page.get_by_role("button", name="Send", exact=True).click()
                self.assertIn(nurse_answer, page.locator('.chat-message').last.inner_text())
                self.assert_composer_in_view(page)
                page.locator('#student-message-input').fill("Yes, I'm ready to proceed.")
                with patch.object(ConversationEngine, "_request_clinician_reply", return_value=reply(intent="ready")):
                    with page.expect_navigation():
                        page.get_by_role("button", name="Send", exact=True).click()
                self.assertTrue(page.locator('.closed-banner').is_visible())
                self.assertIn("Thank you for talking through your concerns with me, Rachel", page.locator('.chat-message').last.inner_text())
            finally:
                browser.close()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    def test_student_login_retry_and_complete_session(self):
        user = get_user_model().objects.create_user("browserlearner", password="browser-test-password")
        user.groups.add(Group.objects.create(name="Class: Browser test"))
        RolePrompt.objects.create(title="Browser scenario", content=scenario_text(), is_active=True)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
            try:
                page = browser.new_page(viewport={"width": 390, "height": 844})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script("Object.defineProperty(window, 'localStorage', {get() {throw new Error('Storage disabled');}})")
                page.goto(self.live_server_url)
                page.locator('[name="username"]').fill("browserlearner")
                page.locator('[name="password"]').fill("browser-test-password")
                with page.expect_navigation():
                    page.locator('button[type="submit"]').click()
                with page.expect_navigation():
                    page.get_by_role("button", name="Start New Chat", exact=True).click()
                page.locator('#student-message-input').fill("Hello, I'm your nurse today.")
                with patch("vip.views._generate_assistant_response", side_effect=RuntimeError("synthetic outage")):
                    with self.assertLogs("vip.views", level="ERROR"):
                        with page.expect_navigation():
                            page.get_by_role("button", name="Send", exact=True).click()
                self.assertTrue(page.get_by_role("button", name="Retry response").is_visible())
                self.assertTrue(page.locator('#student-message-input').get_attribute("readonly") is not None)
                self.assert_composer_in_view(page)
                with patch.object(ConversationEngine, "_assess_core_reply", return_value=selection()):
                    with page.expect_navigation():
                        page.get_by_role("button", name="Retry response").click()
                    self.assert_composer_in_view(page)
                    for turn in range(6):
                        page.locator('#student-message-input').fill("Here is a reasonable explanation of that concern.")
                        with page.expect_navigation():
                            page.get_by_role("button", name="Send", exact=True).click()
                        if turn < 5:
                            self.assert_composer_in_view(page)
                self.assertTrue(page.locator('.closed-banner').is_visible())
                page.wait_for_function("""() => {
                    const rect = document.querySelector('.closed-banner').getBoundingClientRect();
                    return rect.top >= 0 && rect.bottom <= window.innerHeight;
                }""")
                self.assertEqual(errors, [])
            finally:
                browser.close()
        self.assertEqual(ChatSession.objects.get().messages.filter(sender="student").count(), 7)

    def assert_composer_in_view(self, page):
        page.wait_for_function("""() => {
            const input = document.querySelector('#student-message-input').getBoundingClientRect();
            const reply = document.querySelector('.chat-window').lastElementChild.getBoundingClientRect();
            return input.top >= 0 && input.bottom <= window.innerHeight
                && reply.bottom > 0 && reply.top < window.innerHeight;
        }""")
