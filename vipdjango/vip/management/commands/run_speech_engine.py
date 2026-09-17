"""Runtime entrypoint for the authenticated ElevenLabs upstream adapter.

The external Speech Engine resource and its ID are managed outside Django;
this command only starts the local WebSocket process that resource connects to.
"""

import asyncio

from django.core.management.base import BaseCommand, CommandError

from vip.speech_engine.adapter import serve_speech_engine
from vip.speech_engine.service import SpeechEngineConfigurationError


class Command(BaseCommand):
    help = "Run the isolated, authenticated ElevenLabs Speech Engine upstream server."

    def add_arguments(self, parser):
        parser.add_argument("--port", type=int, default=None, help="WebSocket port (default: SPEECH_ENGINE_PORT or 3001).")
        parser.add_argument("--debug", action="store_true", help="Enable SDK protocol logs; do not use in production.")

    def handle(self, *args, **options):
        try:
            asyncio.run(serve_speech_engine(port=options["port"], debug=options["debug"]))
        except SpeechEngineConfigurationError as error:
            raise CommandError(str(error)) from error
