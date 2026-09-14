"""An owned streaming response that closes upstream resources on disconnect."""
from contextlib import ExitStack


class SpeechStream:
    def __init__(self, api_key, **kwargs):
        from openai import OpenAI

        self.resources = ExitStack()
        try:
            client = self.resources.enter_context(OpenAI(api_key=api_key, timeout=30.0, max_retries=0))
            response = self.resources.enter_context(client.audio.speech.with_streaming_response.create(**kwargs))
            self.chunks = response.iter_bytes(chunk_size=4096)
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.chunks)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.resources.close()
