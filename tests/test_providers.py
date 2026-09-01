import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from retrofit_rag.providers import (
    ConfigurationError,
    OllamaLLM,
    OpenAICompatibleLLM,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return json.dumps(self.payload).encode("utf-8")


class ProviderTest(unittest.TestCase):
    @patch("retrofit_rag.providers.urlopen")
    def test_openai_compatible_adapter(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse(
            {"choices": [{"message": {"content": '{"answer":"x [1]","citations":[1]}'}}]}
        )
        client = OpenAICompatibleLLM("test-key", "test-model", "https://llm.example/v1")
        output = client.generate([{"role": "user", "content": "question"}])
        self.assertIn('"citations"', output)
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://llm.example/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    @patch("retrofit_rag.providers.urlopen")
    def test_ollama_adapter(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse(
            {"message": {"content": '{"answer":"x [1]","citations":[1]}'}}
        )
        client = OllamaLLM("local-model", "http://ollama.example")
        output = client.generate([{"role": "user", "content": "question"}])
        self.assertIn('"answer"', output)
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://ollama.example/api/chat")
        payload = json.loads(request.data)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], "json")

    @patch("retrofit_rag.providers.urlopen")
    def test_ollama_readiness_requires_configured_model(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse(
            {"models": [{"name": "llama3.2:3b-instruct-q4_K_M"}]}
        )
        client = OllamaLLM("llama3.2:3b-instruct-q4_K_M", "http://ollama.example")
        self.assertTrue(client.ready())
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://ollama.example/api/tags")

    @patch("retrofit_rag.providers.urlopen")
    def test_openai_readiness_checks_models_endpoint(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse({"data": []})
        client = OpenAICompatibleLLM("test-key", "test-model", "https://llm.example/v1")
        self.assertTrue(client.ready())
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://llm.example/v1/models")

    def test_openai_environment_requires_key_and_model(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigurationError):
                OpenAICompatibleLLM.from_env()

    def test_ollama_environment_requires_model(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigurationError):
                OllamaLLM.from_env()


if __name__ == "__main__":
    unittest.main()
