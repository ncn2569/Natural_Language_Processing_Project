"""Gemini REST contract and fallback, no real key or network required."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from src.hcmut.iaslab.nlp.app.rag import DEFAULT_MODEL, SYSTEM, call_llm, get_api_key

URL_OPEN = "src.hcmut.iaslab.nlp.app.rag.urllib.request.urlopen"


def reply(parts=None, finish="STOP"):
    return io.BytesIO(json.dumps({"candidates": [{"finishReason": finish, "content": {
        "parts": parts if parts is not None else [{"text": "3 tín chỉ. [faq:Q1]"}]
    }}]}).encode())


class GeminiTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.env_file = Path(temp.name) / ".env"
        patcher = patch("src.hcmut.iaslab.nlp.app.rag.ENV_FILE", self.env_file)
        patcher.start()
        self.addCleanup(patcher.stop)
        timer = patch("src.hcmut.iaslab.nlp.app.rag._next_request_at", 0.0)
        timer.start()
        self.addCleanup(timer.stop)
        sleep = patch("src.hcmut.iaslab.nlp.app.rag.time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def test_google_key_from_dotenv_and_process_precedence(self):
        self.env_file.write_text("# settings\nGOOGLE_API_KEY='file-test-key'\nRAG_MODEL=gemini-test\nUNRELATED=ignore\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_api_key(), "file-test-key")
            self.assertEqual(os.environ["RAG_MODEL"], "gemini-test")
            self.assertNotIn("UNRELATED", os.environ)
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "process-test-key"}, clear=True):
            self.assertEqual(get_api_key(), "process-test-key")

    def test_google_key_is_used_for_request(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-test-key"}, clear=True), patch(URL_OPEN, return_value=reply()) as request:
            self.assertIsNotNone(call_llm("test"))
            self.assertEqual(request.call_args.args[0].get_header("X-goog-api-key"), "google-test-key")

    def test_request_and_response_follow_gemini_contract(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
            URL_OPEN, return_value=reply()
        ) as request:
            self.assertEqual(call_llm("Câu hỏi thử"), "3 tín chỉ. [faq:Q1]")
        req = request.call_args.args[0]
        self.assertEqual(req.full_url, f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_MODEL}:generateContent")
        self.assertEqual(req.get_header("X-goog-api-key"), "test-key")
        self.assertNotIn("test-key", req.full_url)
        body = json.loads(req.data)
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], SYSTEM)
        self.assertEqual(body["contents"], [{"role": "user", "parts": [{"text": "Câu hỏi thử"}]}])
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], 1024)
        self.assertEqual(request.call_args.kwargs["timeout"], 30)

    def test_model_override_and_thought_parts_are_not_answers(self):
        parts = [{"thought": True, "text": "internal"}, {"text": "60 "}, {"text": "phút."}]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test", "RAG_MODEL": "gemini-test"}), patch(
            URL_OPEN, return_value=reply(parts)
        ) as request:
            self.assertEqual(call_llm("test"), "60 phút.")
            self.assertIn("models/gemini-test:generateContent", request.call_args.args[0].full_url)

    def test_no_key_never_calls_network(self):
        with patch.dict(os.environ, {}, clear=True), patch(URL_OPEN) as request:
            self.assertIsNone(call_llm("test"))
            request.assert_not_called()

    def test_blocked_truncated_empty_or_malformed_response_falls_back(self):
        samples = [b'{"promptFeedback":{"blockReason":"SAFETY"}}',
                   b'{"candidates":[]}', b'{"candidates":"invalid"}', b'not json']
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            for raw in samples:
                with self.subTest(raw=raw), patch(URL_OPEN, return_value=io.BytesIO(raw)), redirect_stdout(io.StringIO()):
                    self.assertIsNone(call_llm("test"))
            for response in (reply(finish="MAX_TOKENS"), reply(finish="SAFETY"), reply([]), reply([{"thought": True, "text": "internal"}])):
                with patch(URL_OPEN, return_value=response), redirect_stdout(io.StringIO()):
                    self.assertIsNone(call_llm("test"))

    def test_http_and_network_errors_fall_back_without_exposing_details(self):
        secret = "private-value-do-not-log"
        errors = [urllib.error.HTTPError("https://example", code, secret, {}, None) for code in (401, 404, 429, 500)]
        errors += [urllib.error.URLError(secret), TimeoutError(secret)]
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
            for error in errors:
                output = io.StringIO()
                with self.subTest(error=type(error).__name__), patch(URL_OPEN, side_effect=error), redirect_stdout(output):
                    self.assertIsNone(call_llm("test"))
                self.assertNotIn(secret, output.getvalue())

    def test_rate_limit_retries_and_paces_next_model(self):
        error = urllib.error.HTTPError(
            "https://example", 429, "rate limit", {"Retry-After": "25"}, None
        )
        clock = "src.hcmut.iaslab.nlp.app.rag.time.monotonic"
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True),
            patch(clock, return_value=100.0),
            patch(URL_OPEN, side_effect=[error, reply(), reply()]) as request,
        ):
            self.assertIsNotNone(call_llm("answer"))
            self.assertIsNotNone(call_llm("judge", model="gemma-3-27b-it"))
        self.assertEqual(request.call_count, 3)
        delays = [call.args[0] for call in self.sleep.call_args_list]
        self.assertEqual(delays, [25.0, 10.0])

    def test_rate_limit_retries_are_bounded(self):
        error = urllib.error.HTTPError("https://example", 429, "limit", {}, None)
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True),
            patch(URL_OPEN, side_effect=error) as request,
            redirect_stdout(io.StringIO()),
        ):
            self.assertIsNone(call_llm("answer"))
        self.assertEqual(request.call_count, 3)

    def test_invalid_model_cannot_change_request_target(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test", "RAG_MODEL": "https://example"}), patch(URL_OPEN) as request:
            with self.assertRaises(ValueError):
                call_llm("test")
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
