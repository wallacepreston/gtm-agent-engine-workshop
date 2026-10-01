import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


def call_send(prospect_id):
    return send_prospect_email.func(
        prospect_id,
        "Book a demo",
        "Please book a demo.",
        SimpleNamespace(config={"metadata": {}}),
        {"name": "Rep", "email": "rep@example.com"},
    )


def test_disqualified_prospect_is_blocked_and_prompt_requires_transparency():
    result = call_send("LEAD-50001")

    assert result == {
        "status": "blocked",
        "error": "Prospect is disqualified; email not sent.",
    }
    assert "do not retry or work around the block" in SYSTEM_PROMPT
    assert "email was not sent and why" in SYSTEM_PROMPT


def test_qualified_prospect_is_sent_from_source_record():
    result = call_send("LEAD-12853")

    assert result["status"] == "sent"
    assert result["to"] == "omar.okafor@lakesideanalytics.com"
    assert result["to_name"] == "Omar Okafor"
    assert result["message_id"].startswith("msg-")


def test_unknown_prospect_is_blocked_without_message_id():
    result = call_send("LEAD-40404")

    assert result == {
        "status": "blocked",
        "error": "Prospect not found; email not sent.",
    }
