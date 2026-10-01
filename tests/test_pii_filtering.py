import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect
from gtm_agent import data_service


PROSPECT_ID = "LEAD-12853"
SENSITIVE_KEYS = {"billing_qualification", "tax_id", "date_of_birth", "card_on_file", "credit_check_ref"}


def assert_no_sensitive_keys(test_case, value):
    if isinstance(value, dict):
        test_case.assertTrue(SENSITIVE_KEYS.isdisjoint(value))
        for item in value.values():
            assert_no_sensitive_keys(test_case, item)
    elif isinstance(value, list):
        for item in value:
            assert_no_sensitive_keys(test_case, item)


class ProspectPiiFilteringTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()

    def test_get_prospect_allowlists_contact_fields(self):
        result = get_prospect.invoke({"prospect_id": PROSPECT_ID})

        self.assertEqual(
            set(result["prospect"]),
            {"prospect_id", "name", "email", "disqualified"},
        )
        assert_no_sensitive_keys(self, result)

    def test_build_prospect_profile_filters_fresh_profile(self):
        fresh = build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

        assert_no_sensitive_keys(self, fresh)

    def test_build_prospect_profile_filters_stale_cached_profile(self):
        data_service._PROFILES[PROSPECT_ID] = {
            "prospect_id": PROSPECT_ID,
            "name": "Omar Okafor",
            "billing_qualification": {"tax_id": "541-88-2307"},
            "nested": [{"card_on_file": "4539882100437712"}],
        }

        cached = build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

        assert_no_sensitive_keys(self, cached)

    def test_score_prospect_filters_sensitive_nested_profile_data(self):
        profile = {
            "prospect_id": PROSPECT_ID,
            "name": "Omar Okafor",
            "billing_qualification": {
                "tax_id": "541-88-2307",
                "date_of_birth": "1984-03-11",
                "card_on_file": "4539882100437712",
                "credit_check_ref": "EXPN-88213",
            },
            "nested": [{"tax_id": "nested-secret"}],
        }
        offering = {
            "required_tech_stack": ["Salesforce"],
            "min_annual_revenue": 1,
            "description": "An offering",
        }
        llm_result = Mock(model_dump=lambda: {"score": 80})

        scoring_llm = Mock(invoke=Mock(return_value=llm_result))
        with patch("gtm_agent.gtm_agent._scoring_llm", scoring_llm):
            score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        serialized_profile = user_message.split("\n\nProspect profile:\n", 1)[1]
        assert_no_sensitive_keys(self, json.loads(serialized_profile))


if __name__ == "__main__":
    unittest.main()
