import json
import os
import sys
import types
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

package = types.ModuleType("gtm_agent")
package.__path__ = [str(Path(__file__).parent)]
sys.modules.setdefault("gtm_agent", package)
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent import gtm_agent as agent_module
from gtm_agent.gtm_agent import score_prospect
from gtm_agent.gtm_records import OFFERINGS, PROSPECTS


class _FakeScore:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


class _FakeScoringLlm:
    def invoke(self, messages):
        return _FakeScore(_score_payload(messages))


def _score_payload(messages):
    profile = json.loads(messages[1]["content"].split("\n\nProspect profile:\n", 1)[1])
    offering = json.loads(messages[1]["content"].split("Offering:\n", 1)[1].split("\n\nProspect profile:\n", 1)[0])
    required = offering["required_tech_stack"]
    matched = [technology for technology in required if technology in profile["tech_stack"]]
    missing = [technology for technology in required if technology not in profile["tech_stack"]]
    return {
        "score": 50 + 10 * len(matched),
        "max_score": 100,
        "justification": f"Matched: {', '.join(matched)}. Missing: {', '.join(missing)}.",
        "rubric_breakdown": {
            "revenue_fit": 25,
            "tech_stack_match": 10 * len(matched),
            "segment_fit": 25,
            "component_max": 100,
        },
    }


def test_update_and_score_lead_71001_with_kafka():
    prospect_id = "LEAD-71001"
    original_record = deepcopy(PROSPECTS[prospect_id])
    original_profile = data_service._PROFILES.pop(prospect_id, None)
    try:
        before_profile = agent_module.build_prospect_profile.invoke(prospect_id)["prospect_profile"]
        with patch.object(agent_module, "_scoring_llm", _FakeScoringLlm()):
            before = score_prospect.invoke({
                "prospect_profile": before_profile,
                "offering": OFFERINGS["OFFER-10005"],
            })
            update = data_service.update_prospect_info(prospect_id, "Kafka")
            after_profile = agent_module.build_prospect_profile.invoke(prospect_id)
            after = score_prospect.invoke({
                "prospect_profile": after_profile["prospect_profile"],
                "offering": OFFERINGS["OFFER-10005"],
            })

        assert update["updated"] is True
        assert "Kafka" in data_service.fetch_tech_stack(prospect_id)
        assert "Kafka" not in after["justification"].split("Missing: ", 1)[-1]
        assert after["rubric_breakdown"]["tech_stack_match"] > before["rubric_breakdown"]["tech_stack_match"]
        assert after["score"] > before["score"]
    finally:
        PROSPECTS[prospect_id] = original_record
        if original_profile is None:
            data_service._PROFILES.pop(prospect_id, None)
        else:
            data_service._PROFILES[prospect_id] = original_profile


def test_update_and_score_lead_39002_with_terraform():
    prospect_id = "LEAD-39002"
    original_record = deepcopy(PROSPECTS[prospect_id])
    original_profile = data_service._PROFILES.pop(prospect_id, None)
    try:
        before_profile = agent_module.build_prospect_profile.invoke(prospect_id)["prospect_profile"]
        with patch.object(agent_module, "_scoring_llm", _FakeScoringLlm()):
            before = score_prospect.invoke({
                "prospect_profile": before_profile,
                "offering": OFFERINGS["OFFER-10004"],
            })
            update = data_service.update_prospect_info(prospect_id, "Terraform")
            after_profile = agent_module.build_prospect_profile.invoke(prospect_id)
            after = score_prospect.invoke({
                "prospect_profile": after_profile["prospect_profile"],
                "offering": OFFERINGS["OFFER-10004"],
            })

        assert update["updated"] is True
        assert "Terraform" in data_service.fetch_tech_stack(prospect_id)
        assert "Terraform" not in after["justification"].split("Missing: ", 1)[-1]
        assert after["rubric_breakdown"]["tech_stack_match"] > before["rubric_breakdown"]["tech_stack_match"]
        assert after["score"] > before["score"]
    finally:
        PROSPECTS[prospect_id] = original_record
        if original_profile is None:
            data_service._PROFILES.pop(prospect_id, None)
        else:
            data_service._PROFILES[prospect_id] = original_profile


def test_update_existing_technology_reports_already_present():
    prospect_id = "LEAD-71001"
    original_record = deepcopy(PROSPECTS[prospect_id])
    try:
        result = data_service.update_prospect_info(prospect_id, original_record["tech_stack"][0])
        assert result["updated"] is False
        assert result["already_present"] is True
    finally:
        PROSPECTS[prospect_id] = original_record
