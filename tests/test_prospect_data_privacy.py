import json
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["OPENAI_API_KEY"] = "test"
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import gtm_agent as agent_module


class ProspectDataPrivacyTests(unittest.TestCase):
    def test_get_prospect_returns_only_contact_allowlist(self):
        record = {
            "name": "Jordan Lee",
            "email": "jordan@example.com",
            "billing_qualification": {"tax_id": "123"},
            "annual_revenue": 5_000_000,
            "disqualified": True,
        }
        with patch.object(agent_module.data_service, "get_prospect_record", return_value=record):
            result = agent_module.get_prospect.invoke({"prospect_id": "LEAD-1"})

        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": "LEAD-1",
                "name": "Jordan Lee",
                "email": "jordan@example.com",
                "disqualified": True,
            },
        )
        self.assertNotIn("billing_qualification", str(result))

    def test_build_profile_filters_returned_and_cached_values(self):
        record = {
            "name": "Jordan Lee",
            "email": "jordan@example.com",
            "annual_revenue": 5_000_000,
            "billing_qualification": {"tax_id": "123"},
        }
        with (
            patch.object(agent_module.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(agent_module.data_service, "get_prospect_record", return_value=record),
            patch.object(agent_module.data_service, "fetch_engagement_history", return_value=[]),
            patch.object(agent_module.data_service, "fetch_account_details", return_value={"industry": "software"}),
            patch.object(agent_module.data_service, "fetch_tech_stack", return_value=["CRM"]),
            patch.object(agent_module.data_service, "save_profile_to_db") as save_profile,
        ):
            result = agent_module.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        cached_profile = save_profile.call_args.args[1]
        self.assertEqual(result["prospect_profile"], cached_profile)
        self.assertEqual(cached_profile["annual_revenue"], 5_000_000)
        self.assertEqual(cached_profile["tech_stack"], ["CRM"])
        self.assertNotIn("billing_qualification", str(result))
        self.assertNotIn("billing_qualification", str(cached_profile))

    def test_build_profile_sanitizes_existing_cached_values(self):
        cached = {
            "prospect_id": "LEAD-1",
            "name": "Jordan Lee",
            "billing_qualification": {"tax_id": "123"},
        }
        with (
            patch.object(agent_module.data_service, "get_profile_from_db", return_value={"prospect_profile": cached}),
            patch.object(agent_module.data_service, "save_profile_to_db") as save_profile,
        ):
            result = agent_module.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        self.assertEqual(result["prospect_profile"], save_profile.call_args.args[1])
        self.assertEqual(result["prospect_profile"], {"prospect_id": "LEAD-1", "name": "Jordan Lee"})
        self.assertNotIn("billing_qualification", str(result))

    def test_score_filters_extra_fields_and_keeps_scoring_fields(self):
        scoring_llm = MagicMock()
        scoring_llm.invoke.return_value.model_dump.return_value = {"score": 80}
        profile = {
            "prospect_id": "LEAD-1",
            "annual_revenue": 5_000_000,
            "tech_stack": ["caller value"],
            "billing_qualification": {"tax_id": "123"},
        }
        offering = {
            "required_tech_stack": ["CRM"],
            "min_annual_revenue": 1_000_000,
            "description": "GTM software",
        }
        with (
            patch.object(agent_module.data_service, "fetch_tech_stack", return_value=["CRM"]),
            patch.object(agent_module, "_scoring_llm", scoring_llm),
        ):
            result = agent_module.score_prospect.invoke(
                {"prospect_profile": profile, "offering": offering}
            )

        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertEqual(result, {"score": 80})
        self.assertNotIn("billing_qualification", user_message)
        self.assertIn('"annual_revenue": 5000000', user_message)
        self.assertIn('"tech_stack": [\n    "CRM"', user_message)

    def test_missing_optional_allowlisted_fields_are_omitted(self):
        with (
            patch.object(agent_module.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(agent_module.data_service, "get_prospect_record", return_value={"name": "Jordan Lee"}),
            patch.object(agent_module.data_service, "fetch_engagement_history", return_value=[]),
            patch.object(agent_module.data_service, "fetch_account_details", return_value={}),
            patch.object(agent_module.data_service, "fetch_tech_stack", return_value=[]),
            patch.object(agent_module.data_service, "save_profile_to_db"),
        ):
            result = agent_module.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        self.assertEqual(
            result["prospect_profile"],
            {
                "prospect_id": "LEAD-1",
                "name": "Jordan Lee",
                "engagement_history": [],
                "account_details": {},
                "tech_stack": [],
            },
        )

    def test_real_record_tool_workflow_never_returns_billing_data(self):
        prospect_id = next(
            prospect_id
            for prospect_id, record in agent_module.data_service.PROSPECTS.items()
            if "billing_qualification" in record
        )
        offering = next(iter(agent_module.data_service.OFFERINGS.values()))
        agent_module.data_service._PROFILES.pop(prospect_id, None)
        scoring_llm = MagicMock()
        scoring_llm.invoke.return_value.model_dump.return_value = {"score": 80}
        with patch.object(agent_module, "_scoring_llm", scoring_llm):
            contact = agent_module.get_prospect.invoke({"prospect_id": prospect_id})
            profile = agent_module.build_prospect_profile.invoke({"prospect_id": prospect_id})
            score = agent_module.score_prospect.invoke(
                {"prospect_profile": profile["prospect_profile"], "offering": offering}
            )

        self.assertNotIn(
            "billing_qualification",
            json.dumps([contact, profile, score]),
        )
        scoring_input = scoring_llm.invoke.call_args.args[0][1]["content"]
        source_record = agent_module.data_service.PROSPECTS[prospect_id]
        self.assertIn(str(source_record["annual_revenue"]), scoring_input)
        for technology in source_record["tech_stack"]:
            self.assertIn(f'"{technology}"', scoring_input)
        self.assertNotIn("billing_qualification", scoring_input)
        agent_module.data_service._PROFILES.pop(prospect_id, None)


if __name__ == "__main__":
    unittest.main()
