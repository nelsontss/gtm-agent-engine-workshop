import json
import os
import re
import unittest
from unittest.mock import Mock, patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from gtm_agent import gtm_agent as agent_module
from gtm_agent.data_service import REP_IDS, get_offering, get_prospect_record
from gtm_agent.gtm_agent import ProspectScore, RubricBreakdown


LEAD_ID = "LEAD-26912"
OFFERING_ID = "OFFER-10001"


class ScriptedChatModel(BaseChatModel):
    _responses: list[AIMessage] = PrivateAttr()
    _calls: list[list] = PrivateAttr()
    _bound_tools: list = PrivateAttr()

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self._calls = []
        self._bound_tools = []

    @property
    def _llm_type(self):
        return "scripted-chat-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self._bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._responses.pop(0))])


def tool_call(name, args, index):
    return {
        "name": name,
        "args": args,
        "id": f"call-{index}",
        "type": "tool_call",
    }


def run_flow(message, calls, score_result=None):
    responses = [
        AIMessage(content="", tool_calls=[tool_call(name, args, index)])
        for name, args, index in calls
    ]
    responses.append(AIMessage(content="Done."))
    model = ScriptedChatModel(responses)
    agent = agent_module._create_gtm_agent(model)
    with patch.object(agent_module, "gtm_agent", agent):
        if score_result is None:
            return agent_module.run_agent(message, user_id=REP_IDS[0]["rep_id"]), model
        scoring_model = Mock(invoke=Mock(return_value=score_result))
        with patch.object(agent_module, "_scoring_llm", scoring_model):
            return agent_module.run_agent(message, user_id=REP_IDS[0]["rep_id"]), model


def count_prompt_tokens(messages, tools):
    tool_text = json.dumps(
        [
            {"name": tool.name, "description": tool.description, "parameters": tool.args}
            for tool in tools
        ]
    )
    messages_with_tools = [*messages, HumanMessage(content=tool_text)]
    try:
        return agent_module.agent_model.get_num_tokens_from_messages(messages_with_tools)
    except Exception:
        prompt_text = " ".join(str(message.content) for message in messages_with_tools)
        return len(re.findall(r"\w+|[^\w\s]", prompt_text))


class GTMAgentTests(unittest.TestCase):
    def test_short_request_uses_only_domain_prompt_and_tools(self):
        model = ScriptedChatModel([AIMessage(content="Ready.")])
        agent = agent_module._create_gtm_agent(model)

        agent.invoke({"messages": [{"role": "user", "content": "Hi."}]})

        self.assertEqual(len(model._calls), 1)
        system_messages = [message for message in model._calls[0] if message.type == "system"]
        self.assertEqual([message.content for message in system_messages], [agent_module.SYSTEM_PROMPT])
        tool_names = [tool.name if hasattr(tool, "name") else tool["name"] for tool in model._bound_tools]
        self.assertEqual(
            tool_names,
            [
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            ],
        )
        prompt_tokens = count_prompt_tokens(model._calls[0], model._bound_tools)
        self.assertLess(prompt_tokens, 2000)

    def test_email_flow_injects_signed_in_rep(self):
        prospect = get_prospect_record(LEAD_ID)
        result, model = run_flow(
            "Email this prospect.",
            [
                ("get_current_rep", {}, 1),
                (
                    "send_prospect_email",
                    {"prospect": prospect, "subject": "Hello", "body": "Let's talk."},
                    2,
                ),
            ],
        )

        tool_messages = [message for message in result["messages"] if message.type == "tool"]
        self.assertEqual([message.name for message in tool_messages], ["get_current_rep", "send_prospect_email"])
        self.assertIn(REP_IDS[0]["email"], str(tool_messages[-1].content))
        self.assertEqual(result["reply"], "Done.")
        self.assertEqual(len(model._calls), 3)

    def test_score_flow_uses_domain_tools(self):
        score_result = ProspectScore(
            score=80,
            justification="Strong fit.",
            rubric_breakdown=RubricBreakdown(
                revenue_fit=80,
                tech_stack_match=80,
                segment_fit=80,
            ),
        )
        result, _ = run_flow(
            "Score this prospect for the offering.",
            [
                ("get_current_rep", {}, 1),
                ("build_prospect_profile", {"prospect_id": LEAD_ID}, 2),
                ("lookup_offering", {"offering_id": OFFERING_ID}, 3),
                (
                    "score_prospect",
                    {
                        "prospect_profile": {"prospect_id": LEAD_ID},
                        "offering": get_offering(OFFERING_ID),
                    },
                    4,
                ),
            ],
            score_result=score_result,
        )

        tool_messages = [message for message in result["messages"] if message.type == "tool"]
        self.assertEqual(
            [message.name for message in tool_messages],
            ["get_current_rep", "build_prospect_profile", "lookup_offering", "score_prospect"],
        )
        self.assertEqual(result["reply"], "Done.")

    def test_update_then_score_flow_uses_domain_tools(self):
        score_result = ProspectScore(
            score=75,
            justification="Good fit.",
            rubric_breakdown=RubricBreakdown(
                revenue_fit=75,
                tech_stack_match=75,
                segment_fit=75,
            ),
        )
        result, _ = run_flow(
            "Update this prospect's technology and score them.",
            [
                ("get_current_rep", {}, 1),
                (
                    "update_prospect_info",
                    {"prospect_id": LEAD_ID, "technology": "Kafka"},
                    2,
                ),
                ("build_prospect_profile", {"prospect_id": LEAD_ID}, 3),
                ("lookup_offering", {"offering_id": OFFERING_ID}, 4),
                (
                    "score_prospect",
                    {
                        "prospect_profile": {"prospect_id": LEAD_ID},
                        "offering": get_offering(OFFERING_ID),
                    },
                    5,
                ),
            ],
            score_result=score_result,
        )

        tool_messages = [message for message in result["messages"] if message.type == "tool"]
        self.assertEqual(
            [message.name for message in tool_messages],
            [
                "get_current_rep",
                "update_prospect_info",
                "build_prospect_profile",
                "lookup_offering",
                "score_prospect",
            ],
        )
        self.assertEqual(result["reply"], "Done.")


if __name__ == "__main__":
    unittest.main()
