# ===============================
# Project: LangGraph Card Agent (Bedrock + Streamlit)
# Files in this single canvas:
#  - app.py               (Streamlit UI)
#  - graph.py             (LangGraph autonomous workflow graph)
#  - agents.py            (LLM calls via AWS Bedrock Titan)
#  - tools.py             (mocked validation + actions)
#  - prompts.py           (planner + intent prompts)
#  - requirements.txt     (deps)
# ===============================

# -------------------------------
# file: prompts.py
# -------------------------------
PLANNER_PROMPT = (
    """
    You are an autonomous planner for a card-service agent. Think step-by-step.
    You have tools: ["validate", "replace", "cancel", "finish"].
    Goal: correctly fulfill the user's request while ensuring ownership is valid before destructive actions.

    State snapshot:
    - last_user_message: {user_input}
    - validated: {validated}

    Rules:
    - If intent seems to be card replacement, plan to `validate` (if not yet validated), then `replace`.
    - If intent seems to be card cancellation, plan to `validate` (if not yet validated), then `cancel`.
    - If unclear, respond with a short clarification and then `finish`.
    - NEVER ask the human for confirmation (run fully autonomously). If risk is high and unclear, return a safe clarification with `finish`.

    Output ONLY strict JSON:
    {
      "next_action": "validate" | "replace" | "cancel" | "finish",
      "intent": "replace" | "cancel" | "unknown",
      "assistant_message": "what you would tell the user (concise)",
      "reason": "why you chose this next_action"
    }
    """
).strip()

INTENT_ONLY_PROMPT = (
    """
    Classify the user's message for card services. Output ONLY JSON:
    {"intent": "replace" | "cancel" | "unknown", "reason": "short"}
    Message: {user_input}
    """
).strip()

# -------------------------------
# file: agents.py
# -------------------------------
import os
import json
import boto3
from typing import Dict, Any

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-text-express-v1")
BEDROCK_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

class BedrockLLM:
    def __init__(self, model_id: str = BEDROCK_MODEL_ID, region: str = BEDROCK_REGION):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256, top_p: float = 1.0) -> str:
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        })
        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        data = json.loads(resp["body"].read())
        return data.get("results", [{}])[0].get("outputText", "").strip()

llm = BedrockLLM()

from prompts import PLANNER_PROMPT, INTENT_ONLY_PROMPT

def intent_agent(user_input: str) -> Dict[str, Any]:
    text = llm.generate(INTENT_ONLY_PROMPT.format(user_input=user_input))
    try:
        parsed = json.loads(text)
        return {"intent": parsed.get("intent", "unknown"), "intent_reason": parsed.get("reason", "")}
    except Exception:
        return {"intent": "unknown", "intent_reason": "parse_error"}


def planner_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PLANNER_PROMPT.format(
        user_input=state.get("user_input", ""),
        validated=state.get("validated", False),
    )
    text = llm.generate(prompt)
    try:
        plan = json.loads(text)
        # Normalize
        next_action = plan.get("next_action", "finish")
        assistant_message = plan.get("assistant_message", "")
        intent = plan.get("intent", state.get("intent", "unknown"))
        reason = plan.get("reason", "")
        return {
            "next_action": next_action,
            "assistant_message": assistant_message,
            "intent": intent,
            "plan_reason": reason,
        }
    except Exception:
        return {"next_action": "finish", "assistant_message": "I'm sorry, I couldn't process that.", "plan_reason": "parse_error"}

# -------------------------------
# file: tools.py
# -------------------------------
from typing import Dict, Any

# Mocked tool: validate ownership (pretend to check a DB)

def validate_ownership_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"validated": True, "validation_reason": "Mock validation passed."}


def replace_card_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"result": "✅ Replacement completed. A new card will arrive in 5–7 business days."}


def cancel_card_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"result": "🛑 Cancellation completed. Your card is now inactive."}

# -------------------------------
# file: graph.py
# -------------------------------
from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END

from agents import intent_agent, planner_agent
from tools import validate_ownership_tool, replace_card_tool, cancel_card_tool

class CardState(TypedDict, total=False):
    user_input: str
    messages: List[Dict[str, str]]
    intent: Literal["replace", "cancel", "unknown"]
    intent_reason: str
    validated: bool
    validation_reason: str
    result: str
    assistant_message: str
    next_action: Literal["validate", "replace", "cancel", "finish"]
    plan_reason: str
    step_count: int


def build_graph(max_steps: int = 6):
    g = StateGraph(CardState)

    # Node: initial classification (optional but helpful)
    def classify_node(state: CardState) -> Dict[str, Any]:
        return intent_agent(state.get("user_input", ""))

    # Node: planner decides next action
    def planner_node(state: CardState) -> Dict[str, Any]:
        out = planner_agent(state)
        # auto-guard: if planner suggests replace/cancel but not validated, flip to validate first
        if out.get("next_action") in ("replace", "cancel") and not state.get("validated", False):
            out["next_action"] = "validate"
            out["plan_reason"] = (out.get("plan_reason", "") + " | Guarded: validating before action")[:500]
        return out

    # Tool dispatcher node
    def tool_node(state: CardState) -> Dict[str, Any]:
        action = state.get("next_action", "finish")
        if action == "validate":
            return validate_ownership_tool(state)
        if action == "replace":
            return replace_card_tool(state)
        if action == "cancel":
            return cancel_card_tool(state)
        return {}

    # Step counter / loop guard
    def step_node(state: CardState) -> Dict[str, Any]:
        n = int(state.get("step_count", 0)) + 1
        return {"step_count": n}

    g.add_node("classify", classify_node)
    g.add_node("planner", planner_node)
    g.add_node("tool", tool_node)
    g.add_node("step", step_node)

    g.set_entry_point("classify")
    g.add_edge("classify", "planner")

    # After planner, either finish immediately (e.g., clarification) or run tool
    def route_after_planner(state: CardState):
        if state.get("next_action") == "finish":
            return END
        return "tool"

    g.add_conditional_edges("planner", route_after_planner)

    # After tool, if we have a terminal result OR exceeded steps, finish; else plan again
    def route_after_tool(state: CardState):
        # If a result exists (action done), finish
        if state.get("result"):
            return END
        # else keep planning until max_steps
        if int(state.get("step_count", 0)) >= max_steps:
            return END
        return "planner"

    g.add_edge("tool", "step")
    g.add_edge("step", "planner")
    g.add_conditional_edges("tool", route_after_tool)  # (conditional also checked via step->planner loop)

    return g.compile()

# -------------------------------
# file: app.py
# -------------------------------
import streamlit as st
from dotenv import load_dotenv
from graph import build_graph

load_dotenv()

st.set_page_config(page_title="Card Agent (Autonomous - LangGraph + Bedrock)", page_icon="🤖")
st.title("🤖 Fully Autonomous Card Agent — LangGraph + Amazon Titan (Bedrock)")
st.caption("Autonomous planner selects tools and executes without user confirmations.")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "state" not in st.session_state:
    st.session_state.state = {}

user_text = st.text_input("Describe what you want to do:", placeholder="Please cancel my card ending 4321")

if st.button("Run Autonomous Agent") and user_text.strip():
    st.session_state.state = {"user_input": user_text}
    result_state = st.session_state.graph.invoke(st.session_state.state)
    st.session_state.state.update(result_state)

# Show assistant message (from planner) and final result
if msg := st.session_state.state.get("assistant_message"):
    st.info(msg)
if res := st.session_state.state.get("result"):
    st.success(res)

with st.expander("🔎 Debug state"):
    st.json(st.session_state.state)

# -------------------------------
# file: requirements.txt
# -------------------------------
# Core
streamlit
python-dotenv

# AWS + LLM
boto3

# Orchestration
langgraph
