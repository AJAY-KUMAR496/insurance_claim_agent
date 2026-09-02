import json
import os
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

# 1. Load Environment Variables / Secrets
load_dotenv()

# Priority: Streamlit secrets (for cloud deployment) -> Environment variable (for local deployment)
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env or Streamlit Secrets.")

# 2. Initialize Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.1,
    response_mime_type="application/json"
)

# Define Graph State
class ClaimState(TypedDict):
    claim_id: str
    policy_holder_name: str
    policy_number: str
    claim_amount: float
    claim_type: str
    incident_description: str
    provided_documents: List[str]
    
    # Parallel Agent Outputs
    doc_verification: Optional[Dict[str, Any]]
    eligibility_check: Optional[Dict[str, Any]]
    fraud_detection: Optional[Dict[str, Any]]
    
    # Aggregation & Decision Outputs
    claim_summary: Optional[str]
    recommendation: Optional[str]
    final_status: Optional[str]
    human_notes: Optional[str]

# -------------------------------------------------------------------
# Agent Node 1: Document Verification Agent
# -------------------------------------------------------------------
def document_verification_node(state: ClaimState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        """You are a Document Verification Agent. Analyze the provided document list for a {claim_type} claim.
Required documents for {claim_type}:
- Health: Medical Invoice, Hospital Discharge Summary, Identity Proof
- Auto: Repair Estimate, Police Report, Driver License
- Property: Damage Assessment Report, Property Deed, Repair Quotes

Provided Documents: {provided_documents}

Return JSON with keys:
- "is_valid": boolean
- "missing_documents": list of strings
- "verification_score": float (0.0 to 1.0)
- "reason": string
"""
    )
    chain = prompt | llm
    response = chain.invoke({
        "claim_type": state["claim_type"],
        "provided_documents": state["provided_documents"]
    })
    
    try:
        data = json.loads(response.content)
    except Exception:
        data = {"is_valid": False, "missing_documents": [], "verification_score": 0.5, "reason": "Failed to parse JSON output"}
        
    return {"doc_verification": data}

# -------------------------------------------------------------------
# Agent Node 2: Eligibility Check Agent
# -------------------------------------------------------------------
def eligibility_check_node(state: ClaimState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        """You are an Eligibility Check Agent. Determine policy coverage.
Policy Number: {policy_number}
Claim Type: {claim_type}
Claim Amount: ${claim_amount}
Description: {incident_description}

Rules:
- Maximum auto-approved claim amount limit is $10,000.
- Claims over $50,000 require strict review.
- Description must match claim type context.

Return JSON with keys:
- "eligible": boolean
- "coverage_limit": float
- "reason": string
"""
    )
    chain = prompt | llm
    response = chain.invoke({
        "policy_number": state["policy_number"],
        "claim_type": state["claim_type"],
        "claim_amount": state["claim_amount"],
        "incident_description": state["incident_description"]
    })
    
    try:
        data = json.loads(response.content)
    except Exception:
        data = {"eligible": True, "coverage_limit": 10000.0, "reason": "Default assessment"}
        
    return {"eligibility_check": data}

# -------------------------------------------------------------------
# Agent Node 3: Fraud Detection Agent
# -------------------------------------------------------------------
def fraud_detection_node(state: ClaimState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        """You are an AI Fraud Detection Specialist. Evaluate potential fraud risk.
Claim Amount: ${claim_amount}
Claim Type: {claim_type}
Description: {incident_description}
Provided Documents: {provided_documents}

Look for red flags (e.g., suspicious round amounts, missing official records, vague descriptions).

Return JSON with keys:
- "risk_score": float (0.0 = zero risk, 1.0 = extreme risk)
- "red_flags": list of strings
- "risk_level": string ("LOW", "MEDIUM", "HIGH")
"""
    )
    chain = prompt | llm
    response = chain.invoke({
        "claim_amount": state["claim_amount"],
        "claim_type": state["claim_type"],
        "incident_description": state["incident_description"],
        "provided_documents": state["provided_documents"]
    })
    
    try:
        data = json.loads(response.content)
    except Exception:
        data = {"risk_score": 0.2, "red_flags": [], "risk_level": "LOW"}
        
    return {"fraud_detection": data}

# -------------------------------------------------------------------
# Agent Node 4: Claim Summary & Decision Aggregator Agent
# -------------------------------------------------------------------
def claim_summary_node(state: ClaimState) -> Dict[str, Any]:
    doc = state.get("doc_verification", {})
    eligibility = state.get("eligibility_check", {})
    fraud = state.get("fraud_detection", {})
    
    risk_score = fraud.get("risk_score", 0.0)
    is_doc_valid = doc.get("is_valid", False)
    is_eligible = eligibility.get("eligible", False)
    
    if not is_doc_valid or not is_eligible:
        recommendation = "AUTO_REJECT"
    elif risk_score >= 0.5 or state["claim_amount"] > 15000:
        recommendation = "HUMAN_REVIEW"
    else:
        recommendation = "AUTO_APPROVE"
        
    summary_text = f"""
    ### Executive Claim Summary
    - **Claim ID**: {state['claim_id']}
    - **Policy Holder**: {state['policy_holder_name']}
    - **Amount**: ${state['claim_amount']:,.2f}
    - **Document Status**: {"PASS" if is_doc_valid else "FAIL"} (Score: {doc.get('verification_score', 0)})
    - **Eligibility Status**: {"ELIGIBLE" if is_eligible else "INELIGIBLE"}
    - **Fraud Risk Level**: {fraud.get('risk_level', 'UNKNOWN')} (Risk Score: {risk_score})
    - **Red Flags**: {", ".join(fraud.get('red_flags', [])) if fraud.get('red_flags') else "None"}
    - **Recommendation**: {recommendation}
    """
    
    return {
        "claim_summary": summary_text,
        "recommendation": recommendation,
        "final_status": recommendation if recommendation != "HUMAN_REVIEW" else "PENDING_HUMAN_REVIEW"
    }

# -------------------------------------------------------------------
# Agent Node 5: Human Approval Agent (Human-in-the-Loop)
# -------------------------------------------------------------------
def human_approval_node(state: ClaimState) -> Dict[str, Any]:
    review_response = interrupt({
        "task": "HUMAN_REVIEW_REQUIRED",
        "claim_id": state["claim_id"],
        "summary": state["claim_summary"],
        "recommendation": state["recommendation"]
    })
    
    action = review_response.get("action", "REJECT")
    notes = review_response.get("notes", "No notes provided.")
    
    return {
        "final_status": f"HUMAN_{action.upper()}ED",
        "human_notes": notes
    }

# Dynamic Router Callback
def route_claim(state: ClaimState) -> str:
    rec = state.get("recommendation")
    if rec == "HUMAN_REVIEW":
        return "human_approval"
    return END

# Build LangGraph State Graph
def build_claim_graph():
    builder = StateGraph(ClaimState)
    
    builder.add_node("doc_verification", document_verification_node)
    builder.add_node("eligibility_check", eligibility_check_node)
    builder.add_node("fraud_detection", fraud_detection_node)
    builder.add_node("claim_summary", claim_summary_node)
    builder.add_node("human_approval", human_approval_node)
    
    builder.add_edge(START, "doc_verification")
    builder.add_edge(START, "eligibility_check")
    builder.add_edge(START, "fraud_detection")
    
    builder.add_edge("doc_verification", "claim_summary")
    builder.add_edge("eligibility_check", "claim_summary")
    builder.add_edge("fraud_detection", "claim_summary")
    
    builder.add_conditional_edges(
        "claim_summary",
        route_claim,
        {
            "human_approval": "human_approval",
            END: END
        }
    )
    
    builder.add_edge("human_approval", END)
    
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
