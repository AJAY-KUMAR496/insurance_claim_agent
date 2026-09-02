import streamlit as st
import uuid
from langgraph.types import Command
from claim_agent import build_claim_graph

st.set_page_config(page_title="AI Insurance Claim Processing System", layout="wide")

st.title("🛡️ AI Insurance Claim Processing Agent")
st.markdown("Automated claims handling powered by LangGraph parallel multi-agent execution & **Gemini 2.5 Flash**.")

# Initialize workflow graph state in session
if "graph" not in st.session_state:
    st.session_state.graph = build_claim_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Sidebar Controls
st.sidebar.header("Claim Submission Portal")
claim_id = st.sidebar.text_input("Claim ID", f"CLM-{uuid.uuid4().hex[:6].upper()}")
policy_holder = st.sidebar.text_input("Policy Holder Name", "Jane Doe")
policy_number = st.sidebar.text_input("Policy Number", "POL-998231")
claim_type = st.sidebar.selectbox("Claim Type", ["Health", "Auto", "Property"])
claim_amount = st.sidebar.number_input("Claim Amount ($)", min_value=100.0, max_value=100000.0, value=12500.0, step=500.0)

incident_description = st.sidebar.text_area(
    "Incident Description",
    "Patient underwent emergency appendectomy. Hospitalization lasted 3 days."
)

st.sidebar.subheader("Uploaded Documents")
doc_options = ["Medical Invoice", "Hospital Discharge Summary", "Identity Proof", "Repair Estimate", "Police Report"]
selected_docs = st.sidebar.multiselect("Select attached documents:", doc_options, default=["Medical Invoice", "Identity Proof"])

# Action Buttons
if st.sidebar.button("🚀 Process Claim", type="primary"):
    st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    initial_state = {
        "claim_id": claim_id,
        "policy_holder_name": policy_holder,
        "policy_number": policy_number,
        "claim_amount": claim_amount,
        "claim_type": claim_type,
        "incident_description": incident_description,
        "provided_documents": selected_docs,
        "doc_verification": None,
        "eligibility_check": None,
        "fraud_detection": None,
        "claim_summary": None,
        "recommendation": None,
        "final_status": None,
        "human_notes": None,
    }

    with st.spinner("Agents running parallel checks (Document, Eligibility, Fraud)..."):
        events = list(st.session_state.graph.stream(initial_state, config, stream_mode="values"))
        st.session_state.latest_state = events[-1]

# Display Dashboard Logic
if "latest_state" in st.session_state:
    state = st.session_state.latest_state
    
    st.subheader("📊 Parallel Agent Analysis Results")
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.markdown("### 📄 Document Agent")
        if state.get("doc_verification"):
            st.json(state["doc_verification"])
            
    with res_col2:
        st.markdown("### 📋 Eligibility Agent")
        if state.get("eligibility_check"):
            st.json(state["eligibility_check"])
            
    with res_col3:
        st.markdown("### 🚨 Fraud Agent")
        if state.get("fraud_detection"):
            st.json(state["fraud_detection"])

    st.markdown("---")
    
    if state.get("claim_summary"):
        st.markdown(state["claim_summary"])

    snapshot = st.session_state.graph.get_state(config)
    
    if snapshot.next and snapshot.next[0] == "human_approval":
        st.warning("⚠️ High Risk / High Value Claim Escalated to Human Approval Agent.")
        
        with st.form("human_review_form"):
            st.subheader("Human Officer Decision Interface")
            notes = st.text_area("Review Notes & Justification", "All documents verified manually. Escalation approved.")
            
            c1, c2 = st.columns(2)
            approve_submitted = c1.form_submit_button("✅ Approve Claim")
            reject_submitted = c2.form_submit_button("❌ Reject Claim")
            
            if approve_submitted or reject_submitted:
                decision = "APPROVE" if approve_submitted else "REJECT"
                
                resume_cmd = Command(resume={"action": decision, "notes": notes})
                resumed_events = list(st.session_state.graph.stream(resume_cmd, config, stream_mode="values"))
                
                st.session_state.latest_state = resumed_events[-1]
                st.rerun()

    elif state.get("final_status"):
        status = state['final_status']
        if "APPROVE" in status:
            st.success(f"Final Claim Decision: **{status}**")
        else:
            st.error(f"Final Claim Decision: **{status}**")
            
        if state.get("human_notes"):
            st.info(f"Officer Notes: {state['human_notes']}")
            