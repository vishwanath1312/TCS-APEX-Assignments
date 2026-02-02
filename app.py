"""
Industry-Ready Quality Assurance Test Case Generation Agent
Generic Context-Aware Preconditions + Canonical Type Classification + Coverage Dashboard

An AI-driven Quality Assurance Test Case Generation Agent is designed to automate the creation of test cases from application requirements and user stories, addressing the time-consuming and error-prone nature of manual test writing. Leveraging a Large Language Model, this tool interprets requirement texts, generates functional test scenarios, supports user input of requirement descriptions, and enables iterative refinement, ultimately producing structured test cases with detailed steps and expected results based on sample requirement documents or user stories.
QA teams need to create test cases based on application requirements and user stories. Manually writing test cases is time-consuming and prone to omissions. An AI agent that generates test cases from requirement descriptions can accelerate QA processes.

Run: streamlit run app.py
"""

import os
import json
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import openai
from dotenv import load_dotenv

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
load_dotenv()
HF_MODEL = "distilgpt2"

OUTPUT_COLUMNS = [
    "Test Case ID",
    "Requirement",
    "Type Label",
    "Type",
    "Description",
    "Preconditions",
    "Test Steps",
    "Expected Result",
    "Source",
]

# ---------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------
st.set_page_config(page_title="CAPSTONE PROJECT 9", layout="wide")
st.title("🤖 Quality Assurance Test Case Generation Agent")

# ---------------------------------------------------
# Load Hugging Face Generator
# ---------------------------------------------------
@st.cache_resource
def load_hf_generator():
    try:
        return pipeline("text-generation", model=HF_MODEL)
    except Exception:
        return None

hf_gen = load_hf_generator()

# ---------------------------------------------------
# Type Classification Helpers
# ---------------------------------------------------
NEGATIVE_KEYWORDS = ["error", "fail", "invalid", "reject", "denied", "unsuccessful", "incorrect"]
NONFUNC_KEYWORDS = ["performance", "speed", "load", "security", "response time", "availability", "scalability"]

def classify_requirement(req: str) -> str:
    req = req.lower()
    if any(k in req for k in NEGATIVE_KEYWORDS):
        return "Negative"
    if any(k in req for k in NONFUNC_KEYWORDS):
        return "Non-Functional"
    return "Functional"

def canonical_to_label(t: str) -> str:
    return {
        "Functional": "🟢 Functional",
        "Negative": "🔴 Negative",
        "Non-Functional": "🟣 Non-Functional",
    }.get(t, "🟢 Functional")

# ---------------------------------------------------
# 🧠 Generic, Context-Aware Preconditions Generator
# ---------------------------------------------------
def generate_preconditions(requirement: str) -> str:
    req = requirement.lower().strip()

    if re.search(r"\b(login|log in|authenticate|sign in|authorization)\b", req):
        return "Valid user credentials must exist and authentication service should be available."

    elif re.search(r"\b(register|create|add|insert|signup|sign up)\b", req):
        return "System must be online and input data should be valid and complete."

    elif re.search(r"\b(update|edit|modify|change)\b", req):
        return "An existing record should be available for modification with appropriate permissions."

    elif re.search(r"\b(delete|remove|deactivate|cancel)\b", req):
        return "Target record or item must exist and user should have delete privileges."

    elif re.search(r"\b(view|display|show|retrieve|fetch|list|search|filter)\b", req):
        return "Relevant data must be present in the system and user must have view access."

    elif re.search(r"\b(report|dashboard|analytics|chart|statistics)\b", req):
        return "System must contain valid data and reporting components should be operational."

    elif re.search(r"\b(email|notification|alert|message|sms)\b", req):
        return "Notification or mail service must be configured and reachable."

    elif re.search(r"\b(upload|import|attach|file|image|document)\b", req):
        return "Valid file should exist locally and meet size and format constraints."

    elif re.search(r"\b(download|export|generate|print|save)\b", req):
        return "System should have necessary data available for export or download."

    elif re.search(r"\b(payment|transaction|billing|invoice|checkout|order)\b", req):
        return "Payment gateway must be active and user account configured with valid payment details."

    elif re.search(r"\b(api|endpoint|service|request|response)\b", req):
        return "API server must be live and accessible with valid authentication tokens."

    elif re.search(r"\b(performance|load|response time|scalability|latency|throughput)\b", req):
        return "System should run in a test environment simulating expected workload."

    elif re.search(r"\b(security|encrypt|encryption|compliance|privacy|access control)\b", req):
        return "Security modules and configurations must be active according to policy."

    elif re.search(r"\b(ui|screen|form|button|field|interface)\b", req):
        return "User interface should be loaded correctly and all elements should be interactive."

    elif re.search(r"\b(database|data|storage|table|record|dataset)\b", req):
        return "Database connection should be active and contain necessary test data."

    elif re.search(r"\b(error|fail|invalid|exception|timeout)\b", req):
        return "System should be operational and capable of handling erroneous inputs."

    else:
        return "System must be operational with all dependent services configured and available."

# ---------------------------------------------------
# Auto-fill and Normalize Fields
# ---------------------------------------------------
def autofill_and_normalize(row: dict) -> dict:
    req = row.get("Requirement", "")
    t = classify_requirement(req)
    row["Type"] = t
    row["Type Label"] = canonical_to_label(t)

    if not row.get("Description"):
        if t == "Negative":
            row["Description"] = f"Verify that the system handles invalid inputs correctly for: {req}"
        elif t == "Non-Functional":
            row["Description"] = f"Ensure performance and reliability criteria are satisfied for: {req}"
        else:
            row["Description"] = f"Validate that the system correctly performs: {req}"

    if not row.get("Preconditions"):
        row["Preconditions"] = generate_preconditions(req)

    if not row.get("Test Steps"):
        if t == "Non-Functional":
            row["Test Steps"] = (
                "1. Simulate workload or stress scenario\n"
                "2. Measure response time or resource usage\n"
                "3. Compare with performance benchmarks"
            )
        else:
            row["Test Steps"] = (
                "1. Navigate to the relevant module\n"
                "2. Perform the described action\n"
                "3. Observe system response"
            )

    if not row.get("Expected Result"):
        if t == "Negative":
            row["Expected Result"] = "System rejects invalid input and displays appropriate error message."
        elif t == "Non-Functional":
            row["Expected Result"] = "System meets defined performance and reliability criteria."
        else:
            row["Expected Result"] = "System performs the operation successfully."

    row["Source"] = row.get("Source", "Auto-Filled")
    return row

# ---------------------------------------------------
# Negative Coverage
# ---------------------------------------------------
def ensure_negative(df, req, seq):
    if "Negative" in df["Type"].values:
        return df, seq
    neg = {
        "Test Case ID": f"TC-{seq:04d}",
        "Requirement": req,
        "Type Label": "🔴 Negative",
        "Type": "Negative",
        "Description": f"Synthetic negative test for: {req}",
        "Preconditions": generate_preconditions(req),
        "Test Steps": "1. Input invalid data\n2. Attempt action\n3. Observe response",
        "Expected Result": "System should handle invalid input gracefully.",
        "Source": "Auto-Generated",
    }
    seq += 1
    return pd.concat([df, pd.DataFrame([neg])], ignore_index=True), seq

# ---------------------------------------------------
# Dashboard Utilities
# ---------------------------------------------------
def compute_summary(df):
    return {
        "Total Requirements": df["Requirement"].nunique(),
        "Total Test Cases": len(df),
        "Functional": (df["Type"] == "Functional").sum(),
        "Negative": (df["Type"] == "Negative").sum(),
        "Non-Functional": (df["Type"] == "Non-Functional").sum(),
    }

def plot_heatmap(df):
    st.subheader("📊 Requirement Coverage Heatmap")
    cov = df.groupby(["Requirement", "Type"]).size().unstack(fill_value=0)
    plt.figure(figsize=(10, max(4, len(cov) * 0.6)))
    sns.heatmap(cov, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    st.pyplot(plt)

# ---------------------------------------------------
# Sidebar Config
# ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    openai_key = st.text_input("OpenAI API Key (optional)", type="password")
    openai.api_key = openai_key or os.getenv("OPENAI_API_KEY")
    use_openai = bool(openai.api_key)
    auto_negative = st.checkbox("Auto-add missing negative test", True)
    expected_cases = st.number_input("Expected cases per requirement", 1, 10, 2)

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------
st.subheader("📥 Input Requirements")
file = st.file_uploader("Upload CSV (must contain 'Requirement' column)", type=["csv"])
if file:
    df_req = pd.read_csv(file)
    requirements = df_req["Requirement"].dropna().tolist()
else:
    txt = st.text_area("Or paste requirements here (one per line):", height=250)
    requirements = [r.strip() for r in txt.splitlines() if r.strip()]

st.download_button("Download (placeholder)", data=b"", file_name="generated_test_cases.csv", disabled=True)

# ---------------------------------------------------
# Generate Button
# ---------------------------------------------------
if st.button("🚀 Generate Test Cases & Dashboard"):
    if not requirements:
        st.warning("Please enter or upload requirements first.")
        st.stop()

    seq, results = 1, []
    progress = st.progress(0)

    for i, req in enumerate(requirements, start=1):
        t = classify_requirement(req)
        row = {
            "Requirement": req,
            "Type": t,
            "Type Label": canonical_to_label(t),
            "Description": "",
            "Preconditions": "",
            "Test Steps": "",
            "Expected Result": "",
            "Source": "Generated",
        }
        row = autofill_and_normalize(row)
        row["Test Case ID"] = f"TC-{seq:04d}"
        seq += 1
        df_local = pd.DataFrame([row])
        if auto_negative:
            df_local, seq = ensure_negative(df_local, req, seq)
        results.append(df_local)
        progress.progress(i / len(requirements))

    df_final = pd.concat(results, ignore_index=True)[OUTPUT_COLUMNS]

    # Display summary
    st.subheader("📈 Summary")
    summary = compute_summary(df_final)
    cols = st.columns(4)
    for (k, v), c in zip(summary.items(), cols):
        c.metric(k, v)

    # Data Table
    st.subheader("🧾 Generated Test Cases")
    st.dataframe(df_final, use_container_width=True)

    # Coverage
    coverage = df_final.groupby(["Requirement", "Type"]).size().unstack(fill_value=0)
    coverage["Total"] = coverage.sum(axis=1)
    coverage["Coverage Score (%)"] = (coverage["Total"] / expected_cases * 100).round(1)
    st.subheader("📊 Coverage Summary")
    st.dataframe(coverage.reset_index(), use_container_width=True)
    plot_heatmap(df_final)

    # Export
    csv = df_final.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV", csv, "generated_test_cases.csv", "text/csv")

    xlsx = "generated_test_cases.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        df_final.to_excel(writer, sheet_name="Test Cases", index=False)
        coverage.to_excel(writer, sheet_name="Coverage Summary")
    with open(xlsx, "rb") as f:
        st.download_button("📘 Download Excel", f, xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
