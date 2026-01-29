
# Hugging Face Chatbot
# Supported Models: Qwen-2.5-72B, Llama-3.3-70B

import streamlit as st
import pandas as pd
import os
from huggingface_hub import InferenceClient

# Global dataframes
_DATASETS = {}
_CURRENT_AGENCY = "TIMES"

AVAILABLE_MODELS = {
    "Qwen-2.5-72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
}


import re

def extract_year_from_question(question: str, available_years: list) -> int:
    """
    Extract year from user's question if they ask about a specific year.
    Returns None if no specific year mentioned.
    """
    # Look for 4-digit years in the question
    year_pattern = r'\b(20\d{2})\b'
    matches = re.findall(year_pattern, question)

    for match in matches:
        year = int(match)
        if year in available_years:
            return year

    return None


def prepare_dataset_context(df: pd.DataFrame, question: str = "") -> str:
    """
    Prepare dataset context for the LLM.
    - If user asks about specific year, send that year's data
    - Otherwise, send latest year data (2025/2026)
    """
    if df is None or df.empty:
        return "No data available."

    # Get available years
    available_years = sorted(df['Year'].unique().tolist())
    latest_year = df['Year'].max()

    # Check if user asked about a specific year
    requested_year = extract_year_from_question(question, available_years)

    if requested_year:
        target_year = requested_year
    else:
        target_year = latest_year

    # Filter to target year
    year_df = df[df['Year'] == target_year].copy()

    # Select key columns based on agency
    key_columns = ['IPEDS_Name', 'Year']

    if _CURRENT_AGENCY == "TIMES":
        key_columns += ['Times_Rank', 'Overall', 'Teaching', 'Research_Quality', 'Research_Environment', 'Industry', 'International_Outlook']
    elif _CURRENT_AGENCY == "QS":
        key_columns += ['QS_Rank', 'Overall_Score', 'Academic_Reputation', 'Employer_Reputation', 'Citations_per_Faculty']
    elif _CURRENT_AGENCY == "USN":
        key_columns += ['Rank', 'Peer_assessment_score', 'Actual_graduation_rate', 'Average_first_year_retention_rate']
    elif _CURRENT_AGENCY == "Washington":
        key_columns += ['Washington_Rank', '8-year_graduation_rate', 'Research_expenditures_(M)']

    available_cols = [c for c in key_columns if c in year_df.columns]
    year_df = year_df[available_cols]

    csv_data = year_df.to_csv(index=False)

    context = f"""
DATASET: {_CURRENT_AGENCY} University Rankings
SHOWING DATA FOR YEAR: {target_year}
AVAILABLE YEARS IN DATASET: {available_years}
TOTAL UNIVERSITIES: {len(year_df)}
COLUMNS: {', '.join(available_cols)}

DATA (CSV format):
{csv_data}
"""
    return context


def call_hf_model(client: InferenceClient, system_prompt: str, user_message: str, max_tokens: int = 300) -> str:
    """Call Hugging Face model with chat completion"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return "Error: Rate limit exceeded. Please wait a moment and try again."
        elif "token" in error_msg.lower() or "length" in error_msg.lower():
            return "Error: Input too long. The dataset might be too large for this model."
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            return "Error: Invalid API key. Please check your HF_API_KEY."
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            return "Error: Access denied. This model may require accepting terms at huggingface.co"
        else:
            return f"Error: {error_msg}"


def get_ai_response(question: str, api_key: str, model_id: str) -> str:
    """
    Main function to get AI response.
    Sends dataset context + question to the model.
    """
    global _DATASETS, _CURRENT_AGENCY

    df = _DATASETS.get(_CURRENT_AGENCY)
    if df is None:
        return "Error: No dataset loaded."

    # Prepare dataset context (auto-detects year from question)
    dataset_context = prepare_dataset_context(df, question)

    # System prompt with ranking rules
    system_prompt = """You are a university rankings data analyst. You help users understand university ranking data.

IMPORTANT RULES FOR RANKINGS:
- LOWER rank number = BETTER (Rank 1 is the BEST, Rank 1500 is very POOR)
- Rank 1-50 = Excellent
- Rank 51-200 = Very Good
- Rank 201-500 = Good
- Rank 501-1000 = Average
- Rank 1000+ = Below Average

- HIGHER scores = BETTER (Score 90 is better than Score 50)

RESPONSE STYLE:
- Keep responses SHORT and PRECISE (2-4 sentences maximum)
- For comparisons: Use bullet points, show only KEY metrics (3-4 metrics max)
- End with a 1-line conclusion summarizing who performs better
- NO lengthy explanations
- Direct answers only with specific numbers from the data"""

    # User message with context and question
    user_message = f"""{dataset_context}

QUESTION: {question}

Answer in 2-4 sentences max. Be brief and direct. For comparisons, use bullet points. Lower rank = better."""

    try:
        client = InferenceClient(model=model_id, token=api_key)
        response = call_hf_model(client, system_prompt, user_message)
        return response
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_hf_chatbot_ui(times_df, qs_df, usn_df, washington_df, selected_universities, selected_years):
    """Render the Hugging Face chatbot UI"""
    global _DATASETS, _CURRENT_AGENCY

    # Set global datasets
    _DATASETS["TIMES"] = times_df
    _DATASETS["QS"] = qs_df
    _DATASETS["USN"] = usn_df
    _DATASETS["Washington"] = washington_df

    api_key = None
    try:
        api_key = st.secrets.get("HF_API_KEY", None)
    except:
        pass
    if not api_key:
        api_key = os.getenv("HF_API_KEY")

    # Initialize session state
    if "hf_chat_messages" not in st.session_state:
        st.session_state.hf_chat_messages = []
    if "hf_current_agency" not in st.session_state:
        st.session_state.hf_current_agency = "TIMES"

    # UI Header
    st.sidebar.markdown("---")
    st.sidebar.header("Open-Source AI Assistant")

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model:",
        list(AVAILABLE_MODELS.keys()),
        key="hf_model_select"
    )

    # Agency selection
    agency_options = ["TIMES", "QS", "USN", "Washington"]
    selected_agency = st.sidebar.selectbox(
        "Select Dataset:",
        agency_options,
        key="hf_agency_select"
    )

    _CURRENT_AGENCY = selected_agency

    # Clear chat if agency changed
    if st.session_state.hf_current_agency != selected_agency:
        st.session_state.hf_chat_messages = []
        st.session_state.hf_current_agency = selected_agency

    # API Key status
    if api_key:
        st.sidebar.success(f"Ready: {selected_model}")
    else:
        st.sidebar.error("HF API Key Required")
        st.sidebar.markdown("""
        Get your free API key:
        """)
        return

    # Chat input
    user_input = st.sidebar.text_input(
        "Ask about rankings:",
        key="hf_chat_input",
        placeholder="e.g., Which university has the best rank?"
    )

    col1, col2 = st.sidebar.columns(2)
    send_clicked = col1.button("Ask", use_container_width=True, key="hf_send")
    clear_clicked = col2.button("Clear", use_container_width=True, key="hf_clear")

    if clear_clicked:
        st.session_state.hf_chat_messages = []
        st.rerun()

    if send_clicked and user_input.strip():
        # Add user message
        st.session_state.hf_chat_messages.append({
            "role": "user",
            "content": user_input
        })

        # Get model ID
        model_id = AVAILABLE_MODELS[selected_model]

        # Get response
        with st.sidebar:
            with st.spinner(f"{selected_model} thinking..."):
                response = get_ai_response(user_input, api_key, model_id)

        # Add assistant message
        st.session_state.hf_chat_messages.append({
            "role": "assistant",
            "content": response
        })

        st.rerun()

    # Display chat history
    st.sidebar.markdown("---")

    if st.session_state.hf_chat_messages:
        for msg in st.session_state.hf_chat_messages[-4:]:
            if msg["role"] == "user":
                st.sidebar.markdown(f"**You:** {msg['content']}")
            else:
                st.sidebar.markdown(f"**AI:** {msg['content']}")
            st.sidebar.markdown("---")
    else:
        st.sidebar.info(f"""{selected_model}""")
