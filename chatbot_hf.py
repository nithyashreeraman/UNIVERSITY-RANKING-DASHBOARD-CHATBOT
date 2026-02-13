
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

def extract_year_from_question(question: str, available_years: list) -> dict:
    """
    Extract year information from user's question.
    Returns dict with:
    - 'type': 'single', 'multiple', 'range', 'relative', 'all'
    - 'years': list of specific years to include
    """
    question_lower = question.lower()

    # Check for "all years" or aggregate queries
    all_years_keywords = ['all years', 'all available years', 'across all years', 'every year',
                          'overall', 'average', 'mean', 'across years', 'over time',
                          'historical', 'trend', 'history']
    if any(keyword in question_lower for keyword in all_years_keywords):
        return {'type': 'all', 'years': available_years}

    # Check for relative time periods
    relative_patterns = [
        (r'last (\d+) years?', 'last_n'),
        (r'past (\d+) years?', 'last_n'),
        (r'previous (\d+) years?', 'last_n'),
        (r'recent (\d+) years?', 'last_n'),
    ]

    for pattern, rel_type in relative_patterns:
        match = re.search(pattern, question_lower)
        if match:
            n = int(match.group(1))
            recent_years = sorted(available_years, reverse=True)[:n]
            return {'type': 'relative', 'years': sorted(recent_years)}

    # Look for 4-digit years in the question
    year_pattern = r'\b(20\d{2})\b'
    matches = re.findall(year_pattern, question)
    found_years = []

    for match in matches:
        year = int(match)
        if year in available_years:
            found_years.append(year)

    if not found_years:
        # No specific year mentioned, return latest year only
        return {'type': 'single', 'years': [max(available_years)]}

    # Check for year range patterns
    range_patterns = [
        r'from (\d{4}) to (\d{4})',
        r'between (\d{4}) and (\d{4})',
        r'(\d{4})\s*-\s*(\d{4})',
        r'(\d{4})\s+to\s+(\d{4})',
    ]

    for pattern in range_patterns:
        match = re.search(pattern, question)
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            range_years = [y for y in available_years if start_year <= y <= end_year]
            if range_years:
                return {'type': 'range', 'years': sorted(range_years)}

    # Multiple specific years found
    if len(found_years) > 1:
        return {'type': 'multiple', 'years': sorted(found_years)}

    # Single specific year
    return {'type': 'single', 'years': found_years}


def prepare_dataset_context(df: pd.DataFrame, question: str = "") -> str:
    """
    Prepare dataset context for the LLM.
    - Detects year requirements from question (single, multiple, range, all)
    - Sends appropriate year(s) data based on question context
    """
    if df is None or df.empty:
        return "No data available."

    # Get available years
    available_years = sorted(df['Year'].unique().tolist())

    # Check if user asked about specific year(s)
    year_info = extract_year_from_question(question, available_years)
    target_years = year_info['years']

    # Filter to target year(s)
    year_df = df[df['Year'].isin(target_years)].copy()

    # Select key columns based on agency
    key_columns = ['IPEDS_Name', 'Year']

    if _CURRENT_AGENCY == "TIMES":
        key_columns += [
            'Times_Rank', 'Overall', 'Teaching', 'Research_Quality', 'Research_Environment',
            'Industry', 'International_Outlook',
            'No_of_FTE_Students', 'No_of_students_per_staff', 'International_Students',
            'Female_Ratio', 'Male_Ratio'
        ]
    elif _CURRENT_AGENCY == "QS":
        key_columns += [
            'QS_Rank', 'Overall_Score', 'Academic_Reputation', 'Employer_Reputation',
            'Citations_per_Faculty', 'Faculty_Student_Ratio', 'Employment_Outcomes',
            'International_Faculty_Ratio', 'International_Student_Ratio',
            'International_Research_Network', 'Sustainability_Score',
            'International_Student_Diversity'
        ]
    elif _CURRENT_AGENCY == "USN":
        key_columns += [
            'Rank', 'Overall_scores', 'Peer_assessment_score',
            'Actual_graduation_rate', 'Average_first_year_retention_rate',
            '6-year_Graduation_Rate', 'Over_/_Under-_Performance',
            'Pell_Graduation_Rate', 'Median_debt_for_grads_with_federal_loans',
            'College_grads_earning_more_than_a_HS_grad', 'Financial_resources_rank',
            'Student-faculty_ratio', 'Percent_of_full-time_faculty',
            'Bibliometric_Rank', 'Social_Mobility_Rank',
            'Alumni_Giving', 'Acceptance_rate'
        ]
    elif _CURRENT_AGENCY == "Washington":
        key_columns += [
            'Washington_Rank', '8-year_graduation_rate', 'Research_expenditures_(M)',
            'Social_mobility_rank', 'Research_rank', 'Service_rank',
            'Access_rank', 'Affordability_rank', 'Outcomes_rank',
            'Number_of_Pell_recipients', 'Actual_vs._predicted_Pell_enrollment',
            'Net_price_of_attendance_for_families_below_$75,000_income',
            'Student_loan_debt_of_graduates', 'Pell/non-Pell_graduation_gap',
            'Grad_rate_performance_rank', 'Earnings_after_9_years',
            'Median_Earnings_after_10_years', "Bachelor's_to_PhD_rank",
            'AmeriCorps/Peace_Corps_rank', 'ROTC_rank',
            'Work-study_service_%', 'Service-oriented_majors_%'
        ]

    available_cols = [c for c in key_columns if c in year_df.columns]
    year_df = year_df[available_cols]

    # Sort by Year and University for consistency
    year_df = year_df.sort_values(['Year', 'IPEDS_Name'])

    csv_data = year_df.to_csv(index=False)

    # Build context message
    if year_info['type'] == 'all':
        year_desc = f"ALL YEARS ({min(target_years)}-{max(target_years)})"
    elif year_info['type'] in ['multiple', 'range', 'relative']:
        year_desc = f"YEARS: {', '.join(map(str, target_years))}"
    else:
        year_desc = f"YEAR: {target_years[0]}"

    context = f"""
DATASET: {_CURRENT_AGENCY} University Rankings
SHOWING DATA FOR {year_desc}
AVAILABLE YEARS IN DATASET: {available_years}
TOTAL UNIVERSITIES PER YEAR: ~{len(year_df) // len(target_years) if len(target_years) > 0 else len(year_df)}
COLUMNS: {', '.join(available_cols)}

DATA (CSV format):
{csv_data}
"""
    return context


def call_hf_model(client: InferenceClient, system_prompt: str, user_message: str, max_tokens: int = 500) -> str:
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
    system_prompt = """You are a university rankings data analyst. You help users understand university ranking data with EXTREME ACCURACY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 CRITICAL RULES - READ CAREFULLY BEFORE EVERY RESPONSE 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. RANKING INTERPRETATION (MOST IMPORTANT):
   ⚠️ LOWER RANK NUMBER = BETTER RANKING ⚠️

   Examples:
   • Rank 50 is BETTER than Rank 200
   • Rank 100 is BETTER than Rank 500
   • Rank 1 is BETTER than Rank 1000

   When comparing:
   • University with Rank 150 BEATS University with Rank 300
   • University with Rank 80 BEATS University with Rank 100

   ❌ NEVER say "higher rank" when you mean "better ranking"
   ✅ ALWAYS say "lower rank number" or "better ranking"

2. SCORES INTERPRETATION:
   • HIGHER score = BETTER performance
   • Score 90 is BETTER than Score 50
   • Score 75.5 is BETTER than Score 68.0

3. DATA VALIDATION - MANDATORY BEFORE ANSWERING:
   ✅ Double-check ALL numbers before stating them
   ✅ Verify year data matches the question
   ✅ For calculations (averages, changes), verify using actual data values
   ✅ For comparisons, confirm which university has the LOWER rank number
   ✅ If data seems inconsistent, acknowledge it

4. COMPARISON LOGIC:
   When asked "Is X better than Y?":
   STEP 1: Find X's rank number
   STEP 2: Find Y's rank number
   STEP 3: Compare: Lower number wins
   STEP 4: State conclusion clearly

   Example: "NJIT Rank 120 vs WPI Rank 250 → NJIT is ranked BETTER (120 < 250)"

5. MULTI-YEAR QUESTIONS:
   • "Last 3 years" → Check data for 3 most recent years
   • "Compare 2024 and 2025" → Show both years explicitly
   • "Average across all years" → Calculate using ALL year values provided
   • "Trend over time" → Show year-by-year progression

6. MISSING DATA:
   • If year is not in the provided data, say "Data not available for [year]"
   • Do NOT make up or extrapolate data
   • Do NOT assume consistent values across years

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESPONSE STYLE:
- Keep responses SHORT and PRECISE (2-4 sentences maximum)
- For comparisons: Use bullet points, show only KEY metrics
- End with a 1-line CONCLUSION that clearly states who performs better
- Use ONLY data from the provided CSV - NO external knowledge
- Show specific numbers from the data to support your answer"""

    # User message with context and question
    user_message = f"""{dataset_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION: {question}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REMINDER BEFORE ANSWERING:
1. ⚠️ LOWER rank number = BETTER ranking (Rank 50 beats Rank 200)
2. ✅ Verify all numbers from the CSV data above
3. ✅ For comparisons, check which university has LOWER rank number
4. ✅ For multi-year questions, use ALL relevant years from data
5. ✅ Never make up data - only use what's provided above

Answer in 2-4 sentences max. Be brief, direct, and ACCURATE."""

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
