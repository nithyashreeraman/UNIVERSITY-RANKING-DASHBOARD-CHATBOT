
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
                          'historical', 'trend', 'history', 'between years', 'any two years',
                          'biggest change', 'largest change', 'most change', 'compare years',
                          'change over', 'difference between years']
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


def extract_universities_from_question(question: str, available_universities: list) -> list:
    """
    Extract university names from question by matching against available universities.
    Handles common abbreviations and partial matches.
    """
    question_lower = question.lower()
    matched_unis = []

    # Common abbreviations mapping
    abbreviations = {
        'njit': 'New Jersey Institute of Technology',
        'mit': 'Massachusetts Institute of Technology',
        'caltech': 'California Institute of Technology',
        'georgia tech': 'Georgia Institute of Technology-Main Campus',
        'rutgers': 'Rutgers University-New Brunswick',
        'cmu': 'Carnegie Mellon University',
        'wpi': 'Worcester Polytechnic Institute',
        'rpi': 'Rensselaer Polytechnic Institute',
        'stevens': 'Stevens Institute of Technology'
    }

    # Check for abbreviations
    for abbr, full_name in abbreviations.items():
        if abbr in question_lower and full_name in available_universities:
            matched_unis.append(full_name)

    # Check for partial matches with actual university names
    for uni in available_universities:
        uni_lower = uni.lower()
        # Check if any significant part of the university name appears in question
        uni_words = [w for w in uni_lower.split() if len(w) > 3 and w not in ['university', 'institute', 'technology', 'college']]
        for word in uni_words:
            if word in question_lower and uni not in matched_unis:
                matched_unis.append(uni)
                break

    return matched_unis


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

    # Filter to target year(s) - ensure we're only getting the requested years
    year_df = df[df['Year'].isin(target_years)].copy()

    # Verify filtering worked - if multiple years still present, force to latest only
    actual_years = sorted(year_df['Year'].unique().tolist())
    if len(actual_years) > 1 and year_info['type'] != 'all':
        # Should not have multiple years unless explicitly requested
        # Force to latest year only to avoid model confusion
        year_df = year_df[year_df['Year'] == max(actual_years)].copy()
        target_years = [max(actual_years)]

    # Extract universities mentioned in question and filter (to reduce token usage)
    available_unis = year_df['IPEDS_Name'].unique().tolist()
    mentioned_unis = extract_universities_from_question(question, available_unis)

    # DIAGNOSTIC: Check if NJIT is in the dataset
    njit_name = "New Jersey Institute of Technology"
    njit_present = njit_name in available_unis

    # Check if question needs discovery/comparison context (not just specific university lookup)
    question_lower = question.lower()
    discovery_keywords = ['best', 'top', 'which', 'who']
    competitor_keywords = ['competitor', 'similar', 'peer', 'comparable']
    needs_context = any(keyword in question_lower for keyword in discovery_keywords)
    needs_competitors = any(keyword in question_lower for keyword in competitor_keywords)

    # Check for geographic filtering (e.g., "in NJ", "in New Jersey")
    nj_keywords = [' nj ', ' new jersey', 'in nj', 'in new jersey']
    needs_nj_filter = any(keyword in question_lower for keyword in nj_keywords)

    # Smart filtering logic (priority order matters):
    # 1. Competitor questions - HIGHEST PRIORITY
    if needs_competitors:
        # Competitor question: find universities with similar rank (±25 rank positions)
        # e.g., "Who are NJIT's competitors?"
        njit_name = "New Jersey Institute of Technology"

        # Debug: Check if NJIT is in the year-filtered dataset
        if njit_name not in available_unis:
            # NJIT not in this agency's dataset for this year - send all available data
            pass  # Fall through to else clause to send top 50
        else:
            # Find NJIT's rank and filter to nearby universities
            rank_col = None
            if _CURRENT_AGENCY == "TIMES" and 'Times_Rank' in year_df.columns:
                rank_col = 'Times_Rank'
            elif _CURRENT_AGENCY == "QS" and 'QS_Rank' in year_df.columns:
                rank_col = 'QS_Rank'
            elif _CURRENT_AGENCY == "USN" and 'Rank' in year_df.columns:
                rank_col = 'Rank'
            elif _CURRENT_AGENCY == "Washington" and 'Washington_Rank' in year_df.columns:
                rank_col = 'Washington_Rank'

            if rank_col:
                njit_data = year_df[year_df['IPEDS_Name'] == njit_name]
                if not njit_data.empty:
                    njit_rank_raw = njit_data[rank_col].iloc[0]

                    # Parse rank (handle string ranges like "501-600" or numeric values)
                    try:
                        if pd.isna(njit_rank_raw):
                            # No rank available, skip filtering
                            pass
                        elif isinstance(njit_rank_raw, str) and ('-' in str(njit_rank_raw) or '–' in str(njit_rank_raw)):
                            # Parse range like "501-600" or "501–600" to get midpoint (handles both hyphen and en-dash)
                            parts = str(njit_rank_raw).replace("–", "-").split("-")
                            njit_rank_mid = (int(parts[0]) + int(parts[1])) // 2

                            # For string ranges, we need to parse all ranks to filter properly
                            # Create a temporary numeric column for filtering
                            def parse_rank_to_mid(rank_val):
                                if pd.isna(rank_val):
                                    return float('inf')
                                if isinstance(rank_val, str) and ('-' in str(rank_val) or '–' in str(rank_val)):
                                    try:
                                        p = str(rank_val).replace("–", "-").split("-")
                                        return (int(p[0]) + int(p[1])) // 2
                                    except:
                                        return float('inf')
                                try:
                                    return float(rank_val)
                                except:
                                    return float('inf')

                            year_df['_temp_numeric_rank'] = year_df[rank_col].apply(parse_rank_to_mid)
                            # Filter to universities within ±50 ranks of NJIT (wider range for string ranks)
                            year_df = year_df[
                                (year_df['_temp_numeric_rank'] >= njit_rank_mid - 50) &
                                (year_df['_temp_numeric_rank'] <= njit_rank_mid + 50)
                            ].copy()
                            year_df = year_df.drop(columns=['_temp_numeric_rank'])

                            # Ensure NJIT itself is in the results
                            if njit_name not in year_df['IPEDS_Name'].values:
                                # Add NJIT back if it was filtered out
                                njit_row = df[(df['IPEDS_Name'] == njit_name) & (df['Year'].isin(target_years))]
                                if not njit_row.empty:
                                    year_df = pd.concat([year_df, njit_row], ignore_index=True)
                        else:
                            # Numeric rank
                            njit_rank = float(njit_rank_raw)

                            # Convert rank column to numeric for comparison (handles string ranks like USN)
                            year_df['_temp_rank'] = pd.to_numeric(year_df[rank_col], errors='coerce')

                            # Filter to universities within ±25 ranks of NJIT
                            year_df = year_df[
                                (year_df['_temp_rank'] >= njit_rank - 25) &
                                (year_df['_temp_rank'] <= njit_rank + 25)
                            ].copy()

                            # Drop temporary column
                            year_df = year_df.drop(columns=['_temp_rank'])

                            # Ensure NJIT itself is in the results
                            if njit_name not in year_df['IPEDS_Name'].values:
                                # Add NJIT back if it was filtered out
                                njit_row = df[(df['IPEDS_Name'] == njit_name) & (df['Year'].isin(target_years))]
                                if not njit_row.empty:
                                    year_df = pd.concat([year_df, njit_row], ignore_index=True)
                    except (ValueError, TypeError, IndexError):
                        # If parsing fails, send top 50
                        try:
                            year_df = year_df.nsmallest(50, rank_col)
                        except (TypeError, ValueError, KeyError):
                            try:
                                year_df = year_df.sort_values(rank_col).head(50)
                            except (TypeError, ValueError, KeyError):
                                # If both fail, take first 50 rows
                                year_df = year_df.head(50)
    # 2. Geographic filtering
    elif needs_nj_filter and 'New_Jersey_University' in year_df.columns:
        year_df = year_df[year_df['New_Jersey_University'] == 'Yes'].copy()
    # 3. Direct comparison between specific universities
    elif mentioned_unis and len(mentioned_unis) >= 2:
        year_df = year_df[year_df['IPEDS_Name'].isin(mentioned_unis)].copy()
    # 4. Single university lookup
    elif mentioned_unis and len(mentioned_unis) == 1 and not needs_context:
        year_df = year_df[year_df['IPEDS_Name'].isin(mentioned_unis)].copy()
    # 5. General discovery - send top 50
    else:
        # General discovery question → send top 50 by rank
        # This handles: "Which is best?", "Top universities"
        rank_col = None
        if _CURRENT_AGENCY == "TIMES" and 'Times_Rank' in year_df.columns:
            rank_col = 'Times_Rank'
        elif _CURRENT_AGENCY == "QS" and 'QS_Rank' in year_df.columns:
            rank_col = 'QS_Rank'
        elif _CURRENT_AGENCY == "USN" and 'Rank' in year_df.columns:
            rank_col = 'Rank'
        elif _CURRENT_AGENCY == "Washington" and 'Washington_Rank' in year_df.columns:
            rank_col = 'Washington_Rank'

        if rank_col:
            try:
                # Try numeric sorting first (works for USN, Washington)
                year_df = year_df.nsmallest(50, rank_col)
            except (TypeError, ValueError, KeyError):
                # If that fails, try sort_values (for string ranks)
                try:
                    year_df = year_df.sort_values(rank_col).head(50)
                except (TypeError, ValueError, KeyError):
                    # If both sorting methods fail, just take first 50 rows
                    year_df = year_df.head(50)

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

    # Get list of universities in the filtered dataset for debug visibility
    unis_in_data = sorted(year_df['IPEDS_Name'].unique().tolist())

    # Check if NJIT is in filtered data
    njit_check = "YES - New Jersey Institute of Technology is in this data" if njit_present else "NO - New Jersey Institute of Technology is NOT in this data"

    # Add filtering note for competitor questions
    filter_note = ""
    if needs_competitors and njit_present:
        filter_note = "\n⚠️ IMPORTANT: This data has been PRE-FILTERED to show only universities with similar ranks to New Jersey Institute of Technology. All universities in this CSV are appropriate competitors - just list them directly from the data below."

    context = f"""
DATASET: {_CURRENT_AGENCY} University Rankings
SHOWING DATA FOR {year_desc}
AVAILABLE YEARS IN DATASET: {available_years}
TOTAL UNIVERSITIES IN THIS DATA: {len(unis_in_data)}
NJIT PRESENT: {njit_check}
UNIVERSITIES INCLUDED: {', '.join(unis_in_data[:20])}{'...' if len(unis_in_data) > 20 else ''}{filter_note}
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


def expand_abbreviations_in_question(question: str) -> str:
    """
    Expand university abbreviations to full IPEDS names in the question.
    This helps the model find universities in the CSV data.
    """
    # Map abbreviations to full IPEDS names
    abbreviations = {
        'NJIT': 'New Jersey Institute of Technology',
        'MIT': 'Massachusetts Institute of Technology',
        'Caltech': 'California Institute of Technology',
        'Georgia Tech': 'Georgia Institute of Technology-Main Campus',
        'Rutgers': 'Rutgers University-New Brunswick',
        'CMU': 'Carnegie Mellon University',
        'WPI': 'Worcester Polytechnic Institute',
        'RPI': 'Rensselaer Polytechnic Institute',
        'Stevens': 'Stevens Institute of Technology',
        'UT Austin': 'The University of Texas at Austin',
        'Penn State': 'The Pennsylvania State University',
        'Ohio State': 'Ohio State University-Main Campus',
        'Michigan State': 'Michigan State University',
        'Case Western': 'Case Western Reserve University'
    }

    expanded_question = question
    for abbr, full_name in abbreviations.items():
        # Case-insensitive replacement, preserve original case in output
        import re
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        expanded_question = pattern.sub(full_name, expanded_question)

    return expanded_question


def get_ai_response(question: str, api_key: str, model_id: str) -> str:
    """
    Main function to get AI response.
    Sends dataset context + question to the model.
    """
    global _DATASETS, _CURRENT_AGENCY

    df = _DATASETS.get(_CURRENT_AGENCY)
    if df is None:
        return "Error: No dataset loaded."

    # Expand abbreviations in question so model can find universities in CSV
    expanded_question = expand_abbreviations_in_question(question)

    # Prepare dataset context (auto-detects year from question)
    dataset_context = prepare_dataset_context(df, expanded_question)

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

3. METRIC TYPES - CRITICAL FOR CORRECT INTERPRETATION:
   ⚠️ DISTINGUISH BETWEEN COUNTS, RATES, AND GAPS ⚠️

   A. COUNTS (Raw Numbers):
      • Examples: Number_of_Pell_recipients, No_of_FTE_Students
      • These are ABSOLUTE NUMBERS (e.g., 233 students, 5000 students)
      • ❌ NEVER call these "rates" or "percentages"
      • Interpretation: Larger institutions naturally have higher counts
      • For "best rate" questions, DO NOT use count columns

   B. RATES/PERCENTAGES (0-100 or 0.0-1.0):
      • Examples: Graduation_rate, Pell_Graduation_Rate, Retention_rate
      • Format: 0.85 = 85%, or already shown as 85
      • HIGHER rate = BETTER performance (e.g., 85% grad rate > 60% grad rate)
      • These measure SUCCESS relative to population size

   C. GAPS (Differences Between Groups):
      • Examples: Pell/non-Pell_graduation_gap, Earnings_gap
      • SMALLER gap (closer to 0) = BETTER equity
      • NEGATIVE gap may be GOOD (e.g., -0.05 means Pell students graduate MORE)
      • Zero gap = perfect equity

   D. PERFORMANCE METRICS (Actual vs Predicted):
      • Examples: Actual_vs._predicted_Pell_enrollment, Grad_rate_performance_rank
      • HIGHER than 1.0 = EXCEEDING expectations (GOOD)
      • LOWER than 1.0 = UNDERPERFORMING expectations (BAD)

   E. EXPENDITURES/FINANCIAL:
      • Examples: Research_expenditures_(M), Net_price
      • Context-dependent: Higher spending may be good (research), lower may be good (net price/affordability)

   When answering "which has good [metric]" questions:
   ✅ Identify if they're asking about COUNT, RATE, or GAP
   ✅ Use the CORRECT column type (rate columns for rate questions)
   ✅ Explain what the number means (e.g., "40% Pell rate" not "40 Pell rate")

4. DATA SOURCE - USE PROVIDED CSV
   The CSV data provided contains all relevant information. Answer questions based on this data.
   • If asked about a university in the CSV, provide the data shown
   • If a university isn't in the filtered CSV but the question is about competitors/comparisons, use the universities that ARE in the CSV
   • Avoid using general knowledge about universities - focus on the specific data provided

4a. TIED RANKS (STRING RANGES):
   Some agencies use rank ranges (e.g., "501-600") where multiple universities share the same rank band.
   When multiple universities have identical ranks, they are equal competitors.

   Example: If NJIT and several other universities all show rank "501-600", list any 5 as competitors
   and note they're "all tied at rank 501-600".

5. DATA VALIDATION - MANDATORY BEFORE ANSWERING:
   ✅ Double-check ALL numbers before stating them
   ✅ Verify year data matches the question
   ✅ For calculations (averages, changes), verify using actual data values
   ✅ For comparisons, confirm which university has the LOWER rank number
   ✅ If data seems inconsistent, acknowledge it

6. COMPARISON LOGIC:
   When asked "Is X better than Y?":
   STEP 1: Find X's rank number
   STEP 2: Find Y's rank number
   STEP 3: Compare: Lower number wins
   STEP 4: State conclusion clearly

   Example: "NJIT Rank 120 vs WPI Rank 250 → NJIT is ranked BETTER (120 < 250)"

   When asked "Who are X's competitors/peers?":
   ⚠️ COMPETITORS = SIMILAR RANK, NOT BETTER RANK ⚠️
   STEP 1: Find X's rank number
   STEP 2: Find universities with ranks CLOSEST to X's rank (within ±5 positions)
   STEP 3: Prioritize TIED ranks (exact same rank) first
   STEP 4: Then include ranks just above and below

   Example: "NJIT Rank 80 → Competitors are Drexel (80-tied), Stevens (80-tied), WPI (84)"
   ❌ WRONG: "NJIT Rank 80 → Competitors are Villanova (57)" - 57 is BETTER, not a peer

7. MULTI-YEAR QUESTIONS:
   • "Last 3 years" → Check data for 3 most recent years
   • "Compare 2024 and 2025" → Show both years explicitly
   • "Average across all years" → Calculate using ALL year values provided
   • "Trend over time" → Show year-by-year progression

8. MISSING DATA:
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
QUESTION: {expanded_question}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REMINDER BEFORE ANSWERING:
1. ⚠️ ONLY use universities listed in the CSV above - do not mention any university not in the CSV
2. LOWER rank number = BETTER ranking (Rank 50 beats Rank 200)
3. Distinguish metric types: Number_of_X = COUNT, X_rate = PERCENTAGE, X_gap = DIFFERENCE
4. For tied ranks (same rank like "501-600"), list universities as equal competitors
5. For comparisons, check which university has LOWER rank number
6. For multi-year questions, use ALL relevant years from data

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

def render_hf_chatbot_ui(times_df, qs_df, usn_df, washington_df, sidebar_selected_unis, selected_years):
    """Render the Hugging Face chatbot UI"""
    global _DATASETS, _CURRENT_AGENCY

    # Store full datasets (filtering happens in prepare_dataset_context based on question)
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
