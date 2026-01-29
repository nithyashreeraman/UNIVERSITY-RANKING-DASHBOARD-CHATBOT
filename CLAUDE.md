# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University Rankings Dashboard - A Streamlit application that compares NJIT (New Jersey Institute of Technology) against peer universities across four ranking agencies: TIMES, QS, US News (USN), and Washington Monthly. Features an AI chatbot powered by Hugging Face open-source models for natural language queries about the rankings data.

## Common Commands

```bash
# Run the Streamlit dashboard
streamlit run main.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Main Application (`main.py`)
- Streamlit dashboard with 5 tabs: Overview, TIMES, QS, USN, Washington
- Loads data from 4 Excel files: `TIMES.xlsx`, `QS.xlsx`, `USN.xlsx`, `Washington.xlsx`
- AI chatbot: Imports `chatbot_hf.py` for Hugging Face open-source model integration
- Peer group system: Users can select entire peer groups or individual universities
  - Peer groups defined in `peer.csv`: BENCHMARK PEERS, ASPIRATIONAL PEERS, NJ PEERS
  - Manual selections persist per tab in session state (e.g., `manual_times_selected_unis`)
  - Rutgers University-New Brunswick is default comparison when no peer groups selected
- Color mapping: Fixed color assignments ensure consistent university colors across all visualizations
  - NJIT always red (#E10600)
  - 19 predefined colors for peer universities
  - Fallback hash-based colors for additional universities

### AI Chatbot System (`chatbot_hf.py`)
Powered by Hugging Face Inference API with open-source models:
- **Supported models**: Qwen-2.5-72B (72B params), Llama-3.3-70B (70B params)
- **Year-aware context**: Automatically detects year mentions in questions and filters data
- **Token efficiency**: Sends only year-specific dataset (latest year by default) to reduce context size
- **Free to use**: No API costs, only requires free HF account

Key functions:
- `extract_year_from_question()` - Parse year from user query (e.g., "2024" → filters to 2024 data)
- `prepare_dataset_context()` - Build CSV context for specific year with agency-specific columns
- `get_ai_response()` - Main inference call to selected HF model
- `render_hf_chatbot_ui()` - Streamlit sidebar UI with model selection and chat interface

### Data Files
- **Excel files**: TIMES.xlsx, QS.xlsx, USN.xlsx, Washington.xlsx - Historical ranking data (2020-2026)
- **peer.csv**: Maps universities to peer types (BENCHMARK PEERS, ASPIRATIONAL PEERS, NJ PEERS)
- **IPEDS_Name column**: Standardized university names from IPEDS database
  - All university references in code use IPEDS format (e.g., "New Jersey Institute of Technology" not "NJIT")
  - Common mappings: NJIT → "New Jersey Institute of Technology", Rutgers → "Rutgers University-New Brunswick"

## Configuration

- **API Key**: Stored in `.streamlit/secrets.toml`
  - `HF_API_KEY` - Hugging Face API token (get free from https://huggingface.co/settings/tokens)
  - Falls back to environment variable `HF_API_KEY` if secrets.toml not available
  - **IMPORTANT**: Never commit actual API keys to version control
- **Streamlit config**:
  - Page layout: "wide" mode
  - Caching: `@st.cache_data` on data loading functions
  - Session state: Tracks chat history (`hf_chat_messages`), current agency, and per-tab university selections

## Key Conventions

### Data Semantics
- **University names**: Always use IPEDS standardized format (e.g., "New Jersey Institute of Technology", not "NJIT")
- **Rankings**: Lower number = Better (Rank 1 is best, Rank 1000+ is poor)
  - Rankings may contain ranges (e.g., "501-600") which are parsed into low/high/mid for visualization
- **Scores**: Higher number = Better (90 > 50)
- **NJIT**: Primary comparison target, always included and colored red (#E10600)

### Visualization Patterns
- Line charts: Used for trends over time, with markers and text labels showing values
- Range charts: TIMES and QS rankings displayed as shaded bands with transparent fill (rgba opacity 0.15)
- Bar charts: Used for gender distribution (faceted by university)
- KPI boxes: Grid layout showing latest year metrics for all selected universities
- Tabs: Each ranking agency has dedicated tab with agency-specific metrics

### Session State Management
- Peer group selections in sidebar affect all tabs globally
- Manual university selections persist per tab independently (e.g., `manual_times_selected_unis`)
- Chat history maintained in `hf_chat_messages` state variable
- Agency selection for chatbot clears chat history on change
