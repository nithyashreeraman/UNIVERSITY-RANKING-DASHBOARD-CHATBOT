import streamlit as st
import pandas as pd
import plotly.express as px
import re
import plotly.graph_objects as go
import matplotlib.colors as mcolors

# Only use Hugging Face chatbot (open-source models)
try:
    from chatbot_hf import render_hf_chatbot_ui
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False

def rgba_with_opacity(color, alpha=0.15):
    try:
        # Convert to rgba using matplotlib
        rgba = mcolors.to_rgba(color, alpha=alpha)
        return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"
    except:
        # Fallback for any color conversion issues
        return f"rgba(128, 128, 128, {alpha})"

st.set_page_config(page_title="University Dashboard", layout="wide")
st.title("🏛️ University Rankings Dashboard")

@st.cache_data
def load_data():
    times_df = pd.read_excel("TIMES.xlsx", sheet_name="Sheet1")
    qs_df = pd.read_excel("QS.xlsx", sheet_name="Sheet1")
    usn_df = pd.read_excel("USN.xlsx", sheet_name="Sheet1")
    washington_df = pd.read_excel("Washington.xlsx", sheet_name="Sheet1")
    return times_df, qs_df, usn_df, washington_df

@st.cache_data
def load_peer_groups():
    try:
        peer_df = pd.read_csv("peer.csv")
        return peer_df
    except FileNotFoundError:
        st.error("❌ File not found.")
        return pd.DataFrame(columns=['PEER_TYPE', 'PEER_NAME'])

@st.cache_data
def get_common_universities(times_df, qs_df, usn_df, washington_df):
    return set(times_df["IPEDS_Name"]) & set(qs_df["IPEDS_Name"]) & set(usn_df["IPEDS_Name"]) & set(washington_df["IPEDS_Name"])

@st.cache_data
def get_filtered_combined_df(_combined_df, common_universities, nj_filter):
    combined_common_df = _combined_df[_combined_df["IPEDS_Name"].isin(common_universities)]
    if nj_filter == "Yes":
        combined_common_df = combined_common_df[combined_common_df["New_Jersey_University"] == "Yes"]
    elif nj_filter == "No":
        combined_common_df = combined_common_df[combined_common_df["New_Jersey_University"] == "No"]
    return combined_common_df

def get_peer_type(university_name, peer_df):
    match = peer_df[peer_df['PEER_NAME'] == university_name]
    return match['PEER_TYPE'].iloc[0] if not match.empty else None

def create_color_map(universities_list):
    """Create consistent color map where each university always gets the same color"""
    # Fixed color assignment based on university name - using HEX colors only
    university_color_mapping = {
        NJIT_NAME: "#E10600",  # NJIT - Red
        # Benchmark Peers
        "Clarkson University": "#FF7F0E",  # Orange
        "Colorado School of Mines": "#2CA02C",  # Green
        "Florida Institute of Technology": "#D62728",  # Red
        "Illinois Institute of Technology": "#9467BD",  # Purple
        "Michigan Technological University": "#8C564B",  # Brown
        "Missouri University of Science and Technology": "#E377C2",  # Pink
        "Rensselaer Polytechnic Institute": "#7F7F7F",  # Gray
        "Stevens Institute of Technology": "#BCBD22",  # Yellow-Green
        "Worcester Polytechnic Institute": "#17BECF",  # Cyan
        # Aspirational Peers
        "California Institute of Technology": "#FF9896",  # Light Red
        "Carnegie Mellon University": "#98DF8A",  # Light Green
        "Georgia Institute of Technology-Main Campus": "#FFBB78",  # Light Orange
        "Massachusetts Institute of Technology": "#C5B0D5",  # Light Purple
        # NJ Peers
        "Montclair State University": "#C49C94",  # Tan
        "Rowan University": "#F7B6D2",  # Light Pink
        "Rutgers University-New Brunswick": "#1F77B4",  # Blue
        "Rutgers University-Newark": "#C7C7C7",  # Light Gray
        "Seton Hall University": "#DBDB8D"  # Light Yellow
    }
    
    color_map = {}
    for uni in universities_list:
        if uni in university_color_mapping:
            color_map[uni] = university_color_mapping[uni]
        else:
            fallback_colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
            ]
            color_map[uni] = fallback_colors[hash(uni) % len(fallback_colors)]
    
    return color_map

times_df, qs_df, usn_df, washington_df = load_data()
peer_groups_df = load_peer_groups()

for df in [times_df, qs_df, usn_df, washington_df]:
    df["Year"] = df["Year"].astype(int)
    df["IPEDS_Name"] = df["IPEDS_Name"].astype(str)

NJIT_NAME = "New Jersey Institute of Technology"
DEFAULT_RUTGERS = "Rutgers University-New Brunswick"

common_universities = get_common_universities(times_df, qs_df, usn_df, washington_df)

st.sidebar.header("🔍 Filters")

years = sorted(
    list(
        set(times_df["Year"].unique()) |
        set(qs_df["Year"].unique()) |
        set(usn_df["Year"].unique()) |
        set(washington_df["Year"].unique())
    )
)
selected_years = st.sidebar.multiselect("Select Years", years, default=years)

nj_filter = st.sidebar.selectbox("Include Only NJ Universities?", ["All", "Yes", "No"])

# --- NEW: Peer Group Selection ---
st.sidebar.markdown("---")
st.sidebar.header("🎯 Peer Groups")

peer_types = sorted(peer_groups_df['PEER_TYPE'].unique())

selected_peer_types = st.sidebar.multiselect(
    "Select Peer Groups:",
    options=peer_types,
    default=[],
    help="Select peer groups to compare with NJIT"
)

peer_group_universities = []
if selected_peer_types:
    peer_group_universities = peer_groups_df[
        peer_groups_df['PEER_TYPE'].isin(selected_peer_types)
    ]['PEER_NAME'].tolist()

st.sidebar.markdown("---")
st.sidebar.header("🏫 Individual Universities")

#Full Base Dataset from All Agencies
combined_df = pd.concat([times_df, qs_df, usn_df, washington_df], ignore_index=True)

# Filter Combined Dataset
combined_common_df = get_filtered_combined_df(combined_df, common_universities, nj_filter)

# Final Universities for Dropdown
common_universities_filtered = sorted([u for u in combined_common_df["IPEDS_Name"].unique() if u != NJIT_NAME])

# Filter available universities (excluding those already in peer groups)
available_for_manual = [u for u in common_universities_filtered if u not in peer_group_universities]

# Only show Rutgers in manual selection if no peer groups are selected
if selected_peer_types:
    available_for_manual = [u for u in available_for_manual if u != DEFAULT_RUTGERS]

manual_selected_unis = st.sidebar.multiselect(
    "Add individual universities:",
    available_for_manual,
    default=[],
    help="Select additional universities to compare"
)

# Clear per-tab widget state when sidebar selection changes (peer groups or individual unis)
if "previous_peer_types" not in st.session_state:
    st.session_state.previous_peer_types = []
if "previous_manual_unis" not in st.session_state:
    st.session_state.previous_manual_unis = []

sidebar_changed = (
    st.session_state.previous_peer_types != selected_peer_types or
    sorted(st.session_state.previous_manual_unis) != sorted(manual_selected_unis)
)

if sidebar_changed:
    for key in ["times_optional_unis", "qs_optional_unis", "usn_optional_unis", "washington_optional_unis"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.previous_peer_types = selected_peer_types
    st.session_state.previous_manual_unis = manual_selected_unis

# Combine peer groups and manual selections (Rutgers is excluded when peer groups are selected)
all_selected_unis = list(set(peer_group_universities + manual_selected_unis))

# Only include Rutgers by default if no peer groups are selected
if not selected_peer_types and not all_selected_unis and DEFAULT_RUTGERS in common_universities_filtered:
    all_selected_unis = [DEFAULT_RUTGERS]

# Display active peer groups
if selected_peer_types:
    st.sidebar.markdown("---")
    st.sidebar.header("✅ Active Peer Groups")
    for peer_type in selected_peer_types:
        st.sidebar.write(f"**{peer_type}**")
        peers_in_type = peer_groups_df[peer_groups_df['PEER_TYPE'] == peer_type]['PEER_NAME'].tolist()
        for peer in peers_in_type:
            status = "✅" if peer in all_selected_unis else "❌"
            #st.sidebar.write(f"{status} {peer}")
            st.sidebar.write(f"{peer}")

# ============================================================================
# CHATBOT INTEGRATION (Hugging Face Open-Source Models)
# ============================================================================
if CHATBOT_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 AI Assistant")
    render_hf_chatbot_ui(
        times_df, qs_df, usn_df, washington_df,
        all_selected_unis,  # Pass global sidebar selections (chatbot will merge with per-tab selections)
        selected_years
    )

#Extra Universities Per Agency 
extra_times_unis = sorted([u for u in times_df["IPEDS_Name"].unique() if (u not in common_universities and u != NJIT_NAME)])
extra_qs_unis = sorted([u for u in qs_df["IPEDS_Name"].unique() if (u not in common_universities and u != NJIT_NAME)])
extra_usn_unis = sorted([u for u in usn_df["IPEDS_Name"].unique() if (u not in common_universities and u != NJIT_NAME)])
extra_washington_unis = sorted([u for u in washington_df["IPEDS_Name"].unique() if (u not in common_universities and u != NJIT_NAME)])

# Helper Function for KPIs 
def get_metric_value(df, university, column):
    try:
        val = df[df["IPEDS_Name"] == university][column].values[0]
        if isinstance(val, (int, float)):
            return round(val, 2)
        return val if pd.notna(val) else "N/A"
    except:
        return "N/A"

def plot_chart_sorted(df, metric_col, title_label, description, color_map, height=400):
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.sort_values("Year")
    df["Year"] = df["Year"].astype(str)

    fig = px.line(
        df,
        x="Year",
        y=metric_col,
        text=metric_col,
        color="IPEDS_Name",
        markers=True,
        color_discrete_map=color_map,
        title=title_label
    )
    fig.update_traces(
        textposition="top center",
        texttemplate="%{text:.2f}",
        textfont_size=10,
        connectgaps=True
    )
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=70, l=30, r=30),
        title_font=dict(size=15, color="#333"),
        title_x=0.0,
        xaxis=dict(type='category'),
        xaxis_title="Year",
        yaxis_title=title_label,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
            bgcolor='rgba(0,0,0,0)',
            title_text=None
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Chart Description Below
    st.markdown(f"""
        <div style='text-align:center; font-size:0.85rem; font-weight:bold; color:#555; margin-top:8px; margin-bottom:20px;'>
            {description}
        </div>
    """, unsafe_allow_html=True)

# Global KPI Box Styling 
st.markdown("""
    <style>
    .kpi-box {
        background-color: #F6F6F6;
        padding: 10px 8px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border: 1px solid #ddd;
    }
    .kpi-box h4 {
        font-size: 0.78rem;
        margin-bottom: 6px;
        color: #333;
    }
    .kpi-box .kpi-value {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-label {
        font-size: 0.78rem;
        margin-bottom: 6px;
        color: #333;
    }
    .methodology-link {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #f0f2f6;
        padding: 8px 12px;
        border-radius: 5px;
        font-size: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    </style>
""", unsafe_allow_html=True)

#Setup Tabs 
tabs = st.tabs(["📊 Overview", "🟣 TIMES", "🟨 QS", "📘 USN", "🔵 Washington"])

with tabs[0]:
    st.markdown("""
        <h2 style='text-align: center; color: #4B4B4B;'>Overall Ranking</h2>
    """, unsafe_allow_html=True)
    
    universities_to_compare = [NJIT_NAME] + all_selected_unis

 
    color_map = create_color_map(universities_to_compare)

    latest_years = {
        "TIMES": max(times_df[times_df["IPEDS_Name"].isin(universities_to_compare)]["Year"].unique(), default=None),
        "QS": max(qs_df[qs_df["IPEDS_Name"].isin(universities_to_compare)]["Year"].unique(), default=None),
        "USN": max(usn_df[usn_df["IPEDS_Name"].isin(universities_to_compare)]["Year"].unique(), default=None),
        "Washington": max(washington_df[washington_df["IPEDS_Name"].isin(universities_to_compare)]["Year"].unique(), default=None)
    }

    overview_kpi_metrics = {
        "Times_Rank": "TIMES Rank",
        "QS_Rank": "QS Rank",
        "Rank": "USN Rank",
        "Washington_Rank": "Washington Rank"
    }

    kpi_cols = st.columns(len(overview_kpi_metrics))
    for idx, (metric, label) in enumerate(overview_kpi_metrics.items()):
        df_for_kpi = None
        if metric == "Times_Rank":
            df_for_kpi = times_df
        elif metric == "QS_Rank":
            df_for_kpi = qs_df
        elif metric == "Rank":
            df_for_kpi = usn_df
        elif metric == "Washington_Rank":
            df_for_kpi = washington_df

        if df_for_kpi is not None:
            year = latest_years.get(label.split(" ")[0], None)
            
            kpi_html = f"<h4>{label} ({year})</h4>"
            if year:
                kpi_row = df_for_kpi[df_for_kpi["Year"] == year]
                for uni in universities_to_compare:
                    val = get_metric_value(kpi_row, uni, metric)
                    kpi_html += f"<div class='kpi-value' style='color:{color_map.get(uni)}'>{uni}: {val}</div>"
            
            with kpi_cols[idx]:
                st.markdown(f"<div class='kpi-box'>{kpi_html}</div>", unsafe_allow_html=True)
                
    st.divider()

    def parse_rank_range(rank_str):
        try:
            parts = str(rank_str).replace("–", "-").split("-")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]), (int(parts[0]) + int(parts[1])) // 2)
            else:
                val = int(rank_str)
                return (val, val, val)
        except:
            return (None, None, None)

    def build_rank_range_df(df, metric_col):
        df = df.copy()
        df = df[df[metric_col].notna()]
        df[["low", "high", "mid"]] = df[metric_col].apply(lambda r: pd.Series(parse_rank_range(str(r))))
        return df[df["mid"].notna()]

    metrics_tabs = st.tabs(["TIMES Rank", "QS Rank", "USN Rank", "Washington Rank"])
    
    with metrics_tabs[0]:
        times_filtered_for_chart = times_df[
        (times_df["Year"].isin(selected_years)) &
        (times_df["IPEDS_Name"].isin(universities_to_compare))
    ]
        times_ranks = build_rank_range_df(times_filtered_for_chart, "Times_Rank")
        times_ranks = times_ranks.sort_values("Year")

        fig = go.Figure()

        for uni in universities_to_compare:
            uni_df = times_ranks[times_ranks["IPEDS_Name"] == uni]
            base_color = color_map.get(uni)

            fig.add_trace(go.Scatter(
        x=uni_df["Year"],
        y=uni_df["high"],
        mode="lines",
        line=dict(color=base_color),
        name=f"{uni} range",
        showlegend=True,
    ))

    #Low line with transparent fill
            fig.add_trace(go.Scatter(
        x=uni_df["Year"],
        y=uni_df["low"],
        mode="lines",
        line=dict(color=base_color),
        fill='tonexty',
        fillcolor=rgba_with_opacity(base_color, alpha=0.15),
        name=f"{uni} band",
        showlegend=False
    ))

    #Text labels
            fig.add_trace(go.Scatter(
        x=uni_df["Year"],
        y=(uni_df["low"] + uni_df["high"]) / 2,
        mode="text",
        text=uni_df["Times_Rank"],
        textposition="middle center",
        textfont=dict(size=14, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))
        

        fig.update_layout(
        title="TIMES Rank",
        height=450,
        margin=dict(t=30, b=30, l=30, r=30),
        title_font=dict(size=15),
        title_x=0.0,
        xaxis=dict(type='category'),
        yaxis_title="Rank",
        yaxis_autorange="reversed",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
    )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
        "<div style='text-align:center; font-size:0.85rem; margin-top:-5px;'>"
        "TIMES rankings are shown as shaded ranges with reduced opacity. "
        "</div>",
        unsafe_allow_html=True
    )

    with metrics_tabs[1]:
        qs_filtered_for_chart = qs_df[
        (qs_df["Year"].isin(selected_years)) &
        (qs_df["IPEDS_Name"].isin(universities_to_compare))
    ]
        qs_ranks = build_rank_range_df(qs_filtered_for_chart, "QS_Rank")
        qs_ranks = qs_ranks.sort_values("Year")

        fig = go.Figure()

        for uni in universities_to_compare:
            uni_df = qs_ranks[qs_ranks["IPEDS_Name"] == uni]
            base_color = color_map.get(uni)

        # High line 
            fig.add_trace(go.Scatter(
            x=uni_df["Year"],
            y=uni_df["high"],
            mode="lines",
            line=dict(color=base_color),
            name=f"{uni} range",
            showlegend=True,
        ))

        #Low line with transparent fill
            fig.add_trace(go.Scatter(
            x=uni_df["Year"],
            y=uni_df["low"],
            mode="lines",
            line=dict(color=base_color),
            fill='tonexty',
            fillcolor=rgba_with_opacity(base_color, alpha=0.15),
            name=f"{uni} band",
            showlegend=False
        ))

        #Text labels 
            fig.add_trace(go.Scatter(
            x=uni_df["Year"],
            y=(uni_df["low"] + uni_df["high"]) / 2,
            mode="text",
            text=uni_df["QS_Rank"],
            textposition="middle center",
            textfont=dict(size=14, color="black"),
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.update_layout(
        title="QS Rank",
        height=450,
        margin=dict(t=30, b=30, l=30, r=30),
        title_font=dict(size=15),
        title_x=0.0,
        xaxis=dict(type='category'),
        yaxis_title="Rank",
        yaxis_autorange="reversed",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
    )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
        "<div style='text-align:center; font-size:0.85rem; margin-top:-5px;'>"
        "QS rankings are shown as shaded ranges with reduced opacity. "
        "</div>",
        unsafe_allow_html=True
    )

    with metrics_tabs[2]:
        usn_filtered_for_chart = usn_df[(usn_df["Year"].isin(selected_years)) & (usn_df["IPEDS_Name"].isin(universities_to_compare))]
        fig = px.line(
            usn_filtered_for_chart.sort_values("Year"),
            x="Year",
            y="Rank",
            color="IPEDS_Name",
            markers=True,
            text="Rank",
            color_discrete_map=color_map,
            title="USN Rank"
        )
        fig.update_traces(textposition="top center", texttemplate="%{text}")
        fig.update_layout(
            height=450,
            margin=dict(t=30, b=30, l=30, r=30),
            title_font=dict(size=15),
            title_x=0.0,
            xaxis=dict(type='category'),
            yaxis_title="Rank",
            yaxis_autorange="reversed",
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div style='text-align:center; font-size:0.85rem; margin-top:-5px;'>USN ranking is displayed directly. Lower rank indicates better performance</div>", unsafe_allow_html=True)

    with metrics_tabs[3]:
        washington_filtered_for_chart = washington_df[(washington_df["Year"].isin(selected_years)) & (washington_df["IPEDS_Name"].isin(universities_to_compare))]
        fig = px.line(
            washington_filtered_for_chart.sort_values("Year"),
            x="Year",
            y="Washington_Rank",
            color="IPEDS_Name",
            markers=True,
            text="Washington_Rank",
            color_discrete_map=color_map,
            title="Washington Monthly Rank"
        )
        fig.update_traces(textposition="top center", texttemplate="%{text}")
        fig.update_layout(
            height=450,
            margin=dict(t=30, b=30, l=30, r=30),
            title_font=dict(size=15),
            title_x=0.0,
            xaxis=dict(type='category'),
            yaxis_title="Rank",
            yaxis_autorange="reversed",
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div style='text-align:center; font-size:0.85rem; margin-top:-5px;'>Washington Monthly rankings are plotted yearly. Lower ranks indicate stronger outcomes</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("<h2 style='text-align: center; color: #4B4B4B;'>TIMES Ranking</h2>", unsafe_allow_html=True)

    # Build full options list (global + extra TIMES)
    times_options = list(dict.fromkeys(all_selected_unis + extra_times_unis))

    # Get previously selected manual universities for TIMES
    manual_times_selected_unis = st.session_state.get("manual_times_selected_unis", [])

    # Merge peer groups + manual selections -> ensures peer groups are always included
    merged_selected_unis = list(set(all_selected_unis + manual_times_selected_unis))

    # Multi-select for TIMES
    current_times_selected_unis = st.multiselect(
        "🔎 Select universities to compare with NJIT:",
        options=times_options,
        default=merged_selected_unis,
        key="times_optional_unis"
    )

    # Compute manual selections = current - peer groups 
    st.session_state["manual_times_selected_unis"] = [
        uni for uni in current_times_selected_unis if uni not in all_selected_unis
    ]

    # Final unis = NJIT + all selected
    final_times_unis = [NJIT_NAME] + current_times_selected_unis

    color_map = create_color_map(final_times_unis)

    #Filter Data 
    times_filtered_tab = times_df[
        (times_df["Year"].isin(selected_years)) &
        (times_df["IPEDS_Name"].isin(final_times_unis))
    ]

    latest_times_year = max([y for y in selected_years if y in times_filtered_tab["Year"].unique()], default=None)

       #KPI Metrics 
    kpi_metrics = {
        "Times_Rank": "🏅 Rank",
        "Overall": "📊 Overall Score",
        "Teaching": "📖 Teaching",
        "Research_Quality": "🔬 Research Quality",
        "Research_Environment": "🏛️ Research Environment",
        "International_Students": "🌍 Intl. Students %",
        "No_of_students_per_staff": "👩‍🏫 Student/Staff Ratio",
        "No_of_FTE_Students": "🎓 FTE Students"
    }

    kpi_keys = list(kpi_metrics.keys())
    for i in range(0, len(kpi_keys), 4):
        row = st.columns(4)
        for j in range(4):
            if i + j < len(kpi_keys):
                col_key = kpi_keys[i + j]
                label = kpi_metrics[col_key] + (f" ({latest_times_year})" if latest_times_year else "")
                
                kpi_html = f"<h4>{label}</h4>"
                if latest_times_year:
                    kpi_row = times_filtered_tab[times_filtered_tab["Year"] == latest_times_year]
                    for uni in final_times_unis:
                        val = get_metric_value(kpi_row, uni, col_key)
                        kpi_html += f"<div class='kpi-value' style='color:{color_map.get(uni)}'>{uni}: {val}</div>"
                
                with row[j]:
                    st.markdown(f"<div class='kpi-box'>{kpi_html}</div>", unsafe_allow_html=True)

    st.divider()

    section = st.radio(
        "Choose TIMES Section",
        ["📖 Teaching", "🔬 Research Performance", "🌍 Global Engagement & Gender"],
        horizontal=True, key="times_section"
    )

    if section == "📖 Teaching":
        teaching_data = times_filtered_tab[["Year", "IPEDS_Name", "Teaching"]]
        plot_chart_sorted(
            df=teaching_data,
            metric_col="Teaching",
            title_label="📖 Teaching (29.5%)",
            description="Quality of learning environment via teaching reputation and staff ratios",
            color_map=color_map,
        )

    elif section == "🔬 Research Performance":
        col1, col2 = st.columns(2)
        with col1:
            rq_data = times_filtered_tab[["Year", "IPEDS_Name", "Research_Quality"]]
            plot_chart_sorted(
                df=rq_data,
                metric_col="Research_Quality",
                title_label="🔬 Research Quality (30%)",
                description="Research excellence through citation impact and scholarly influence",
                color_map=color_map,
            )
        with col2:
            re_data = times_filtered_tab[["Year", "IPEDS_Name", "Research_Environment"]]
            plot_chart_sorted(
                df=re_data,
                metric_col="Research_Environment",
                title_label="🏛️ Research Environment (29%)",
                description="Research funding, reputation, and output volume",
                color_map=color_map,
            )

    elif section == "🌍 Global Engagement & Gender":
        col1, col2 = st.columns(2)
        with col1:
            intl_data = times_filtered_tab[["Year", "IPEDS_Name", "International_Outlook"]]
            plot_chart_sorted(
                df=intl_data,
                metric_col="International_Outlook",
                title_label="🌍 International Outlook (7.5%)",
                description="Global faculty, international students, and collaboration strength",
                color_map=color_map,
            )
        with col2:
            industry_data = times_filtered_tab[["Year", "IPEDS_Name", "Industry"]]
            plot_chart_sorted(
                df=industry_data,
                metric_col="Industry",
                title_label="🏢 Industry Income (4%)",
                description="Ability to attract industry-sponsored research income",
                color_map=color_map,
            )

        gender_data = times_filtered_tab[["Year", "IPEDS_Name", "Male_Ratio", "Female_Ratio"]]
        gender_melted = gender_data.melt(
            id_vars=["Year", "IPEDS_Name"],
            value_vars=["Male_Ratio", "Female_Ratio"],
            var_name="Gender",
            value_name="Percentage"
        )
        gender_melted["Year"] = gender_melted["Year"].astype(str)

        fig = px.bar(
            gender_melted,
            x="Year",
            y="Percentage",
            color="Gender",
            barmode="group",
            facet_col="IPEDS_Name",
            color_discrete_map={
                "Male_Ratio": "#E10600",
                "Female_Ratio": "#1F77B4"
            },
            text="Percentage",
            title="👥 Gender Distribution"
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1] if "=" in a.text else ""))
        fig.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=10)
        fig.update_layout(
            height=450,
            margin=dict(t=30, b=20, l=30, r=30),
            title_font=dict(size=15, color="#333"),
            title_x=0.0,
            xaxis_title="Year",
            yaxis_title="Percentage",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.35,
                xanchor="center",
                x=0.5,
                font=dict(size=9),
                bgcolor='rgba(0,0,0,0)',
                title_text=None
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
            <div style='text-align:center; font-size:0.85rem; font-weight:bold; color:#555; margin-top:4px; margin-bottom:8px;'>
                Gender distribution across male and female student ratios per year
            </div>
        """, unsafe_allow_html=True)

    # Methodology Link for TIMES Tab
    st.markdown("""
        <div class='methodology-link'>
            📚 <a href='https://www.timeshighereducation.com/world-university-rankings/world-university-rankings-2025-methodology' target='_blank'>TIMES Methodology</a>
        </div>
    """, unsafe_allow_html=True)

with tabs[2]:
    st.markdown("<h2 style='text-align: center; color: #4B4B4B;'>QS Ranking</h2>", unsafe_allow_html=True)

    qs_options = list(dict.fromkeys(all_selected_unis + extra_qs_unis))

    manual_qs_selected_unis = st.session_state.get("manual_qs_selected_unis", [])

    merged_selected_unis = list(set(all_selected_unis + manual_qs_selected_unis))

    current_qs_selected_unis = st.multiselect(
        "🔎 Select universities to compare with NJIT:",
        options=qs_options,
        default=merged_selected_unis,
        key="qs_optional_unis"
    )

    st.session_state["manual_qs_selected_unis"] = [
        uni for uni in current_qs_selected_unis if uni not in all_selected_unis
    ]

    final_qs_unis = [NJIT_NAME] + current_qs_selected_unis

    color_map = create_color_map(final_qs_unis)

    #Filter Data 
    qs_filtered_tab = qs_df[
        (qs_df["Year"].isin(selected_years)) &
        (qs_df["IPEDS_Name"].isin(final_qs_unis))
    ]

    latest_qs_year = max([y for y in selected_years if y in qs_filtered_tab["Year"].unique()], default=None)

    #KPI Metrics 
    kpi_metrics = {
        "QS_Rank": "🏅 QS Rank",
        "Overall_Score": "📊 Overall Score",
        "Academic_Reputation": "🎓 Academic Reputation",
        "Employer_Reputation": "🏢 Employer Reputation",
        "Citations_per_Faculty": "📖 Citations/Faculty",
        "Faculty_Student_Ratio": "👩‍🏫 Faculty-Student Ratio",
        "Employment_Outcomes": "💼 Employment Outcomes",
        "Sustainability_Score": "🌱 Sustainability Score"
    }

    kpi_keys = list(kpi_metrics.keys())
    for i in range(0, len(kpi_keys), 4):
        row = st.columns(4)
        for j in range(4):
            if i + j < len(kpi_keys):
                col_key = kpi_keys[i + j]
                label = kpi_metrics[col_key] + (f" ({latest_qs_year})" if latest_qs_year else "")
                
                kpi_html = f"<h4>{label}</h4>"
                if latest_qs_year:
                    kpi_row = qs_filtered_tab[qs_filtered_tab["Year"] == latest_qs_year]
                    for uni in final_qs_unis:
                        val = get_metric_value(kpi_row, uni, col_key)
                        kpi_html += f"<div class='kpi-value' style='color:{color_map.get(uni)}'>{uni}: {val}</div>"
                
                with row[j]:
                    st.markdown(f"<div class='kpi-box'>{kpi_html}</div>", unsafe_allow_html=True)

    st.divider()

    chart_selection = st.radio(
        "Choose QS Section",
        ["🎓 Research & Learning", "🌍 Global Engagement"],
        horizontal=True, key="qs_section"
    )

    if chart_selection.startswith("🎓"):
        col1, col2 = st.columns(2)
        with col1:
            ar_data = qs_filtered_tab[["Year", "IPEDS_Name", "Academic_Reputation"]]
            plot_chart_sorted(
                df=ar_data,
                metric_col="Academic_Reputation",
                title_label="🎓 Academic Reputation (30%)",
                description="Global survey of academic prestige.",
                color_map=color_map,
            )
        with col2:
            citations_data = qs_filtered_tab[["Year", "IPEDS_Name", "Citations_per_Faculty"]]
            plot_chart_sorted(
                df=citations_data,
                metric_col="Citations_per_Faculty",
                title_label="📖 Citations per Faculty (20%)",
                description="Research strength via faculty citation rates",
                color_map=color_map,
            )

    elif chart_selection.startswith("🌍"):
        col1, col2 = st.columns(2)
        with col1:
            intl_student_data = qs_filtered_tab[["Year", "IPEDS_Name", "International_Student_Ratio"]]
            plot_chart_sorted(
                df=intl_student_data,
                metric_col="International_Student_Ratio",
                title_label="🌎 International Student Ratio (5%)",
                description="Global student diversity at the institution",
                color_map=color_map,
            )
        with col2:
            intl_faculty_data = qs_filtered_tab[["Year", "IPEDS_Name", "International_Faculty_Ratio"]]
            plot_chart_sorted(
                df=intl_faculty_data,
                metric_col="International_Faculty_Ratio",
                title_label="👩‍🏫 International Faculty Ratio (5%)",
                description="International diversity of faculty members",
                color_map=color_map,
            )

    # Methodology Link for QS Tab
    st.markdown("""
        <div class='methodology-link'>
            📚 <a href='https://www.topuniversities.com/world-university-rankings/methodology' target='_blank'>QS Methodology</a>
        </div>
    """, unsafe_allow_html=True)

with tabs[3]:
    st.markdown("<h2 style='text-align: center; color: #4B4B4B;'>USN Ranking</h2>", unsafe_allow_html=True)

    usn_options = list(dict.fromkeys(all_selected_unis + extra_usn_unis))

    manual_usn_selected_unis = st.session_state.get("manual_usn_selected_unis", [])

    merged_selected_unis = list(set(all_selected_unis + manual_usn_selected_unis))

    current_usn_selected_unis = st.multiselect(
        "🔎 Select universities to compare with NJIT:",
        options=usn_options,
        default=merged_selected_unis,
        key="usn_optional_unis"
    )

    st.session_state["manual_usn_selected_unis"] = [
        uni for uni in current_usn_selected_unis if uni not in all_selected_unis
    ]

    final_usn_unis = [NJIT_NAME] + current_usn_selected_unis

    color_map = create_color_map(final_usn_unis)

    usn_filtered_tab = usn_df[
        (usn_df["Year"].isin(selected_years)) &
        (usn_df["IPEDS_Name"].isin(final_usn_unis))
    ]
    
    latest_usn_year = max([y for y in selected_years if y in usn_filtered_tab["Year"].unique()], default=None)
    
    kpi_metrics = {
    "Rank": "🏅 USN_Rank",
    "Peer_assessment_score": "🤝 Peer Assessment",
    "Actual_graduation_rate": "🎓 Graduation Rate", 
    "Average_first_year_retention_rate": "📚 First-Year Retention",
    "Faculty_resources_rank": "🏫 Faculty Resources Rank",
    "Financial_resources_rank": "💰 Financial Resources Rank",
    "Pell_Graduation_Rate": "🎓 Pell Grad Rate",
    "College_grad_income_benefit_(%)": "💼 Income Benefit"
}

    kpi_keys = list(kpi_metrics.keys())
    for i in range(0, len(kpi_keys), 4):
        row = st.columns(4)
        for j in range(4):
            if i + j < len(kpi_keys):
                col_key = kpi_keys[i + j]
                label = kpi_metrics[col_key] + (f" ({latest_usn_year})" if latest_usn_year else "")
                
                kpi_html = f"<h4>{label}</h4>"
                if latest_usn_year:
                    kpi_row = usn_filtered_tab[usn_filtered_tab["Year"] == latest_usn_year]
                    for uni in final_usn_unis:
                        val = get_metric_value(kpi_row, uni, col_key)
                        kpi_html += f"<div class='kpi-value' style='color:{color_map.get(uni)}'>{uni}: {val}</div>"
                
                with row[j]:
                    st.markdown(f"<div class='kpi-box'>{kpi_html}</div>", unsafe_allow_html=True)

    st.divider()

    chart_selection = st.radio(
        "Choose USN Section",
        ["🎓 Student Success", "👩‍🏫 Faculty & Financials", "🎯 Admissions & Selectivity", "🎓 Alumni Outcomes"],
        horizontal=True, key="usn_section"
    )

    if chart_selection == "🎓 Student Success":
        col1, col2 = st.columns(2)
        with col1:
            plot_chart_sorted(
                df=usn_filtered_tab[["Year", "IPEDS_Name", "Graduation_and_retention_rank"]],
                metric_col="Graduation_and_retention_rank",
                title_label="🎯 Graduation & Retention Rank",
                description="Combined ranking on student graduation and retention success.",
                color_map=color_map,
            )
        with col2:
            plot_chart_sorted(
                df=usn_filtered_tab[["Year", "IPEDS_Name", "Pell_Graduation_Rate"]],
                metric_col="Pell_Graduation_Rate",
                title_label="🎓 Pell Graduation Rate",
                description="Graduation rate of low-income Pell Grant students.",
                color_map=color_map,
            )

    elif chart_selection == "👩‍🏫 Faculty & Financials":
        col1, col2 = st.columns(2)
        with col1:
            plot_chart_sorted(
                df=usn_filtered_tab[["Year", "IPEDS_Name", "Percent_of_full-time_faculty"]],
                metric_col="Percent_of_full-time_faculty",
                title_label="👩‍🏫 % Full-Time Faculty",
                description="Ratio of full-time instructional faculty.",
                color_map=color_map,
            )
        with col2:
            plot_chart_sorted(
                df=usn_filtered_tab[["Year", "IPEDS_Name", "Faculty_resources_rank"]],
                metric_col="Faculty_resources_rank",
                title_label="🏛️ Faculty Resources Rank",
                description="Ranking based on class size, salary, and staff ratios.",
                color_map=color_map,
            )

    elif chart_selection == "🎯 Admissions & Selectivity":
        col1, col2 = st.columns(2)
        with col1:
            plot_chart_sorted(
                df=usn_filtered_tab[["Year", "IPEDS_Name", "Top_10%_of_HS_Class"]],
                metric_col="Top_10%_of_HS_Class",
                title_label="📘 Top 10% HS Class",
                description="Percentage of students in top decile of their class.",
                color_map=color_map,
            )
        with col2:
            plot_chart_sorted(
                df=usn_filtered_tab[["Year", "IPEDS_Name", "%_students_submitting_SAT_scores"]],
                metric_col="%_students_submitting_SAT_scores",
                title_label="📝 % Submitted SAT",
                description="SAT submission ratio indicating selectivity.",
                color_map=color_map,
            )

    elif chart_selection == "🎓 Alumni Outcomes":
        plot_chart_sorted(
            df=usn_filtered_tab[["Year", "IPEDS_Name", "Alumni_Giving"]],
            metric_col="Alumni_Giving",
            title_label="🎓 Alumni Giving Rate",
            description="Measures alumni engagement through donations.",
            color_map=color_map,
        )

with tabs[4]:
    st.markdown("<h2 style='text-align: center; color: #4B4B4B;'>Washington Ranking</h2>", unsafe_allow_html=True)

    washington_options = list(dict.fromkeys(all_selected_unis + extra_washington_unis))

    manual_washington_selected_unis = st.session_state.get("manual_washington_selected_unis", [])

    merged_selected_unis = list(set(all_selected_unis + manual_washington_selected_unis))

    current_washington_selected_unis = st.multiselect(
        "🔎 Select universities to compare with NJIT:",
        options=washington_options,
        default=merged_selected_unis,
        key="washington_optional_unis"
    )

    st.session_state["manual_washington_selected_unis"] = [
        uni for uni in current_washington_selected_unis if uni not in all_selected_unis
    ]

    final_washington_unis = [NJIT_NAME] + current_washington_selected_unis

    color_map = create_color_map(final_washington_unis)

    washington_filtered_tab = washington_df[
        (washington_df["Year"].isin(selected_years)) &
        (washington_df["IPEDS_Name"].isin(final_washington_unis))
    ]

    latest_wash_year = max([y for y in selected_years if y in washington_filtered_tab["Year"].unique()], default=None)
    
    kpi_metrics = {
    "Washington_Rank": "🏅 Washington_Rank",
    "8-year_graduation_rate": "🎓 8-Year_Graduation_Rate",
    "Pell/non-Pell_graduation_gap": "📚 Pell_vs_Non-Pell_Grad_Gap",
    "Affordability_rank": "💸 Affordability_Rank",  
    "Earnings_after_9_years": "💼 Earnings_after_9_years",
    "Service-oriented_majors_%": "🔬 Service-Oriented_Majors_%",  
    "Work-study_service_%": "🎓 Work-Study_Service %",  
    "Net_price_rank": "🏆 Net_Price_Rank"
    }

    kpi_keys = list(kpi_metrics.keys())
    for i in range(0, len(kpi_keys), 4):
        row = st.columns(4)
        for j in range(4):
            if i + j < len(kpi_keys):
                col_key = kpi_keys[i + j]
                label = kpi_metrics[col_key] + (f" ({latest_wash_year})" if latest_wash_year else "")
                
                kpi_html = f"<h4>{label}</h4>"
                if latest_wash_year:
                    kpi_row = washington_filtered_tab[washington_filtered_tab["Year"] == latest_wash_year]
                    for uni in final_washington_unis:
                        val = get_metric_value(kpi_row, uni, col_key)
                        kpi_html += f"<div class='kpi-value' style='color:{color_map.get(uni)}'>{uni}: {val}</div>"
                
                with row[j]:
                    st.markdown(f"<div class='kpi-box'>{kpi_html}</div>", unsafe_allow_html=True)
    
    st.divider()

    chart_selection = st.radio(
        "Choose Washington Monthly Section",
        ["📊 Social Mobility", "🔬 Research", "🤝 Service"],
        horizontal=True, key="washington_section"
    )

    if chart_selection == "📊 Social Mobility":
        col1, col2 = st.columns(2)
        with col1:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "8-year_graduation_rate"]],
                metric_col="8-year_graduation_rate",
                title_label="🎓 8-Year Graduation Rate",
                description="Percentage of students graduating within 8 years",
                color_map=color_map,
            )
        with col2:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "Pell/non-Pell_graduation_gap"]],
                metric_col="Pell/non-Pell_graduation_gap",
                title_label="📚 Pell vs Non-Pell Grad Gap",
                description="Gap in graduation rates between Pell and non-Pell students",
                color_map=color_map,
            )
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "Actual_vs._predicted_Pell_enrollment"]],
                metric_col="Actual_vs._predicted_Pell_enrollment",
                title_label="📈 Pell Enrollment Performance",
                description="Difference between actual and predicted Pell student enrollment",
                color_map=color_map,
            )
        with col2:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "Net_price_of_attendance_for_families_below_$75,000_income"]],
                metric_col="Net_price_of_attendance_for_families_below_$75,000_income",
                title_label="💸 Net Price for <$75k Income",
                description="Average net price for low-income families",
                color_map=color_map,
            )

    elif chart_selection == "🔬 Research":
        col1, col2 = st.columns(2)
        with col1:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "Research_expenditures_(M)"]],
                metric_col="Research_expenditures_(M)",
                title_label="🔬 Research Expenditures (M$)",
                description="Total institutional research spending in millions",
                color_map=color_map,
            )
        with col2:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "Science_&_engineering_PhDs_awarded"]],
                metric_col="Science_&_engineering_PhDs_awarded",
                title_label="🎓 S&E PhDs Awarded",
                description="Number of science and engineering PhDs awarded",
                color_map=color_map,
            )
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "Bachelor's_to_PhD_rank"]],
                metric_col="Bachelor's_to_PhD_rank",
                title_label="🎓 Alumni Earning PhDs",
                description="Rank of undergraduate alumni earning PhDs relative to size",
                color_map=color_map,
            )
        with col2:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "Faculty_receiving_significant_awards"]],
                metric_col="Faculty_receiving_significant_awards",
                title_label="🏆 Faculty Awards",
                description="Number of faculty receiving prestigious awards",
                color_map=color_map,
            )

    elif chart_selection == "🤝 Service":
        col1, col2 = st.columns(2)
        with col1:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "Work-study_service_%"]],
                metric_col="Work-study_service_%",
                title_label="🧰 Fed Work-Study for Service",
                description="Percentage of work-study funds spent on service",
                color_map=color_map,
            )
        with col2:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "Service-oriented_majors_%"]],
                metric_col="Service-oriented_majors_%",
                title_label="📘 Service-Oriented Majors",
                description="% of students graduating in service-oriented disciplines",
                color_map=color_map,
            )
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "AmeriCorps/Peace_Corps_rank"]],
                metric_col="AmeriCorps/Peace_Corps_rank",
                title_label="🌍 AmeriCorps/Peace Corps",
                description="Rank of participation in AmeriCorps and Peace Corps programs",
                color_map=color_map,
            )
        with col2:
            plot_chart_sorted(
                df=washington_filtered_tab[["Year", "IPEDS_Name", "ROTC_rank"]],
                metric_col="ROTC_rank",
                title_label="🎖️ ROTC Program",
                description="Rank of ROTC program size relative to enrollment",
                color_map=color_map,
            )

    # Methodology Link for Washington Tab
    st.markdown("""
        <div class='methodology-link'>
            📚 <a href='https://washingtonmonthly.com/2024/08/25/a-note-on-methodology-four-year-colleges-and-universities/' target='_blank'>Washington Monthly Methodology</a>
        </div>
    """, unsafe_allow_html=True)