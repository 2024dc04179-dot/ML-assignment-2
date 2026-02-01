"""CSS styles for the Streamlit app"""

SIDEBAR_CSS = """
<style>
    /* FORCE SIDEBAR TO ALWAYS BE VISIBLE - CANNOT COLLAPSE */
    section[data-testid="stSidebar"] {
        min-width: 280px !important;
        width: 280px !important;
        transform: none !important;
        position: relative !important;
        visibility: visible !important;
    }
    
    /* Hide the collapse button entirely */
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[kind="header"],
    button[data-testid="baseButton-headerNoPadding"],
    .stSidebar > div:first-child > button,
    div[data-testid="stSidebarCollapsedControl"] > button,
    section[data-testid="stSidebar"] button[kind="header"],
    header button[data-testid="stSidebarCollapseButton"],
    button[aria-label*="Close"],
    button[aria-label*="close"],
    button[aria-label*="Collapse"],
    button[aria-label*="collapse"],
    button[data-baseweb="button"][aria-label*="Close"],
    button[data-baseweb="button"][aria-label*="close"],
    button[data-baseweb="button"][aria-label*="Collapse"],
    button[data-baseweb="button"][aria-label*="collapse"],
    header > div > button,
    header button,
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] > div > button,
    button:has(svg[viewBox*="0 0 24 24"]),
    button svg[viewBox*="0 0 24 24"],
    section[data-testid="stSidebar"] > div:first-child > button:first-child,
    section[data-testid="stSidebar"] > div:first-child > button:last-child {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
        position: absolute !important;
        left: -9999px !important;
        z-index: -9999 !important;
    }
    
    /* Hide any SVG icons in collapse buttons */
    button[data-testid="stSidebarCollapseButton"] svg,
    button[kind="header"] svg,
    header button svg {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }
    
    /* Ensure sidebar content is always visible */
    section[data-testid="stSidebar"] > div {
        width: 100% !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
</style>
"""

MAIN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
    
    /* ===== GLOBAL ===== */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1200px;
    }
    
    /* ===== HEADER ===== */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.35);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
    }
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0 0 0.5rem 0;
        position: relative;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.05rem;
        font-weight: 500;
        position: relative;
    }
    
    /* ===== SIDEBAR - DARK PROFESSIONAL ===== */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #181825 100%) !important;
    }
    
    /* ALL sidebar text white */
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] div {
        color: #cdd6f4 !important;
    }
    
    section[data-testid="stSidebar"] h2 {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #6366f1;
        margin-bottom: 1rem !important;
    }
    
    section[data-testid="stSidebar"] h3 {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #a6adc8 !important;
        margin: 1.5rem 0 0.75rem 0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: #313244 !important;
        margin: 1.25rem 0 !important;
    }
    
    /* Sidebar Radio Buttons */
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        color: #cdd6f4 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        background: #313244 !important;
        padding: 0.85rem 1rem !important;
        border-radius: 10px !important;
        margin: 5px 0 !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #45475a !important;
        border-color: #6366f1 !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border-color: transparent !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Sidebar Select Box */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        color: #cdd6f4 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background: #313244 !important;
        border: 2px solid #45475a !important;
        border-radius: 10px !important;
        transition: border-color 0.2s !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div:hover {
        border-color: #6366f1 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div > div {
        color: #ffffff !important;
    }
    
    /* Sidebar File Uploader */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #313244 !important;
        border: 2px dashed #6366f1 !important;
        border-radius: 14px !important;
        padding: 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        background: #3b3d54 !important;
        border-color: #8b5cf6 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        color: #bac2de !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.7rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Sidebar Download Button - Compact */
    section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.4rem 0.8rem !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 2px 6px rgba(99,102,241,0.25) !important;
        transition: all 0.15s ease !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%) !important;
        box-shadow: 0 3px 10px rgba(99,102,241,0.35) !important;
    }
    
    /* ===== CARDS ===== */
    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.05);
        border-left: 5px solid #6366f1;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .info-card h3 {
        color: #1e293b !important;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-card p {
        color: #475569 !important;
        line-height: 1.7;
        margin: 0;
        font-size: 0.95rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1rem 0;
        border-left: 5px solid #10b981;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    .success-card h4 {
        color: #065f46 !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .success-card p {
        color: #047857 !important;
        font-size: 0.95rem;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
        border: 1px solid rgba(99, 102, 241, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .metric-icon { font-size: 1.75rem; margin-bottom: 0.5rem; }
    .metric-value { font-size: 1.8rem; font-weight: 800; font-family: 'Fira Code', monospace; background: linear-gradient(135deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .metric-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-top: 0.3rem; }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b !important;
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid transparent;
        background: linear-gradient(90deg, #6366f1, #8b5cf6) padding-box, linear-gradient(90deg, #6366f1, #8b5cf6) border-box;
        border-image: linear-gradient(90deg, #6366f1, #8b5cf6) 1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* ===== MODEL BADGES ===== */
    .model-badge {
        display: inline-block;
        padding: 0.6rem 1.25rem;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: #ffffff !important;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
        transition: all 0.25s ease;
    }
    
    .model-badge:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45);
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        padding: 8px;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
        color: #6366f1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
    }
    
    /* ===== STREAMLIT METRICS ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    [data-testid="stMetric"] label { 
        color: #64748b !important; 
        font-weight: 600; 
        text-transform: uppercase; 
        font-size: 0.7rem; 
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] { 
        color: #1e293b !important; 
        font-family: 'Fira Code', monospace; 
        font-weight: 700; 
    }
    
    /* ===== DATAFRAME ===== */
    [data-testid="stDataFrame"] { 
        background: #ffffff; 
        border-radius: 14px; 
        overflow: hidden; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    /* ===== EXPANDER ===== */
    [data-testid="stExpander"] { 
        background: #ffffff; 
        border: 1px solid #e2e8f0; 
        border-radius: 14px;
        overflow: hidden;
        transition: all 0.2s ease;
    }
    
    [data-testid="stExpander"]:hover {
        border-color: #6366f1;
    }
    
    [data-testid="stExpander"] summary { 
        color: #1e293b; 
        font-weight: 600; 
    }
    
    /* ===== MARKDOWN ===== */
    .main .stMarkdown h1 { color: #0f172a !important; font-weight: 800; font-size: 1.75rem; }
    .main .stMarkdown h2 { color: #1e293b !important; font-weight: 700; font-size: 1.4rem; }
    .main .stMarkdown h3 { color: #334155 !important; font-weight: 700; font-size: 1.15rem; }
    .main .stMarkdown p, .main .stMarkdown li { color: #475569 !important; line-height: 1.7; font-size: 0.95rem; }
    .main .stMarkdown strong { color: #1e293b !important; font-weight: 600; }
    .main .stMarkdown code { 
        background: linear-gradient(135deg, #f1f5f9, #e2e8f0); 
        color: #6366f1; 
        padding: 0.2rem 0.5rem; 
        border-radius: 6px; 
        font-family: 'Fira Code', monospace; 
        font-size: 0.85em;
        border: 1px solid #e2e8f0;
    }
    
    /* ===== ALERTS ===== */
    .stSuccess { 
        background: linear-gradient(135deg, #ecfdf5, #d1fae5) !important; 
        border: 1px solid #a7f3d0 !important; 
        border-radius: 12px;
        border-left: 4px solid #10b981 !important;
    }
    .stError { 
        background: linear-gradient(135deg, #fef2f2, #fee2e2) !important; 
        border: 1px solid #fecaca !important; 
        border-radius: 12px;
        border-left: 4px solid #ef4444 !important;
    }
    .stInfo { 
        background: linear-gradient(135deg, #eff6ff, #dbeafe) !important; 
        border: 1px solid #bfdbfe !important; 
        border-radius: 12px;
        border-left: 4px solid #3b82f6 !important;
    }
    
    /* ===== HIDE BRANDING ===== */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #94a3b8, #64748b); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #6366f1, #8b5cf6); }
    
    /* ===== ANIMATIONS ===== */
    .animate-fade { animation: fadeIn 0.6s ease-out; }
    @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(15px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    
    /* ===== PULSE GLOW ===== */
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 5px rgba(99, 102, 241, 0.3); }
        50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.5); }
    }
</style>
"""


