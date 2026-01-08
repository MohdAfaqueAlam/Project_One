import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="HistoryBot - AI Historical Expert",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 3em;
        margin: 0;
        font-weight: bold;
    }
    
    .main-header p {
        font-size: 1.2em;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    /* Chat message text color */
    .stChatMessage p {
        color: #333333 !important;
    }
    
    /* Avatar styling */
    .stChatMessage .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Example questions styling */
    .example-question {
        background-color: white;
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .example-question:hover {
        background-color: #667eea;
        color: white;
        transform: translateX(5px);
    }
    </style>
""", unsafe_allow_html=True)

