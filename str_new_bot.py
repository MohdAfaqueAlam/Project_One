import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

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

# --- Load environment variables ---
load_dotenv()

# --- Initialize Session State ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.chat_histories = {}
    st.session_state.messages = []

# --- Initialization Function ---
@st.cache_resource
def initialize_system():
    """Initialize the vector store and QA chain"""
    try:
        # PDF Ingestion
        loader = PyPDFLoader("Notable_Historical_Figures.pdf")
        documents = loader.load()

        # Document Splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        docs = splitter.split_documents(documents)

        # Embeddings
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # Vector Store
        vector_store_path = "historical_figures_vector_store_v3"
        
        # Check if vector store exists
        if not os.path.exists(vector_store_path):
            vectorstore = Chroma(
                collection_name="historical_figures_v3",
                embedding_function=embedding_model,
                persist_directory=vector_store_path
            )
            vectorstore.add_documents(docs)
        else:
            vectorstore = Chroma(
                collection_name="historical_figures_v3",
                embedding_function=embedding_model,
                persist_directory=vector_store_path
            )

        # LLM Setup
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Prompt Template
        prompt_template = PromptTemplate.from_template(
            """You are HistoryBot, an expert on historical figures.
Use the following context to answer the question.
if you don't know the answer, say you don't know.
Context:
{context}
Question: {question}
Answer:"""
        )

        # RetrievalQA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )

        return vectorstore, qa_chain, True
    except Exception as e:
        return None, None, str(e)

# --- Memory Functions ---
def get_chat_history(session_id: str):
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = InMemoryChatMessageHistory()
    return st.session_state.chat_histories[session_id]

def chat_interface(query, session_id):
    """Process chat query and return response"""
    try:
        chat_history = get_chat_history(session_id)
        result = st.session_state.qa_chain.run(query)  # Changed from input= to just query
        chat_history.add_user_message(query)
        chat_history.add_ai_message(result)
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Main App ---
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>📚 HistoryBot</h1>
            <p>Your AI Expert on Historical Figures</p>
            <p style="font-size: 1em; margin-top: 0.5rem;">Ask me anything about notable people from history!</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/book.png", width=80)
        st.title("⚙️ Settings")
        
        # Session ID
        session_id = st.text_input("Session ID", value="default", help="Use different IDs for separate conversations")
        
        st.divider()
        
        # System Status
        st.subheader("📊 System Status")
        
        if not st.session_state.initialized:
            with st.spinner("🔄 Initializing HistoryBot..."):
                vectorstore, qa_chain, status = initialize_system()
                
                if status == True:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_chain = qa_chain
                    st.session_state.initialized = True
                    st.success("✅ System Ready!")
                else:
                    st.error(f"❌ Initialization Error: {status}")
                    return
        else:
            st.success("✅ System Ready!")
        
        st.divider()
        
        # Statistics
        st.subheader("📈 Statistics")
        st.metric("Total Messages", len(st.session_state.messages))
        st.metric("Active Sessions", len(st.session_state.chat_histories))
        
        st.divider()
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if session_id in st.session_state.chat_histories:
                del st.session_state.chat_histories[session_id]
            st.rerun()
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About")
        st.info("""
        **HistoryBot** uses advanced AI to answer your questions about historical figures.
        
        **Powered by:**
        - OpenAI GPT-4
        - LangChain
        - ChromaDB
        """)

    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat Container
        st.subheader("💬 Chat")
        
        # Display chat messages
        chat_container = st.container(height=500)
        with chat_container:
            if len(st.session_state.messages) == 0:
                st.info("👋 Hello! I'm HistoryBot. Ask me anything about historical figures!")
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat Input
        if prompt := st.chat_input("Type your question here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get bot response
            with st.spinner("🤔 Thinking..."):
                response = chat_interface(prompt, session_id)
            
            # Add assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update chat
            st.rerun()
    
    with col2:
        # Example Questions
        st.subheader("💡 Example Questions")
        
        examples = [
            "Who was Leonardo da Vinci?",
            "Tell me about Cleopatra's reign",
            "What were Albert Einstein's major contributions?",
            "When did William Shakespeare live?",
            "What was Marie Curie known for?"
        ]
        
        for example in examples:
            if st.button(example, key=example, use_container_width=True):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": example})
                
                # Get bot response
                with st.spinner("🤔 Thinking..."):
                    response = chat_interface(example, session_id)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to update chat
                st.rerun()
        
        st.divider()
        
        # Tips
        st.subheader("💭 Tips")
        st.markdown("""
        <div class="info-box">
        <strong>For best results:</strong>
        <ul>
            <li>Ask specific questions</li>
            <li>Use clear language</li>
            <li>One topic at a time</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>Powered by OpenAI GPT-4 | Built with LangChain & Streamlit</strong></p>
            <p style='font-size: 0.9em; margin-top: 10px;'>📚 Built for exploring historical knowledge</p>
        </div>
    """, unsafe_allow_html=True)

# --- Run App ---
if __name__ == "__main__":
    main()