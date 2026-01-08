import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from langsmith import traceable
import os
import shutil

# --- Load environment variables ---
load_dotenv()

# --- Delete old vector store if it exists ---
vector_store_path = "historical_figures_vector_store_test_v2"
if os.path.exists(vector_store_path):
    shutil.rmtree(vector_store_path)
    print(f"Deleted old vector store: {vector_store_path}")

# --- PDF Ingestion ---
loader = PyPDFLoader("Notable_Historical_Figures.pdf")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = splitter.split_documents(documents)

# --- Vector Store ---
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="historical_figures_test_v2",
    embedding_function=embedding,
    persist_directory=vector_store_path
)
vectorstore.add_documents(docs)

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Prompt Template for Vector Store QA ---
prompt_template = PromptTemplate.from_template(
    """You are HistoryBot, an expert on historical figures.
Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question: {question}
Answer:"""
)

# --- RetrievalQA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --- Web Search Tool Setup ---
try:
    web_search = TavilySearchResults(max_results=3)
    web_search_available = True
except Exception as e:
    print(f"Warning: Tavily search not available: {e}")
    web_search_available = False

# --- Intelligent Search Function ---
def intelligent_search(query: str) -> str:
    """Search vector store first, then web if needed"""
    # First, try vector store
    try:
        result = qa_chain({"query": query})
        answer = result["result"]
        
        # Check if the answer is meaningful
        if "don't know" not in answer.lower() and len(answer.strip()) > 30:
            return f"📚 **From Knowledge Base:**\n\n{answer}"
    except Exception as e:
        print(f"Vector store error: {e}")
    
    # If vector store didn't have good answer, try web search
    if web_search_available:
        try:
            search_results = web_search.invoke(query)
            if search_results:
                # Format web search results
                web_answer = "🌐 **From Web Search:**\n\n"
                for i, result in enumerate(search_results, 1):
                    content = result.get('content', '')
                    url = result.get('url', '')
                    if content:
                        web_answer += f"{i}. {content}\n"
                        if url:
                            web_answer += f"   Source: {url}\n\n"
                
                # Use LLM to synthesize the web results
                synthesis_prompt = f"""Based on these web search results, provide a clear and concise answer to: {query}

Web search results:
{web_answer}

Synthesized answer:"""
                
                synthesized = llm.invoke(synthesis_prompt)
                return f"🌐 **From Web Search:**\n\n{synthesized.content}\n\n---\n{web_answer}"
        except Exception as e:
            print(f"Web search error: {e}")
            return "❌ I couldn't find information in my knowledge base, and web search is not available. Please check if TAVILY_API_KEY is set in your .env file."
    
    return "❌ I don't have information about this in my knowledge base. Web search is not configured."

# --- Memory Setup ---
chat_histories = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

# --- Chat Interface Function ---
@traceable(name="HistoryBot Chat")
def chat_interface(message, history, session_id):
    """Handle chat messages and return updated history"""
    try:
        # Save message to memory
        chat_history = get_history(session_id)
        chat_history.add_user_message(message)

        # Get intelligent search response
        answer = intelligent_search(message)

        # Save response to memory
        chat_history.add_ai_message(answer)

        # Update chat history for display
        history.append((message, answer))
        return history, ""
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append((message, error_msg))
        return history, ""

def clear_chat(session_id):
    """Clear chat history"""
    chat_histories.pop(session_id, None)
    return [], ""

# --- Custom CSS ---
custom_css = """
#header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 20px;
    color: white;
}

#header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: bold;
}

#header p {
    margin: 10px 0 0 0;
    font-size: 1.2em;
    opacity: 0.9;
}

#chatbot-container {
    border: 2px solid #667eea;
    border-radius: 10px;
    padding: 10px;
    background: #f8f9fa;
}

.message-wrap {
    padding: 10px !important;
}

#examples {
    margin-top: 20px;
}

footer {
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    color: #666;
}
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    # Header
    with gr.Row(elem_id="header"):
        gr.Markdown(
            """
            # 📚 HistoryBot
            ### Your AI Expert on Historical Figures
            Ask me anything about notable people from history! I'll search my knowledge base first, then the web if needed.
            """
        )
    
    # Session ID (hidden state)
    session_id = gr.State(value="default")
    
    # Main Chat Interface
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                value=[],
                elem_id="chatbot-container",
                height=500,
                show_label=False,
                avatar_images=(None, "🤖"),
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your question about historical figures here...",
                    show_label=False,
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button("Send 📤", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", scale=1, variant="secondary")
                
    # Example Questions
    with gr.Row(elem_id="examples"):
        gr.Examples(
            examples=[
                "Who was Leonardo da Vinci?",
                "Tell me about Cleopatra's reign",
                "What were Albert Einstein's major contributions?",
                "When did William Shakespeare live?",
                "What was Marie Curie known for?"
            ],
            inputs=msg,
            label="💡 Example Questions"
        )
    
    # Footer
    gr.Markdown(
        """
        ---
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Powered by OpenAI GPT-4 + Tavily Web Search | Built with LangChain & Gradio</p>
            <p style='font-size: 0.9em; margin-top: 10px;'>💡 I search my knowledge base first, then the web for additional information</p>
        </div>
        """,
        elem_id="footer"
    )
    
    # Event Handlers
    submit_btn.click(
        fn=chat_interface,
        inputs=[msg, chatbot, session_id],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        fn=chat_interface,
        inputs=[msg, chatbot, session_id],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[session_id],
        outputs=[chatbot, msg]
    )

# --- Run App ---
if __name__ == "__main__":
    demo.launch(share=False)