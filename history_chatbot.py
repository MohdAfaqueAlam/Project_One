# imports
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langsmith import traceable

# --- Load environment variables ---
load_dotenv()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# --- PDF Ingestion ---
loader = PyPDFLoader("Notable_Historical_Figures.pdf")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = splitter.split_documents(documents)

# --- Vector Store ---
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="historical_figures_test",
    embedding_function=embedding,
    persist_directory="historical_figures_vector_store_test"
)
vectorstore.add_documents(docs)

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Prompt Template ---
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
    return_source_documents=False
)

# --- Memory Setup ---
chat_histories = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

# --- Gradio UI ---
@traceable(name="HistoryBot Chat")
def chat_interface(query, session_id):
    try:
        # Save message to history
        history = get_history(session_id)
        history.add_user_message(query)

        # Run QA chain directly with string input
        response = qa_chain.run(query)

        # Save response to history
        history.add_ai_message(response)

        return response
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("### 👋 Hello, I am **HistoryBot**, your expert on historical figures. How can I assist you today?")
    session_id = gr.State(value="default")
    chatbot = gr.Textbox(label="Your Question")
    output = gr.Textbox(label="HistoryBot's Answer", lines=10, max_lines=20, scale=2)
    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear History")

    submit_btn.click(fn=chat_interface, inputs=[chatbot, session_id], outputs=output)
    clear_btn.click(fn=lambda: chat_histories.pop(session_id.value, None), inputs=[], outputs=[])

# --- Run App ---
if __name__ == "__main__":
    demo.launch()