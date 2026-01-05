from langchain import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.document_loaders import pyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- PDF Ingestion ---
loader = pyPDFLoader("Notable_Historical_Figures.pdf")
documents = loader.load()

# --- Document Splitting ---

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = splitter.split_documents(documents)

# --- Embeddings ---

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Vector Store ---

vectorstore = Chroma(
    collection_name="historical_figures",
    embedding_function=embedding_model,
    persist_directory="historical_figures_vector_store"
)
# ---- Document Addition ----

vectorstore.add_documents(docs)

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0) 

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template(
    """You are HistoryBot, an expert on historical figures.
Use the following context to answer the question.
if you don't know the answer, say you don't know.
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
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

# --- streamlit UI ---
def chat_interface(query, session_id):
    try:
        chat_history = get_chat_history(session_id)
        result = qa_chain.run(input=query)
        chat_history.add_user_message(query)
        chat_history.add_ai_message(result)
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"
