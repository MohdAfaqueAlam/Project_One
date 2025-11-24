Overview of the app
- Purpose: Answer questions about historical figures using a PDF as the knowledge base.
- Flow:
Load PDF → Split into chunks → Embed with Ollama → Store in Chroma → Retrieve relevant chunks → Answer with ChatOllama → Gradio UI.
- State: Minimal in-memory chat history per session, no long-term memory persistence.

Prerequisites and installation
1) System and runtime
- Python: 3.10+ recommended.
- OS: Works on Windows 11 (your current device), macOS, Linux.
2) Local LLM and embeddings via Ollama
- Install Ollama:
- Windows: Download and install from the Ollama site, then ensure ollama is in PATH.
3) Python packages:
-gradio 
-python-dotenv 
-langchain 
-langchain-community 
-langchain-core
-langchain-chroma 
-langchain-ollama 
-langchain-text-splitters 
-langsmith
-chromadb 
-pypdf

Prepare data and environment
4) Project structure
- Place files:
- Notable_Historical_Figures.pdf in the project root (same folder as your script).
- .env file in root for optional LangSmith tracing (keys commented in code).
5) Optional: LangSmith tracing
- .env contents (optional):

Configure components in code
6) PDF ingestion and chunking
- Loader: PyPDFLoader("Notable_Historical_Figures.pdf") reads the document.
- Splitter: CharacterTextSplitter(chunk_size=200, chunk_overlap=30) creates small, overlapping chunks for better recall.
7) Embeddings and vector store
- Embeddings: OllamaEmbeddings(model="all-minilm") runs locally via Ollama.
- Chroma store:
- Collection name: "historical_figures_test"
- Persistence: "historical_figures_vector_store_test"
- Action: vectorstore.add_documents(docs) embeds and writes chunks to disk for reuse.

8) LLM and prompt
- LLM: ChatOllama(model="llama3.2:latest")
- Prompt: Constrained answer style with “say you don’t know” to avoid hallucinations.
9) RetrievalQA chain
- Chain type: "stuff" (simple approach that inserts retrieved text directly).
- Retriever: vectorstore.as_retriever()
- Return: Answer text only (return_source_documents=False).
10) Session memory
- Per-session: InMemoryChatMessageHistory stored in a dict keyed by session_id.
- Current UI: Uses a single gr.State(value="default"), so one shared session unless you later vary the state.

