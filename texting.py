
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

video_id = "Gfr50f6ZBvo"

try:
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)  # this is a FetchedTranscript (iterable of FetchedTranscriptSnippet)

    # Each snippet has .text, .start, .duration
    transcript = " ".join(snippet.text for snippet in fetched_transcript)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.create_documents([transcript])


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(texts, embedding_model)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever.invoke("What is the main topic discussed in the video?")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
query = "Provide a brief summary of the video's content."

