import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent  # Modern v1.x agent factory
from langchain_core.tools import tool      # Modern tool decorator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_tavily import TavilySearchResults

load_dotenv()


def get_agri_agent():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)

    # TOOL 1: Local Document Search (For your Aadhaar)
    @tool
    def document_lookup(query: str) -> str:
        """Searches the uploaded personal documents like Aadhaar cards or land records."""
        docs = db.similarity_search(query, k=2)
        return "\n\n".join([d.page_content for d in docs])

    # TOOL 2: Web Search (For real-time Government Schemes)
    # You need TAVILY_API_KEY in your .env
    web_search = TavilySearchResults(k=3)

    tools = [document_lookup, web_search]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Or "gemini-3-flash-preview" for the newest engine
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    agent = create_agent(
        model=llm,  # Your Gemini model
        tools=tools,
        system_prompt=(
            "You are a multilingual Agri-Consultant. "
            "If the user asks in English, Gujatai, Hindi, Tamil, or Telugu, search for the rules in English (using web_search) "
            "but translate your final eligibility verdict back into the user's native language. "
            "Maintain a helpful and empathetic tone for farmers."
        )
    )
    return agent
