import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


def get_agri_agent():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)

    # --- TOOL 1: Document Search ---
    def document_lookup_fn(query: str) -> str:
        docs = db.similarity_search(query, k=5)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join([d.page_content for d in docs])

    # --- TOOL 2: Extract User Info ---
    def get_user_profile_fn(query: str = "") -> str:
        docs = db.similarity_search("Aadhaar name address state location", k=5)

        if not docs:
            return "No user info found."

        return "\n".join([d.page_content for d in docs])

    # --- TOOL 3: Tavily (context-aware) ---
    tavily = TavilySearchResults(k=3)

    def smart_web_search_fn(query: str) -> str:
        docs = db.similarity_search("state address location", k=3)
        context = " ".join([d.page_content for d in docs])

        state = "India"
        if "Gujarat" in context:
            state = "Gujarat"
        elif "Tamil" in context:
            state = "Tamil Nadu"
        elif "Telugu" in context:
            state = "Telangana"
        elif "Bengali" in context:
            state = "West Bengal"
        elif "Malayalam" in context:
            state = "Kerala"
        elif "Marathi" in context:
            state = "Maharashtra"

        enhanced_query = f"{query} schemes for farmers in {state} India"
        return tavily.run(enhanced_query)

    # --- CONVERT TO TOOLS (THIS WAS BROKEN BEFORE) ---
    document_lookup = Tool.from_function(
        func=document_lookup_fn,
        name="document_lookup",
        description="Search user uploaded documents like Aadhaar or land records"
    )

    get_user_profile = Tool.from_function(
        func=get_user_profile_fn,
        name="get_user_profile",
        description="Extract user personal details like name, state, address"
    )

    smart_web_search = Tool.from_function(
        func=smart_web_search_fn,
        name="smart_web_search",
        description="Search latest government schemes based on user's state"
    )

    # --- Tools list ---
    tools = [document_lookup, get_user_profile, smart_web_search]

    # --- LLM ---
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # --- SYSTEM PROMPT ---
    system_prompt = """
You are an intelligent agricultural assistant helping Indian farmers.

MANDATORY STEPS:
1. ALWAYS call get_user_profile first
2. Identify user's STATE
3. Then:
   - Use document_lookup for personal data
   - Use smart_web_search for schemes

RULES:
- Always prioritize schemes from user's state (e.g., Gujarat)
- Then include central schemes

OUTPUT:
- Give scheme names
- Explain eligibility clearly
- Keep it practical

LANGUAGE:
- Respond in user's language
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent