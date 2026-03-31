import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI


# --- LLM FOR EXTRACTION ---
def extract_user_info(text):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro")

        prompt = f"""
        Extract the following details from the document:
        - Name
        - State
        - Address

        If not found, return null.

        Document:
        {text}

        Return JSON only.
        """

        response = llm.invoke(prompt)

        return response.content
    except Exception as e:
        print("Extraction error:", e)
        return None


# --- MAIN BUILD FUNCTION ---
def build_vector_db():
    loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        print("⚠️ No documents found!")
        return None

    print(f"Loaded {len(docs)} documents")

    # 🔍 Extract full text for debugging
    full_text = "\n".join([doc.page_content for doc in docs])
    print("DOCUMENT TEXT SAMPLE:", full_text[:500])

    # 🔥 Extract structured info
    user_info = extract_user_info(full_text)
    print("EXTRACTED USER INFO:", user_info)

    # --- Split ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    # --- Attach metadata ---
    for chunk in chunks:
        chunk.metadata["source"] = "user_document"
        if user_info:
            chunk.metadata["user_info"] = user_info

    # --- Embeddings ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- Store ---
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vector_db"
    )

    print("✅ Vector DB created successfully")
    return vector_db


# --- PROCESS FUNCTION ---
def process_and_add_to_db():
    loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        print("⚠️ No documents loaded")
        return None

    # 🔍 Debug text
    full_text = "\n".join([doc.page_content for doc in docs])
    print("DOCUMENT TEXT SAMPLE:", full_text[:500])

    # 🔥 Extract structured info
    user_info = extract_user_info(full_text)
    print("EXTRACTED USER INFO:", user_info)

    # --- Split ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    # --- Attach metadata ---
    for chunk in chunks:
        chunk.metadata["source"] = "user_document"
        if user_info:
            chunk.metadata["user_info"] = user_info

    # --- Embeddings ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- Store ---
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vector_db"
    )

    print("✅ Documents processed and added to DB")
    return vector_db