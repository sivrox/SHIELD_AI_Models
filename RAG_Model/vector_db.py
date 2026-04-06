import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. SET THE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "medical_docs")
DOCS_PATH = PDF_DIR
DB_PATH = "shield_medical_db"

def build_medical_library():
    # A. INITIALIZE THE TRANSLATOR (Embeddings)
    # This model turns human sentences into a list of numbers.
    # 'all-MiniLM-L6-v2' is chosen because it is fast and runs locally for free.
    print("🌍 Initializing the Sentence-to-Math Translator...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # B. LOAD THE KNOWLEDGE (PDFs)
    all_documents = []
    print(f"📚 Scanning {DOCS_PATH} for medical guidelines...")
    
    if not os.path.exists(DOCS_PATH) or not os.listdir(DOCS_PATH):
        print(f"❌ Error: No PDFs found in {DOCS_PATH}. Please add them first!")
        return

    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):
            print(f"📄 Reading: {file}")
            loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
            all_documents.extend(loader.load())

    # C. CHOP INTO SEARCHABLE PIECES (Chunking)
    # We can't search a 600-page book at once. We chop it into 1000-character chunks.
    # overlap=150 ensures that if a fact is cut in half, it appears in both chunks.
    print("✂️  Chipping text into logical paragraphs...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_documents)

    # D. CREATE THE VECTOR DB (The Shelves)
    # This step takes the most time. It 'translates' every chunk into math 
    # and saves it to the hard drive in the 'shield_medical_db' folder.
    print(f"💾 Saving {len(chunks)} chunks to the database at {DB_PATH}...")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("\n✅ Step 1 Complete: The S.H.I.E.L.D. Medical Library is now live!")

if __name__ == "__main__":
    build_medical_library()