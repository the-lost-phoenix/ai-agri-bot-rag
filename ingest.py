import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # --- NEW IMPORT ---
from langchain_community.vectorstores import FAISS         # --- NEW IMPORT ---

# --- Define the path to your data folder ---
DATA_PATH = "data/"
# --- Define the path to where you want to save your "library" (vector store) ---
DB_PATH = "faiss_v2"  # "v1" because this is our first "naive" version

# --- Step 1: Load all PDF documents ---
def load_documents():
    print("Loading documents...")
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="*.pdf", 
        loader_cls=PyMuPDFLoader
    )
    
    documents = loader.load()
    
    if not documents:
        print("No documents found. Did you add your PDFs to the 'data' folder?")
        return None
        
    print(f"Successfully loaded {len(documents)} documents.")
    return documents

# --- Step 2: Split the loaded documents into chunks ---
def split_documents(documents):
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # The maximum size of each chunk
        chunk_overlap=150   # The number of characters to overlap between chunks
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Successfully split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# --- Step 3: Get an embedding model (the "Librarian") ---
def get_embedding_model():
    """
    Loads the "librarian" model from Hugging Face.
    """
    print("Loading embedding model...")
    # This is our "generic" model. It's good, but not an expert.
    # We'll use the "bge-base-en-v1.5" model.
    model_name = "./agri-bot-model-v2/"
    
    # We can use our CPU for this.
    model_kwargs = {'device': 'cpu'}
    
    # 'normalize_embeddings' is a technical step that improves search results.
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded.")
    return embeddings

# --- Step 4: Create and save the "Library" (Vector Store) ---
def create_and_save_vector_db(chunks, embeddings):
    """
    Uses the "librarian" (embeddings) to turn all chunks into vectors
    and stores them in the "library" (FAISS database).
    """
    print("Creating vector database from chunks...")
    
    # This is the magic line. FAISS.from_documents() does all the hard work:
    # 1. It takes all 722 chunks.
    # 2. It uses the 'embeddings' model to turn each one into a vector.
    # 3. It builds the "map" (FAISS index) of all these vectors.
    db = FAISS.from_documents(chunks, embeddings)
    
    # Now, save this "map" to our disk so we can use it later.
    db.save_local(DB_PATH)
    
    print(f"Successfully created and saved vector database to {DB_PATH}")

# --- Main execution ---
if __name__ == "__main__":
    
    # Run Step 1
    documents = load_documents()
    
    if documents:
        # Run Step 2
        chunks = split_documents(documents)
        
        # Run Step 3
        embeddings = get_embedding_model()
        
        # Run Step 4
        create_and_save_vector_db(chunks, embeddings)