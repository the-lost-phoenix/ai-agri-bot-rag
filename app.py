import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from huggingface_hub import hf_hub_download # --- NEW IMPORT ---

# --- DEFINE REPO & PATHS ---
# This is the ONE place your app will get all its files
HF_REPO_ID = "Your-HF-Username-Goes-Here/agri-bot-model-v2" 
# (Remember to change this to your username, e.g., "goldenevil/agri-bot-model-v2")

# We'll create a local folder on the cloud server for the index
LOCAL_DB_PATH = "faiss_deployed" 


# --- SAFE API Key Handling ---
if "HF_TOKEN" not in st.secrets:
    st.error("You need to set your Hugging Face API token in .streamlit/secrets.toml")
    st.stop() 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]


# --- Load Components (with caching) ---
@st.cache_resource
def load_all():
    """
    Loads all the necessary components for the RAG chain.
    """
    print("Loading all components...")
    
    # --- 1. DOWNLOAD THE DATABASE FILES ---
    # This runs *first* to get the index files from the HF Hub
    print(f"Downloading FAISS index from {HF_REPO_ID}...")
    try:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="index.faiss",
            local_dir=LOCAL_DB_PATH,
            token=st.secrets["HF_TOKEN"] # Pass token for auth
        )
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="index.pkl",
            local_dir=LOCAL_DB_PATH,
            token=st.secrets["HF_TOKEN"]
        )
        print("FAISS index downloaded.")
    except Exception as e:
        print(f"Error downloading FAISS index: {e}")
        raise e


    # --- 2. Load the "Librarian" (Embedding Model) ---
    # This will download the model from the *same* HF repo
    print(f"Loading embedding model from {HF_REPO_ID}...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_REPO_ID, # Use the repo ID directly
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # --- 3. Load the "Library" (Vector Database) ---
    # Now this path will work, because we just downloaded the files
    print(f"Loading local vector store from: {LOCAL_DB_PATH}")
    db = FAISS.load_local(
        LOCAL_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # --- 4. Load the "Student" (LLM) ---
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
    print(f"Connecting to Hugging Face LLM Endpoint: {repo_id}")
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        temperature=0.1,  
        max_new_tokens=512
    )
    
    chat_model = ChatHuggingFace(llm=llm)

    # --- 5. Create the "Instruction Sheet" (Prompt Template) ---
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based only on the provided context. If the answer is not in the context, say 'I don't know'."),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ])
    
    # --- 6. Create the "Master Machine" (RAG Chain) ---
    document_chain = create_stuff_documents_chain(chat_model, template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("Components loaded successfully.")
    return retrieval_chain

# --- Streamlit UI ---
st.set_page_config(page_title="AI Agronomist's Assistant", layout="wide")
st.title("🌱 AI Agronomist's Assistant (v2 - Custom Trained & Deployed)")

# Load the RAG chain
try:
    qa_chain = load_all()
except Exception as e:
    st.error(f"Error loading components. This can happen on the first boot. Please try refreshing. Error: {e}")
    st.stop()

# Chat Input
question = st.text_input("Ask a question about your crop symptoms:")

if question:
    with st.spinner("Finding an answer..."):
        try:
            result = qa_chain.invoke({"input": question})
            st.write("**Answer:**")
            st.write(result["answer"])
            st.write("**Sources:**")
            for doc in result["context"]:
                st.info(f"Source: {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")
                st.write(doc.page_content[:250] + "...")
                
        except Exception as e:
            st.error(f"Error during inference: {e}")
            print("\n--- FULL ERROR TRACEBACK ---")
            import traceback
            traceback.print_exc()
            print("----------------------------\n")