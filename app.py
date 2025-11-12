import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint  # For the API connection
from langchain_huggingface.chat_models import ChatHuggingFace  # For the chat wrapper
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# --- Define Paths ---
DB_PATH = "faiss_v2"                  # Using our smart v2 library
MODEL_NAME = "./agri-bot-model-v2/" # Using our smart v2 model

# --- SAFE API Key Handling ---
# This checks for your .streamlit/secrets.toml file
if "HF_TOKEN" not in st.secrets:
    st.error("You need to set your Hugging Face API token in a .streamlit/secrets.toml file.")
    st.info("1. Create a folder named '.streamlit' in your project root.")
    st.info("2. Inside it, create a file named 'secrets.toml'.")
    st.info("3. Add this line: HF_TOKEN = 'hf_...your_new_token...'")
    st.stop() 

# This sets the token for the API client to use
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]


# --- Load Components (with caching) ---
@st.cache_resource
def load_all():
    """
    Loads all the necessary components for the RAG chain.
    """
    print("Loading all components...")
    
    # --- 1. Load the "Librarian" (Embedding Model) ---
    print(f"Loading local embedding model from: {MODEL_NAME}")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # --- 2. Load the "Library" (Vector Database) ---
    print(f"Loading local vector store from: {DB_PATH}")
    db = FAISS.load_local(
        DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Create the "Retriever"
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # --- 3. Load the "Student" (LLM) ---
    # We are connecting to the Llama 3 8B Instruct model on the HF API
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
    
    print(f"Connecting to Hugging Face Endpoint: {repo_id}")
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        temperature=0.1,  
        max_new_tokens=512
    )
    
    # --- 4. Wrap the LLM in a CHAT MODEL wrapper ---
    # This formats our request as a "conversation"
    chat_model = ChatHuggingFace(llm=llm)

    # --- 5. Create the "Instruction Sheet" (Chat Prompt Template) ---
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based only on the provided context. If the answer is not in the context, say 'I don't know'."),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ])
    
    # --- 6. Create the "Master Machine" (New RAG Chain) ---
    document_chain = create_stuff_documents_chain(chat_model, template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    print("Components loaded successfully.")
    return retrieval_chain

# --- Streamlit UI ---
st.set_page_config(page_title="AI Agronomist's Assistant", layout="wide")
st.title("🌱 AI Agronomist's Assistant (v2 - Custom Trained API)")

# Load the RAG chain
try:
    qa_chain = load_all()
except Exception as e:
    st.error(f"Error loading the model. Error: {e}")
    st.stop()

# Chat Input
question = st.text_input("Ask a question about your crop symptoms:")

if question:
    with st.spinner("Connecting to Hugging Face and finding an answer..."):
        try:
            # 1. Ask the "Master Machine"
            result = qa_chain.invoke({"input": question})
            
            # 2. Display the answer
            st.write("**Answer:**")
            st.write(result["answer"])
            
            # 3. (Optional) Display the sources
            st.write("**Sources:**")
            for doc in result["context"]:
                st.info(f"Source: {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")
                st.write(doc.page_content[:250] + "...")
                
        except Exception as e:
            st.error(f"Error during inference: {e}")
            
            # This prints the full error to your TERMINAL for debugging
            print("\n--- FULL ERROR TRACEBACK ---")
            import traceback
            traceback.print_exc()
            print("----------------------------\n")