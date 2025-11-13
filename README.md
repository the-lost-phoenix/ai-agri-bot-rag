# 🌱 AI Agronomist's Assistant (v2 - Custom Trained)

**[Link to your deployed Streamlit App will go here]**

This project is an end-to-end RAG (Retrieval-Augmented Generation) application built in 20 days. It serves as an expert assistant that answers complex agricultural questions by reading a knowledge base of technical farming guides.

The key feature of this project is a **custom-trained embedding model** that bridges the "semantic gap" between "farmer language" and "scientific language," resulting in dramatically more accurate answers than a generic model.

---

## Problem: The "Semantic Gap"

Generic RAG systems are a good start, but they fail when faced with specialized domains. In agriculture, a farmer's query and a scientist's answer look completely different:

* **Farmer:** "My corn leaves look rusty."
* **Scientist:** "Symptoms of *Puccinia sorghi* include cinnamon-brown pustules..."

A "naive" RAG system using a generic embedding model (like `BAAI/bge-base-en-v1.5`) **fails** to connect these two. It can't find the right documents, so it gives an "I don't know" answer.

## Solution: A Custom-Trained "Expert" Model

To solve this, I built a two-version system to prove the value of custom fine-tuning.

1.  **Data Curation:** I manually read 16+ agricultural PDF guides and created a high-quality, 250+ entry dataset of `(query, positive_passage)` pairs. This dataset (my "ground truth") teaches the model this new, specialized agricultural language.
2.  **Model Fine-Tuning:** I used Google Colab and the `sentence-transformers` library to fine-tune the `BAAI/bge-base-en-v1.5` model on my new dataset.
3.  **RAG Pipeline:** I built a full RAG application using **LangChain**, **FAISS** (for the vector store), and a **Llama 3** model from the Hugging Face API.

---

## 🏆 The Results: Before vs. After

The fine-tuned **v2 "Expert" Model** dramatically outperforms the **v1 "Naive" Model** on real-world farmer queries.

### Test 1: The "Rusty Corn" Problem

**Query:** `my corn leaves look rusty`

| v1 (Naive) Model | v2 (Custom "Expert") Model |
| :--- | :--- |
| **Answer:** "I don't know." | **Answer:** "It sounds like you might have... Rust disease... caused by the organism Puccinia sorghi... its symptoms include brown pustules... which could be described as 'rusty' in appearance." |
| ![v1 Fails](httpsIn-your-v1-app-ask-the-question-and-take-a-screenshot-of-the-failure) | ![v2 Succeeds](Upload-your-v2-success-screenshot-to-GitHub-and-paste-the-link-here) |

### Test 2: The "Low Pressure" Problem

**Query:** `My sprayer has low pressure. Will that still work for spraying fungicides?`

| v1 (Naive) Model | v2 (Custom "Expert") Model |
| :--- | :--- |
| **Answer:** "I don't know." or an irrelevant passage. | **Answer:** "...large droplets are formed when pressure is low, which are more likely to run off leaves... and provide less coverage... So, low pressure might not work as well..." |
| (v1 Screenshot) | (Upload-your-v2-success-screenshot-to-GitHub-and-paste-the-link-here) |

---

## 🛠️ Technology Stack

* **RAG & Orchestration:** LangChain
* **LLM:** Llama 3 8B-Instruct (via Hugging Face API)
* **Model/Data Hosting:** Hugging Face Hub
* **Vector Store:** FAISS
* **Embedding Model:** `BAAI/bge-base-en-v1.5`
* **Fine-Tuning:** `sentence-transformers`, Google Colab (T4 GPU)
* **Web App & Deployment:** Streamlit, Streamlit Community Cloud
* **Core Language:** Python

---

## 🚀 How to Run Locally

This app downloads the custom model and pre-built FAISS index from the Hugging Face Hub, so no local training or ingestion is required.

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/the-lost-phoenix/ai-agri-bot-rag.git](https://github.com/the-lost-phoenix/ai-agri-bot-rag.git)
    cd ai-agri-bot-rag
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add your Hugging Face API Token:**
    * Create a folder named `.streamlit`
    * Inside that folder, create a file named `secrets.toml`
    * Add your token (with "write" permissions) to the file:
        ```toml
        HF_TOKEN = "hf_...your_api_token_here..."
        ```

5.  **Run the app!**
    ```bash
    streamlit run app.py
    ```
The app will automatically download the "expert" model and database on its first run.