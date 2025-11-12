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

1.  **Data Curation:** I manually read 16+ agricultural PDF guides and created a high-quality, 254-entry dataset of `(query, positive_passage)` pairs. This dataset (my "ground truth") teaches the model this new, specialized agricultural language.
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
| ![v1 Fails](URL_to_your_v1_fail_screenshot) | ![v2 Succeeds](URL_to_your_v2_success_screenshot) |

*(**Your job:** You'll need to upload your screenshots to GitHub or an image host to get a URL to put here. You can also just use a text table.)*

### Test 2: The "Low Pressure" Problem

**Query:** `My sprayer has low pressure. Will that still work for spraying fungicides?`

| v1 (Naive) Model | v2 (Custom "Expert") Model |
| :--- | :--- |
| **Answer:** "I don't know." or an irrelevant passage. | **Answer:** "...large droplets are formed when pressure is low, which are more likely to run off leaves... and provide less coverage... So, low pressure might not work as well..." |
| (v1 Screenshot) | (Your Screenshot) |

---

## 🛠️ Technology Stack

* **RAG & Orchestration:** LangChain
* **LLM:** Llama 3 8B-Instruct (via Hugging Face API)
* **Vector Store:** FAISS
* **Embedding Model:** `BAAI/bge-base-en-v1.5`
* **Fine-Tuning:** `sentence-transformers`, Google Colab (T4 GPU)
* **Web App:** Streamlit
* **Core Language:** Python

---

## 🚀 How to Run Locally

1.  Clone this repository.
2.  Create a virtual environment: `python -m venv venv`
3.  Activate it: `source venv/bin/activate`
4.  Install dependencies: `pip install -r requirements.txt`
5.  Create your Hugging Face API key and save it in `.streamlit/secrets.toml`:
    ```toml
    HF_TOKEN = "hf_..."
    ```
6.  (Re-build the v2 index): `python ingest.py`
7.  Run the app: `streamlit run app.py`