# DocuLens AI

A LangChain-powered RAG (Retrieval Augmented Generation) app that lets you upload PDF documents and ask questions about them. Built with Streamlit, ChromaDB, HuggingFace embeddings, and Mistral AI.

## Features

- Upload and ingest PDF documents
- Semantic search with MMR retrieval
- Streaming AI answers with source citations
- Abbreviation expansion for better retrieval (BFS, NLP, RAG, etc.)
- PDF upload and ingestion

## Setup (Local)

1. **Clone the repository**

   ```bash
   git clone https://github.com/Hamza07-debug/DocuLens-AI.git
   cd DocuLens-AI
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy `.env.example` to `.env` and add your API keys:

   ```bash
   copy .env.example .env   # Windows
   # cp .env.example .env   # macOS/Linux
   ```

   - `MISTRAL_API_KEY` — required for chat responses ([Mistral AI](https://console.mistral.ai/))
   - `GROQ_API_KEY` — optional, for Groq-based models

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
├── app.py              # Streamlit web app (main entry point)
├── main.py             # CLI RAG interface
├── create_database.py  # Script to ingest a sample PDF into ChromaDB
├── requirements.txt
├── document loaders/   # Sample documents and loader scripts
├── retrievers/         # Retriever experiments
└── vector store/       # Vector store utilities
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo, branch `main`, and main file `app.py`.
4. Under **Advanced settings → Secrets**, add:

   ```toml
   MISTRAL_API_KEY = "your_mistral_api_key"
   GROQ_API_KEY = "your_groq_api_key"
   ```

5. Click **Deploy**. The app will build and go live at a `*.streamlit.app` URL.

> **Note:** On Streamlit Cloud, the ChromaDB folder is ephemeral. Users need to upload and ingest PDFs each session (which the app supports).

## License

MIT
