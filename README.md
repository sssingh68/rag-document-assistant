# 🧠 RAG Document Assistant (Offline AI PDF Summarizer)

An AI-powered **document assistant** that lets you upload PDFs or text files, ask questions, and get instant summaries — all running locally via **FastAPI + Ollama + ChromaDB**.

---

## 🚀 Features

✅ Upload PDF or text files  
✅ Summarize or ask context-based questions  
✅ Fully local (no OpenAI API or internet needed)  
✅ Uses **SentenceTransformer embeddings + ChromaDB vector store**  
✅ Interactive **FastAPI + HTML UI**  
✅ Works seamlessly with **Ollama models (Mistral, Llama2, etc.)**

---

## 🧩 Tech Stack

| Layer | Technology |
|--------|-------------|
| Backend | FastAPI (Python 3.11) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent) |
| Model | Ollama (`mistral:7b` by default) |
| Frontend | HTML, CSS, Vanilla JS |
| Deployment | Local / Docker-ready |

---

## 🛠️ Setup & Run

```bash
# Clone repo
git clone https://github.com/sssingh68/rag-document-assistant.git
cd rag-document-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the offline demo
./start_offline_demo.sh
