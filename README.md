# AmbedkarGPT – Speech Q&A using LangChain, Chroma & Ollama

AmbedkarGPT is a command-line Question & Answer chatbot built using:
- **LangChain**
- **ChromaDB vector database**
- **HuggingFace sentence embeddings**
- **Ollama (Mistral model)**

The application loads a speech text, generates embeddings, stores them in Chroma, and performs semantic search to answer questions from the speech.

---

## Features
- Local embeddings & vector search (no cloud required)
- Works fully **offline** after model is downloaded
- Simple CLI interface for asking questions interactively
- Uses **Mistral** via Ollama for answer generation

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama `mistral` |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| Framework | LangChain |

---

## 📂 Project Structure

AmbedkarGPT-Intern-Task
│── main.py # Main Q&A script
│── speech.txt # Speech text
│── requirements.txt # Dependencies
│── chroma_db/ # Auto-generated vector database (ignored in git)
│── README.md # Documentation
└── venv/ # Virtual environment (ignored in git)


---

## 🛠 Installation & Setup

Clone the repository
git clone https://github.com/yashkhopkar3/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task

Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

Install requirements
pip install -r requirements.txt

Install Ollama and pull Mistral model

curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral

Run the Application
python3 main.py


Example:

AmbedkarGPT ready. Type questions about the speech. Type 'exit' or 'quit' to stop.

Question: Real Remedy?
