"""
AmbedkarGPT - simple CLI Q&A using LangChain, Chroma, HuggingFaceEmbeddings, and Ollama (Mistral).
Make sure 'ollama' and model 'mistral' are installed (ollama pull mistral).
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
import sys


def build_vectorstore(speech_path: str, persist_directory: str = "chroma_db"):
    print("Loading documents...")
    loader = TextLoader(speech_path, encoding="utf-8")
    docs = loader.load()

    print("Splitting into chunks...")
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=40)

    split_docs = splitter.split_documents(docs)

    print("Creating embeddings (this can take 20–40 sec first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building Chroma DB...")
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    print("Done building vectorstore.")
    return vectordb



def create_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    llm = Ollama(model="mistral", verbose=False)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    return qa


def main():
    speech_file = "speech.txt"
    chroma_dir = "chroma_db"

    if not os.path.exists(speech_file):
        print(f"Missing {speech_file}. Please create it and paste the speech text.")
        sys.exit(1)

    print("Building / loading vectorstore (this may take a moment)...")
    vectordb = build_vectorstore(speech_file, persist_directory=chroma_dir)

    print("Creating QA chain using Ollama (Mistral)...")
    qa = create_qa_chain(vectordb)

    print("\nAmbedkarGPT ready. Type questions about the speech. Type 'exit' or 'quit' to stop.\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Bye!")
            break
        try:
            # Debug retrieved docs
            docs = vectordb.as_retriever(search_kwargs={"k": 1}).get_relevant_documents(query)
            print("\nRetrieved Docs:\n", docs, "\n")

            # Get model response
            answer = qa.run(query)
            print("\nAnswer:\n", answer, "\n")

        except Exception as e:
            print("Error when calling the LLM:", str(e))
            if "ollama" in str(e).lower():
                print("Hint: is the 'ollama' binary installed and is model 'mistral' pulled? Try 'ollama pull mistral'.")

if __name__ == "__main__":
    main()
