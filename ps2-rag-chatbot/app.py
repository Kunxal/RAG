from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

# Load the PDF
loader = PyPDFLoader("example.pdf")
documents = loader.load()

# Split text with more context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
docs = text_splitter.split_documents(documents)

# Embed and store vectors
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="db")

# Load GGUF model
llm = LlamaCpp(
    model_path="./mistral-7b-instruct.gguf",
    n_ctx=2048,
    max_tokens=512,
    temperature=0.7,
    verbose=False
)

# Create RAG chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Start chat
print("\n🤖 Chatbot ready. Ask anything from the PDF.")
print("🔚 Type 'exit' to quit.\n")

while True:
    query = input("You: ").strip()
    if query.lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye!")
        break

    # 👇 Add prompt wrapper to reduce hallucinations
    prompt = f"Answer only using the PDF. Do not guess. Question: {query}"
    
    try:
        answer = qa.run(prompt)
        print("🤖 Answer:", answer, "\n")
    except Exception as e:
        print("⚠️ Error:", e)
