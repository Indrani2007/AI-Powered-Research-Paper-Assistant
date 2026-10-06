from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


CHROMA_DIR = "chroma_db"

LLM_MODEL = "llama3.2"

EMBEDDING_MODEL = "nomic-embed-text"


embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


def load_pdfs(pdf_files):

    documents = []

    for pdf_file in pdf_files:

        loader = PyPDFLoader(pdf_file)

        docs = loader.load()

        documents.extend(docs)

    return documents


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


def create_vector_database(chunks):

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_db


def get_vector_database():

    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    return vector_db


def ask_question(question):

    vector_db = get_vector_database()

    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    documents = retriever.invoke(question)

    if not documents:

        return (
            "I could not find relevant information "
            "in the uploaded research papers.",
            []
        )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
You are an AI-powered Research Paper Assistant.

Answer the user's question using ONLY
the information given in the research paper context.

Do not make up information.

If the answer is not available in the context,
say:

"I could not find this information in
the uploaded research papers."

Research Paper Context:
----------------------------

{context}

----------------------------

User Question:

{question}

Give a clear, accurate and simple answer.
"""

    response = llm.invoke(prompt)

    return response.content, documents