### IMPORTS ###
import os

from dotenv import load_dotenv, find_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv(find_dotenv(), override=True)

EMBEDDING_MODEL = "text-embedding-3-small"

def load_pdf(file_path:str) -> list:
    """ Load a PDF and return a list of documents.
        Args:
            file_path: path to the PDF file
        
        Returns:
            list of document objects (each page of the pdf in a separate document)
    """

    # Get the file path and loader
    loader = PyPDFLoader(file_path= file_path)
    docs = loader.load()

    # Add the filename to the metadata
    filename = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["title"] = filename

    return docs


def chunk_documents(documents: list,
                     chunk_size: int = 1000,
                    chunk_overlap: int = 200) -> list:
    """Split documents into smaller chunks
    
        Args:
            documents: list of the documents objecto from the PDF loader
            chunk_size: size of each chunk in characters
            chunck_overlap: how much overlap between chunks
        
        Returns:
            list of smaller document chunks 
    """
    # Initiate text splitters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    
    return text_splitter.split_documents(documents)


def create_vector_store(chunks: list, 
                        persist_directory: str = "./geology_kb")-> Chroma:
    """ Create a Chroma vector store from document chunks.

    Args:
        chunks: lis of chunk documents
        persist_directory: where to save the vector database
    
    Returns:
        chroma vector store object
    """
    # Create embeddings with OpenAI
    embeddings = OpenAIEmbeddings(model = EMBEDDING_MODEL )

    # Create the chroma
    chroma= Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_directory)
    
    return chroma


def load_vector_store(persist_directory: str = "./geology_kb") -> Chroma:
    """Load an existing Chroma vector store.
        Args:
            persist_directory: whre the vector database is saved
        Returns:
            Chroma vector store object
    """
    # Create embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Load the documents
    vector_store = Chroma(
        embedding_function = embeddings,
        persist_directory = persist_directory)
    
    return vector_store


# ONLY RUNS WHEN THE FILE IS RUN
if __name__ == "__main__":
    # Ensure the source directory exists
    source_dir = "source_documents"
    test_file = os.path.join(source_dir, "test_doc.pdf")
    
    if not os.path.exists(test_file):
        print(f"❌ Error: Could not find {test_file}. Please check the path.")
    else:
        print("=== 🛠️ Building Knowledge Base ===")
        docs = load_pdf(test_file)
        chunks = chunk_documents(docs)
        
        # Build the store
        vectorstore = create_vector_store(chunks)
        print(f"✓ Successfully processed '{test_file}' into {len(chunks)} chunks.")
        
        # Quick verify
        print("\n=== 🧪 Testing Search Capability ===")
        query = "What is the primary topic?"
        results = vectorstore.similarity_search(query, k=1)
        if results:
            print(f"Top Result: {results[0].page_content[:100]}...")
            print(f"Metadata: {results[0].metadata}")
