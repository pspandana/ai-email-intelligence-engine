# build_vector_db.py

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the documents from the knowledge_base directory
loader = DirectoryLoader('knowledge_base/', glob="**/*.txt", show_progress=True)
documents = loader.load()

# 2. Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. Create the embedding model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the vector database
# This will create a new 'db' directory to store the vectors
db = Chroma.from_documents(texts, embeddings, persist_directory="db")

print("Vector database created successfully.")