"""Prepare similarity search and keyword search retrivers and merge their results with
RRF(Reciprocal Rank Fusion)."""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

loader = PyPDFLoader("data/ITOps_Knowledge_Base_Troubleshooting_Guides.pdf")
documents = loader.load()

# print(documents)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", google_api_key=os.getenv("GOOGLE_GENAI_API_KEY")
)
vectorstore = FAISS.from_documents(chunks, embeddings)

similarity_search_retriever = vectorstore.as_retriever()

# ------------------------------------------------------------------------------------

keyword_retriever = BM25Retriever.from_documents(documents)

ensemble_retriever = EnsembleRetriever(
    retrievers=[similarity_search_retriever, keyword_retriever],
    weights=[0.5, 0.5],
    c=60,
)
