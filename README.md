# ITOps Agent

An AI-powered IT operations assistant built using LangGraph and Retrieval-Augmented Generation (RAG) to understand IT support requests, retrieve relevant troubleshooting guidance, and assist with ticket preparation through human-in-the-loop interactions.

## Overview

ITOps Agent is designed to assist users with common IT support issues by retrieving relevant information from an IT troubleshooting knowledge base.

Users can describe their IT-related problems in natural language. The system classifies the user's intent, retrieves relevant troubleshooting documents using hybrid search, and generates context-grounded responses.

When a user explicitly requests a support ticket, the workflow prepares a structured ticket summary and asks for human confirmation before presenting the ticket confirmation message.

The workflow is orchestrated using LangGraph, allowing the application to maintain state and pause execution for human input.

## Architecture

```text
             User Query
                  |
                  v
          Intent Classification
                  |
                  v
          Document Retrieval
                  |
          Hybrid Search Engine
          /               \
   Semantic Search      BM25 Search
        FAISS                |
          \                 /
           Reciprocal Rank
                Fusion
                  |
                  v
          Context-Grounded
             Response
                  |
                  v
            Ticket Node
                  |
          Human-in-the-Loop
             Interaction
                  |
          Ticket Summary
                  |
          User Confirmation
                  |
                  v
          Confirmation Message
```

## Core Components

### 1. Intent Classification

Uses Google's Gemini model to classify incoming user requests into three categories:

- Information query
- IT support request
- IT support request with explicit ticket request

This classification determines how the workflow handles the user's request.

### 2. Knowledge Base Ingestion

The system loads IT troubleshooting documents in PDF format using PyPDFLoader.

The ingestion pipeline:

- Loads troubleshooting documents.
- Splits documents into chunks using Recursive Character Text Splitter.
- Uses a chunk size of 1,000 characters and an overlap of 200 characters.
- Generates document embeddings using Google's `gemini-embedding-001` model.
- Creates a FAISS vector store for semantic retrieval.

### 3. Hybrid Search and Retrieval

The project combines semantic search and keyword-based search to retrieve relevant troubleshooting information.

**Semantic Search**
- Uses Google Generative AI embeddings.
- Stores and retrieves vectors using FAISS.

**Keyword Search**
- Uses BM25Retriever to retrieve documents based on keyword relevance.

**Ensemble Retrieval**
- Combines both retrievers using LangChain's EnsembleRetriever.
- Uses Reciprocal Rank Fusion (RRF) to combine retrieval rankings.
- Configures equal weights for semantic and keyword retrieval.

### 4. Context-Grounded Response Generation

The retrieved documents are passed to Gemini to generate step-by-step troubleshooting instructions.

The response generation process is instructed to:

- Use only the provided knowledge-base documents.
- Provide clear, sequential troubleshooting steps.
- Avoid assumptions or unsupported information.
- Explicitly communicate when the available documents do not contain sufficient information.

### 5. Human-in-the-Loop Ticket Workflow

For explicit ticket requests, LangGraph pauses execution and requests human input.

The workflow includes:

1. Asking the user whether the retrieved troubleshooting documents are relevant.
2. Preparing a structured ticket summary.
3. Asking the user to confirm the proposed ticket details.
4. Allowing the user to approve or provide an edited ticket summary.
5. Returning a ticket confirmation message.

**Integration note:** The current implementation handles ticket preparation and confirmation within the workflow. Actual ServiceNow ticket creation is not integrated.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Agent Orchestration | LangGraph |
| LLM | Google Gemini |
| LLM Integration | Google GenAI SDK |
| RAG Framework | LangChain |
| Document Loading | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | Google Generative AI Embeddings |
| Vector Store | FAISS |
| Keyword Retrieval | BM25 |
| Retrieval Strategy | Hybrid Search, Reciprocal Rank Fusion |
| State Management | LangGraph StateGraph |
| Checkpointing | InMemorySaver |
| Human-in-the-Loop | LangGraph interrupt and Command resume |

## Project Structure

```text
ITOps_Agent/
│
├── data/
│   └── ITOps_Knowledge_Base_Troubleshooting_Guides.pdf
│
├── graph.py
├── nodes.py
├── state.py
├── ingestion.py
└── README.md
```

## Configuration

### Prerequisites

- Python environment
- Google AI API key
- IT troubleshooting knowledge-base PDF

### Environment Variables

Configure the following environment variable:

| Variable | Description |
|---|---|
| `GOOGLE_GENAI_API_KEY` | API key used for Gemini interactions and Google Generative AI embeddings. |

Ensure that the troubleshooting PDF is available at the path expected by the ingestion module:

`data/ITOps_Knowledge_Base_Troubleshooting_Guides.pdf`

## Running the Project

1. Clone the repository.
2. Set up the Python environment and install the project's dependencies.
3. Configure `GOOGLE_GENAI_API_KEY`.
4. Place the troubleshooting PDF in the expected data directory.
5. Run the LangGraph workflow through its graph entry point.
6. Interact with the agent through the terminal when human input is requested.

## Future Enhancements

- Integrate with ServiceNow for actual ticket creation and status tracking.
- Add additional IT operations knowledge sources.
- Introduce more advanced intent routing and workflow branches.
- Improve persistent checkpointing and conversation management.
- Add evaluation for retrieval quality and answer accuracy.
- Expand support for additional IT support scenarios.
