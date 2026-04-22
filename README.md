# 🪨 Rocky — AI Geology Assistant

> *"Ask the rocks. They've been around longer than anyone."*

Rocky is an AI-powered geology assistant built with **LangGraph**, **FastAPI**, and **RAG (Retrieval-Augmented Generation)**. It answers geological questions by searching a curated knowledge base, retrieving real-time data, and reasoning across multiple tools — all streamed live to the user.

🌍 **Live Demo:** [Rocky — Geology Assistant](https://geology-agent-frontend-git-main-piriscs-projects.vercel.app)  
🖥️ **Frontend Repository:** [pirisc/geology_agent_frontend](https://github.com/pirisc/geology_agent_frontend)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Build the Knowledge Base](#build-the-knowledge-base)
  - [Run the Server](#run-the-server)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Roadmap](#roadmap)

---

## Overview

Rocky is a domain-specific AI agent specializing in Earth Sciences. It uses a **Retrieval-Augmented Generation (RAG)** pipeline to ground its answers in real geological documents, supplemented by live web search, real-time earthquake data from USGS, and an interactive quiz mode.

The system is designed for **students, educators, and geology enthusiasts** who want accurate, educational, and accessible explanations of geological topics — from rock types to tectonic plate dynamics.

---

## Architecture

Rocky's brain is a **LangGraph state machine** with two nodes: a chatbot node (the LLM + tools) and a tools node (tool execution). The flow is:

```
User Input
    │
    ▼
┌─────────────┐     tool call?     ┌─────────────┐
│   Chatbot   │ ─────────────────► │    Tools    │
│    Node     │ ◄───────────────── │    Node     │
└─────────────┘    tool result     └─────────────┘
    │
    ▼
Streamed Response (SSE via FastAPI)
```

Conversation history is persisted per `thread_id` using either **SQLite** (local) or **PostgreSQL** (production), giving Rocky multi-turn memory across sessions.

### RAG Pipeline

```
PDF Documents
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter → OpenAI Embeddings
    │
    ▼
Chroma Vector Store (persisted locally at ./geology_kb)
    │
    ▼
similarity_search_with_relevance_scores() at query time
```

---

## Project Structure

```
geology_agent/
│
├── agent.py                  # LangGraph graph, tools, state, streaming logic
├── app.py                    # FastAPI server, endpoints, rate limiting
├── build_knowledge_base.py   # RAG pipeline: load PDFs → chunk → embed → store
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── geology_kb/               # ⚠️ GITIGNORED — Chroma vector DB (auto-generated)
│                             #    Build locally with: python build_knowledge_base.py
│
└── source_documents/         # ⚠️ GITIGNORED — Your PDF source files go here
                              #    Add geological PDFs before building the KB
```

> **Note:** `geology_kb/` and `source_documents/` are excluded from version control.  
> You must populate `source_documents/` with your PDFs and run the build script locally.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | LangGraph (StateGraph) |
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Store** | Chroma (persistent) |
| **Web Search** | Tavily Search API |
| **API Server** | FastAPI + Uvicorn |
| **Streaming** | Server-Sent Events (SSE) |
| **Persistence (local)** | SQLite via LangGraph SqliteSaver |
| **Persistence (prod)** | PostgreSQL via LangGraph PostgresSaver |
| **Document Loading** | LangChain PyPDFLoader |

---

## Features

### 🔍 Geology Knowledge Base (RAG)
Rocky searches a curated vector database built from geological PDFs before answering. Sources include geological associations and public domain Earth Science resources. Results include source title and page number citations.

### 🌐 Web Search Fallback
If the knowledge base doesn't have a direct match, Rocky falls back to **Tavily Search** for live web results.

### 🗺️ Geological Image Finder
Rocky proactively suggests relevant geological images (minerals, rock samples, process diagrams) sourced from the web when describing physical objects or processes.

### 🌋 Live Earthquake Data
Rocky queries the **USGS GeoJSON Feed** in real time to retrieve recent earthquake events, filterable by location and minimum magnitude.

### 🧠 Interactive Quiz Mode
Users can enter study mode for any geological topic, with configurable difficulty (`easy`, `intermediate`, `advanced`) and number of questions. Rocky waits for answers before revealing explanations.

### 🌊 Streaming Responses
All responses are streamed token-by-token via **Server-Sent Events**, giving users a live, responsive experience without waiting for the full answer.

### 💾 Persistent Conversation Memory
Rocky remembers your full conversation history per `thread_id`, enabling genuine multi-turn dialogues. Backed by SQLite locally and PostgreSQL in production.

### 🛡️ Rate Limiting
Built-in IP-based rate limiting (20 requests/minute) protects the API from abuse.

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key
- A Tavily API key
- PDF documents to populate the knowledge base

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/pirisc/geology_agent.git
cd geology_agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys (see Environment Variables section)
```

### Build the Knowledge Base

Before running Rocky, you need to build the vector store from your geological PDFs.

```bash
# 1. Place your PDF files inside the source_documents/ folder
mkdir source_documents
# (copy your PDFs here)

# 2. Run the build script
python build_knowledge_base.py
```

This will:
1. Load and parse all PDFs from `source_documents/`
2. Split them into overlapping text chunks (1000 chars, 200 overlap)
3. Generate embeddings via OpenAI
4. Persist the Chroma vector store to `geology_kb/`

> **First run will take a few minutes** depending on the number and size of your PDFs.

### Run the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## API Reference

### `GET /`
Returns API status and available endpoints.

### `GET /health`
Health check endpoint. Returns current timestamp and status.

```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00.000000"
}
```

### `POST /chat`
Main chat endpoint. Returns a **Server-Sent Events (SSE)** stream.

**Request body:**
```json
{
  "message": "What causes volcanic eruptions?",
  "thread_id": "optional-uuid-for-conversation-continuity"
}
```

**Response stream (SSE):**
```
data: {"thread_id": "abc-123"}

data: {"content": "Volcanic"}
data: {"content": " eruptions"}
data: {"content": " occur when..."}

data: [DONE]
```

**Error response:**
```
data: {"error": "An error occurred during response generation"}
data: [DONE]
```

**Rate limit:** 20 requests per minute per IP. Returns `429` if exceeded.

---

## Environment Variables

Create a `.env` file in the root directory with the following:

```env
# Required
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Optional — defaults shown
MAX_INPUT_LENGTH=1000
WEB_SCRAPER_CHAR_LIMIT=8000
WEB_SCRAPER_TIMEOUT=20
VECTOR_DB_DIR=geology_kb

# Production only (Render PostgreSQL)
DATABASE_URL=postgresql://user:password@host:port/dbname

# Local SQLite path (used when DATABASE_URL is not set)
ROCKY_DB_PATH=rocky_conversations.db
```

---

## Deployment

Rocky is configured for deployment on **Render**.

The app auto-detects the environment:
- If `DATABASE_URL` is set → uses **PostgreSQL** (production persistence)
- If not → falls back to **SQLite** (local development)

**Render setup steps:**
1. Connect your GitHub repository
2. Set environment variables in the Render dashboard
3. Set the start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Build the knowledge base as part of your build command or pre-deploy hook

> **Important:** `geology_kb/` is gitignored, so you need to either commit a pre-built KB, include a build step in your deploy pipeline, or mount a persistent disk on Render and build the KB there.

---

## Author

Built by **[Iris](https://github.com/pirisc)** as a portfolio project exploring AI Agents, RAG systems, and production-ready LLM applications.

---

*Rocky was built for all the geology students who want to have everything in one place. I know how much better is not to search everywhere for answers. Enjoy!*
