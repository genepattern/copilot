<h1 align="center">🧬 GenePattern Copilot</h1>

<p align="center">
  <strong>Conversational AI for genomic science — ask questions in plain English, get expert-level bioinformatic answers.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+"/></a>
  <a href="https://www.djangoproject.com/"><img src="https://img.shields.io/badge/Django-5.2-092E20?logo=django&logoColor=white" alt="Django 5.2"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-green" alt="License: BSD 3 Clause"/></a>
  <a href="https://www.genepattern.org"><img src="https://img.shields.io/badge/Powered%20by-GenePattern-003865" alt="Powered by GenePattern"/></a>
</p>

---

## Why GenePattern Copilot?

Bioinformatics pipelines are powerful — but they've never been easy to *talk to*. Researchers spend hours navigating documentation, chasing pipeline parameters, and decoding opaque error messages.

We built GenePattern Copilot to change that. It wraps the full GenePattern analysis ecosystem in a multi-turn conversational AI layer, so you can:

- 🗣️ **Ask** *"Run a GSEA analysis on my expression data"* and get a step-by-step guided answer.
- 🔍 **Query** the GenePattern module library using natural language instead of clicking through menus.
- 🤖 **Automate** job submission, result retrieval, and pipeline orchestration via an AI agent with live tool-use.
- 📜 **Audit** every reasoning step — the model's chain-of-thought, tool calls, and retrieved context are all stored and queryable.

This is not a generic chatbot bolted onto a bioinformatics portal. GenePattern Copilot is a purpose-built agentic AI that understands GenePattern's job model, module catalog, file system, and user permissions — and acts on them in real time.

---

## ✨ Features

| | Feature | Description                                                                                                                                                            |
|---|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🧬 | **Genomic Integration** | Deep integration with GenePattern modules, tasks, and pipelines. The agent can submit jobs, fetch results, and interpret outputs.                                      |
| 🤖 | **Multi-Model AI** | Swap between the most expert LLM models with a single menu click.                                                                                                      |
| 🔧 | **Agentic Tool Use** | Powered by Pydantic AI + MCP (Model Context Protocol), the assistant has 30+ live tools covering job management, file I/O, user stats, and server introspection.       |
| 🗂️ | **RAG Pipeline** | A ChromaDB vector store indexes GenePattern documentation and forum posts so every answer is grounded in verified sources.                                             |
| ☁️ | **Cloud Scalability** | Containerized with Docker and deployed behind a Django ASGI server, ready to scale horizontally on GCP, AWS, or any Kubernetes cluster.                                |
| 📊 | **Reproducible Science** | Every conversation, query, reasoning step, token count, and user rating is persisted to a relational database — your interactions are a first-class research artifact. |
| 🔐 | **Auth & Multi-User** | Session + token authentication, per-user GenePattern API key storage, and admin-only analytics endpoints.                                                          |
| 👍 | **Human Feedback Loop** | Thumbs-up / thumbs-down ratings on every response feed directly into a queryable evaluation dataset.                                                                   |

---

## 🚀 Quick Start

### 1 · Clone & create environment

```bash
git clone https://github.com/genepattern/GPCopilot.git
cd GPCopilot

# Using Conda (recommended)
conda create -n gp_copilot python=3.11
conda activate gp_copilot

# Or using venv
python -m venv venv && source venv/bin/activate
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

### 3 · Configure environment variables

Create a `.env` file in the project root:

```dotenv
# ── Django ────────────────────────────────────────────────────────────────────
SECRET_KEY='your-strong-django-secret-key'   # python -c "import secrets; print(secrets.token_hex(50))"
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
CSRF_TRUSTED_ORIGINS=http://localhost:3000,http://127.0.0.1:8000

# ── LLM API Keys (add only the providers you use) ─────────────────────────────
OPENAI_API_KEY='sk-...'
GOOGLE_GEMINI_API_KEY='AIza...'
AWS_ACCESS_KEY_ID='AKIA...'
AWS_SECRET_ACCESS_KEY='...'

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS_ALLOWED_ORIGINS=http://localhost:3000
```

> **Tip:** You only need the API keys for the LLM providers you intend to use. The server will gracefully skip unconfigured providers.

### 4 · Initialize the database

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser   # optional — enables the /admin dashboard
```

### 5 · Start the server

```bash
python manage.py runserver
# → http://127.0.0.1:8000/
```

---

## 💬 API at a Glance

GenePattern Copilot can also be embedded in other apps via a simple RESTful API. For example, to ask a question directly:

```python
import httpx

response = httpx.post("http://127.0.0.1:8000/api/chat/", json={
    "query": "What GenePattern modules are available for RNA-seq differential expression?",
    "model_id": "gpt-4o",
})
print(response.json()["response"])
```

**Start a pipeline job via the agent:**

```python
response = httpx.post("http://127.0.0.1:8000/api/chat/", json={
    "query": "Run GSEA on my expression file at /uploads/expr.gct using the Hallmarks gene set.",
    "model_id": "gpt-4o",
    "conversation_id": "existing-uuid-or-omit-for-new",
}, cookies={"sessionid": your_session_cookie})
print(response.json()["response"])
```

The agent will automatically invoke the appropriate GenePattern MCP tools, submit the job, and report back — all in one conversational turn.

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/` | Send a message; starts or continues a conversation |
| `GET` | `/api/conversations/` | List the authenticated user's conversations |
| `GET` | `/api/conversations/<id>/` | Retrieve full turn-by-turn history for a conversation |
| `POST` | `/api/rate/<query_id>/` | Submit a thumbs-up / thumbs-down rating |
| `GET` | `/api/models/` | List all configured and enabled LLM models |
| `GET` | `/api/token-summary/` | Admin: aggregate token usage statistics |

---

## 🧩 Ecosystem & Interoperability

GenePattern Copilot is designed to *fit in*, not to be a silo:

- **GenePattern Notebook** — Run the backend locally alongside a [GenePattern Notebook](https://notebook.genepattern.org/) instance to combine programmatic LLM access with interactive notebook workflows.
- **MCP Tool Protocol** — Any MCP-compatible tool server (local or remote) can be plugged in, making it trivial to extend the agent with new capabilities.
- **ChromaDB / RAG** — Swap in your own document corpus (publications, SOPs, lab wikis) by pointing the vector store builder at a new source directory.
- **Django REST Framework** — The browsable API (`/api/`) makes it easy to explore all endpoints interactively without a frontend.
- **Docker / Kubernetes** — The included `Dockerfile` makes cloud deployment a single command away.

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t genepattern/copilot .

# Run (pass your .env at runtime)
docker run -p 8000:8000 --env-file .env genepattern/copilot
```

---

## 🤝 Join the Conversation

We built this for the bioinformatics community — researchers, analysts, and developers alike. Contributions are warmly welcomed.

- 🐛 **Found a bug?** [Open an issue](https://github.com/genepattern/copilot/issues) — the more detail, the better.
- 💡 **Have a feature idea?** Start a [GitHub Discussion](https://github.com/genepattern/copilot/discussions) — we read everything.
- 🔧 **Want to contribute code?** Fork the repo, make your changes, and open a PR. Please include tests.
- 💬 **GenePattern community support:** [groups.google.com/g/genepattern-help](https://groups.google.com/g/genepattern-help)

---

## 📖 Citing This Work

If GenePattern Copilot contributes to published research, please cite:

Reich M, Liefeld T, Gould J, Lerner J, Tamayo P, Mesirov JP. [GenePattern 2.0](http://www.nature.com/ng/journal/v38/n5/full/ng0506-500.html) Nature Genetics 38 no. 5 (2006): pp500-501 [Google Scholar](http://scholar.google.com/citations?user=lREO6vMAAAAJ&hl=en)

---

## ⚙️ Configuration Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `SECRET_KEY` | ✅ | — | Django secret key. Generate with `secrets.token_hex(50)`. |
| `DEBUG` | | `True` | Set `False` in production. |
| `ALLOWED_HOSTS` | | `127.0.0.1,localhost` | Comma-separated list of allowed hostnames. |
| `OPENAI_API_KEY` | | — | OpenAI API key for GPT models. |
| `GOOGLE_GEMINI_API_KEY` | | — | Google API key for Gemini models. |
| `AWS_ACCESS_KEY_ID` | | — | AWS credentials for Bedrock models. |
| `AWS_SECRET_ACCESS_KEY` | | — | AWS credentials for Bedrock models. |
| `CORS_ALLOWED_ORIGINS` | | `http://localhost:3000` | Frontend origins permitted for cross-site requests. |
| `DATABASE_URL` | | SQLite | Any `dj-database-url`-compatible connection string. |

