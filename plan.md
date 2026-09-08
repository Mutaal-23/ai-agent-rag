# AI Engineer Gig – Complete Build Guide

## What This Project Is (In Simple Terms)

You're building a **smart assistant** that can:
- **Search a knowledge base** (like a library) when asked questions
- **Call external APIs** (like fetching weather data)
- **Decide on its own** which tool to use based on the user's question
- **Serve everything via an API** (so other apps can use it)

Think of it like this: You're building a brain that has two hands — one hand can search documents, the other can call websites. The brain decides which hand to use.

---

## Phase 1: Setup & API Keys

### Step 1.1: Get Free LLM Keys

**What is this?**
LLM (Large Language Model) is the AI brain. You need a key (like a password) to use Google's Gemini model for free.

**What to do:**

1. Open your browser and go to **Google AI Studio** (search "Google AI Studio" on Google)
2. Sign in with your Google account
3. Look for a button that says **"Get API Key"** or **"Create API Key"**
4. Copy that key — it looks like a long string of random letters and numbers
5. Save it somewhere safe (like a notepad file). You'll use it later

**Why Gemini 1.5 Flash?**
- It's free for small usage
- It's fast
- It works well with LangChain

**Optional – Groq:**
- Go to **console.groq.com**
- Sign up and create an API key
- Groq gives you access to fast open-source models (like Llama 3)
- This is backup in case you want faster responses later

---

### Step 1.2: Setup Local Project Environment

**What is a virtual environment?**
Imagine your project is a house. A virtual environment puts a fence around that house so the packages (libraries) you install don't mess up other projects on your computer.

**What to do in your terminal:**

1. **Navigate to your project folder:**
   ```bash
   cd /home/mutaal/Work/upwork_gig1
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```
   - This creates a folder called `venv` inside your project
   - Inside this folder, Python will keep its own isolated packages

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```
   - After running this, you'll see `(venv)` appear at the start of your terminal line
   - This means you're now "inside" the virtual environment
   - Every package you install now goes inside `venv/` folder only

4. **Verify it worked:**
   ```bash
   which python
   ```
   - It should show something like `/home/mutaal/Work/upwork_gig1/venv/bin/python`
   - If it shows a system path, the activation didn't work

---

### Step 1.3: Install Dependencies

**What are dependencies?**
These are pre-built code packages that other people wrote. You're borrowing their work so you don't have to build everything from scratch.

**What each package does (in simple terms):**

| Package | What It Does |
|---------|--------------|
| `fastapi` | Creates web APIs (like a waiter that takes orders and brings food) |
| `uvicorn` | Runs the web server (like the restaurant itself) |
| `langgraph` | Helps build AI workflows with steps and decisions |
| `langchain-google-genai` | Connects LangChain to Google's Gemini AI |
| `chromadb` | Local vector database (like a search engine for your documents) |
| `pydantic` | Validates and structures your data (like a form that checks if fields are filled correctly) |
| `requests` | Makes HTTP calls to external APIs (like visiting a website from your code) |
| `python-dotenv` | Reads your `.env` file to load secret keys |

**Run this command:**
```bash
pip install fastapi uvicorn langgraph langchain-google-genai chromadb pydantic requests python-dotenv
```

**How to verify it worked:**
```bash
pip list
```
You should see all the packages listed with their version numbers.

---

### Step 1.4: Configure Secrets & Safety

**What is a `.env` file?**
It's a special file where you store secret information (like API keys). You never want to share this file publicly because anyone with your key can use your AI credits.

**What to do:**

1. **Create `.env` file:**
   ```bash
   touch .env
   ```

2. **Open it and add this line (replace with your actual key):**
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```

3. **Create `.gitignore` file:**
   ```bash
   touch .gitignore
   ```

4. **Open it and add these lines:**
   ```
   .env
   venv/
   __pycache__/
   .chroma_db/
   ```
   **Why these lines?**
   - `.env` — contains your secret API keys
   - `venv/` — huge folder, don't need to upload it
   - `__pycache__` — Python's temporary files
   - `.chroma_db/` — your local database files

**How to test your `.env` is working:**

Create a quick test script:
```python
# test_env.py
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print(f"Key loaded: {'Yes' if key else 'No'}")
print(f"First 10 chars: {key[:10]}..." if key else "No key found")
```

Run it:
```bash
python test_env.py
```

You should see "Key loaded: Yes" and the first 10 characters of your key.

**Delete this test file after testing** — it was just for verification.

---

## Phase 2: Building the 4 Core Modules

### Step 2.1: Module 1 – Knowledge Base & RAG Tool

**What is RAG?**
RAG stands for **Retrieval-Augmented Generation**. Here's the simple version:

- **Without RAG:** You ask AI a question → AI uses only what it already knows → might make things up
- **With RAG:** You ask AI a question → AI first searches your documents → uses those documents to give an accurate answer

It's like open-book exams vs closed-book exams.

**What you need to create:**

1. **A `data/` folder** with a text file containing some domain knowledge
   - Example: Create `data/company_policies.txt` with 10-15 lines about company rules
   - Keep it simple — just plain text

2. **A script that does 3 things:**
   - **Reads** the text file
   - **Splits** it into small chunks (like cutting a book into paragraphs)
   - **Converts** each chunk into numbers (embeddings) and saves them in ChromaDB

**Key concepts you need to understand:**

**Chunking:**
- You can't feed an entire document to the AI at once
- So you cut it into smaller pieces (chunks)
- Each chunk should be 500-1000 characters
- You also overlap chunks slightly so you don't lose context at the edges

**Embeddings:**
- AI doesn't understand text directly — it understands numbers
- An embedding converts text into a list of numbers (like [0.23, -0.45, 0.67, ...])
- Similar texts get similar numbers
- This lets the AI compare and find relevant chunks

**ChromaDB:**
- A local database that stores these number vectors
- When you ask a question, it converts your question to numbers too
- Then finds the closest matching chunks
- Returns those chunks to the AI

**What your RAG tool function should do:**

```
Function: search_knowledge_base(query: str) -> str
- Takes a question as input
- Converts question to embedding
- Searches ChromaDB for top 3 most similar chunks
- Returns those chunks as a single string
```

**Build this step by step:**

**Step A:** Create your data folder and a sample text file with some policies or rules

**Step B:** Create a script that:
1. Loads the text file
2. Splits it into chunks (use LangChain's `CharacterTextSplitter`)
3. Creates ChromaDB collection
4. Adds chunks to the collection
5. Saves the database locally

**Step C:** Create the search tool function:
1. Takes a user question
2. Searches the ChromaDB collection
3. Returns the top matching chunks

**Test it:**
- Run the script to build the database
- Call the search function with a sample question
- Verify it returns relevant chunks

---

### Step 2.2: Module 2 – REST API Action Tool

**What is a REST API?**
It's a way to talk to other websites/services using code. Like when you check the weather on your phone — the app calls a weather API to get data.

**What you need to create:**

A Python function that:
1. Takes parameters (like a city name)
2. Makes an HTTP request to an API
3. Returns the response in a clean format

**Example API to use (free, no key needed):**
- **JSONPlaceholder** (`https://jsonplaceholder.typicode.com`) — for testing
- Or **Open-Meteo** (`https://api.open-meteo.com`) — for real weather data

**What your function should look like (conceptually):**

```
Function: call_rest_api(endpoint: str, params: dict) -> str
- Takes an API endpoint URL and parameters
- Makes a GET request
- Returns the response as a formatted string
```

**OR** you can make it more specific:

```
Function: get_weather(city: str) -> str
- Takes a city name
- Calls a weather API
- Returns temperature, humidity, conditions
```

**Build this step by step:**

**Step A:** Choose which API you want to use

**Step B:** Write a function that:
1. Accepts input parameters
2. Uses the `requests` library to call the API
3. Handles errors (what if the API is down? what if the city doesn't exist?)
4. Returns a clean, readable string

**Step C:** Test it:
- Call the function directly
- Make sure it returns real data
- Test error cases (invalid inputs)

---

### Step 2.3: Module 3 – LangGraph Agent & State Machine

**What is LangGraph?**
It's a tool that helps you build AI workflows as a **flowchart**. Instead of writing one long script, you define steps (nodes) and connections (edges), and the AI moves through them.

**What is a State Machine?**
Think of it like a GPS navigation:
- You're at a point (state)
- You have options (go left, go right, go straight)
- Based on the situation (traffic, destination), you choose a path
- You move to the next point and repeat

**Key components you need to build:**

**1. AgentState (The Memory)**
This is a dictionary that stores everything during the conversation:
```
{
    "messages": [],        # conversation history
    "tool_used": None,     # which tool was called
    "tool_output": None,   # what the tool returned
    "final_answer": None   # the AI's response
}
```

**2. Agent Node (The Brain)**
This is where the AI:
1. Reads the user's question
2. Decides: "Do I need a tool, or can I answer directly?"
3. If tool needed → picks which tool → calls it
4. If no tool needed → generates answer directly

**3. Tool Node (The Hands)**
This executes the actual tools:
- If RAG tool was chosen → calls your knowledge base search
- If API tool was chosen → calls your REST API function

**4. Conditional Edge (The Router)**
This decides the flow:
- After the Agent speaks → Did it use a tool? → Yes: go to Tool Node → No: finish
- After the Tool Node runs → Go back to Agent with the tool's output

**Build this step by step:**

**Step A:** Define your `AgentState` using Pydantic or TypedDict

**Step B:** Import your two tools (RAG search + REST API) and bind them to the Gemini LLM

**Step C:** Create the Agent node function:
- Receives the current state
- Sends the conversation to the LLM with tool descriptions
- Returns the updated state

**Step D:** Create the Tool node function:
- Receives the state
- Checks which tool was requested
- Runs the appropriate tool
- Returns the output in the state

**Step E:** Build the graph:
1. Start with a `StateGraph`
2. Add your Agent node
3. Add your Tool node
4. Add an edge from Agent → conditional function
5. Add edges from conditional: either → Tool or → END
6. Add edge from Tool → back to Agent

**Step F:** Compile and test the graph

**Test it:**
- Ask a question that needs RAG → verify it searches and answers
- Ask a question that needs the API → verify it calls the API
- Ask a general question → verify it answers without tools

---

### Step 2.4: Module 4 – FastAPI Server & Endpoints

**What is FastAPI?**
It's a framework for building web APIs. It creates "endpoints" (URLs) that other apps can talk to.

**What is an endpoint?**
Think of it like a mail slot:
- Someone sends a letter (HTTP request) to a specific address (URL)
- Your code catches it, processes it, and sends back a response

**What you need to create:**

**1. A POST endpoint at `/api/v1/chat`**

POST means you're sending data to the server. The endpoint `/api/v1/chat` is where chat requests go.

**2. Request format (what the client sends):**
```json
{
    "message": "What is the company policy on remote work?"
}
```

**3. Response format (what your server returns):**
```json
{
    "status": "success",
    "response": "According to the policy document...",
    "tools_used": ["knowledge_base_search"]
}
```

**Build this step by step:**

**Step A:** Import FastAPI and create the app:
```python
from fastapi import FastAPI
app = FastAPI()
```

**Step B:** Create a Pydantic model for the request:
```
class ChatRequest:
    message: str
```

**Step C:** Create a Pydantic model for the response:
```
class ChatResponse:
    status: str
    response: str
    tools_used: list
```

**Step D:** Create the POST endpoint:
```
@app.post("/api/v1/chat")
async function chat(request: ChatRequest):
    - Take the message from request
    - Pass it to your LangGraph runner
    - Get the result
    - Return it as ChatResponse
```

**Step E:** Add a health check endpoint:
```
@app.get("/health")
async function health():
    return {"status": "ok"}
```

**Test it:**
- Start the server: `uvicorn main:app --reload`
- Go to `http://localhost:8000/docs` (Swagger UI)
- Send a test request through Swagger
- Verify you get a proper response

---

## Phase 3: Testing & Containerization

### Step 3.1: Local Terminal & API Testing

**Start your server:**
```bash
uvicorn main:app --reload
```
- `--reload` means if you change code, the server restarts automatically

**Test RAG Query:**
Open a new terminal and run:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the policy on remote work?"}'
```

**Expected result:** JSON with the policy information from your documents

**Test API Call:**
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in New York?"}'
```

**Expected result:** JSON with weather data from the API

**Test General Question:**
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 2 + 2?"}'
```

**Expected result:** JSON with answer "4" (no tools used)

---

### Step 3.2: Dockerization

**What is Docker?**
Docker packages your entire application into a "container" — like a shipping box that has everything inside it. Anyone can run this box without worrying about installing Python, dependencies, or setting up environment variables.

**What you need to create:**

**1. `Dockerfile`**

This is a recipe for building your container:

```
Step 1: Start with a Python base image (like choosing which foundation to build on)
Step 2: Set the working directory inside the container
Step 3: Copy your requirements file (list of packages)
Step 4: Install all packages
Step 5: Copy your application code
Step 6: Expose the port your server runs on
Step 7: Define the command to start the server
```

**2. `docker-compose.yml`**

This makes running Docker easier with one command:

```
Step 1: Define the service (your app)
Step 2: Set the build context (where your code is)
Step 3: Map ports (container port → your computer's port)
Step 4: Set environment variables
Step 5: Define startup command
```

**Build and run:**
```bash
docker compose up --build
```

**Test it:**
- Open browser to `http://localhost:8000/docs`
- Send test requests
- Verify everything works the same as local

---

### Step 3.3: Configuration Template

**Create `.env.example`:**
```
GEMINI_API_KEY=your_gemini_api_key_here
```
This shows collaborators what variables they need without exposing your actual keys.

---

## Phase 4: Delivery & Client Handoff

### Step 4.1: GitHub Repository Setup

**Steps:**
1. `git init` — initialize git
2. `git add .` — stage all files
3. `git commit -m "Initial commit: AI agent with RAG and API tools"` — first commit
4. Create a private repo on GitHub
5. `git remote add origin <your-repo-url>` — connect to GitHub
6. `git push -u origin main` — push your code

**Make sure `.gitignore` is working** — no `.env` or `venv/` should appear in the repo.

---

### Step 4.2: Record a Loom Walkthrough Video

**What to show in the video (2 minutes max):**

1. **Project structure** — show the folder layout
2. **Server starting** — show terminal running `uvicorn`
3. **RAG query demo** — send a question, show the agent searching and answering
4. **API tool demo** — send a weather question, show the agent calling the API
5. **Code walkthrough** — briefly explain each module

**Tips:**
- Speak clearly
- Keep it under 2 minutes
- Show the actual terminal output
- Don't read code line by line — focus on the demo

---

### Step 4.3: Submit on Upwork

**What to deliver:**

1. **README.md** with:
   - Project overview
   - Prerequisites
   - Setup instructions
   - API documentation
   - Architecture diagram (simple text-based)

2. **GitHub repo link** (private repo, add client as collaborator)

3. **Loom video link**

4. **Brief description** of what you built and how it meets their requirements

---

## File Structure

Here's what your final project should look like:

```
upwork_gig1/
├── data/
│   └── company_policies.txt      # Your knowledge base document
├── .env                           # Secret API keys (DO NOT COMMIT)
├── .env.example                   # Template for collaborators
├── .gitignore                     # Ignores venv, .env, __pycache__
├── main.py                        # FastAPI app + LangGraph agent
├── rag_tool.py                    # RAG knowledge base search function
├── api_tool.py                    # REST API action tool
├── requirements.txt               # List of all packages
├── Dockerfile                     # Docker recipe
├── docker-compose.yml             # One-command deployment
├── README.md                      # Setup & usage guide
└── test_env.py                    # Quick test (delete after use)
```

---

## Important Notes

1. **You write the code, I guide you** — I won't write code for you. I'll tell you exactly what to do and where to put it.

2. **Test after every step** — Don't move to the next step until the current one works.

3. **Keep it simple** — If something isn't working, simplify it. Don't add more complexity.

4. **Use `print()` statements** — Add print statements to see what's happening inside your code while debugging.

5. **Read error messages** — When something fails, Python usually tells you why. Read the error message carefully.

6. **One step at a time** — Complete Phase 1 before starting Phase 2. Complete Step 2.1 before starting Step 2.2.

---

## Quick Reference: Key Concepts

| Concept | Simple Explanation |
|---------|-------------------|
| RAG | AI searches your documents before answering |
| LangGraph | Flowchart builder for AI workflows |
| ChromaDB | Local database that stores document embeddings |
| Embedding | Converting text to numbers for comparison |
| Chunking | Cutting documents into small pieces |
| FastAPI | Framework for building web APIs |
| Docker | Packages your app into a portable container |
| Vector Database | Database that searches by meaning, not just keywords |
| State | The memory/variables the AI carries during a conversation |
| Node | A step in the workflow (like a box in a flowchart) |
| Edge | A connection between nodes (like an arrow in a flowchart) |
| Conditional Edge | A decision point (like a fork in the road) |
| Function Calling | AI deciding which tool to use |
| Structured Output | AI returning data in a specific JSON format |

---

## Your Progress Tracker

Use this checklist to track your progress:

- [ ] Phase 1: Got Gemini API key
- [ ] Phase 1: Created virtual environment
- [ ] Phase 1: Installed all dependencies
- [ ] Phase 1: Created `.env` and `.gitignore`
- [ ] Phase 2.1: Built RAG knowledge base
- [ ] Phase 2.1: RAG search tool works
- [ ] Phase 2.2: REST API tool works
- [ ] Phase 2.3: LangGraph agent works
- [ ] Phase 2.4: FastAPI server works
- [ ] Phase 3.1: All 3 test queries pass
- [ ] Phase 3.2: Docker builds and runs
- [ ] Phase 3.3: `.env.example` created
- [ ] Phase 4.1: Git repo set up
- [ ] Phase 4.2: Loom video recorded
- [ ] Phase 4.3: Submitted on Upwork
