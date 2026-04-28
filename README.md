# Autonomous Agent

An AI-powered task automation agent that **plans before it acts**: it decomposes the user's request into a step-by-step plan, lets the user critique and refine that plan, then executes each step with a Think → Act → Observe loop.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       AutonomousAgent                        │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │   1. PLAN          (initial plan from request)      │   │
│   │   2. CRITIQUE      (reviewer surfaces gaps)         │   │
│   │   3. REFINE        (user iterates until satisfied)  │   │
│   └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │   For each plan step:                               │   │
│   │   ┌───────┐   ┌───────┐   ┌─────────┐               │   │
│   │   │ THINK │ → │  ACT  │ → │ OBSERVE │ → repeat      │   │
│   │   └───────┘   └───────┘   └─────────┘               │   │
│   └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │   CRITIC (final post-execution quality review)      │   │
│   └─────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Create Virtual Environment (if not done)

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install openai python-dotenv
```

### 3. Configure API

```bash
cp config/.env.example config/.env
```

Edit `config/.env`:

```
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4
MAX_ITERATIONS=10
DEBUG=true

# Planning loop
AGENT_PLANNING_ENABLED=true
AGENT_MAX_PLAN_REFINEMENTS=5

# Per-loop max_tokens (tune to control truncation vs. cost)
AGENT_MAX_TOKENS_TAO=15000
AGENT_MAX_TOKENS_CRITIC=5000
AGENT_MAX_TOKENS_PLAN_INITIAL=10000
AGENT_MAX_TOKENS_PLAN_CRITIQUE=15000
AGENT_MAX_TOKENS_PLAN_REFINE=15000
```

## Usage

### Command Line

```bash
# Single-task mode
python src/agent.py "build a small neural network from scratch"

# Interactive mode (recommended)
python src/agent.py
```

### As a Module

```python
from src.agent import AutonomousAgent

agent = AutonomousAgent()
result = agent.run("Create a file called test.txt with 'Hello World' content")
print(result)
```

## Planning Loop

When you submit a task, the agent first proposes a plan and shows it to you with the reviewer's suggested improvements:

```
═══════════════════════════════════════════════════════
📋 PROPOSED PLAN
═══════════════════════════════════════════════════════
Summary: Build a small NN from scratch with backprop, train, document.

Steps:
  1. Create project directory
  2. Write neural_network.py with NN class, forward pass, backprop
  3. Load real dataset (e.g., Iris)
  4. Train and report test accuracy
  5. Write README.md with run instructions

💡 Suggested improvements:
  [1] Step 3 doesn't specify which dataset
      → Pick a concrete dataset (Iris, MNIST, etc.)
  [2] No test/eval split mentioned
      → Add a step to evaluate on held-out test data

Type 'go' / 'ok' / Enter to approve and execute,
     'apply 1' or 'apply 1,2' to adopt suggestions,
     or describe changes in your own words.
> _
```

You can:

- Type `go` / `ok` / Enter — approve and execute the current plan.
- Type `apply 1` or `apply 1,3` — adopt one or more critic suggestions.
- Describe changes freely — e.g. *"merge steps 2 and 3, and use MNIST instead"*.

The plan re-renders after each refinement until you approve. After approval, each step runs through the Think → Act → Observe loop with its `success_criterion` injected as guidance.

To skip planning for trivial one-shot tasks, set `AGENT_PLANNING_ENABLED=false`.

## Available Tools

1. **execute_shell**: Run shell commands
   ```json
   {"tool": "execute_shell", "parameters": {"command": "ls -la"}}
   ```
2. **read_file**: Read file contents
   ```json
   {"tool": "read_file", "parameters": {"path": "README.md"}}
   ```
3. **write_file**: Write to a file
   ```json
   {"tool": "write_file", "parameters": {"path": "output.txt", "content": "Hello"}}
   ```
4. **list_directory**: List directory contents
   ```json
   {"tool": "list_directory", "parameters": {"path": "."}}
   ```

## How It Works

1. **PLAN**: Decompose the user's request into ordered steps with success criteria.
2. **CRITIQUE**: A reviewer LLM surfaces gaps, ambiguity, and missing implicit work.
3. **REFINE**: User iterates on the plan (apply suggestions, free-form edits) until satisfied.
4. **EXECUTE**: For each step, run Think → Act → Observe until the step's success criterion is met.
5. **CRITIC**: Final LLM pass reviews the overall result for quality and accuracy.

## Example Session

```
> You: build a small NN with backprop from scratch

🤖 Starting agent for task: build a small NN with backprop from scratch
🧭 Generating initial plan...
🔍 Critiquing plan (round 1)...

[plan + suggestions shown — see "Planning Loop" section]

> apply 1,2

✏️  Refining plan based on feedback...
🔍 Critiquing plan (round 2)...

[refined plan shown]

> go
✅ Plan approved. Beginning execution.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶️  STEP 1/5: Create project directory
   Success: Directory exists
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[T-A-O loop runs...]

[steps 2-5 follow]

--- CRITIC REVIEW ---
Critic: APPROVED
✅ Output approved by critic
```

## Project Structure

```
agent/
├── src/
│   ├── agent.py        # Agent implementation
│   └── test_agent.py   # Test scripts
├── config/
│   ├── .env            # Local config (not committed)
│   └── .env.example    # Configuration template
├── tests/              # (reserved for future tests)
├── venv/               # Virtual environment
├── GUIDE.md            # Architecture & extension guide
├── progress.md         # Development progress tracker
└── README.md           # This file
```

## Roadmap

See [progress.md](progress.md) for the detailed roadmap. Recent additions:

- ✅ Phase 1: Single-agent Think → Act → Observe loop
- ✅ Phase 1.5: **Plan → Critique → Refine → Execute orchestration** (new)
- 🔮 Phase 2: Memory System (lessons across sessions, vector DB)
- 🔮 Phase 3: Multi-Agent Orchestration
- 🔮 Phase 4: Advanced Tools (web, APIs)
- 🔮 Phase 5: Production Readiness (Docker sandbox, monitoring)

## Security Notes

- Absolute paths are blocked for `read_file` / `write_file`.
- Shell commands run with the permissions of the current user — review the proposed plan before approving.
- Consider Docker sandboxing (Phase 5) for production use.
