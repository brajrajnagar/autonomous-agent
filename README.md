# Autonomous Agent

A highly autonomous AI-powered task automation agent with Think → Act → Observe loop architecture.

## Architecture

```
┌─────────────────────────────────────────┐
│           Single Agent Loop             │
│                                         │
│  ┌─────────┐    ┌─────────┐    ┌──────┐│
│  │  THINK  │ →  │  ACT    │ →  │OBSERVE││
│  │(Plan)   │    │(Execute)│    │(Result)│
│  └─────────┘    └─────────┘    └──────┘│
│       ↑                                  │
│       └──────────────────────────────────┘
│              (Repeat until done)         │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   CRITIC (Second Pass Review)   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Setup

### 1. Create Virtual Environment (if not done)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install openai python-dotenv
```

### 3. Configure API

Copy the example config and fill in your values:

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
```

## Usage

### Command Line

```bash
# Run with a task as argument
python src/agent.py "List all files in the current directory"

# Run interactively
python src/agent.py
```

### As a Module

```python
from src.agent import AutonomousAgent

agent = AutonomousAgent()
result = agent.run("Create a file called test.txt with 'Hello World' content")
print(result)
```

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

1. **THINK**: The agent considers what needs to be done next based on the task and previous observations
2. **ACT**: The agent executes a tool (shell command, file operation, etc.)
3. **OBSERVE**: The agent observes the result of its action
4. **REPEAT**: Steps 1-3 continue until the task is complete
5. **CRITIC**: A second LLM pass reviews the output for quality and accuracy

## Example Session

```
🤖 Starting agent for task: Create a file called hello.txt with "Hello World" inside
Max iterations: 10

--- Iteration 1 ---
THINKING...
LLM Response: {"tool": "write_file", "parameters": {"path": "hello.txt", "content": "Hello World"}}
Thought: I need to create a file called hello.txt with "Hello World" content
ACT: Executing tool 'write_file'...
OBSERVE: Successfully wrote 11 characters to hello.txt
✅ Task marked as complete

--- CRITIC REVIEW ---
Critic: APPROVED
✅ Output approved by critic

FINAL RESULT:
Task completed successfully
```

## Project Structure

```
agent/
├── src/
│   └── agent.py       # Main agent implementation
├── config/
│   └── .env.example   # Configuration template
├── tests/             # Test files (future)
├── venv/              # Virtual environment
├── progress.md        # Development progress tracker
└── README.md          # This file
```

## Roadmap

See [progress.md](progress.md) for detailed roadmap including:
- Phase 2: Memory System (vector DB, context retention)
- Phase 3: Multi-Agent Orchestration
- Phase 4: Advanced Tools (web, APIs)
- Phase 5: Production Readiness (Docker sandbox, monitoring)

## Security Notes

- Absolute paths are blocked for file operations
- Shell commands run with the permissions of the current user
- Consider using Docker sandboxing (Phase 5) for production use