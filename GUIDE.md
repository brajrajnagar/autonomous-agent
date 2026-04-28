# Autonomous Agent Code Guide

This guide explains the flow and architecture of the autonomous agent codebase. It's designed to help you understand how the agent works and how to extend it.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [File Structure](#file-structure)
4. [Execution Flow](#execution-flow)
5. [Component Deep Dive](#component-deep-dive)
6. [Extending the Agent](#extending-the-agent)

---

## Overview

The autonomous agent implements a **Think → Act → Observe** loop pattern. This is a classic AI agent architecture where the agent:

1. **Thinks** about what to do next based on the task and current knowledge
2. **Acts** by executing a tool (shell command, file operation, etc.)
3. **Observes** the result of the action
4. **Repeats** until the task is complete
5. **Reviews** the output with a critic pass

### Key Design Decisions

- **Single LLM**: Uses the same LLM for both action selection and critic review (different prompts)
- **Stateful Execution**: Maintains complete history of thoughts, actions, and observations
- **Tool-Based**: All interactions with the environment happen through defined tools
- **Configurable**: API endpoint, model, and iteration limits are environment-based

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     AutonomousAgent                             │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    run(task)                              │ │
│  │                                                           │ │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐ │ │
│  │  │   THINK     │     │    ACT      │     │   OBSERVE   │ │ │
│  │  │ Build prompt│────▶│ Execute tool│────▶│ Get result  │ │ │
│  │  │ Call LLM    │     │ Tools class │     │ Store in    │ │ │
│  │  │ Parse JSON  │     │             │     │ history     │ │ │
│  │  └─────────────┘     └─────────────┘     └─────────────┘ │ │
│  │         ▲                                              │   │ │
│  │         └──────────────────────────────────────────────┘   │ │
│  │                    (Repeat loop)                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              _critic_review(result)                       │ │
│  │              Second LLM pass for quality check            │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Tools                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │execute_shell│ │ read_file   │ │ write_file  │ │list_dir   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
agent/
├── src/
│   ├── agent.py        # Main agent implementation
│   └── test_agent.py   # Test scripts
├── config/
│   └── .env            # Configuration (API keys, etc.)
├── GUIDE.md            # This file
├── README.md           # Quick start guide
└── progress.md         # Development roadmap
```

### Key Classes and Their Roles

| Class/File | Purpose | Key Methods |
|------------|---------|-------------|
| `AutonomousAgent` | Main agent controller | `run()`, `_build_prompt()`, `_critic_review()` |
| `Tools` | Tool implementations | `execute_shell()`, `read_file()`, `write_file()`, `list_directory()` |
| `AgentState` | State container | N/A (dataclass) |
| `ToolResult` | Result container | N/A (dataclass) |

---

## Execution Flow

Let's trace through a complete execution with an example task: **"List all files in the current directory"**

### Step-by-Step Flow

```
1. User calls: agent.run("List all files in the current directory")
   │
   ▼
2. AgentState is initialized:
   - task = "List all files in the current directory"
   - iteration_count = 0
   - All history lists empty
   │
   ▼
3. ┌─────────────────────────────────────────┐
   │     ITERATION 1 LOOP STARTS             │
   └─────────────────────────────────────────┘
   │
   ▼
4. THINK Phase:
   - _build_prompt() creates prompt with:
     * Original task
     * Empty history (first iteration)
     * Tool instructions
   │
   ▼
5. LLM Call:
   - Prompt sent to LLM
   - LLM responds: {"tool": "list_directory", "parameters": {"path": "."}}
   │
   ▼
6. Parse Response:
   - _parse_response() extracts JSON
   - Returns: {"tool": "list_directory", "parameters": {"path": "."}}
   │
   ▼
7. ACT Phase:
   - _execute_tool("list_directory", {"path": "."})
   - Calls: Tools.list_directory(".")
   │
   ▼
8. OBSERVE Phase:
   - Result: [DIR] src, [FILE] README.md, ...
   - Stored in observation_history
   │
   ▼
9. ┌─────────────────────────────────────────┐
   │     ITERATION 2 LOOP STARTS             │
   └─────────────────────────────────────────┘
   │
   ▼
10. THINK Phase:
    - _build_prompt() now includes:
      * Original task
      * History from iteration 1
    │
    ▼
11. LLM Call:
    - LLM responds: {"complete": true, "answer": "Files: src, README.md..."}
    │
    ▼
12. Completion Detected:
    - state.is_complete = True
    - state.final_answer = "Files: src, README.md..."
    - Loop exits
    │
    ▼
13. CRITIC Phase:
    - _critic_review(final_answer)
    - LLM reviews: "APPROVED"
    │
    ▼
14. Return Result to User
```

---

## Component Deep Dive

### 1. AutonomousAgent Class

The main controller that orchestrates everything.

#### Key Attributes
```python
self.client        # OpenAI API client
self.model         # Model name (e.g., "gpt-4")
self.max_iterations # Max loop iterations (default: 10)
self.state         # Current AgentState
self.tools         # Tools instance
```

#### Key Methods

**`run(task: str) -> str`**
- Main entry point
- Runs the Think → Act → Observe loop
- Returns final result with critic feedback

**`_build_prompt() -> str`**
- Constructs the prompt for the LLM
- Includes task, history, and instructions
- Called each iteration

**`_parse_response(response: str) -> Dict`**
- Extracts JSON from LLM response
- Handles both tool calls and completion signals

**`_execute_tool(name, params) -> ToolResult`**
- Routes to appropriate tool method
- Returns ToolResult object

**`_critic_review(result: str) -> str`**
- Second LLM pass for quality assurance
- Returns "APPROVED" or feedback

### 2. Tools Class

Provides all available actions for the agent.

#### Available Tools

| Tool | Purpose | Parameters | Example |
|------|---------|------------|---------|
| `execute_shell` | Run shell commands | `command: str` | `{"command": "ls -la"}` |
| `read_file` | Read file contents | `path: str` | `{"path": "config.txt"}` |
| `write_file` | Write to file | `path: str, content: str` | `{"path": "out.txt", "content": "Hello"}` |
| `list_directory` | List directory | `path: str` (default ".") | `{"path": "."}` |

#### Adding New Tools

```python
@staticmethod
def new_tool(param1: str, param2: int = 0) -> ToolResult:
    """
    Description of the tool.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        ToolResult with success/output/error
    """
    try:
        # Implementation
        return ToolResult(success=True, output="Result")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))
```

Then update `TOOLS_PROMPT` in `AutonomousAgent`:
```python
TOOLS_PROMPT = """
...
5. new_tool: Description. Parameters: {"param1": "string", "param2": "number"}
"""
```

And update `_execute_tool`:
```python
elif tool_name == "new_tool":
    return self.tools.new_tool(
        parameters.get("param1", ""),
        parameters.get("param2", 0)
    )
```

### 3. AgentState Dataclass

Tracks the complete execution state.

```python
@dataclass
class AgentState:
    task: str = ""                          # Current task
    thought_history: List[str] = field(default_factory=list)
    action_history: List[Dict] = field(default_factory=list)
    observation_history: List[str] = field(default_factory=list)
    iteration_count: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None
```

### 4. ToolResult Dataclass

Standardized result format.

```python
@dataclass
class ToolResult:
    success: bool      # Did execution succeed?
    output: str        # Output content (stdout or file content)
    error: Optional[str] = None  # Error message if failed
```

---

## Extending the Agent

### Adding a New Tool

1. **Add method to Tools class:**
```python
@staticmethod
def http_get(url: str) -> ToolResult:
    """Fetch content from a URL."""
    import requests
    try:
        response = requests.get(url)
        return ToolResult(
            success=response.status_code == 200,
            output=response.text
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))
```

2. **Update TOOLS_PROMPT:**
```python
TOOLS_PROMPT = """
...
5. http_get: Fetch URL content. Parameters: {"url": "string"}
"""
```

3. **Add routing in _execute_tool:**
```python
elif tool_name == "http_get":
    return self.tools.http_get(parameters.get("url", ""))
```

### Adding Memory

To add long-term memory:

1. **Create a Memory class:**
```python
# src/memory.py
class Memory:
    def __init__(self):
        self.memories = []
    
    def add(self, content: str):
        self.memories.append(content)
    
    def search(self, query: str) -> List[str]:
        # Implement search logic
        return relevant_memories
```

2. **Integrate with AgentState:**
```python
@dataclass
class AgentState:
    # ... existing fields ...
    memory: Memory = field(default_factory=Memory)
```

3. **Include in prompt:**
```python
def _build_prompt(self) -> str:
    prompt = f"""Task: {self.state.task}

Relevant Memories:
{self.state.memory.search(self.state.task)}

..."""
```

### Adding Multi-Agent Support

For multiple specialized agents:

1. **Create agent types:**
```python
class PlannerAgent(AutonomousAgent):
    """Specializes in breaking down tasks."""
    pass

class ExecutorAgent(AutonomousAgent):
    """Specializes in executing tools."""
    pass

class CriticAgent(AutonomousAgent):
    """Specializes in reviewing output."""
    pass
```

2. **Create orchestrator:**
```python
class Orchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.critic = CriticAgent()
    
    def run(self, task: str) -> str:
        plan = self.planner.run(f"Plan how to: {task}")
        result = self.executor.run(f"Execute: {plan}")
        review = self.critic.run(f"Review: {result}")
        return review
```

---

## Configuration

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_API_BASE` | API endpoint | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | Authentication key | `sk-xxx` |
| `OPENAI_MODEL` | Model to use | `gpt-4` |
| `MAX_ITERATIONS` | Safety limit | `10` |

### Loading Configuration

```python
# In agent.py
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, "config", ".env")
load_dotenv(env_path)

# Access in code
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4")  # Default to gpt-4
```

---

## Debugging Tips

### Enable Verbose Logging

Add debug prints in the loop:
```python
def run(self, task: str) -> str:
    # ... existing code ...
    print(f"DEBUG: Prompt = {prompt}")
    print(f"DEBUG: LLM Response = {llm_response}")
    print(f"DEBUG: Parsed = {parsed}")
    # ...
```

### Test Individual Components

```python
# Test tools directly
result = Tools.list_directory(".")
print(result)

# Test prompt building
agent = AutonomousAgent()
agent.state.task = "Test task"
prompt = agent._build_prompt()
print(prompt)

# Test response parsing
parsed = agent._parse_response('{"tool": "list_directory", "parameters": {"path": "."}}')
print(parsed)
```

---

## Summary

The autonomous agent is a modular, extensible system that:

1. **Receives a task** from the user
2. **Iterates** through Think → Act → Observe cycles
3. **Uses tools** to interact with the environment
4. **Reviews** output with a critic pass
5. **Returns** the final result

The key to understanding the flow is:
- `run()` is the main loop
- `_build_prompt()` creates context for the LLM
- `_parse_response()` extracts actions from LLM output
- `_execute_tool()` routes to tool implementations
- `_critic_review()` validates the final output

This architecture can be extended with new tools, memory systems, or even multiple specialized agents.