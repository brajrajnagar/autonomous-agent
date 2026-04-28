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
7. [Configuration](#configuration)
8. [Debugging Tips](#debugging-tips)

---

## Overview

The agent operates in two layers:

1. **Outer planning layer** — Plan → Critique → Refine. Before any action is taken, the agent decomposes the user's request into an ordered set of steps, has a reviewer surface gaps, and lets the user iterate on the plan until they approve.
2. **Inner execution layer** — Think → Act → Observe. For each approved step, a classic agent loop selects a tool, executes it, observes the result, and repeats until the step's success criterion is met.

A final **Critic** pass reviews the aggregated result.

### Key Design Decisions

- **Planning before action** — anticipate implicit requirements (data, training, README, examples) that the user usually wants but doesn't say.
- **Single LLM, many call types** — same model is used for planning, critiquing, refinement, the inner T-A-O loop, and final critic review (different prompts and `max_tokens` budgets).
- **Stateful execution** — `AgentState` carries history, the active plan, and the current step across all iterations of one `run()`.
- **Tool-based** — all environment interactions go through the `Tools` class.
- **Configurable** — every LLM call's `max_tokens` and the planning behavior itself are env-driven.

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         AutonomousAgent                            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      run(task)                               │  │
│  │                                                              │  │
│  │   ┌──────────────────────────────────────────────────────┐   │  │
│  │   │          _plan_refinement_loop(task)                 │   │  │
│  │   │                                                      │   │  │
│  │   │   _initial_plan ─▶ _critique_plan ─▶ _present_plan   │   │  │
│  │   │         ▲                                  │         │   │  │
│  │   │         └────── _refine_plan ◀─────────────┘         │   │  │
│  │   │                  (until user types 'go')             │   │  │
│  │   └──────────────────────────────────────────────────────┘   │  │
│  │                              │                               │  │
│  │                              ▼                               │  │
│  │   ┌──────────────────────────────────────────────────────┐   │  │
│  │   │          _execute_plan(task, plan)                   │   │  │
│  │   │   for each step:                                     │   │  │
│  │   │     state.current_step = step                        │   │  │
│  │   │     _run_tao_loop(max_iterations)                    │   │  │
│  │   │       ┌───────┐  ┌───────┐  ┌─────────┐              │   │  │
│  │   │       │ THINK │→ │  ACT  │→ │ OBSERVE │ (repeat)     │   │  │
│  │   │       └───────┘  └───────┘  └─────────┘              │   │  │
│  │   └──────────────────────────────────────────────────────┘   │  │
│  │                              │                               │  │
│  │                              ▼                               │  │
│  │   ┌──────────────────────────────────────────────────────┐   │  │
│  │   │          _critic_review(result)                      │   │  │
│  │   └──────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                              Tools                                 │
│  ┌─────────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────┐  │
│  │execute_shell│  │ read_file │  │ write_file │  │list_directory│  │
│  └─────────────┘  └───────────┘  └────────────┘  └──────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

When `AGENT_PLANNING_ENABLED=false`, `run()` skips the planning layer and calls `_legacy_execute(task)`, which runs `_run_tao_loop` directly with no plan/step context.

---

## File Structure

```
agent/
├── src/
│   ├── agent.py        # Agent implementation
│   └── test_agent.py   # Test scripts
├── config/
│   ├── .env            # Local config (not committed)
│   └── .env.example    # Configuration template
├── GUIDE.md            # This file
├── README.md           # Quick start
└── progress.md         # Roadmap
```

### Key Classes and Their Roles

| Class/File         | Purpose                       | Key Methods                                                                                                                         |
|--------------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `AutonomousAgent`  | Main agent controller         | `run`, `_plan_refinement_loop`, `_initial_plan`, `_critique_plan`, `_refine_plan`, `_execute_plan`, `_run_tao_loop`, `_critic_review` |
| `Tools`            | Tool implementations          | `execute_shell`, `read_file`, `write_file`, `list_directory`                                                                        |
| `AgentState`       | State container (dataclass)   | fields: `task`, history lists, `plan`, `current_step`                                                                               |
| `ToolResult`       | Result container (dataclass)  | fields: `success`, `output`, `error`                                                                                                |

---

## Execution Flow

Trace through a complete run for: **"build a small NN from scratch"**.

```
1. agent.run("build a small NN from scratch")
   │
   ▼
2. Planning layer enabled → _plan_refinement_loop(task)
   │
   ├── _initial_plan ............ LLM #1 emits {"summary":..., "steps":[...]}
   ├── _critique_plan ........... LLM #2 emits {"suggestions":[...]}
   ├── _present_plan_to_user .... prints plan + suggestions, reads user input
   │     │
   │     ├── input == "go"/"ok"/"" → break (APPROVED)
   │     ├── input == "apply 1,2" → _refine_plan with that text
   │     └── free-form text       → _refine_plan with that text
   │
   └── loops up to AGENT_MAX_PLAN_REFINEMENTS rounds
   │
   ▼
3. _execute_plan(task, plan)
   │
   for each step in plan["steps"]:
       state.current_step = step
       _run_tao_loop(max_iterations):
         ┌──────────────────────────────────────┐
         │ Iteration N:                         │
         │   THINK   build prompt (incl. step)  │
         │   ACT     parse JSON, run tool       │
         │   OBSERVE store result in state      │
         │   exit when step marked complete     │
         └──────────────────────────────────────┘
       collect step result string
   │
   ▼
4. _critic_review(aggregated_result)
   │
   ▼
5. return result (+ critic feedback if not approved)
```

`_build_prompt` injects a `--- Current Step ---` section into the inner T-A-O prompt while `state.current_step` is set, so the LLM keeps the step's `description` and `success_criterion` in mind.

---

## Component Deep Dive

### 1. AutonomousAgent — outer controller

#### Key Attributes

```python
self.client                      # OpenAI-compatible client
self.model                       # e.g. "gpt-4"
self.max_iterations              # T-A-O iteration cap per step
self.planning_enabled            # toggle the plan loop
self.max_plan_refinements        # cap on user refinement rounds
self.max_tokens_tao              # T-A-O LLM budget
self.max_tokens_critic           # final-critic LLM budget
self.max_tokens_plan_initial     # plan generation budget
self.max_tokens_plan_critique    # plan critique budget
self.max_tokens_plan_refine      # plan refinement budget
self.state                       # AgentState
self.tools                       # Tools
```

#### Planning methods

| Method                       | What it does                                                               |
|------------------------------|----------------------------------------------------------------------------|
| `_initial_plan(task)`        | LLM call → strict-JSON plan `{summary, steps[]}`. Falls back to a 1-step plan on parse failure. |
| `_critique_plan(task, plan)` | LLM call → list of `{issue, fix}` suggestions; empty list is valid.        |
| `_refine_plan(task, plan, user_feedback, suggestions)` | LLM call → revised plan applying user feedback (and any referenced suggestions). |
| `_present_plan_to_user(plan, suggestions)` | Pretty-prints the plan and reads user input. Returns `"APPROVE"` sentinel for empty/`go`/`ok`/`yes`; returns the raw text otherwise (with `apply N` translated to a directive). |
| `_plan_refinement_loop(task)`| Orchestrates the above until user approves or the refinement cap is hit.   |
| `_execute_plan(task, plan)`  | For each step: sets `state.current_step`, calls `_run_tao_loop`, accumulates the per-step result string. |

#### Inner-loop methods

| Method                 | What it does                                                                                         |
|------------------------|------------------------------------------------------------------------------------------------------|
| `_run_tao_loop(max)`   | Runs Think → Act → Observe iterations until `is_complete` or the cap. Reusable across plan steps and the legacy path. Resets per-step counters but **preserves history** so later steps see earlier observations. |
| `_legacy_execute(task)`| No-plan path: instantiates fresh state and calls `_run_tao_loop` directly.                           |
| `_build_prompt()`      | Builds the prompt for an inner-loop iteration. Includes `--- Current Step ---` when running inside a plan. |
| `_parse_response(text)`| Extracts the first balanced JSON object from the LLM response (3 fallback strategies).               |
| `_safe_json_parse(t)`  | Stricter helper used by planning methods (handles `\`\`\`json fences). Returns `None` on failure.    |
| `_execute_tool(name, params)` | Routes to a method on `Tools`.                                                                |
| `_critic_review(result)`| Final-pass quality review. Returns `"APPROVED"` or feedback.                                        |
| `_hash_action(...)`    | Used by the inner loop to detect repeated tool calls and break stuck-state cycles.                    |

### 2. Tools

| Tool             | Parameters                       | Example                                              |
|------------------|----------------------------------|------------------------------------------------------|
| `execute_shell`  | `command: str`                   | `{"command": "ls -la"}`                              |
| `read_file`      | `path: str`                      | `{"path": "config.txt"}`                             |
| `write_file`     | `path: str, content: str`        | `{"path": "out.txt", "content": "Hello"}`            |
| `list_directory` | `path: str` (default `"."`)      | `{"path": "."}`                                      |

Absolute paths are blocked for file I/O. Shell runs with the current user's permissions.

### 3. AgentState

```python
@dataclass
class AgentState:
    task: str = ""
    thought_history: List[str]
    action_history: List[Dict]
    observation_history: List[str]
    iteration_count: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None         # active plan during run()
    current_step: Optional[Dict[str, Any]] = None # the step being executed
```

### 4. ToolResult

```python
@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None
```

---

## Extending the Agent

### Adding a New Tool

1. Add a static method on `Tools`:
   ```python
   @staticmethod
   def http_get(url: str) -> ToolResult:
       import requests
       try:
           r = requests.get(url, timeout=30)
           return ToolResult(success=r.ok, output=r.text)
       except Exception as e:
           return ToolResult(success=False, output="", error=str(e))
   ```
2. Update `TOOLS_PROMPT`:
   ```
   5. http_get: Fetch a URL. Parameters: {"url": "string"}
   ```
3. Add a route in `_execute_tool`:
   ```python
   elif tool_name == "http_get":
       return self.tools.http_get(parameters.get("url", ""))
   ```

### Adding a Planning Phase Hook

If you want to inject extra context into planning (e.g., past lessons or project conventions), modify `_initial_plan` to prepend additional context before the user request, or wrap `_plan_refinement_loop` to mutate the plan before user presentation.

### Adding Cross-Session Memory

The current agent has no persistent memory between sessions. To add it:

1. Create `src/memory.py` with a `LessonStore` (load/save JSON, search by keyword).
2. Instantiate it in `__init__` from `AGENT_MEMORY_PATH`.
3. Inject relevant lessons into the planning prompt (in `_initial_plan`) or the T-A-O prompt (in `_build_prompt`).
4. Add a reflection pass after `_critic_review` that extracts new lessons from the completed task.

This is on the roadmap (see [progress.md](progress.md) Phase 2).

### Adding Multi-Agent Support

For specialized agents (planner / executor / critic as separate models or prompts), subclass `AutonomousAgent` per role and write an orchestrator that wires them together. The current planner/critic are already separate LLM calls inside one class — they're a natural extraction point.

---

## Configuration

### Environment Variables

| Variable                            | Purpose                                              | Default               |
|-------------------------------------|------------------------------------------------------|-----------------------|
| `OPENAI_API_BASE`                   | API endpoint                                         | —                     |
| `OPENAI_API_KEY`                    | Authentication key                                   | —                     |
| `OPENAI_MODEL`                      | Model to use                                         | `gpt-4`               |
| `MAX_ITERATIONS`                    | T-A-O iteration cap per step                         | `10`                  |
| `DEBUG`                             | Verbose logging flag                                 | `true`                |
| `AGENT_PLANNING_ENABLED`            | Run plan→critique→refine before execution            | `true`                |
| `AGENT_MAX_PLAN_REFINEMENTS`        | User refinement rounds before forcing a go/cancel    | `5`                   |
| `AGENT_MAX_TOKENS_TAO`              | Max tokens for inner Think/Act/Observe LLM call      | `15000`               |
| `AGENT_MAX_TOKENS_CRITIC`           | Max tokens for final critic LLM call                 | `5000`                |
| `AGENT_MAX_TOKENS_PLAN_INITIAL`     | Max tokens for initial plan generation               | `10000`               |
| `AGENT_MAX_TOKENS_PLAN_CRITIQUE`    | Max tokens for plan critique                         | `15000`               |
| `AGENT_MAX_TOKENS_PLAN_REFINE`      | Max tokens for plan revision                         | `15000`               |

### Loading Configuration

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, "config", ".env")
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4")
```

---

## Debugging Tips

### Disable planning for fast smoke tests

```bash
AGENT_PLANNING_ENABLED=false python src/agent.py "list files"
```

This skips the plan/critique/refine layer and runs the legacy T-A-O loop directly — useful when iterating on tool behavior without paying for planning calls.

### Test individual planning components

```python
agent = AutonomousAgent()
plan = agent._initial_plan("build a small NN from scratch")
suggestions = agent._critique_plan("build a small NN from scratch", plan)
refined = agent._refine_plan(
    "build a small NN from scratch", plan,
    user_feedback="also add a step to write a README",
    suggestions=suggestions,
)
print(plan, suggestions, refined)
```

### Test response parsing

```python
agent = AutonomousAgent()
print(agent._parse_response('{"tool": "list_directory", "parameters": {"path": "."}}'))
print(agent._safe_json_parse('```json\n{"summary": "X", "steps": []}\n```'))
```

### Tune `max_tokens` per loop

If you see `⚠️  Tool-call JSON appears truncated; nudging LLM to retry differently.` in the inner loop, raise `AGENT_MAX_TOKENS_TAO` in `.env` (this is what allows large `write_file` payloads to fit). Conversely, lower the planning budgets if your plans are short and you want cheaper / faster runs.

---

## Summary

The agent is a two-layer, modular system:

1. **Plan** the work (with critique + user refinement) before doing anything.
2. **Execute** each step with a Think → Act → Observe loop using the available tools.
3. **Critic** reviews the aggregated result.

Entry points to read first:

- `run()` — outer orchestrator.
- `_plan_refinement_loop()` — the user-facing planning UX.
- `_run_tao_loop()` — the per-step execution loop.
- `_build_prompt()` — how step context is fed back into the inner loop.

This architecture extends naturally to memory, multi-agent setups, and richer toolsets.
