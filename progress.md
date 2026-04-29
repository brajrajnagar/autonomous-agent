# Agent Development Progress

## Current Status: Phase 1.5 — Plan → Critique → Refine → Execute orchestration

### ✅ Completed

1. **Virtual Environment Setup**
   - Created `venv` directory
   - Python 3.x virtual environment ready

2. **Project Structure**
   ```
   agent/
   ├── src/          # Source code (agent.py, test_agent.py)
   ├── config/       # Configuration files (.env, .env.example)
   ├── tests/        # Reserved for future tests
   ├── venv/         # Virtual environment
   ├── GUIDE.md      # Architecture & extension guide
   ├── README.md     # Quick start
   └── progress.md   # This file
   ```

---

## ✅ Phase 1 COMPLETE — Single Agent + T-A-O Loop

To use the agent:

1. Copy `config/.env.example` to `config/.env`
2. Fill in your OpenAI-compatible API credentials
3. Run: `source venv/bin/activate && python src/agent.py "your task"`

### Phase 1 deliverables

- [x] Install dependencies (openai, python-dotenv)
- [x] Create configuration file for OpenAI-compatible API
- [x] Build core agent loop (think → act → observe)
- [x] Implement basic tools:
  - [x] Shell command execution
  - [x] File read/write operations
  - [x] Directory listing
- [x] Add critic/reviewer second pass
- [x] Document usage in README.md
- [x] Create test script
- [x] Add .gitignore

---

## ✅ Phase 1.5 COMPLETE — Planning Loop

The agent now plans before it acts: it decomposes the user's request into ordered steps, has a reviewer surface gaps, lets the user iterate on the plan, then executes step-by-step.

### Phase 1.5 deliverables

- [x] `_initial_plan` — LLM call producing structured `{summary, steps[]}` JSON
- [x] `_critique_plan` — reviewer LLM surfacing `{issue, fix}` suggestions
- [x] `_refine_plan` — applies user feedback (with `apply N` shortcut) to revise the plan
- [x] `_present_plan_to_user` — formatted plan + suggestions, reads user input
- [x] `_plan_refinement_loop` — orchestrates plan → critique → user → refine until approved
- [x] `_run_tao_loop` — extracted reusable Think/Act/Observe loop
- [x] `_execute_plan` — runs T-A-O per step with `current_step` injected into prompt
- [x] `_legacy_execute` — no-plan fallback path when `AGENT_PLANNING_ENABLED=false`
- [x] `AgentState.plan` and `AgentState.current_step` fields added
- [x] Per-loop `max_tokens` env vars (TAO, CRITIC, PLAN_INITIAL, PLAN_CRITIQUE, PLAN_REFINE)
- [x] Truncated-JSON detection in inner loop with corrective feedback to LLM
- [x] Tests added: plan decomposition, critique gaps, refinement applies feedback, legacy fallback
- [x] README, GUIDE updated to document the planning loop

---

## 📋 Active Work

_No active items. Ready for Phase 2._

---

## 🔮 Future Phases

### Phase 2: Memory System
- [ ] Cross-session lesson memory (`LessonStore` in `src/memory.py`)
- [ ] Inject relevant lessons into planning prompts
- [ ] Reflection pass after critic to extract new lessons
- [ ] CLI commands: `lessons`, `forget <id>`, `/learn-off`
- [ ] (Later) vector-based retrieval (ChromaDB) once lesson count grows

### Phase 3: Multi-Agent Orchestration
- [ ] Extract Planner / Critic / Executor into separate classes
- [ ] Implement agent communication protocol
- [ ] Task routing and coordination

### Phase 4: Advanced Tools
- [ ] Web browsing/scraping capabilities
- [ ] API integration tools
- [ ] Code execution sandbox
- [ ] Database query tools

### Phase 5: Production Readiness
- [ ] Docker sandboxing for security
- [ ] Comprehensive logging and monitoring
- [ ] Error handling and retry mechanisms
- [ ] Human-in-the-loop approvals beyond plan-time
- [ ] API endpoint for remote access

### Phase 6: Advanced Features
- [ ] Mid-execution replanning when steps fail repeatedly
- [ ] Pattern recognition across past tasks
- [ ] Autonomous task prioritization
- [ ] Multi-session context persistence (depends on Phase 2)

---

## ✅ Phase 2 COMPLETE — Context Compression & Memory Management

The agent now manages long-running session context to prevent prompt explosion:

### Phase 2 Deliverables

- [x] **Rolling Window Compression** (`src/context.py`)
  - Keeps only last N turns in full detail (default: 6)
  - Older messages replaced with deterministic running summary
  - Triggers when prompt exceeds `TOKEN_BUDGET * TRIGGER_RATIO`

- [x] **Argument Stubbing**
  - Large tool_call arguments (>300 chars) stubbed in old messages
  - Replaced with: `<N chars; included in earlier turn>`
  - Preserves tool_call integrity

- [x] **Step-Boundary Compaction**
  - After each plan step completes, collapse iteration history
  - Replaces all step messages with single summary line
  - Free, deterministic, no LLM call

- [x] **Tool Output Capping**
  - `read_file`: New `mode='head'|'tail'|'slice'` with offset/length
  - `execute_shell`: Head+tail clipping preserves errors at bottom
  - Default cap: 6000 chars per tool result

- [x] **Configuration** (via `.env`)
  - `AGENT_CONTEXT_COMPRESSION_ENABLED=true`
  - `AGENT_CONTEXT_TOKEN_BUDGET=24000`
  - `AGENT_CONTEXT_TRIGGER_RATIO=0.75`
  - `AGENT_CONTEXT_RECENT_TURNS=6`
  - `AGENT_TOOL_OUTPUT_CAP=6000`

---

## ✅ Phase 3 COMPLETE — Mid-Execution Replanning

When a step hits its iteration cap without completing, the planner now chooses a recovery action:

### Phase 3 Deliverables

- [x] **Recovery Actions**
  - `retry`: Same step, fresh iteration budget
  - `revise_step`: Update description/success_criterion
  - `revise_plan`: Replace this step + all subsequent steps
  - `skip`: Mark failed, continue to next step
  - `abort`: Stop with partial results

- [x] **Replan Budgets**
  - `AGENT_MAX_REPLANS_PER_STEP=2` (per-step limit)
  - `AGENT_MAX_REPLANS_PER_RUN=5` (per-run limit)
  - Interactive mode allows user override

- [x] **Execution Flow Rewrite**
  - `execute_plan()` uses cursor-based iteration
  - Nested retry loop per step
  - Tracks `replans_this_step` and `replans_this_run`

- [x] **Testing**
  - 6 new tests: retry→succeed, revise_step, per-step cap, abort, skip, global cap

---

## 📝 Notes

- Using OpenAI-compatible API (provider-swappable).
- Single LLM is reused for planning, critique, refinement, T-A-O, replanning, and final critic — different prompts and `max_tokens` budgets per call.
- Each LLM call's `max_tokens` is configurable from `.env` to tune for cost vs. truncation risk.
- Starting with simple venv; Docker sandbox is Phase 5.
- Planning is on by default; set `AGENT_PLANNING_ENABLED=false` for trivial one-shot tasks.
- Context compression is on by default; set `AGENT_CONTEXT_COMPRESSION_ENABLED=false` to disable.
- Replanning is on by default; set `AGENT_REPLANNING_ENABLED=false` to skip recovery.

---

*Last Updated: Phase 3 — Context compression + Mid-execution replanning shipped*
