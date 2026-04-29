"""
Simple test script for the Autonomous Agent

This script tests the agent with a simple task to verify it's working correctly.
"""

import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import AutonomousAgent
from src.tools import Tools
from src.feedback import (
    FeedbackEngine, PythonSyntaxVerifier, JsonSyntaxVerifier,
    make_feedback_engine,
)


def test_list_directory():
    """Test: List current directory"""
    print("\n" + "="*60)
    print("TEST 1: List Directory")
    print("="*60)
    
    agent = AutonomousAgent()
    result = agent.run("List all files in the current directory")
    
    print("\nResult:", result[:500] if len(result) > 500 else result)
    return result


def test_create_file():
    """Test: Create a simple file"""
    print("\n" + "="*60)
    print("TEST 2: Create File")
    print("="*60)
    
    agent = AutonomousAgent()
    result = agent.run("Create a file called 'test_output.txt' with the content 'Agent test successful!'")
    
    print("\nResult:", result[:500] if len(result) > 500 else result)
    
    # Verify file was created
    if os.path.exists("test_output.txt"):
        with open("test_output.txt", "r") as f:
            content = f.read()
        print(f"\n✅ File verified! Content: {content}")
        os.remove("test_output.txt")  # Cleanup
        print("🗑️  Cleaned up test file")
    else:
        print("\n⚠️  File was not created")
    
    return result


def test_shell_command():
    """Test: Execute a shell command"""
    print("\n" + "="*60)
    print("TEST 3: Shell Command")
    print("="*60)
    
    agent = AutonomousAgent()
    result = agent.run("Run the command 'pwd' and tell me the current working directory")
    
    print("\nResult:", result[:500] if len(result) > 500 else result)
    return result


def test_initial_plan_decomposes_task():
    """Test: _initial_plan produces a multi-step decomposition that anticipates implicit work."""
    print("\n" + "="*60)
    print("TEST 4: Initial plan decomposition")
    print("="*60)

    agent = AutonomousAgent()
    plan = agent.planner.initial_plan("build a small NN from scratch with backprop")
    print(f"\nPlan summary: {plan.get('summary', '')}")
    for step in plan.get("steps", []):
        print(f"  {step['id']}. {step['description']}")

    assert isinstance(plan, dict) and "steps" in plan, "Plan must be a dict with 'steps'"
    assert len(plan["steps"]) >= 2, f"Expected multi-step plan, got {len(plan['steps'])} step(s)"
    joined = " ".join(s.get("description", "").lower() for s in plan["steps"])
    assert any(kw in joined for kw in ("data", "train", "readme")), \
        "Plan should anticipate at least one of: data loading, training, README"
    print("\n✅ Plan is multi-step and anticipates implicit work")
    return plan


def test_plan_critique_returns_suggestions():
    """Test: _critique_plan flags gaps in an obviously-incomplete plan."""
    print("\n" + "="*60)
    print("TEST 5: Plan critique surfaces gaps")
    print("="*60)

    agent = AutonomousAgent()
    skeletal_plan = {
        "summary": "Build a model",
        "steps": [
            {"id": 1, "description": "Write model code", "success_criterion": "File exists"}
        ],
    }
    suggestions = agent.planner.critique_plan("build a small NN from scratch with backprop", skeletal_plan)
    print(f"\nSuggestions returned: {len(suggestions)}")
    for s in suggestions:
        print(f"  - {s.get('issue', '')} → {s.get('fix', '')}")

    assert len(suggestions) >= 1, "Critic should flag at least one gap in a 1-step plan"
    print("\n✅ Critic flagged at least one gap")
    return suggestions


def test_plan_refinement_applies_user_feedback():
    """Test: _refine_plan grows the plan when the user asks for an extra step."""
    print("\n" + "="*60)
    print("TEST 6: Plan refinement applies user feedback")
    print("="*60)

    agent = AutonomousAgent()
    base_plan = {
        "summary": "Build a model",
        "steps": [
            {"id": 1, "description": "Create project directory", "success_criterion": "Directory exists"},
            {"id": 2, "description": "Write model code", "success_criterion": "File exists"},
            {"id": 3, "description": "Train the model", "success_criterion": "Training runs"},
        ],
    }
    refined = agent.planner.refine_plan(
        "build a small NN from scratch with backprop",
        base_plan,
        "also add a step to write a README with run instructions",
        suggestions=[],
    )
    print(f"\nRefined plan now has {len(refined.get('steps', []))} steps:")
    for step in refined.get("steps", []):
        print(f"  {step['id']}. {step['description']}")

    assert len(refined.get("steps", [])) >= len(base_plan["steps"]), \
        "Refinement should add (not lose) steps"
    joined = " ".join(s.get("description", "").lower() for s in refined["steps"])
    assert "readme" in joined, "Refined plan should include a README step"
    print("\n✅ Refinement added a README step")
    return refined


def test_planning_disabled_falls_back_to_legacy():
    """Test: with AGENT_PLANNING_ENABLED=false, run() skips the plan loop."""
    print("\n" + "="*60)
    print("TEST 7: Planning disabled → legacy execution path")
    print("="*60)

    os.environ["AGENT_PLANNING_ENABLED"] = "false"
    try:
        agent = AutonomousAgent()
        assert agent.planning_enabled is False, "planning_enabled should reflect env var"
        result = agent.run("List all files in the current directory")
        print("\nResult:", result[:300] if len(result) > 300 else result)
        assert isinstance(result, str) and len(result) > 0, "Legacy path must return a result"
        print("\n✅ Legacy path completed without plan loop")
    finally:
        os.environ.pop("AGENT_PLANNING_ENABLED", None)
    return result


def test_search_in_files():
    """Test: search_in_files finds a known string in this project (no API call)."""
    print("\n" + "="*60)
    print("TEST 8: search_in_files")
    print("="*60)
    # Tool requires relative paths — run from the agent project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old_cwd = os.getcwd()
    os.chdir(project_root)
    try:
        # `Tools` class definition itself contains "class Tools" — a stable target.
        result = Tools.search_in_files(r"class Tools", path="src", file_glob="*.py")
        print("\n", result.output[:300])
        assert result.success, f"search failed: {result.error}"
        assert "tools.py" in result.output, "expected tools.py in matches"
    finally:
        os.chdir(old_cwd)
    print("\n✅ search_in_files found 'class Tools' in tools.py")


def test_edit_file():
    """Test: edit_file replaces a unique string and rejects non-unique matches."""
    print("\n" + "="*60)
    print("TEST 9: edit_file")
    print("="*60)
    with tempfile.TemporaryDirectory() as td:
        # tempfile gives an absolute path, but tools require relative — chdir into td.
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            with open("sample.txt", "w") as f:
                f.write("hello world\nhello again\n")

            # Unique-match path: edit "hello world" → "hi world".
            r = Tools.edit_file("sample.txt", "hello world", "hi world")
            assert r.success, f"unique edit failed: {r.error}"
            with open("sample.txt") as f:
                content = f.read()
            assert content == "hi world\nhello again\n", f"unexpected content: {content!r}"

            # Non-unique-match path: "hello" appears twice now ("hi world" + "hello again")
            # — actually only once. Let's make it twice deliberately.
            with open("sample.txt", "w") as f:
                f.write("foo\nfoo\n")
            r2 = Tools.edit_file("sample.txt", "foo", "bar")
            assert not r2.success, "non-unique edit should fail"
            assert "appears" in r2.error.lower() or "more" in r2.error.lower(), \
                f"unexpected error: {r2.error}"

            # Missing-string path.
            r3 = Tools.edit_file("sample.txt", "nothing-here", "x")
            assert not r3.success
            assert "not found" in r3.error.lower()
        finally:
            os.chdir(old_cwd)
    print("\n✅ edit_file: unique-match works, non-unique and missing both rejected")


def test_run_python():
    """Test: run_python executes inline code and a script file."""
    print("\n" + "="*60)
    print("TEST 10: run_python")
    print("="*60)
    # Inline code path.
    r = Tools.run_python(code="print(2 + 2)")
    assert r.success, f"inline run failed: {r.error}"
    assert "4" in r.output, f"expected '4' in output, got {r.output!r}"

    # Script path.
    with tempfile.TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            with open("hello.py", "w") as f:
                f.write("print('script ran')\n")
            r2 = Tools.run_python(script_path="hello.py")
            assert r2.success, f"script run failed: {r2.error}"
            assert "script ran" in r2.output

            # Non-zero exit captures stderr separator and exit code.
            with open("boom.py", "w") as f:
                f.write("raise SystemExit(2)\n")
            r3 = Tools.run_python(script_path="boom.py")
            assert not r3.success
            assert "code 2" in (r3.error or "")
        finally:
            os.chdir(old_cwd)

    # Both/neither validation paths.
    r4 = Tools.run_python()
    assert not r4.success and "Provide" in r4.error
    r5 = Tools.run_python(code="x", script_path="y.py")
    assert not r5.success and "only one" in r5.error
    print("\n✅ run_python: inline + script + non-zero + arg validation all pass")


def test_feedback_python_syntax_pass_silent():
    """Test: PythonSyntaxVerifier passes silently on valid Python (no observation noise)."""
    print("\n" + "="*60)
    print("TEST 11: feedback — Python syntax pass is silent")
    print("="*60)
    with tempfile.TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            with open("ok.py", "w") as f:
                f.write("def foo():\n    return 42\n")
            engine = FeedbackEngine([PythonSyntaxVerifier()])
            extra = engine.run_after("write_file", {"path": "ok.py"}, None)
            assert extra == "", f"expected silent pass, got: {extra!r}"
        finally:
            os.chdir(old_cwd)
    print("\n✅ Valid Python produces no observation noise on pass")


def test_feedback_python_syntax_fail():
    """Test: PythonSyntaxVerifier surfaces syntax errors with line info."""
    print("\n" + "="*60)
    print("TEST 12: feedback — Python syntax error caught")
    print("="*60)
    with tempfile.TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            with open("broken.py", "w") as f:
                f.write("def foo(:\n    return 42\n")  # invalid syntax
            engine = FeedbackEngine([PythonSyntaxVerifier()])
            extra = engine.run_after("write_file", {"path": "broken.py"}, None)
            print("\n", extra)
            assert extra, "expected non-empty feedback for broken file"
            assert "[verify] py_syntax:" in extra
            assert "broken.py" in extra
        finally:
            os.chdir(old_cwd)
    print("\n✅ Broken Python produces an actionable [verify] line")


def test_feedback_json_syntax_fail():
    """Test: JsonSyntaxVerifier catches malformed JSON."""
    print("\n" + "="*60)
    print("TEST 13: feedback — JSON syntax error caught")
    print("="*60)
    with tempfile.TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            with open("bad.json", "w") as f:
                f.write('{"key": "missing close')
            engine = FeedbackEngine([JsonSyntaxVerifier()])
            extra = engine.run_after("write_file", {"path": "bad.json"}, None)
            assert "[verify] json_syntax:" in extra, f"got: {extra!r}"
            assert "bad.json" in extra
        finally:
            os.chdir(old_cwd)
    print("\n✅ Broken JSON produces an actionable [verify] line")


def test_feedback_disabled_returns_none():
    """Test: AGENT_FEEDBACK_ENABLED=false skips the engine entirely."""
    print("\n" + "="*60)
    print("TEST 14: feedback — disabled via env var")
    print("="*60)
    os.environ["AGENT_FEEDBACK_ENABLED"] = "false"
    try:
        engine = make_feedback_engine()
        assert engine is None, f"expected None, got {engine}"
    finally:
        os.environ.pop("AGENT_FEEDBACK_ENABLED", None)
    print("\n✅ Disabled feedback returns None")


def test_browser_invalid_url_rejected():
    """Test: visit_url rejects non-http URLs and empty strings (no network)."""
    print("\n" + "="*60)
    print("TEST 16: browser_visit — invalid URL rejected")
    print("="*60)
    from src.browser import visit_url
    for bad in ("", "not-a-url", "ftp://example.com", "javascript:alert(1)"):
        r = visit_url(bad)
        assert not r.success, f"expected rejection for {bad!r}, got success"
        assert "http" in (r.error or "").lower()
    print("\n✅ Invalid URLs rejected before any network call")


def test_browser_deny_host_blocks_localhost():
    """Test: AGENT_BROWSER_DENY_HOSTS blocks matching hostnames pre-fetch."""
    print("\n" + "="*60)
    print("TEST 17: browser_visit — deny-host blocks localhost")
    print("="*60)
    from src.browser import visit_url
    os.environ["AGENT_BROWSER_DENY_HOSTS"] = "localhost,127.0.0.1,internal.corp"
    try:
        r1 = visit_url("http://localhost:8080/admin")
        assert not r1.success and "deny list" in r1.error
        r2 = visit_url("http://api.internal.corp/secret")
        assert not r2.success and "deny list" in r2.error
        # Non-matching host should not be blocked by the env var (will then
        # fail for some *other* reason because we don't actually fetch here,
        # but pre-network rejection should not fire).
        # Use a clearly fake host so trafilatura/requests doesn't actually do
        # anything heavy — we just want to confirm the deny check returns None.
        from src.browser import _is_denied
        assert not _is_denied("https://example.com/")
    finally:
        os.environ.pop("AGENT_BROWSER_DENY_HOSTS", None)
    print("\n✅ Deny-host list blocks matching URLs pre-network")


def test_browser_pagination_via_offset():
    """Test: offset parameter slices the cached content with a 'use offset=X' footer."""
    print("\n" + "="*60)
    print("TEST 18: browser_visit — pagination via offset")
    print("="*60)
    from src import browser
    from src.browser import visit_url, _hash_url

    # Pre-populate cache to avoid any network call.
    fake_url = "https://example.com/big-doc"
    fake_content = ("ABC123" * 2000)  # 12000 chars
    browser._URL_CACHE[_hash_url(fake_url)] = fake_content

    try:
        # First page: offset 0, max 100.
        r1 = visit_url(fake_url, max_chars=100, offset=0)
        assert r1.success
        assert "[browser_visit]" in r1.output and "fake" not in r1.output
        assert "use offset=100" in r1.output.lower(), f"missing pagination hint: {r1.output[-200:]!r}"

        # Mid-document.
        r2 = visit_url(fake_url, max_chars=100, offset=5000)
        assert r2.success
        assert "use offset=5100" in r2.output.lower()

        # Past the end.
        r3 = visit_url(fake_url, max_chars=100, offset=99999)
        assert r3.success
        assert "empty" in r3.output.lower()

        # Invalid offset.
        r4 = visit_url(fake_url, max_chars=100, offset=-1)
        assert not r4.success
        assert "offset" in r4.error.lower()
    finally:
        browser._URL_CACHE.pop(_hash_url(fake_url), None)
    print("\n✅ Pagination, end-of-content and invalid-offset all behave correctly")


def test_feedback_skipped_for_non_matching_tool():
    """Test: Verifiers whose applies_to() returns False produce no output."""
    print("\n" + "="*60)
    print("TEST 15: feedback — verifier skipped for unrelated tool calls")
    print("="*60)
    engine = FeedbackEngine([PythonSyntaxVerifier(), JsonSyntaxVerifier()])
    # list_directory and execute_shell don't match any verifier.
    assert engine.run_after("list_directory", {"path": "."}, None) == ""
    assert engine.run_after("execute_shell", {"command": "ls"}, None) == ""
    # A .txt write has no verifier applicable.
    with tempfile.TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            with open("notes.txt", "w") as f:
                f.write("hello")
            assert engine.run_after("write_file", {"path": "notes.txt"}, None) == ""
        finally:
            os.chdir(old_cwd)
    print("\n✅ Engine produces no output for non-matching tool/file combinations")


def test_is_approved_handles_negation():
    """Test: is_approved correctly rejects 'NOT APPROVED' and similar negations."""
    print("\n" + "="*60)
    print("TEST 19: is_approved handles 'NOT APPROVED' correctly")
    print("="*60)
    from src.critic import is_approved
    # Approved cases.
    assert is_approved("APPROVED")
    assert is_approved("approved")
    assert is_approved("Looks great. APPROVED.")
    assert is_approved("APPROVED (no feedback generated)")
    # Negated cases — these previously slipped through with substring matching.
    assert not is_approved("NOT APPROVED")
    assert not is_approved("Not approved — needs more work.")
    assert not is_approved("DISAPPROVED")
    assert not is_approved("This is NOT_APPROVED.")
    # Non-verdict text.
    assert not is_approved("")
    assert not is_approved(None)
    assert not is_approved("The task needs improvements.")
    print("\n✅ is_approved correctly distinguishes approval from negation")


def test_system_prefix_contains_today():
    """Test: system_prefix returns today's date in ISO form."""
    print("\n" + "="*60)
    print("TEST 20: system_prefix injects today's date")
    print("="*60)
    from src.prompts import system_prefix
    from datetime import datetime
    prefix = system_prefix()
    today = datetime.now().strftime("%Y-%m-%d")
    assert today in prefix, f"expected {today} in prefix, got: {prefix!r}"
    assert "fabricat" in prefix.lower() or "guess" in prefix.lower(), \
        "prefix should warn against fabrication/guessing"
    print(f"\n✅ system_prefix contains {today} and an anti-fabrication clause")


def test_planner_prompt_lists_browser_tools():
    """Test: INITIAL_PLAN_PROMPT mentions web_search/browser_visit so the planner can use them."""
    print("\n" + "="*60)
    print("TEST 21: planner prompt advertises web tools")
    print("="*60)
    from src.prompts import INITIAL_PLAN_PROMPT
    formatted = INITIAL_PLAN_PROMPT.format(task="anything")
    for tool in ("web_search", "browser_visit", "edit_file", "search_in_files", "run_python"):
        assert tool in formatted, f"INITIAL_PLAN_PROMPT missing {tool}"
    print("\n✅ Planner prompt lists web_search, browser_visit, and the dev tools")


def test_browser_blocked_content_detection():
    """Test: _looks_blocked flags JS-required / captcha / 403 short pages."""
    print("\n" + "="*60)
    print("TEST 22: browser — blocked / JS-required content detected")
    print("="*60)
    from src.browser import _looks_blocked
    # Should flag.
    assert _looks_blocked("You need to enable JavaScript to view this site.") == "you need to enable javascript"
    assert _looks_blocked("Access Denied — 403 Forbidden") in ("403 forbidden", "access denied")
    assert _looks_blocked("Please verify you are a human.\nCheck the box.") == "please verify you are a human"
    # Should NOT flag — long real article that incidentally mentions JS.
    long_article = ("Article on JavaScript security. " * 200) + "Check the captcha challenge."
    assert _looks_blocked(long_article) is None, \
        "long articles mentioning sentinels in passing should not be flagged"
    # Empty / short benign.
    assert _looks_blocked("") is None
    assert _looks_blocked("Hello world") is None
    print("\n✅ Blocked-content detection works on JS shells, 403s, and captchas; ignores long articles")


def test_planner_prompt_warns_against_unrequested_files():
    """Test: INITIAL_PLAN_PROMPT tells the planner not to write files unless asked."""
    print("\n" + "="*60)
    print("TEST 23: planner prompt — no unrequested files")
    print("="*60)
    from src.prompts import INITIAL_PLAN_PROMPT
    formatted = INITIAL_PLAN_PROMPT.format(task="anything")
    lower = formatted.lower()
    assert "do not" in lower and "file" in lower, "missing 'do not' / 'file' guidance"
    assert "explicit" in lower or "unless" in lower, \
        "should condition file-writing on explicit user request"
    print("\n✅ Plan prompt warns against writing files for Q&A tasks")


def test_triage_prompt_lists_three_buckets():
    """Test: TRIAGE_PROMPT defines simple/standard/complex with examples."""
    print("\n" + "="*60)
    print("TEST 24: triage prompt — 3 buckets defined")
    print("="*60)
    from src.prompts import TRIAGE_PROMPT
    formatted = TRIAGE_PROMPT.format(task="x")
    for label in ("simple", "standard", "complex"):
        assert label in formatted, f"TRIAGE_PROMPT missing bucket: {label}"
    assert "Examples:" in formatted, "buckets should have examples"
    print("\n✅ Triage prompt defines all three buckets with examples")


def test_autonomy_env_var_validation():
    """Test: AGENT_AUTONOMY accepts auto/interactive/silent, falls back to auto."""
    print("\n" + "="*60)
    print("TEST 25: AGENT_AUTONOMY env var validation")
    print("="*60)
    for valid in ("auto", "interactive", "silent"):
        os.environ["AGENT_AUTONOMY"] = valid
        try:
            a = AutonomousAgent()
            assert a.autonomy == valid, f"expected {valid}, got {a.autonomy}"
        finally:
            os.environ.pop("AGENT_AUTONOMY", None)
    # Garbage falls back to "auto".
    os.environ["AGENT_AUTONOMY"] = "garbage"
    try:
        a = AutonomousAgent()
        assert a.autonomy == "auto", f"garbage should fall back to auto, got {a.autonomy}"
    finally:
        os.environ.pop("AGENT_AUTONOMY", None)
    # Default is "auto".
    a = AutonomousAgent()
    assert a.autonomy == "auto"
    print("\n✅ AGENT_AUTONOMY accepts valid modes, defaults/falls back to 'auto'")


def test_tool_schemas_well_formed():
    """Test: every TOOL_DEFINITIONS entry has the OpenAI function-calling shape."""
    print("\n" + "="*60)
    print("TEST 26: tool schemas — all entries well-formed")
    print("="*60)
    from src.tool_schemas import TOOL_DEFINITIONS
    assert len(TOOL_DEFINITIONS) >= 10, f"expected 10+ tools, got {len(TOOL_DEFINITIONS)}"
    for entry in TOOL_DEFINITIONS:
        assert entry.get("type") == "function"
        fn = entry.get("function", {})
        assert "name" in fn and isinstance(fn["name"], str) and fn["name"]
        assert "description" in fn and len(fn["description"]) > 10
        params = fn.get("parameters", {})
        assert params.get("type") == "object"
        assert "properties" in params and isinstance(params["properties"], dict)
        # Every property must declare its type.
        for prop_name, prop_schema in params["properties"].items():
            assert "type" in prop_schema, f"{fn['name']}.{prop_name} missing type"
    print(f"\n✅ {len(TOOL_DEFINITIONS)} tool schemas all well-formed")


def test_complete_task_schema_present():
    """Test: synthetic complete_task tool exists with `answer` parameter."""
    print("\n" + "="*60)
    print("TEST 27: complete_task tool present")
    print("="*60)
    from src.tool_schemas import TOOL_DEFINITIONS, TOOL_NAMES
    assert "complete_task" in TOOL_NAMES
    ct = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "complete_task")
    params = ct["function"]["parameters"]
    assert "answer" in params["properties"]
    assert "answer" in params.get("required", [])
    assert params["properties"]["answer"]["type"] == "string"
    print("\n✅ complete_task schema requires a string `answer`")


def test_executor_dispatches_all_schema_tools():
    """Test: every non-synthetic tool in TOOL_DEFINITIONS is routed by Executor._execute_tool."""
    print("\n" + "="*60)
    print("TEST 28: Executor dispatch covers all schema tools")
    print("="*60)
    from src.tool_schemas import TOOL_NAMES
    from src.executor import Executor
    import inspect

    # Read the source of _execute_tool and check each tool name appears.
    source = inspect.getsource(Executor._execute_tool)
    expected = TOOL_NAMES - {"complete_task"}  # complete_task is handled in run_tao_loop
    missing = [n for n in expected if f'"{n}"' not in source]
    assert not missing, f"_execute_tool missing dispatch for: {missing}"
    print(f"\n✅ All {len(expected)} non-synthetic tools have a dispatch branch")


def test_initial_messages_includes_complete_task_instruction():
    """Test: initial system message tells the model to call complete_task to terminate."""
    print("\n" + "="*60)
    print("TEST 29: initial messages mention complete_task")
    print("="*60)
    # Construct an Executor with throwaway args; we're only calling _initial_messages.
    from src.executor import Executor
    # Skip full agent init — Executor only needs the args for execution paths,
    # not for _initial_messages which is pure-text.
    e = Executor(client=None, model="x", tools=None,
                 max_iterations=1, max_tokens_tao=1)
    msgs = e._initial_messages("write a haiku")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "complete_task" in msgs[0]["content"]
    assert "tools" in msgs[0]["content"].lower()
    assert msgs[1]["role"] == "user"
    assert "haiku" in msgs[1]["content"]
    print("\n✅ Initial messages have system + user shape and mention complete_task")


def test_xml_tool_call_fallback_parser():
    """Test: parse_xml_tool_calls extracts Qwen-style <tool_call> blocks."""
    print("\n" + "="*60)
    print("TEST 30: XML tool-call fallback parser")
    print("="*60)
    from src.parsing import parse_xml_tool_calls

    # Realistic Qwen3.5 output.
    text = """<tool_call>
<function=list_directory>
<parameter=path>
src
</parameter>
</function>
</tool_call>"""
    calls = parse_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "list_directory"
    assert calls[0]["arguments"] == {"path": "src"}

    # Multi-arg case.
    text2 = """<tool_call>
<function=write_file>
<parameter=path>foo.py</parameter>
<parameter=content>print('hi')</parameter>
</function>
</tool_call>"""
    calls2 = parse_xml_tool_calls(text2)
    assert len(calls2) == 1
    assert calls2[0]["name"] == "write_file"
    assert calls2[0]["arguments"]["path"] == "foo.py"
    assert "print" in calls2[0]["arguments"]["content"]

    # No tool calls — empty list.
    assert parse_xml_tool_calls("Just thinking out loud.") == []
    assert parse_xml_tool_calls("") == []
    print("\n✅ XML tool-call fallback parser handles single, multi-arg, and empty cases")


def _make_state(task="t", action_history=None, observation_history=None, plan=None):
    """Build a minimal AgentState-like for context tests."""
    from src.state import AgentState
    s = AgentState(task=task)
    if action_history:
        s.action_history = list(action_history)
    if observation_history:
        s.observation_history = list(observation_history)
    if plan:
        s.plan = plan
    return s


def test_context_token_estimator():
    """Test: token estimator uses len/4 and aggregates messages + tool_call args."""
    print("\n" + "="*60)
    print("TEST 31: ContextManager token estimator")
    print("="*60)
    from src.context import ContextManager
    cm = ContextManager(token_budget=1000)
    assert cm.estimate_tokens("") == 0
    assert cm.estimate_tokens("abcd") == 1
    assert cm.estimate_tokens("a" * 400) == 100

    # Tool-call arguments contribute to the total.
    msgs = [
        {"role": "system", "content": "x" * 400},  # 100 tokens
        {"role": "assistant", "content": None,
         "tool_calls": [{"function": {"arguments": '{"x": "' + ("y" * 392) + '"}'}}]},
    ]
    total = cm.estimate_messages_tokens(msgs)
    assert total >= 200, f"expected >=200 tokens, got {total}"
    print("\n✅ Token estimator counts content + tool_call arguments")


def test_context_no_compress_below_budget():
    """Test: maybe_compress is a no-op when prompt size is under threshold."""
    print("\n" + "="*60)
    print("TEST 32: no compression below budget")
    print("="*60)
    from src.context import ContextManager
    cm = ContextManager(token_budget=10000, trigger_ratio=0.75)
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "thinking"},
        {"role": "tool", "tool_call_id": "x", "content": "result"},
    ]
    state = _make_state(task="task")
    out = cm.maybe_compress(msgs, state)
    assert out is msgs, "should return the same list reference when not compressing"
    print("\n✅ No compression below budget threshold")


def test_context_compresses_above_budget():
    """Test: maybe_compress reduces token count and inserts a summary marker."""
    print("\n" + "="*60)
    print("TEST 33: compression triggers above budget")
    print("="*60)
    from src.context import ContextManager, SUMMARY_MARKER

    # Tiny budget so we trigger easily.
    cm = ContextManager(token_budget=200, trigger_ratio=0.5, recent_turns=2)

    # Build many turns with bulky content.
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "do the thing"}]
    for i in range(8):
        # Assistant message with a tool call
        msgs.append({
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "execute_shell",
                              "arguments": '{"command": "' + ("a" * 200) + '"}'},
            }],
        })
        # Matching tool result
        msgs.append({
            "role": "tool", "tool_call_id": f"call_{i}",
            "content": "tool output: " + ("b" * 200),
        })

    state = _make_state(
        task="do the thing",
        action_history=[
            {"tool": "execute_shell", "params": {"command": f"cmd-{i}"}}
            for i in range(8)
        ],
        observation_history=[f"output-{i}" for i in range(8)],
    )

    before = cm.estimate_messages_tokens(msgs)
    out = cm.maybe_compress(msgs, state)
    after = cm.estimate_messages_tokens(out)

    assert out is not msgs, "should return a new list when compression fires"
    assert after < before, f"compression should reduce tokens: before={before} after={after}"
    # Summary message should be present.
    summary_msgs = [m for m in out if SUMMARY_MARKER in (m.get("content") or "")]
    assert len(summary_msgs) == 1, f"expected one summary message, got {len(summary_msgs)}"
    # System message preserved.
    assert out[0].get("role") == "system"
    print(f"\n✅ Compression: {before} → {after} tokens; summary marker present")


def test_context_preserves_tool_call_integrity():
    """Test: compressor doesn't orphan tool messages from their assistant."""
    print("\n" + "="*60)
    print("TEST 34: tool-call integrity preserved after compression")
    print("="*60)
    from src.context import ContextManager
    cm = ContextManager(token_budget=200, trigger_ratio=0.5, recent_turns=2)

    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "task"}]
    for i in range(5):
        msgs.append({
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "x",
                              "arguments": '{"k": "' + ("v" * 300) + '"}'},
            }],
        })
        msgs.append({
            "role": "tool", "tool_call_id": f"call_{i}",
            "content": "obs " + ("o" * 300),
        })

    state = _make_state()
    out = cm.maybe_compress(msgs, state)

    # For every assistant message with tool_calls in the output, the matching
    # tool result must appear before the next assistant message.
    assert cm._is_tool_call_integrity_intact(out)
    print("\n✅ Compressed messages keep every tool_call paired with its result")


def test_context_arg_stubbing_for_old_writes():
    """Test: huge `content` in old write_file tool_calls gets stubbed."""
    print("\n" + "="*60)
    print("TEST 35: argument stubbing on old assistant messages")
    print("="*60)
    import json as _json
    from src.context import ContextManager
    cm = ContextManager(token_budget=200, trigger_ratio=0.5, recent_turns=1)

    big_content = "x" * 2000
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "task"},
        {  # OLD assistant message with huge write_file
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "call_old", "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": _json.dumps({"path": "foo.py", "content": big_content}),
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": "wrote 2000 chars"},
        # Recent assistant (within recent_turns=1)
        {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "call_new", "type": "function",
                "function": {
                    "name": "execute_shell",
                    "arguments": '{"command": "ls"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call_new", "content": "src/"},
    ]

    state = _make_state()
    out = cm.maybe_compress(msgs, state)

    # Find the recent assistant — its arguments should remain intact.
    recent = [m for m in out if m.get("role") == "assistant"
              and any(tc.get("id") == "call_new" for tc in (m.get("tool_calls") or []))]
    assert recent, "recent assistant message should be preserved"
    recent_args = recent[0]["tool_calls"][0]["function"]["arguments"]
    assert "ls" in recent_args, "recent args should stay unstubbed"

    # The old assistant's args should be stubbed (or the message dropped via summarization).
    old_in_output = [m for m in out if m.get("role") == "assistant"
                     and any(tc.get("id") == "call_old"
                             for tc in (m.get("tool_calls") or []))]
    if old_in_output:
        old_args = _json.loads(
            old_in_output[0]["tool_calls"][0]["function"]["arguments"])
        # content field should be a stub, not the original 2000 chars
        assert isinstance(old_args.get("content"), str)
        assert "chars" in old_args["content"] and len(old_args["content"]) < 100, \
            "old write_file content should be stubbed"
    print("\n✅ Old write_file args stubbed; recent args preserved verbatim")


def test_context_step_boundary_compaction():
    """Test: compact_step_boundary collapses a finished step's messages to one summary line."""
    print("\n" + "="*60)
    print("TEST 36: step-boundary compaction")
    print("="*60)
    from src.context import ContextManager
    cm = ContextManager(token_budget=10000)

    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Task: build something"},
        # Step 2 scope marker (the executor uses this exact prefix)
        {"role": "user", "content": "Now do Step 2: write the foo module\nSuccess criterion: foo.py exists"},
        # Some iterations within step 2
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "write_file",
                                       "arguments": '{"path":"foo.py","content":"..."}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "wrote 5 chars"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "complete_task",
                                       "arguments": '{"answer":"created foo.py"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "(step terminated)"},
    ]

    step = {"id": 2, "description": "write the foo module",
            "success_criterion": "foo.py exists"}
    out = cm.compact_step_boundary(msgs, _make_state(), step, "created foo.py")

    # Step messages collapsed to a single user summary.
    assert len(out) == 3, f"expected 3 messages after compaction, got {len(out)}"
    assert out[0]["role"] == "system"
    assert out[1]["content"].startswith("Task:")
    assert out[2]["role"] == "user"
    assert "[Step 2 complete]" in out[2]["content"]
    assert "created foo.py" in out[2]["content"]
    print("\n✅ Step-boundary compaction collapses iteration history to one summary")


def test_context_disabled_returns_input_unchanged():
    """Test: AGENT_CONTEXT_COMPRESSION_ENABLED=false yields a no-op manager."""
    print("\n" + "="*60)
    print("TEST 37: context manager disabled via env")
    print("="*60)
    import os
    from src.context import make_context_manager
    os.environ["AGENT_CONTEXT_COMPRESSION_ENABLED"] = "false"
    try:
        cm = make_context_manager()
        assert cm is None, f"expected None when disabled, got {cm}"
    finally:
        os.environ.pop("AGENT_CONTEXT_COMPRESSION_ENABLED", None)
    print("\n✅ Disabled context manager returns None")


def test_read_file_modes():
    """Test: read_file head/tail/slice modes return the right slice + pagination hint."""
    print("\n" + "="*60)
    print("TEST 38: read_file head/tail/slice modes")
    print("="*60)
    import tempfile
    from src.tools import Tools
    with tempfile.TemporaryDirectory() as td:
        old = os.getcwd()
        os.chdir(td)
        try:
            content = "".join(f"line{i:03d}\n" for i in range(500))  # 500 lines
            with open("big.txt", "w") as f:
                f.write(content)

            # head mode
            r1 = Tools.read_file("big.txt", mode="head", length=100)
            assert r1.success
            assert r1.output.startswith("line000")
            assert "Use mode='slice'" in r1.output  # pagination hint

            # tail mode
            r2 = Tools.read_file("big.txt", mode="tail", length=100)
            assert r2.success
            # Last bit should contain end-of-file lines
            assert "line499" in r2.output

            # slice mode
            r3 = Tools.read_file("big.txt", mode="slice", offset=100, length=100)
            assert r3.success
            assert "Use mode='slice'" in r3.output

            # invalid mode
            r4 = Tools.read_file("big.txt", mode="bogus")
            assert not r4.success and "mode" in r4.error.lower()
        finally:
            os.chdir(old)
    print("\n✅ read_file head/tail/slice all behave correctly")


def _build_executor_with_mocks(planner_decisions, run_results,
                                max_replans_per_step=2,
                                max_replans_per_run=5):
    """Build an Executor with a mocked planner and a mocked run_tao_loop.

    `planner_decisions` is a list of decision dicts the planner returns in order.
    `run_results` is a list of (state.is_complete, return_string) tuples; the
    Nth call to run_tao_loop pulls the Nth tuple.
    """
    from src.executor import Executor

    class _MockPlanner:
        def __init__(self, decisions):
            self._decisions = list(decisions)
            self.calls = []

        def replan(self, task, plan, failed_step, reason, recent_obs):
            self.calls.append({
                "task": task, "step_id": failed_step.get("id"), "reason": reason,
            })
            return self._decisions.pop(0) if self._decisions else {"action": "abort"}

    planner = _MockPlanner(planner_decisions)

    class _CountingExecutor(Executor):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._results = list(run_results)
            self.tao_calls: list[dict] = []

        def run_tao_loop(self, state, max_iters, step_id=None):
            ok, output = self._results.pop(0) if self._results else (False, "no result")
            self.tao_calls.append({"step_id": step_id, "ok": ok})
            state.is_complete = ok
            state.iteration_count = max_iters if not ok else 1
            if ok:
                state.final_answer = output
            return output

    return _CountingExecutor(
        client=None, model="x", tools=None,
        max_iterations=3, max_tokens_tao=1,
        planner=planner,
        replanning_enabled=True,
        max_replans_per_step=max_replans_per_step,
        max_replans_per_run=max_replans_per_run,
        autonomy="auto",
    ), planner


def test_replan_retry_then_succeed():
    """Test: step fails once → replan retry → step succeeds on attempt 2."""
    print("\n" + "="*60)
    print("TEST 39: replan — retry → succeed")
    print("="*60)
    from src.state import AgentState
    plan = {"summary": "x", "steps": [
        {"id": 1, "description": "do thing", "success_criterion": "thing done"},
    ]}
    decisions = [{"action": "retry", "reasoning": "transient"}]
    results = [(False, "iteration cap"), (True, "done!")]
    executor, planner = _build_executor_with_mocks(decisions, results)
    state = AgentState(task="t")
    out = executor.execute_plan(state, "t", plan)

    assert len(executor.tao_calls) == 2, f"expected 2 tao calls, got {len(executor.tao_calls)}"
    assert len(planner.calls) == 1, "planner should be called once"
    assert "Step 1" in out and "done!" in out
    assert "FAILED" not in out and "ABORTED" not in out
    print("\n✅ Retry path: step rerun and completed on second attempt")


def test_replan_revise_step_updates_description():
    """Test: revise_step replaces the step's description before retrying."""
    print("\n" + "="*60)
    print("TEST 40: replan — revise_step updates step")
    print("="*60)
    from src.state import AgentState
    plan = {"summary": "x", "steps": [
        {"id": 1, "description": "vague", "success_criterion": "vague"},
    ]}
    revised = {"id": 1, "description": "concrete & specific",
               "success_criterion": "specific check"}
    decisions = [{"action": "revise_step", "reasoning": "too vague",
                  "revised_step": revised}]
    results = [(False, "cap"), (True, "ok")]
    executor, planner = _build_executor_with_mocks(decisions, results)
    state = AgentState(task="t")
    out = executor.execute_plan(state, "t", plan)

    # The 2nd run_tao_loop saw the revised step (we can't inspect step directly,
    # but the user message should reference the new description in self._messages).
    msgs_text = " ".join(m.get("content") or "" for m in executor._messages)
    assert "concrete & specific" in msgs_text, "revised description should appear in messages"
    assert "[Replan]" in msgs_text, "replan note should appear in messages"
    assert "ok" in out
    print("\n✅ Revise_step: step description was updated and retried")


def test_replan_abort_after_step_cap():
    """Test: planner returns retry repeatedly; cap is enforced and step is failed."""
    print("\n" + "="*60)
    print("TEST 41: replan — per-step cap forces failure")
    print("="*60)
    from src.state import AgentState
    plan = {"summary": "x", "steps": [
        {"id": 1, "description": "impossible", "success_criterion": "never"},
    ]}
    decisions = [{"action": "retry", "reasoning": "trying"}] * 5
    # Step always fails; with max_replans_per_step=2 that means: initial run +
    # 2 replans = 3 tao_calls before the step is marked failed.
    results = [(False, "cap")] * 10
    executor, planner = _build_executor_with_mocks(
        decisions, results, max_replans_per_step=2,
    )
    state = AgentState(task="t")
    out = executor.execute_plan(state, "t", plan)

    assert len(executor.tao_calls) == 3, f"expected 3 attempts, got {len(executor.tao_calls)}"
    assert len(planner.calls) == 2, f"expected 2 replans, got {len(planner.calls)}"
    assert "FAILED" in out, f"step should be marked failed: {out!r}"
    print("\n✅ Per-step replan cap respected; step ends as FAILED after 2 retries")


def test_replan_abort_action_returns_immediately():
    """Test: planner returns abort → execute_plan stops with partial results."""
    print("\n" + "="*60)
    print("TEST 42: replan — abort action stops execution")
    print("="*60)
    from src.state import AgentState
    plan = {"summary": "x", "steps": [
        {"id": 1, "description": "a", "success_criterion": "a done"},
        {"id": 2, "description": "b", "success_criterion": "b done"},
    ]}
    decisions = [{"action": "abort", "reasoning": "unrecoverable"}]
    # Step 1 fails. Planner aborts. Step 2 should never run.
    results = [(False, "cap")]
    executor, planner = _build_executor_with_mocks(decisions, results)
    state = AgentState(task="t")
    out = executor.execute_plan(state, "t", plan)

    assert len(executor.tao_calls) == 1, "step 2 should NOT run after abort"
    assert "ABORTED" in out
    assert "Step 2" not in out, "step 2 should not appear in output"
    print("\n✅ Abort action: execute_plan halts before later steps")


def test_replan_skip_continues_to_next_step():
    """Test: planner returns skip → step is logged as skipped, next step runs."""
    print("\n" + "="*60)
    print("TEST 43: replan — skip continues")
    print("="*60)
    from src.state import AgentState
    plan = {"summary": "x", "steps": [
        {"id": 1, "description": "cosmetic", "success_criterion": "x"},
        {"id": 2, "description": "real work", "success_criterion": "y"},
    ]}
    decisions = [{"action": "skip", "reasoning": "non-critical"}]
    # Step 1 fails. Planner skips. Step 2 runs and succeeds.
    results = [(False, "cap"), (True, "step 2 done")]
    executor, planner = _build_executor_with_mocks(decisions, results)
    state = AgentState(task="t")
    out = executor.execute_plan(state, "t", plan)

    assert len(executor.tao_calls) == 2, "step 2 should run after skip"
    assert "SKIPPED" in out and "step 2 done" in out
    print("\n✅ Skip action: failed step marked skipped, next step ran")


def test_replan_global_cap_enforced():
    """Test: AGENT_MAX_REPLANS_PER_RUN caps total replans across all steps."""
    print("\n" + "="*60)
    print("TEST 44: replan — global per-run cap")
    print("="*60)
    from src.state import AgentState
    plan = {"summary": "x", "steps": [
        {"id": 1, "description": "a", "success_criterion": "a"},
        {"id": 2, "description": "b", "success_criterion": "b"},
    ]}
    # Each step always fails; both want retries. Global cap=1 means after the
    # first replan we must stop replanning entirely.
    decisions = [{"action": "retry", "reasoning": "a"}] * 10
    results = [(False, "cap")] * 10
    executor, planner = _build_executor_with_mocks(
        decisions, results,
        max_replans_per_step=5, max_replans_per_run=1,
    )
    state = AgentState(task="t")
    out = executor.execute_plan(state, "t", plan)

    # Step 1: initial run + 1 replan retry = 2 calls; then global cap hit, step
    # marked FAILED. Step 2: initial run; no more replans allowed → FAILED.
    assert len(planner.calls) == 1, f"expected 1 replan total, got {len(planner.calls)}"
    assert out.count("FAILED") == 2, f"both steps should fail: {out!r}"
    print("\n✅ Global per-run replan cap respected across multiple steps")


def test_rewrite_query_disabled_returns_original():
    """Test: AGENT_SEARCH_REWRITE_ENABLED=false bypasses rewriting entirely."""
    print("\n" + "="*60)
    print("TEST 45: rewrite_query — disabled returns original")
    print("="*60)
    from src import browser
    os.environ["AGENT_SEARCH_REWRITE_ENABLED"] = "false"
    browser._REWRITE_CACHE.clear()
    try:
        assert browser.rewrite_query("latest news in ai") == "latest news in ai"
    finally:
        os.environ.pop("AGENT_SEARCH_REWRITE_ENABLED", None)
    print("\n✅ Disabled rewrite returns the original query unchanged")


def test_rewrite_query_skips_explicit_operators():
    """Test: queries with quotes / site:/ etc. are passed through unchanged."""
    print("\n" + "="*60)
    print("TEST 46: rewrite_query — skip queries with operators")
    print("="*60)
    from src.browser import rewrite_query, _looks_already_specific
    # Heuristic detector
    assert _looks_already_specific('"exact phrase"')
    assert _looks_already_specific('site:python.org dataclasses')
    assert _looks_already_specific('intitle:tutorial pandas')
    assert _looks_already_specific('python -django')
    assert not _looks_already_specific('latest news in ai')
    assert not _looks_already_specific('how does asyncio work')

    # rewrite_query honors the heuristic — these stay unchanged with no LLM call.
    for q in ('"exact phrase"', 'site:python.org dataclasses', 'pytorch -tensorflow'):
        assert rewrite_query(q) == q, f"expected passthrough for {q!r}"
    print("\n✅ Queries with explicit operators bypass the rewriter")


def test_rewrite_query_caches_result():
    """Test: identical queries hit the per-process cache (no second LLM call)."""
    print("\n" + "="*60)
    print("TEST 47: rewrite_query — caching")
    print("="*60)
    from src import browser
    browser._REWRITE_CACHE.clear()
    # Pre-populate the cache to simulate a prior rewrite, without firing the LLM.
    browser._REWRITE_CACHE["latest python release"] = "Python 3.13 release notes 2026"
    out = browser.rewrite_query("latest python release")
    assert out == "Python 3.13 release notes 2026"
    print("\n✅ Cached query returns the stored rewrite without calling the LLM")


def test_rewrite_query_falls_back_on_no_api_key():
    """Test: when OPENAI_API_KEY is unset, rewrite returns the original query."""
    print("\n" + "="*60)
    print("TEST 48: rewrite_query — falls back without API key")
    print("="*60)
    from src import browser
    browser._REWRITE_CACHE.clear()
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        out = browser.rewrite_query("uncached vague query")
        assert out == "uncached vague query"
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved
    print("\n✅ Missing API key → rewrite gracefully returns original")


def test_search_web_output_shows_rewrite_when_changed():
    """Test: when the rewriter changes the query, the output mentions both."""
    print("\n" + "="*60)
    print("TEST 49: search_web output annotates rewrites")
    print("="*60)
    from src import browser
    # Inject a known rewrite into the cache so the LLM call is bypassed.
    original = "rewrite-test-query-abc"
    rewritten = "rewritten test ABC keywords 2026"
    browser._REWRITE_CACHE[original] = rewritten

    # Stub the network call so we don't actually hit DDG.
    class _StubResp:
        status_code = 200
        text = "<html></html>"  # No results — that's fine for header check.
    saved_post = browser.requests.post
    captured = {}
    def fake_post(url, data=None, headers=None, timeout=None):
        captured["data"] = data
        return _StubResp()
    browser.requests.post = fake_post
    try:
        result = browser.search_web(original, max_results=3)
    finally:
        browser.requests.post = saved_post

    # The DDG fetch should have used the REWRITTEN query.
    assert captured["data"]["q"] == rewritten, \
        f"expected DDG to receive rewritten query, got {captured['data']['q']!r}"
    # And the result output should annotate it for the agent to see.
    assert original in result.output, "original query should appear in header"
    assert "[rewritten as:" in result.output, \
        f"output should annotate the rewrite: {result.output[:200]!r}"
    assert rewritten in result.output
    print("\n✅ search_web sends the rewritten query + annotates the output")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("AUTONOMOUS AGENT TEST SUITE")
    print("="*60)
    print("\nNote: API tests require a valid OpenAI configuration in config/.env.")
    print("Tool unit tests (8-10) run offline.\n")

    # Offline tool unit tests first — fast, no API.
    test_search_in_files()
    test_edit_file()
    test_run_python()

    # Offline feedback-engine tests — also fast, no API.
    test_feedback_python_syntax_pass_silent()
    test_feedback_python_syntax_fail()
    test_feedback_json_syntax_fail()
    test_feedback_disabled_returns_none()
    test_feedback_skipped_for_non_matching_tool()

    # Offline browser tests — pre-populate cache, no real network.
    test_browser_invalid_url_rejected()
    test_browser_deny_host_blocks_localhost()
    test_browser_pagination_via_offset()

    # Offline correctness fixes (no API).
    test_is_approved_handles_negation()
    test_system_prefix_contains_today()
    test_planner_prompt_lists_browser_tools()
    test_browser_blocked_content_detection()

    # Triage / autonomy tests.
    test_planner_prompt_warns_against_unrequested_files()
    test_triage_prompt_lists_three_buckets()
    test_autonomy_env_var_validation()

    # Structured tool calling.
    test_tool_schemas_well_formed()
    test_complete_task_schema_present()
    test_executor_dispatches_all_schema_tools()
    test_initial_messages_includes_complete_task_instruction()
    test_xml_tool_call_fallback_parser()

    # Context management.
    test_context_token_estimator()
    test_context_no_compress_below_budget()
    test_context_compresses_above_budget()
    test_context_preserves_tool_call_integrity()
    test_context_arg_stubbing_for_old_writes()
    test_context_step_boundary_compaction()
    test_context_disabled_returns_input_unchanged()
    test_read_file_modes()

    # Mid-execution replanning.
    test_replan_retry_then_succeed()
    test_replan_revise_step_updates_description()
    test_replan_abort_after_step_cap()
    test_replan_abort_action_returns_immediately()
    test_replan_skip_continues_to_next_step()
    test_replan_global_cap_enforced()

    # Web-search query rewriting.
    test_rewrite_query_disabled_returns_original()
    test_rewrite_query_skips_explicit_operators()
    test_rewrite_query_caches_result()
    test_rewrite_query_falls_back_on_no_api_key()
    test_search_web_output_shows_rewrite_when_changed()

    input("\nPress Enter to run API-backed tests (or Ctrl+C to stop here)...")

    # Legacy execution tests — disable planning so they don't prompt for input.
    os.environ["AGENT_PLANNING_ENABLED"] = "false"
    try:
        test_list_directory()
        test_create_file()
        test_shell_command()
    finally:
        os.environ.pop("AGENT_PLANNING_ENABLED", None)

    # Planning-loop tests — invoke planning methods directly (no interactive prompt).
    test_initial_plan_decomposes_task()
    test_plan_critique_returns_suggestions()
    test_plan_refinement_applies_user_feedback()
    test_planning_disabled_falls_back_to_legacy()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()