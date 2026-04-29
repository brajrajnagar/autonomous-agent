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