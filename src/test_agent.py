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