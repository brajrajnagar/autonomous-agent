"""
Simple test script for the Autonomous Agent

This script tests the agent with a simple task to verify it's working correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import AutonomousAgent


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


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("AUTONOMOUS AGENT TEST SUITE")
    print("="*60)
    print("\nNote: These tests require a valid OpenAI API configuration.")
    print("Make sure config/.env is set up with your API credentials.\n")
    
    input("Press Enter to run tests...")
    
    test_list_directory()
    test_create_file()
    test_shell_command()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()