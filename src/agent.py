"""
Autonomous Agent with Think → Act → Observe Loop

This module implements a single autonomous agent that:
1. THINK: Plans what to do next
2. ACT: Executes actions using available tools
3. OBSERVE: Observes the results
4. REPEAT: Continues until task is complete
5. CRITIC: Reviews the output in a second pass

Example Usage:
    ```python
    from agent import AutonomousAgent
    
    # Create an agent instance
    agent = AutonomousAgent()
    
    # Run a task
    result = agent.run("List all files in the current directory")
    print(result)
    ```
"""

import os
import subprocess
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Project root is one level up
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, "config", ".env")
load_dotenv(env_path)


@dataclass
class ToolResult:
    """
    Result from executing a tool.
    
    Attributes:
        success (bool): Whether the tool execution succeeded
        output (str): The output from the tool
        error (Optional[str]): Error message if execution failed
    
    Example:
        ```python
        result = ToolResult(success=True, output="File contents here")
        result = ToolResult(success=False, output="", error="File not found")
        ```
    """
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class AgentState:
    """
    Maintains the agent's state during execution.
    
    Tracks the complete history of the agent's thought process,
    actions taken, and observations made during task execution.
    
    Attributes:
        task (str): The current task description
        thought_history (List[str]): History of thoughts/reasoning
        action_history (List[Dict]): History of actions taken
        observation_history (List[str]): History of observations
        iteration_count (int): Current iteration number
        is_complete (bool): Whether task is marked complete
        final_answer (Optional[str]): The final answer if complete
    
    Example:
        ```python
        state = AgentState(task="List files in directory")
        state.thought_history.append("I need to list the directory first")
        state.action_history.append({"tool": "list_directory", "params": {"path": "."}})
        state.observation_history.append("[FILE] test.txt")
        ```
    """
    task: str = ""
    thought_history: List[str] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    observation_history: List[str] = field(default_factory=list)
    iteration_count: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None


class Tools:
    """
    Available tools for the agent.
    
    This class provides static methods that the agent can use to interact
    with the environment. Each tool returns a ToolResult object.
    
    Example:
        ```python
        # List directory contents
        result = Tools.list_directory(".")
        
        # Read a file
        result = Tools.read_file("config.txt")
        
        # Write to a file
        result = Tools.write_file("output.txt", "Hello World")
        
        # Execute shell command
        result = Tools.execute_shell("ls -la")
        ```
    """
    
    @staticmethod
    def execute_shell(command: str, timeout: int = 60) -> ToolResult:
        """
        Execute a shell command and capture its output.
        
        Args:
            command (str): The shell command to execute
            timeout (int): Maximum execution time in seconds (default: 60)
        
        Returns:
            ToolResult: Contains success status, stdout, and any error
        
        Example:
            ```python
            # List files
            result = Tools.execute_shell("ls -la")
            if result.success:
                print(result.output)  # Command output
            else:
                print(result.error)   # Error message
            
            # Get current directory
            result = Tools.execute_shell("pwd")
            
            # Create a directory
            result = Tools.execute_shell("mkdir new_folder")
            ```
        
        Security Note:
            Commands run with the permissions of the current user.
            Be cautious with destructive commands like rm -rf.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )
    
    @staticmethod
    def read_file(path: str) -> ToolResult:
        """
        Read the contents of a file.
        
        Args:
            path (str): Relative path to the file (absolute paths blocked for security)
        
        Returns:
            ToolResult: Contains file contents on success, error message on failure
        
        Example:
            ```python
            # Read a config file
            result = Tools.read_file("config/settings.json")
            if result.success:
                content = result.output
                print(content)
            else:
                print(f"Error: {result.error}")
            
            # Read source code
            result = Tools.read_file("src/main.py")
            ```
        
        Security Note:
            Absolute paths (starting with /) are blocked to prevent
            reading files outside the project directory.
        """
        try:
            # Security: prevent absolute paths outside project
            if path.startswith("/"):
                return ToolResult(
                    success=False,
                    output="",
                    error="Absolute paths not allowed. Use relative paths."
                )
            if not os.path.exists(path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}"
                )
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    @staticmethod
    def write_file(path: str, content: str) -> ToolResult:
        """
        Write content to a file (creates new file or overwrites existing).
        
        Args:
            path (str): Relative path to the file
            content (str): Content to write to the file
        
        Returns:
            ToolResult: Success message with character count, or error
        
        Example:
            ```python
            # Create a new file
            result = Tools.write_file("output.txt", "Hello World!")
            
            # Write JSON config
            import json
            config = {"name": "test", "value": 42}
            result = Tools.write_file("config.json", json.dumps(config))
            
            # Write multi-line content
            content = "Line 1\\nLine 2\\nLine 3"
            result = Tools.write_file("multiline.txt", content)
            ```
        
        Security Note:
            Absolute paths are blocked. File will be created if it doesn't exist.
        """
        try:
            # Security: prevent absolute paths outside project
            if path.startswith("/"):
                return ToolResult(
                    success=False,
                    output="",
                    error="Absolute paths not allowed. Use relative paths."
                )
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to {path}"
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    @staticmethod
    def list_directory(path: str = ".") -> ToolResult:
        """
        List all files and directories in a given path.
        
        Args:
            path (str): Directory path to list (default: current directory ".")
        
        Returns:
            ToolResult: Formatted list with [DIR] and [FILE] prefixes
        
        Example:
            ```python
            # List current directory
            result = Tools.list_directory()
            print(result.output)
            # Output:
            # [DIR]  src
            # [DIR]  config
            # [FILE] README.md
            
            # List specific directory
            result = Tools.list_directory("src")
            
            # List parent directory
            result = Tools.list_directory("..")
            ```
        
        Output Format:
            Each item is prefixed with [DIR] for directories or [FILE] for files,
            sorted alphabetically.
        """
        try:
            if not os.path.exists(path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Directory not found: {path}"
                )
            if not os.path.isdir(path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a directory: {path}"
                )
            items = os.listdir(path)
            output = []
            for item in sorted(items):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    output.append(f"[DIR]  {item}")
                else:
                    output.append(f"[FILE] {item}")
            return ToolResult(success=True, output="\n".join(output))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class AutonomousAgent:
    """
    Autonomous agent with Think → Act → Observe loop.
    
    The agent operates in a continuous loop:
    1. THINK: Analyze the task and current state, decide next action
    2. ACT: Execute a tool to perform the action
    3. OBSERVE: Process the result of the action
    4. REPEAT: Continue until task is complete
    5. CRITIC: Review the final output for quality
    
    Example:
        ```python
        # Create agent instance
        agent = AutonomousAgent()
        
        # Run a simple task
        result = agent.run("What files are in the current directory?")
        
        # Run a complex multi-step task
        result = agent.run("Create a file called report.txt with a summary of all .py files")
        ```
    """
    
    TOOLS_PROMPT = """
You have access to these tools. To use a tool, respond with exactly this JSON format:
{"tool": "tool_name", "parameters": {"param1": "value1"}}

Available tools:
1. execute_shell: Run shell commands. Parameters: {"command": "string"}
2. read_file: Read file contents. Parameters: {"path": "string"}
3. write_file: Write to a file. Parameters: {"path": "string", "content": "string"}
4. list_directory: List directory contents. Parameters: {"path": "string"} (default: ".")

When you have completed the task, respond with:
{"complete": true, "answer": "your final answer"}
"""
    
    def __init__(self):
        """
        Initialize the AutonomousAgent.
        
        Sets up the OpenAI-compatible API client and loads configuration
        from environment variables.
        
        Environment Variables Required:
            OPENAI_API_BASE: API endpoint URL (e.g., https://api.openai.com/v1)
            OPENAI_API_KEY: API authentication key
            OPENAI_MODEL: Model to use (default: "gpt-4")
            MAX_ITERATIONS: Maximum loop iterations (default: 10)
        
        Example:
            ```python
            # With environment variables set:
            # OPENAI_API_BASE=https://api.openai.com/v1
            # OPENAI_API_KEY=sk-xxx
            # OPENAI_MODEL=gpt-4
            
            agent = AutonomousAgent()
            # Agent is ready to use
            ```
        """
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
        self.state = AgentState()
        self.tools = Tools()
        self._last_action_hash = None  # For loop detection
    
    def _build_prompt(self) -> str:
        """
        Build the current prompt with full context for the LLM.
        
        Constructs a prompt that includes:
        - The original task
        - Complete history of thoughts, actions, and observations
        - Instructions for the current iteration
        
        Returns:
            str: The complete prompt to send to the LLM
        
        Example:
            ```python
            # After some iterations, the prompt will include:
            # Task: List files in directory
            # 
            # --- History ---
            # Iteration 1:
            # Thought: I need to list the directory
            # Action: {"tool": "list_directory", "params": {"path": "."}}
            # Observation: [FILE] test.txt
            #
            # --- Current Turn ---
            # Iteration 2:
            # What do you think needs to be done next?
            ```
        
        Note:
            This method is called internally during each iteration of run().
        """
        prompt = f"""Task: {self.state.task}

You are an autonomous agent. Complete the task using a Think → Act → Observe loop.

THINK: Consider what you know and what you need to do next.
ACT: Use one of your tools to take action.
OBSERVE: See the result of your action.
REPEAT: Continue until the task is complete.

"""
        
        # Add history
        if self.state.thought_history:
            prompt += "\n--- History ---\n"
            for i, (thought, action, observation) in enumerate(zip(
                self.state.thought_history,
                self.state.action_history,
                self.state.observation_history
            )):
                prompt += f"\nIteration {i+1}:\n"
                prompt += f"Thought: {thought}\n"
                prompt += f"Action: {action}\n"
                prompt += f"Observation: {observation}\n"
        
        prompt += "\n--- Current Turn ---\n"
        prompt += f"Iteration {self.state.iteration_count + 1}:\n"
        prompt += "\nWhat do you think needs to be done next?\n"
        prompt += self.TOOLS_PROMPT
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract tool usage or completion status.
        
        Attempts to find and parse JSON in the response. If no valid JSON
        is found, treats the response as a thought.
        
        Args:
            response (str): Raw response text from the LLM
        
        Returns:
            Dict[str, Any]: Parsed response, which may contain:
                - "tool": Name of tool to use
                - "parameters": Tool parameters
                - "complete": True if task is complete
                - "answer": Final answer if complete
                - "thought": Reasoning text
        
        Example:
            ```python
            # Tool usage response
            response = 'Sure! {"tool": "list_directory", "parameters": {"path": "."}}'
            parsed = _parse_response(response)
            # Result: {"tool": "list_directory", "parameters": {"path": "."}}
            
            # Completion response
            response = '{"complete": true, "answer": "Found 5 files"}'
            parsed = _parse_response(response)
            # Result: {"complete": True, "answer": "Found 5 files"}
            
            # Thought-only response
            response = "I think I should list the directory first"
            parsed = _parse_response(response)
            # Result: {"thought": "I think I should list the directory first"}
            ```
        """
        import json
        
        response = response.strip()
        
        # Try multiple strategies to extract JSON
        
        # Strategy 1: Look for complete JSON objects
        try:
            # Find all potential JSON objects
            brace_count = 0
            start = -1
            for i, char in enumerate(response):
                if char == '{':
                    if brace_count == 0:
                        start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        json_str = response[start:i+1]
                        return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Simple find first { and last }
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Try to find JSON after common prefixes
        prefixes = ["Here's the result:", "Sure!", "OK", "Certainly", "Response:"]
        for prefix in prefixes:
            idx = response.find(prefix)
            if idx != -1:
                try:
                    remaining = response[idx:]
                    start = remaining.find("{")
                    end = remaining.rfind("}") + 1
                    if start != -1 and end > start:
                        json_str = remaining[start:end]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # If no valid JSON, treat as thought
        return {"thought": response}
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool with given parameters.
        
        Routes the request to the appropriate tool method based on tool name.
        
        Args:
            tool_name (str): Name of the tool to execute
            parameters (Dict[str, Any]): Parameters for the tool
        
        Returns:
            ToolResult: Result of the tool execution
        
        Example:
            ```python
            # Execute shell command
            result = _execute_tool("execute_shell", {"command": "ls -la"})
            
            # Read a file
            result = _execute_tool("read_file", {"path": "config.txt"})
            
            # Write to a file
            result = _execute_tool("write_file", {"path": "out.txt", "content": "Hello"})
            
            # List directory
            result = _execute_tool("list_directory", {"path": "."})
            
            # Unknown tool
            result = _execute_tool("unknown_tool", {})
            # Returns: ToolResult(success=False, error="Unknown tool: unknown_tool")
            ```
        """
        if tool_name == "execute_shell":
            return self.tools.execute_shell(parameters.get("command", ""))
        elif tool_name == "read_file":
            return self.tools.read_file(parameters.get("path", ""))
        elif tool_name == "write_file":
            return self.tools.write_file(
                parameters.get("path", ""),
                parameters.get("content", "")
            )
        elif tool_name == "list_directory":
            return self.tools.list_directory(parameters.get("path", "."))
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}"
            )
    
    def _critic_review(self, result: str) -> str:
        """
        Second pass: Critic reviews the output for quality assurance.
        
        After the agent completes a task, this method sends the result
        to the LLM for review, asking it to evaluate:
        1. Is the task actually complete?
        2. Are there any errors or issues?
        3. Is the answer clear and accurate?
        4. Any improvements needed?
        
        Args:
            result (str): The agent's completed result
        
        Returns:
            str: Critic feedback, typically "APPROVED" or suggestions
        
        Example:
            ```python
            # Good result
            feedback = _critic_review("Found 3 Python files: main.py, utils.py, config.py")
            # Returns: "APPROVED"
            
            # Incomplete result
            feedback = _critic_review("There are some files...")
            # Returns: "The answer is vague. Please provide specific file names."
            ```
        
        Note:
            Returns "APPROVED (no feedback generated)" if the API returns
            empty content, allowing the agent to proceed.
        """
        critic_prompt = f"""
You are a critic/reviewer. Review the following task completion:

ORIGINAL TASK: {self.state.task}

PROPOSED ANSWER:
{result}

ACTION HISTORY:
{self.state.action_history}

Please review this output:
1. Is the task actually complete?
2. Are there any errors or issues?
3. Is the answer clear and accurate?
4. Any improvements needed?

Respond with either:
- "APPROVED" if the output is satisfactory
- Or describe what needs to be fixed
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a critical reviewer."},
                {"role": "user", "content": critic_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        if content is None:
            return "APPROVED (no feedback generated)"
        return content.strip()
    
    def _hash_action(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Create a hash of an action for loop detection.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
        
        Returns:
            Hash string representing this action
        """
        import hashlib
        action_str = f"{tool_name}:{str(sorted(parameters.items()))}"
        return hashlib.md5(action_str.encode()).hexdigest()
    
    def run(self, task: str) -> str:
        """
        Run the agent loop to complete a task.
        
        Main execution method that runs the Think → Act → Observe loop
        until the task is complete or max iterations is reached.
        
        Args:
            task (str): Natural language description of the task to complete
        
        Returns:
            str: The final result, including any critic feedback
        
        Example:
            ```python
            agent = AutonomousAgent()
            
            # Simple task
            result = agent.run("List all files in the current directory")
            print(result)
            
            # Multi-step task
            result = agent.run("Create a file called summary.txt that contains a list of all .py files in the src directory")
            print(result)
            
            # Complex task
            result = agent.run("Read the README.md file, count the number of lines, and create a new file called stats.txt with the count")
            ```
        
        Output:
            The method prints progress information during execution:
            - Iteration number
            - LLM response (thought/tool selection)
            - Tool execution result
            - Critic review at the end
        
        Returns:
            If successful: The final answer
            If incomplete: Best attempt with last observations
            Always includes: Critic feedback appended
        """
        self.state = AgentState(task=task)
        self._last_action_hash = None
        repeat_count = 0
        
        print(f"\n🤖 Starting agent for task: {task}")
        print(f"Max iterations: {self.max_iterations}\n")
        
        while not self.state.is_complete and self.state.iteration_count < self.max_iterations:
            self.state.iteration_count += 1
            
            # THINK
            print(f"\n--- Iteration {self.state.iteration_count} ---")
            print("THINKING...")
            
            prompt = self._build_prompt()
            
            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an autonomous agent. Use tools to complete tasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=8000
            )

            llm_response = response.choices[0].message.content
            if llm_response is None:
                llm_response = "I need to think more about this task."
            llm_response = llm_response.strip()
            print(f"LLM Response: {llm_response[:200]}...")

            # Parse response
            parsed = self._parse_response(llm_response)

            # Detect truncated tool-call JSON: response looks like a tool call
            # but parser couldn't extract it (likely cut off by max_tokens).
            looks_like_tool_call = '"tool"' in llm_response and '"parameters"' in llm_response
            parser_only_got_thought = set(parsed.keys()) == {"thought"}
            if looks_like_tool_call and parser_only_got_thought:
                print("⚠️  Tool-call JSON appears truncated; nudging LLM to retry differently.")
                self.state.thought_history.append(llm_response[:200])
                self.state.action_history.append({"parse_error": True})
                self.state.observation_history.append(
                    "ERROR: Your previous response was truncated mid-JSON, likely because "
                    "the 'content' field was too large. For big files, use execute_shell "
                    "with a heredoc (e.g. cat > path << 'EOF' ... EOF), or split the write "
                    "into multiple smaller write_file calls."
                )
                continue
            
            # Check for completion
            if parsed.get("complete"):
                self.state.is_complete = True
                self.state.final_answer = parsed.get("answer", llm_response)
                print(f"✅ Task marked as complete")
                break
            
            # Extract thought
            thought = parsed.get("thought", llm_response)
            self.state.thought_history.append(thought)
            print(f"Thought: {thought[:100]}...")
            
            # Check for tool usage
            if "tool" in parsed and "parameters" in parsed:
                tool_name = parsed["tool"]
                parameters = parsed["parameters"]
                
                # Loop detection
                current_hash = self._hash_action(tool_name, parameters)
                if current_hash == self._last_action_hash:
                    repeat_count += 1
                    print(f"⚠️  Detected repeated action (count: {repeat_count})")
                    
                    if repeat_count >= 3:
                        print("🔄 Breaking loop - forcing progress...")
                        # Force the agent to move on by adding a nudge to the prompt
                        self.state.observation_history.append(
                            f"NOTE: You've repeated the action '{tool_name}' multiple times. "
                            "Please try a different approach or mark the task as complete."
                        )
                        repeat_count = 0
                        continue
                else:
                    repeat_count = 0
                    self._last_action_hash = current_hash
                
                # ACT
                print(f"ACT: Executing tool '{tool_name}'...")
                self.state.action_history.append({"tool": tool_name, "params": parameters})
                
                result = self._execute_tool(tool_name, parameters)
                
                # OBSERVE
                observation = result.output if result.success else f"ERROR: {result.error}"
                self.state.observation_history.append(observation)
                print(f"OBSERVE: {observation[:200]}...")
            else:
                # No tool used, just a thought
                self.state.action_history.append({"thought_only": True})
                self.state.observation_history.append("Continuing...")
        
        # Final result
        if self.state.is_complete:
            result = self.state.final_answer
        else:
            result = f"Task incomplete after {self.max_iterations} iterations. Best attempt:\n"
            result += "\n".join(self.state.observation_history[-3:])
        
        # CRITIC PASS
        print("\n--- CRITIC REVIEW ---")
        critic_feedback = self._critic_review(result)
        print(f"Critic: {critic_feedback}")
        
        if "APPROVED" in critic_feedback.upper():
            print("✅ Output approved by critic")
        else:
            print("⚠️ Critic suggested improvements")
            result += f"\n\n[Critic Feedback]: {critic_feedback}"
        
        return result


def main():
    """
    Main entry point for command-line usage.
    
    Supports two modes:
    
    1. **Single-task mode**: Run one task and exit
       ```bash
       python src/agent.py "List all files"
       ```
    
    2. **Interactive mode**: Chat with the agent, maintaining conversation context
       ```bash
       python src/agent.py --interactive
       # or just: python src/agent.py
       ```
    
    Interactive Mode Features:
    - Multi-turn conversations
    - Context memory across turns
    - Follow-up questions
    - Session-based history
    
    Example Interactive Session:
        ```bash
        $ python src/agent.py
        
        🤖 Autonomous Agent - Interactive Mode
        Type your task, or use commands:
          - 'history' - Show conversation history
          - 'clear' - Clear history and start fresh
          - 'quit' or 'exit' - Exit the agent
        
        > You: List files in current directory
        🤖 [Agent executes and shows result]
        
        > You: Now create a file called test.txt with "Hello"
        🤖 [Agent remembers context and creates the file]
        
        > You: What was in that file?
        🤖 [Agent knows about test.txt from previous turn]
        ```
    """
    import sys
    
    # Check for interactive mode flag
    interactive_mode = "--interactive" in sys.argv or len(sys.argv) == 1
    
    if interactive_mode:
        # Interactive mode with conversation history
        print("\n" + "="*60)
        print("🤖  Autonomous Agent - Interactive Mode")
        print("="*60)
        print("""
Type your task, or use commands:
  - 'history'  - Show conversation history
  - 'clear'    - Clear history and start fresh
  - 'quit' or 'exit' or Ctrl+D - Exit the agent

The agent remembers context across multiple turns!
""")
        
        agent = AutonomousAgent()
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n\033[92m> You:\033[0m ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ('quit', 'exit'):
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'history':
                    print("\n--- Conversation History ---")
                    for i, (task, result) in enumerate(conversation_history, 1):
                        print(f"\n{i}. Task: {task[:50]}...")
                        print(f"   Result: {result[:100]}...")
                    if not conversation_history:
                        print("(No history yet)")
                    continue
                
                if user_input.lower() == 'clear':
                    conversation_history = []
                    print("\n🗑️  History cleared. Starting fresh!")
                    continue
                
                # Build context-aware task
                if conversation_history:
                    # Include recent context for follow-up understanding
                    recent_context = "\n".join([
                        f"Previous task {i}: {task}"
                        for i, (task, _) in enumerate(conversation_history[-3:], 1)
                    ])
                    context_task = f"""You are in an ongoing conversation. Here is recent context:

{recent_context}

Current request from user: {user_input}

If the current request refers to previous items (like "that file", "the content", etc.),
use the context above to understand what is being referenced."""
                else:
                    context_task = user_input
                
                # Run the agent
                result = agent.run(context_task)
                
                # Show result
                print("\n" + "="*50)
                print("🤖 Result:")
                print("="*50)
                print(result)
                
                # Store in history
                conversation_history.append((user_input, result))
                
                # Keep history manageable (last 10 turns)
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
    else:
        # Single-task mode
        task = " ".join(sys.argv[1:])
        
        if task.lower() in ('quit', 'exit'):
            return
        
        agent = AutonomousAgent()
        result = agent.run(task)
        print("\n" + "="*50)
        print("FINAL RESULT:")
        print("="*50)
        print(result)


if __name__ == "__main__":
    main()