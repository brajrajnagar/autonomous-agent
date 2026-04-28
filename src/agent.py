"""
Autonomous Agent with Think → Act → Observe Loop

This module implements a single autonomous agent that:
1. THINK: Plans what to do next
2. ACT: Executes actions using available tools
3. OBSERVE: Observes the results
4. REPEAT: Continues until task is complete
5. CRITIC: Reviews the output in a second pass
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
    """Result from executing a tool"""
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class AgentState:
    """Maintains the agent's state during execution"""
    task: str = ""
    thought_history: List[str] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    observation_history: List[str] = field(default_factory=list)
    iteration_count: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None


class Tools:
    """Available tools for the agent"""
    
    @staticmethod
    def execute_shell(command: str, timeout: int = 60) -> ToolResult:
        """Execute a shell command"""
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
        """Read contents of a file"""
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
        """Write content to a file"""
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
        """List contents of a directory"""
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
    Autonomous agent with Think → Act → Observe loop
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
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
        self.state = AgentState()
        self.tools = Tools()
    
    def _build_prompt(self) -> str:
        """Build the current prompt with full context"""
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
        """Parse the LLM response to extract tool usage or completion"""
        import json
        
        response = response.strip()
        
        # Try to extract JSON
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If no valid JSON, treat as thought
        return {"thought": response}
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a tool with given parameters"""
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
        """Second pass: Critic reviews the output"""
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
    
    def run(self, task: str) -> str:
        """Run the agent loop to complete a task"""
        self.state = AgentState(task=task)
        
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
                max_tokens=1000
            )
            
            llm_response = response.choices[0].message.content.strip()
            print(f"LLM Response: {llm_response[:200]}...")
            
            # Parse response
            parsed = self._parse_response(llm_response)
            
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
    """Main entry point"""
    import sys
    
    agent = AutonomousAgent()
    
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("Enter your task (or 'quit' to exit):")
        task = input("> ")
    
    if task.lower() == "quit":
        return
    
    result = agent.run(task)
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print("="*50)
    print(result)


if __name__ == "__main__":
    main()