import asyncio
import json
import os
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any, Type

from pydantic import BaseModel, Field

class Manus:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "Manus")
        self.description = kwargs.get("description", "Base Manus class")
        self.system_prompt = kwargs.get("system_prompt", "")
        self.next_step_prompt = kwargs.get("next_step_prompt", "")
        self.tools = ToolCollection([])
        self.memory = []
        self.event_q = None

    async def think(self):
        pass

    async def act(self):
        pass

    async def run(self, prompt: str, event_q: Optional[queue.Queue] = None):
        if event_q:
            self.event_q = event_q

        while True:
            await self._send_event("log", f"{self.name} received prompt: {prompt}")
            self.memory.append({"role": "user", "content": prompt})

            if "exit" in prompt.lower() or "quit" in prompt.lower():
                final_response = await self.tools.get_tool("Terminate").execute(
                    message="User requested termination"
                )
                await self._send_event("final_result", final_response)
                break

            # Process different command types
            if "search" in prompt.lower() or "find" in prompt.lower():
                await self.process_search(prompt)
            elif "code" in prompt.lower() or "develop" in prompt.lower():
                await self.process_code(prompt)
            else:
                await self.process_general(prompt)

            # Get next input (in a real implementation, this would come from user/API)
            prompt = await self.get_next_input()

    async def process_search(self, prompt: str):
        await self._send_event("log", "Delegating to WebSearchAgent")
        response = await self.tools.get_tool("delegate_task_to_specialist").execute(
            agent_role="WebSearchAgent", 
            sub_task_prompt=prompt
        )
        await self._send_event("log", f"WebSearchAgent response: {response}")

    async def process_code(self, prompt: str):
        await self._send_event("log", "Delegating to CodingAgent")
        response = await self.tools.get_tool("delegate_task_to_specialist").execute(
            agent_role="CodingAgent", 
            sub_task_prompt=prompt
        )
        await self._send_event("log", f"CodingAgent response: {response}")

    async def process_general(self, prompt: str):
        await self._send_event("log", "Handling request directly")
        await asyncio.sleep(1)
        response = "Request processed directly by Orchestrator"
        await self._send_event("log", response)

    async def get_next_input(self) -> str:
        """Simulated input method - replace with actual input collection"""
        await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
        return "continue"  # Replace with actual input collection

    async def _send_event(self, event_type: str, content: Any, source: Optional[str] = None):
        if self.event_q:
            event = {
                "type": event_type,
                "source": source or self.name,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            self.event_q.put(event)

class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, **kwargs) -> Any:
        raise NotImplementedError

class ToolCollection:
    def __init__(self, tools: List[Tool]):
        self.tools = {tool.name: tool for tool in tools}

    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator Agent for the OpenManus Multi-Agent System..."""  # (Keep your existing prompt)

ORCHESTRATOR_NEXT_STEP_PROMPT = """Review the user's request..."""  # (Keep your existing prompt)

class DelegateTaskTool(Tool):
    def __init__(self):
        super().__init__(
            name="delegate_task_to_specialist",
            description="Delegates a sub-task to a specialized agent"
        )

    async def execute(self, agent_role: str, sub_task_prompt: str) -> str:
        # Simulate processing delay
        await asyncio.sleep(0.5)
        return f"Delegated to {agent_role}: {sub_task_prompt}"

class TerminateTool(Tool):
    def __init__(self):
        super().__init__(
            name="Terminate",
            description="Terminates the current task"
        )

    async def execute(self, message: str) -> str:
        await asyncio.sleep(0.1)  # Small processing delay
        return f"Terminated: {message}"

class OrchestratorAgent(Manus):
    def __init__(self, event_q: Optional[queue.Queue] = None, **kwargs):
        super().__init__(**kwargs)
        self.name = "OrchestratorAgent"
        self.event_q = event_q
        self.system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(directory=os.getcwd())
        self.next_step_prompt = ORCHESTRATOR_NEXT_STEP_PROMPT
        self.tools = ToolCollection([
            DelegateTaskTool(),
            TerminateTool()
        ])
        
        # Schedule initialization event
        if self.event_q:
            asyncio.create_task(
                self._send_event("log", f"{self.name} initialized")
            )

async def main():
    event_q = queue.Queue()
    orchestrator = OrchestratorAgent(event_q=event_q)
    
    try:
        # Start with initial task
        initial_prompt = "Find information about AI advancements"
        await orchestrator.run(initial_prompt)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("System stopped")

if __name__ == "__main__":
    asyncio.run(main())
