# Orchestrator Agent for OpenManus Multi-Agent System

import asyncio
import queue # For event queue
import os # Added for os.getcwd()
import json # Added for json.loads in run method
from typing import Dict, List, Optional, Any, Type

from pydantic import BaseModel, Field # Added BaseModel

from app.agent.manus import Manus # Base agent class
from app.config import config
from app.logger import logger
from app.memory.file_memory import FileBasedLongTermMemory
from app.tool import ToolCollection, Terminate, Tool # Added Tool
from app.schema import Message # For history structure

# Import specialized agent classes
from app.agent.specialized.brainstormer import BrainstormingAgent
from app.agent.specialized.coder import CodingAgent
from app.agent.specialized.web_researcher import WebResearchAgent

# Define a type for specialized agent classes
SpecializedAgentType = Type[Manus]

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the Orchestrator Agent for the OpenManus Multi-Agent System. "
    "Your primary role is to understand complex user requests, decompose them into manageable sub-tasks, "
    "develop a coherent, step-by-step plan, and delegate these sub-tasks to specialized agents. "
    "You must manage the overall workflow, synthesize results from specialized agents, and present the final solution."
    "Key principles for your operation:"
    "1. **Decomposition & Planning:** Break down complex goals. Clearly outline your plan. For each step, decide if you will perform it or delegate it."
    "2. **Delegation:** If delegating, clearly define the sub-task and provide all necessary context. Use the `delegate_task_to_specialist` tool. Specify the `agent_role` (e.g., \"BrainstormingAgent\", \"CodingAgent\", \"WebResearchAgent\") and the `sub_task_prompt`."
    "3. **Synthesis:** Combine outputs from specialized agents and your own work into a cohesive final result."
    "4. **State Management:** Maintain awareness of the overall task progress and the status of sub-tasks."
    "5. **Error Handling:** If a specialized agent fails or returns an unsatisfactory result, re-evaluate your plan, re-delegate, or try an alternative approach."
    "The initial working directory is: {directory}."
)

ORCHESTRATOR_NEXT_STEP_PROMPT = """Review the user's request, your current plan, and any previous results. Determine the next action.
1. **Current Goal:** Briefly state the overall user goal.
2. **Current Plan Step:** What is the current step in your plan?
3. **Action Choice:** Will you perform this step yourself or delegate it?
   - If performing yourself: Select an appropriate tool from your own toolset.
   - If delegating: Use the `delegate_task_to_specialist` tool. Specify `agent_role` and `sub_task_prompt`.

If the overall task is complete, use the `Terminate` tool."""

class DelegateTaskArgs(BaseModel):
    agent_role: str = Field(description="The role of the specialized agent to delegate to (e.g., BrainstormingAgent, CodingAgent, WebResearchAgent).")
    sub_task_prompt: str = Field(description="The detailed prompt or instructions for the specialized agent to perform the sub-task.")

class DelegateTaskTool(Tool):
    name: str = "delegate_task_to_specialist"
    description: str = "Delegates a sub-task to a specialized agent. Provide the agent_role and the sub_task_prompt."
    args_schema: Type[BaseModel] = DelegateTaskArgs

    async def _execute(self, agent_role: str, sub_task_prompt: str) -> str:
        return f"Delegation to {agent_role} with prompt '{sub_task_prompt}' will be handled by Orchestrator."

class OrchestratorAgent(Manus):
    name: str = "OrchestratorAgent"
    description: str = "Coordinates specialized agents to solve complex tasks."

    system_prompt_template: str = ORCHESTRATOR_SYSTEM_PROMPT
    next_step_prompt_template: str = ORCHESTRATOR_NEXT_STEP_PROMPT
    event_q: Optional[queue.Queue] = None

    specialized_agent_map: Dict[str, SpecializedAgentType] = {
        "BrainstormingAgent": BrainstormingAgent,
        "CodingAgent": CodingAgent,
        "WebResearchAgent": WebResearchAgent,
    }

    def __init__(self, event_q: Optional[queue.Queue] = None, **data: Any):
        super().__init__(**data)
        self.event_q = event_q
        self.system_prompt = self.system_prompt_template.format(directory=config.workspace_root)
        self.next_step_prompt = self.next_step_prompt_template

        if not hasattr(self, 'tools') or self.tools is None:
            logger.warning(f"OrchestratorAgent.__init__: self.tools not initialized by super().__init__. Initializing now.")
            self.tools = ToolCollection([])
        
        # Ensure DelegateTaskTool and Terminate are always part of the tools
        current_tool_names = {tool.name for tool in self.tools.tools}
        if "delegate_task_to_specialist" not in current_tool_names:
            self.tools.add_tool(DelegateTaskTool())
        if "Terminate" not in current_tool_names:
            self.tools.add_tool(Terminate())

        logger.info(f"{self.name} initialized with tools: {[tool.name for tool in self.tools.tools]}")
        if self.event_q:
            self.event_q.put({"type": "log", "source": self.name, "content": f"{self.name} initialized."})

    @classmethod
    async def create(cls, event_q: Optional[queue.Queue] = None, **kwargs) -> "OrchestratorAgent":
        instance = await super(OrchestratorAgent, cls).create(**kwargs)
        instance.event_q = event_q
        instance.system_prompt = instance.system_prompt_template.format(directory=config.workspace_root)
        instance.next_step_prompt = instance.next_step_prompt_template
        
        if not hasattr(instance, 'tools') or instance.tools is None:
            logger.warning(f"OrchestratorAgent.create: instance.tools not initialized by super().create. Initializing now.")
            instance.tools = ToolCollection([])
        
        current_tool_names = {tool.name for tool in instance.tools.tools}
        if "delegate_task_to_specialist" not in current_tool_names:
            instance.tools.add_tool(DelegateTaskTool())
            logger.info(f"{instance.name} added/verified delegation tool after create.")
        
        if "Terminate" not in current_tool_names:
            instance.tools.add_tool(Terminate())
            logger.info(f"{instance.name} added Terminate tool after create.")

        if instance.event_q:
            instance.event_q.put({"type": "log", "source": instance.name, "content": f"{instance.name} instance created and tools configured."})
        return instance

    async def _send_event(self, event_type: str, content: Any, source: Optional[str] = None):
        if self.event_q:
            event = {"type": event_type, "source": source or self.name, "content": content, "timestamp": datetime.now().isoformat()} # Added timestamp
            self.event_q.put(event)

    async def _call_specialized_agent(self, agent_role: str, sub_task_prompt: str) -> str:
        await self._send_event("log", f"Delegating to {agent_role} with prompt: {sub_task_prompt[:100]}...")
        
        agent_class = self.specialized_agent_map.get(agent_role)
        if not agent_class:
            err_msg = f"Error: Unknown agent role '{agent_role}'. Cannot delegate."
            await self._send_event("error", err_msg)
            logger.error(f"Unknown agent role for delegation: {agent_role}")
            return err_msg

        specialist_agent_instance = None 
        try:
            # Ensure a unique LTM file for each specialist agent run within an orchestrator task
            ltm_file_name = f"ltm_{agent_role.lower()}_{self.id_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.json"
            specialist_ltm = FileBasedLongTermMemory(workspace_root=config.workspace_root, memory_file_name=ltm_file_name)
            specialist_agent_instance = await agent_class.create(long_term_memory=specialist_ltm, event_q=self.event_q) # Pass event_q
            
            await self._send_event("log", f"Executing {agent_role} for sub-task...", source=agent_role)
            await specialist_agent_instance.run(sub_task_prompt)
            
            result_summary = f"{agent_role} task completed."
            # Check history for Terminate tool usage by specialist
            if specialist_agent_instance.memory.messages:
                last_message_from_specialist = specialist_agent_instance.memory.messages[-1]
                if last_message_from_specialist.tool_calls:
                    for tc in last_message_from_specialist.tool_calls:
                        if tc.function.name == "Terminate":
                            try:
                                args = json.loads(tc.function.arguments)
                                result_summary = args.get("message", f"{agent_role} terminated task.")
                            except json.JSONDecodeError:
                                result_summary = f"{agent_role} terminated task with invalid arguments."
                            break
                elif last_message_from_specialist.role == "assistant" and isinstance(last_message_from_specialist.content, str):
                     result_summary = f"{agent_role} completed. Last thought: {last_message_from_specialist.content[:200]}..."
            
            await self._send_event("log", f"{agent_role} execution finished. Result: {result_summary[:100]}...", source=agent_role)
            logger.info(f"{agent_role} execution finished. Result: {result_summary[:100]}...")
            return result_summary

        except Exception as e:
            err_msg = f"Error: Failed to execute sub-task with {agent_role}. Reason: {str(e)}"
            await self._send_event("error", err_msg, source=agent_role)
            logger.error(f"Error running specialized agent {agent_role}: {e}", exc_info=True)
            return err_msg
        finally:
            if specialist_agent_instance:
                await specialist_agent_instance.cleanup()

    async def process_tool_call(self, tool_name: str, tool_arguments: Dict[str, Any]) -> str:
        await self._send_event("log", f"Orchestrator processing tool call: {tool_name} with args: {tool_arguments}")
        if tool_name == "delegate_task_to_specialist":
            agent_role = tool_arguments.get("agent_role")
            sub_task_prompt = tool_arguments.get("sub_task_prompt")
            if not agent_role or not sub_task_prompt:
                err_msg = "Error: `agent_role` and `sub_task_prompt` are required for delegation."
                await self._send_event("error", err_msg)
                return err_msg
            return await self._call_specialized_agent(agent_role, sub_task_prompt)
        else:
            if not hasattr(self, 'tools') or self.tools is None:
                 logger.error("CRITICAL: self.tools not found in process_tool_call before super call.")
                 return "Error: Tool system not initialized."
            # Ensure the tool exists before calling super().process_tool_call
            if not self.tools.get_tool(tool_name):
                err_msg = f"Error: Tool '{tool_name}' not found in OrchestratorAgent's toolset."
                await self._send_event("error", err_msg)
                logger.error(err_msg)
                return err_msg
            result = await super().process_tool_call(tool_name, tool_arguments)
            await self._send_event("log", f"Orchestrator tool {tool_name} result: {str(result)[:100]}...")
            return result

    async def run(self, prompt: str, event_q: Optional[queue.Queue] = None) -> None:
        if event_q:
            self.event_q = event_q