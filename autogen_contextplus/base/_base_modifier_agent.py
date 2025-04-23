from abc import ABC, abstractmethod
from typing import List, Any, Mapping
from pydantic import BaseModel

from autogen_core.models import ChatCompletionClient, LLMMessage, SystemMessage
from autogen_core import Component, ComponentBase, ComponentModel, CancellationToken
from autogen_agentchat.utils import remove_images


class BaseModifierAgentConfig(BaseModel):
    """The declarative configuration for the base modifier agent."""

    name: str
    model_client: ComponentModel
    system_message: str = "Summarize the conversation so far."
    model_client_stream: bool = False
    kwargs: Mapping[str, Any] = {}


class BaseModifierAgent(ABC, ComponentBase[BaseModel], Component[BaseModifierAgentConfig]):
    """
    Base class for modifier agents.
    """
    component_config_schema = BaseModifierAgentConfig
    component_type = "modifier_agent"

    def __init__(
            self,
            name: str,
            model_client: ChatCompletionClient,
            system_message: str = "Summarize the conversation so far.",
            model_client_stream: bool = False,
            **kwargs: Any,
        ):
        self._name = name
        self._model_client = model_client
        self._system_message = [SystemMessage(content=system_message)]
        self._model_client_stream = model_client_stream
        self._kwargs = kwargs


    @property
    def name(self) -> str:
        """The name of the agent."""
        return self._name

    @abstractmethod
    async def run(
        self,
        task: List[LLMMessage] | None = None,
        original_task: List[LLMMessage] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> List[LLMMessage]:
        """
        Run the summary agent.
        Args:
            task: The task to run.
            original_task: The original task to run.
        Returns:
            The result of the run.
        """
        ...

    @staticmethod
    def _get_compatible_context(model_client: ChatCompletionClient, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Ensure that the messages are compatible with the underlying client, by removing images if needed."""
        if model_client.model_info["vision"]:
            return messages
        else:
            return remove_images(messages)
        
    def _to_config(self) -> BaseModifierAgentConfig:
        """Convert the agent to a config."""
        return BaseModifierAgentConfig(
            name=self._name,
            model_client=self._model_client.dump_component(),
            system_message=self._system_message[0].content,
            model_client_stream=self._model_client_stream,
            kwargs=self._kwargs,
        )
    
    @classmethod
    def _from_config(cls, config: BaseModifierAgentConfig) -> "BaseModifierAgent":
        """Create the agent from a config."""
        return cls(
            name=config.name,
            model_client=ChatCompletionClient.load_component(config.model_client),
            system_message=config.system_message,
            model_client_stream=config.model_client_stream,
            **config.kwargs
        )