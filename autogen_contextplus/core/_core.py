from typing import List

from pydantic import BaseModel
from typing_extensions import Self

from autogen_core import ComponentModel, Component
from autogen_core.models import LLMMessage
from autogen_core.model_context import ChatCompletionContext
from ..base import ContextPlusCondition
from ..base.types import ModifierFunction
from ..modifier import Modifier


class ContextPlusChatCompletionContextConfig(BaseModel):
    modifier_func: ComponentModel
    modifier_condition: ComponentModel
    initial_messages: List[LLMMessage] | None = None
    non_modified_messages: List[LLMMessage] | None = None


class ContextPlusChatCompletionContext(ChatCompletionContext, Component[ContextPlusChatCompletionContextConfig]):
    """A summarized chat completion context that summarizes the messages in the context
    using a summarizing function. The summarizing function is set at initialization.
    The summarizing condition is used to determine when to summarize the messages.

    Args:
        summarizing_func (Callable[[List[LLMMessage]], List[LLMMessage]]): The function to summarize the messages.
        summarizing_condition (MessageCompletionCondition): The condition to determine when to summarize the messages.
        initial_messages (List[LLMMessage] | None): The initial messages.

    Example:
        .. code-block:: python

            from typing import List

            from autogen_core.model_context import SummarizedChatCompletionContext
            from autogen_core.models import LLMMessage


            def summarizing_func(messages: List[LLMMessage]) -> List[LLMMessage]:
                # Implement your summarizing function here.
                return messages


            summarizing_condition = MessageCompletionCondition()

            context = SummarizedChatCompletionContext(
                summarizing_func=summarizing_func,
                summarizing_condition=summarizing_condition,
            )

        .. code-block:: python
            import asyncio
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.model_context import SummarizedChatCompletionContext
            from autogen_core.model_context.conditions import MaxMessageCompletion
            from autogen_ext.summary import buffered_summary, buffered_summarized_chat_completion_context


            client = OpenAIChatCompletionClient(model="claude-3-haiku-20240307")

            context = SummarizedChatCompletionContext(
                summarizing_func=buffered_summary(buffer_count=2), summarizing_condition=MaxMessageCompletion(max_messages=2)
            )

            agent = AssistantAgent(
                "helper", model_client=client, system_message="You are a helpful agent", model_context=context
            )

    """

    component_config_schema = ContextPlusChatCompletionContextConfig
    component_provider_override = "autogen_contextplus.ContextPlusChatCompletionContext"

    def __init__(
        self,
        modifier_func: ModifierFunction | Modifier,
        modifier_condition: ContextPlusCondition,
        initial_messages: List[LLMMessage] | None = None,
        non_modified_messages: List[LLMMessage] | None = None,
    ) -> None:
        super().__init__(initial_messages)

        self._non_modified_messages: List[LLMMessage] = []
        if non_modified_messages is not None:
            self._non_modified_messages.extend(non_modified_messages)

        self._non_modified_messages.extend(self._messages)

        self._modifier_func: Modifier
        if isinstance(modifier_func, Modifier):
            # If the summarizing function is a tool, use it directly.
            self._modifier_func = modifier_func
        elif callable(modifier_func):
            self._modifier_func = Modifier(func=modifier_func)
        else:
            raise ValueError("modifier_func must be a callable or a Modifier.")
        self._modifier_condition = modifier_condition

    async def add_message(self, message: LLMMessage) -> None:
        """Add a message to the context."""
        self._non_modified_messages.append(message)
        await super().add_message(message)

        # Check if the summarizing condition is met.
        await self._modifier_condition(self._messages)
        if self._modifier_condition.triggered:
            # If the condition is met, summarize the messages.
            await self.modifiy()
            await self._modifier_condition.reset()

    async def get_messages(self) -> List[LLMMessage]:
        """Get the messages in the context."""
        return self._messages
    
    async def clear(self) -> None:
        """Clear the context."""
        await super().clear()
        self._non_modified_messages = []

    async def modifiy(self) -> None:
        """Modifiy the messages in the context using the summarizing function."""
        modified_message = await self._modifier_func.run(self._messages, self._non_modified_messages)
        self._messages = modified_message

    def _to_config(self) -> ContextPlusChatCompletionContextConfig:
        return ContextPlusChatCompletionContextConfig(
            modifier_func=self._modifier_func.dump_component(),
            modifier_condition=self._modifier_condition.dump_component(),
            initial_messages=self._initial_messages,
        )

    @classmethod
    def _from_config(cls, config: ContextPlusChatCompletionContextConfig) -> Self:
        """Create a summarized chat completion context from a config."""
        return cls(
            modifier_func=Modifier.load_component(config.modifier_func),
            modifier_condition=ContextPlusCondition.load_component(config.modifier_condition),
            initial_messages=config.initial_messages,
            non_modified_messages=config.non_modified_messages,
        )
