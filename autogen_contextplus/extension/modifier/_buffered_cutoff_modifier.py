from typing import List

from pydantic import BaseModel

from autogen_core import Component
from autogen_core.models import LLMMessage
from ...base._base_modifier import BaseModifierFunction


class BufferedCutoffModifierConfig(BaseModel):
    """Configuration for the BufferedCutoffModifier."""
    buffer_count: int = 5


class BufferedCutoffModifier(BaseModifierFunction, Component[BufferedCutoffModifierConfig]):
    """
    A modifier that cuts off messages to a specified buffer size.
    This modifier is useful for limiting the number of messages in a conversation
    to a manageable size, especially when dealing with large contexts.
    """
    component_config_schema = BufferedCutoffModifierConfig
    component_provider_override = "autogen_contextplus.extension.modifier.BufferedCutoffModifier"

    def __init__(self, buffer_count: int):
        self._buffer_count = buffer_count
        super().__init__(name="BufferedCutoffModifier")

    def __call__(self, messages: List[LLMMessage], non_modified_messages: List[LLMMessage]) -> List[LLMMessage]:
        if len(messages) > self._buffer_count:
            return messages[-self._buffer_count:]
        return messages
    
    def _to_config(self) -> BufferedCutoffModifierConfig:
        return BufferedCutoffModifierConfig(buffer_count=self._buffer_count)
    
    @classmethod
    def _from_config(cls, config: BufferedCutoffModifierConfig) -> "BufferedCutoffModifier":
        return cls(buffer_count=config.buffer_count)