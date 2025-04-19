from abc import ABC, abstractmethod
from typing import Any, List, Mapping

from pydantic import BaseModel

from autogen_core import ComponentBase
from autogen_core.models import LLMMessage


class BaseModifier(ABC, ComponentBase[BaseModel]):
    component_type = "modifier"

    def __init__(
        self,
        name: str,
    ) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def run(self, messages: List[LLMMessage], non_modified_messages: List[LLMMessage]) -> List[LLMMessage]: ...

    def save_state_json(self) -> Mapping[str, Any]:
        return {}

    def load_state_json(self, state: Mapping[str, Any]) -> None:
        pass
