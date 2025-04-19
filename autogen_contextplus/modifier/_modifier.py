import functools
import warnings
from textwrap import dedent
from typing import Any, List, Sequence

from pydantic import BaseModel
from typing_extensions import Self

from autogen_core import Component
from autogen_core._function_utils import (
    get_typed_signature,
)
from autogen_core.code_executor import Import
from autogen_core.code_executor._func_with_reqs import import_to_str, to_code
from autogen_core.models import LLMMessage
from ..base import BaseModifier, BaseModifierAgent
from ..base.types import ModifierFunction


class ModifiierConfig(BaseModel):
    """Configuration for a summary function."""

    source_code: str | None = None
    agent: BaseModifierAgent | None = None
    name: str
    global_imports: Sequence[Import]


class Modifiier(BaseModifier, Component[ModifiierConfig]):
    component_provider_override = "autogen_core.model_context.conditions.SummaryFunction"
    component_config_schema = ModifiierConfig

    def __init__(
        self,
        func: ModifierFunction | None = None,
        agent: BaseModifierAgent | None = None,
        name: str | None = None,
        global_imports: Sequence[Import] = [],
        strict: bool = False,
    ) -> None:
        self._func = func
        self._agent = agent
        self._global_imports = global_imports
        if func is not None:
            self._signature = get_typed_signature(func)
            func_name = name or func.func.__name__ if isinstance(func, functools.partial) else name or func.__name__
        if agent is not None:
            if not isinstance(agent, BaseModifierAgent):
                raise TypeError(f"Expected a BaseChatAgent but got {type(agent)}")
            func_name = name or agent.name
        if func is None and agent is None:
            raise ValueError("Either a function or an agent must be provided.")
        if func is not None and agent is not None:
            raise ValueError("Only one of a function or an agent can be provided.")
        super().__init__(func_name)

    async def run(self, messages: List[LLMMessage], non_summary_messages: List[LLMMessage]) -> List[LLMMessage]:
        if self._func is not None:
            result = self._func(messages, non_summary_messages)
        if self._agent is not None:
            result = await self._agent.run(task=messages, original_task=non_summary_messages)
        return result

    def _to_config(self) -> ModifiierConfig:
        if self._func is None:
            return ModifiierConfig(
                source_code=dedent(to_code(self._func)),
                global_imports=self._global_imports,
                name=self.name,
            )
        if self._agent is not None:
            return ModifiierConfig(
                agent=self._agent,
                global_imports=self._global_imports,
                name=self.name,
            )

    @classmethod
    def _from_config(cls, config: ModifiierConfig) -> Self:
        exec_globals: dict[str, Any] = {}

        # Execute imports first
        for import_stmt in config.global_imports:
            import_code = import_to_str(import_stmt)
            try:
                exec(import_code, exec_globals)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Failed to import {import_code}: Module not found. Please ensure the module is installed."
                ) from e
            except ImportError as e:
                raise ImportError(f"Failed to import {import_code}: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error while importing {import_code}: {str(e)}") from e
            
        if config.source_code is not None:
            warnings.warn(
                "\n⚠️  SECURITY WARNING ⚠️\n"
                "Loading a FunctionTool from config will execute code to import the provided global imports and and function code.\n"
                "Only load configs from TRUSTED sources to prevent arbitrary code execution.",
                UserWarning,
                stacklevel=2,
            )

            # Execute function code
            try:
                exec(config.source_code, exec_globals)
                func_name = config.source_code.split("def ")[1].split("(")[0]
            except Exception as e:
                raise ValueError(f"Could not compile and load function: {e}") from e

            # Get function and verify it's callable
            func: ModifierFunction = exec_globals[func_name]
            if not callable(func):
                raise TypeError(f"Expected function but got {type(func)}")

            return cls(func=func, name=config.name, global_imports=config.global_imports)
        if config.agent is not None:
            return cls(agent=config.agent, name=config.name, global_imports=config.global_imports)
