import time
from typing import List, Sequence

from pydantic import BaseModel
from typing_extensions import Self

from autogen_core import Component, ComponentModel
from autogen_core.models import ChatCompletionClient, FunctionExecutionResultMessage, LLMMessage
from autogen_core.tools import ToolSchema
from ..base import ContextPlusCondition, ContextPlusException
from ..base.types import BaseContextMessageTypes, ContextMessage, LLMMessageInstance, TriggerMessage


class TriggerMessageConditionConfig(BaseModel):
    pass


class TriggerMessageCondition(ContextPlusCondition, Component[TriggerMessageConditionConfig]):
    """Trigger the conversation if a TriggerMessage is received."""

    component_config_schema = TriggerMessageConditionConfig
    component_provider_override = "autogen_contextplus.condidions.TriggerMessageCondition"

    def __init__(self) -> None:
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    async def __call__(self, messages: Sequence[ContextMessage]) -> TriggerMessage | None:
        if self._triggered:
            raise ContextPlusException("Trigger condition has already been reached")
        for message in messages:
            if isinstance(message, TriggerMessage):
                self._triggered = True
                return TriggerMessage(content="Triggered message received", source="TriggerMessageCondition")
        return None

    async def reset(self) -> None:
        self._triggered = False

    def _to_config(self) -> TriggerMessageConditionConfig:
        return TriggerMessageConditionConfig()

    @classmethod
    def _from_config(cls, config: TriggerMessageConditionConfig) -> Self:
        return cls()


class MaxMessageConditionConfig(BaseModel):
    max_messages: int
    # TODO : include_agent_event: bool = False


class MaxMessageCondition(ContextPlusCondition, Component[MaxMessageConditionConfig]):
    """Trigger the conversation after a maximum number of messages have been exchanged.

    Args:
        max_messages: The maximum number of messages allowed in the conversation.
    """

    component_config_schema = MaxMessageConditionConfig
    component_provider_override = "autogen_contextplus.conditions.MaxMessageCondition"

    def __init__(self, max_messages: int) -> None:
        self._max_messages = max_messages
        self._message_count = 0

    @property
    def triggered(self) -> bool:
        return self._message_count >= self._max_messages

    async def __call__(self, messages: Sequence[ContextMessage]) -> TriggerMessage | None:
        if self.triggered:
            raise ContextPlusException("Trigger condition has already been reached")
        self._message_count += len([m for m in messages if isinstance(m, BaseContextMessageTypes)])
        if self._message_count >= self._max_messages:
            return TriggerMessage(
                content=f"Maximum number of messages {self._max_messages} reached, current message count: {self._message_count}",
                source="MaxContextPlus",
            )
        return None

    async def reset(self) -> None:
        self._message_count = 0

    def _to_config(self) -> MaxMessageConditionConfig:
        return MaxMessageConditionConfig(max_messages=self._max_messages)

    @classmethod
    def _from_config(cls, config: MaxMessageConditionConfig) -> Self:
        return cls(max_messages=config.max_messages)


class TextMentionConditionConfig(BaseModel):
    text: str
    sources: Sequence[str] | None = None


class TextMentionCondition(ContextPlusCondition, Component[TextMentionConditionConfig]):
    """Trigger the conversation if a specific text is mentioned.


    Args:
        text: The text to look for in the messages.
        sources: Check only messages of the specified agents for the text to look for.
    """

    component_config_schema = TextMentionConditionConfig
    component_provider_override = "autogen_contextplus.conditions.TextMentionCondition"

    def __init__(self, text: str, sources: Sequence[str] | None = None) -> None:
        self._trigger_text = text
        self._triggered = False
        self._sources = sources

    @property
    def triggered(self) -> bool:
        return self._triggered

    async def __call__(self, messages: Sequence[ContextMessage]) -> TriggerMessage | None:
        if self._triggered:
            raise ContextPlusException("Trigger condition has already been reached")
        for message in [m for m in messages if isinstance(m, BaseContextMessageTypes)]:
            if self._sources is not None and message.source not in self._sources:
                continue

            content = message.content
            if self._trigger_text in content:
                self._triggered = True
                return TriggerMessage(
                    content=f"Text '{self._trigger_text}' mentioned", source="TextMentionCondition"
                )
        return None

    async def reset(self) -> None:
        self._triggered = False

    def _to_config(self) -> TextMentionConditionConfig:
        return TextMentionConditionConfig(
            text=self._trigger_text,
            sources=self._sources,
        )

    @classmethod
    def _from_config(cls, config: TextMentionConditionConfig) -> Self:
        return cls(
            text=config.text,
            sources=config.sources,
        )


class TokenUsageConditionConfig(BaseModel):
    model_client: ComponentModel
    token_limit: int | None = None
    tool_schema: List[ToolSchema] | None = None
    initial_messages: List[LLMMessage] | None = None


class TokenUsageCondition(ContextPlusCondition, Component[TokenUsageConditionConfig]):
    """(Experimental) A token based chat completion context maintains a view of the context up to a token limit.

    .. note::

        Added in v0.5.10. This is an experimental component and may change in the future.

    Args:
        model_client (ChatCompletionClient): The model client to use for token counting.
            The model client must implement the :meth:`~autogen_core.models.ChatCompletionClient.count_tokens`
            and :meth:`~autogen_core.models.ChatCompletionClient.remaining_tokens` methods.
        token_limit (int | None): The maximum number of tokens to keep in the context
            using the :meth:`~autogen_core.models.ChatCompletionClient.count_tokens` method.
            If None, the context will be limited by the model client using the
            :meth:`~autogen_core.models.ChatCompletionClient.remaining_tokens` method.
        tool_schema (List[ToolSchema] | None): A list of tool schema to use in the context.
        initial_messages (List[LLMMessage] | None): A list of initial messages to include in the context.

    """

    component_config_schema = TokenUsageConditionConfig
    component_provider_override = "autogen_contextplus.conditions.TokenUsageCondition"

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        token_limit: int | None = None,
        tool_schema: List[ToolSchema] | None = None,
        initial_messages: List[LLMMessage] | None = None,
    ) -> None:
        if token_limit is not None and token_limit <= 0:
            raise ValueError("token_limit must be greater than 0.")
        self._token_limit = token_limit
        self._total_token = 0
        self._model_client = model_client
        self._tool_schema = tool_schema or []
        if initial_messages is not None:
            self._initial_messages = initial_messages
        else:
            self._initial_messages = []

    @property
    def triggered(self) -> bool:
        _triggered = False
        if self._token_limit is None:
            if (
                self._model_client.remaining_tokens(
                    self._initial_messages,
                    tools=self._tool_schema,
                )
                < 0
            ):
                _triggered = True
        else:
            if self._total_token >= self._token_limit:
                _triggered = True
        return _triggered

    async def __call__(self, messages: Sequence[ContextMessage]) -> TriggerMessage | None:
        if self.triggered:
            raise ContextPlusException("Trigger condition has already been reached")

        _messages = [m for m in messages if isinstance(m, LLMMessageInstance)]
        self._initial_messages.extend(_messages)

        self._total_token += self._model_client.count_tokens(_messages, tools=self._tool_schema)

        if self.triggered:
            content = f"Token usage limit reached, total token count: {self._total_token}."
            return TriggerMessage(content=content, source="TokenUsageCondition")
        return None

    async def reset(self) -> None:
        self._total_token = 0
        self._initial_messages = []

    def _to_config(self) -> TokenUsageConditionConfig:
        return TokenUsageConditionConfig(
            model_client=self._model_client.dump_component(),
            token_limit=self._token_limit,
            tool_schema=self._tool_schema,
            initial_messages=self._initial_messages,
        )

    @classmethod
    def _from_config(cls, config: TokenUsageConditionConfig) -> Self:
        return cls(
            model_client=ChatCompletionClient.load_component(config.model_client),
            token_limit=config.token_limit,
            tool_schema=config.tool_schema,
            initial_messages=config.initial_messages,
        )


class TimeoutConditionConfig(BaseModel):
    timeout_seconds: float


class TimeoutCondition(ContextPlusCondition, Component[TimeoutConditionConfig]):
    """Trigger the conversation after a specified duration has passed.

    Args:
        timeout_seconds: The maximum duration in seconds before triggering the conversation.
    """

    component_config_schema = TimeoutConditionConfig
    component_provider_override = "autogen_contextplus.conditions.TimeoutCondition"

    def __init__(self, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._start_time = time.monotonic()
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    async def __call__(self, messages: Sequence[ContextMessage]) -> TriggerMessage | None:
        if self._triggered:
            raise ContextPlusException("Trigger condition has already been reached")

        if (time.monotonic() - self._start_time) >= self._timeout_seconds:
            self._triggered = True
            return TriggerMessage(
                content=f"Timeout of {self._timeout_seconds} seconds reached", source="TimeoutCondition"
            )
        return None

    async def reset(self) -> None:
        self._start_time = time.monotonic()
        self._triggered = False

    def _to_config(self) -> TimeoutConditionConfig:
        return TimeoutConditionConfig(timeout_seconds=self._timeout_seconds)

    @classmethod
    def _from_config(cls, config: TimeoutConditionConfig) -> Self:
        return cls(timeout_seconds=config.timeout_seconds)


class ExternalConditionConfig(BaseModel):
    pass


class ExternalCondition(ContextPlusCondition, Component[ExternalConditionConfig]):
    """A Trigger condition that is externally controlled
    by calling the :meth:`set` method.

    Example:

    .. code-block:: python
        from autogen_contextplus.conditions import ExternalCondition

        trigger_condition = ExternalCondition()
        trigger_condition.set()
        # Trigger condition is now set to True.

    """

    component_config_schema = ExternalConditionConfig
    component_provider_override = "autogen_contextplus.conditions.ExternalCondition"

    def __init__(self) -> None:
        self._triggered = False
        self._setted = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    def set(self) -> None:
        """Set the trigger condition to triggered.

        This method manually sets the trigger condition to `True`, indicating that
        the external trigger has been activated. Once set, the condition will remain
        triggered until the `reset` method is called to clear it.
        """
        """Set the trigger condition to triggered."""
        self._setted = True

    async def __call__(self, messages: Sequence[ContextMessage]) -> TriggerMessage | None:
        if self._triggered:
            raise ContextPlusException("Trigger condition has already been reached")
        if self._setted:
            self._triggered = True
            return TriggerMessage(content="External trigger requested", source="ExternalCondition")
        return None

    async def reset(self) -> None:
        self._triggered = False
        self._setted = False

    def _to_config(self) -> ExternalConditionConfig:
        return ExternalConditionConfig()

    @classmethod
    def _from_config(cls, config: ExternalConditionConfig) -> Self:
        return cls()


class SourceMatchConditionConfig(BaseModel):
    sources: List[str]


class SourceMatchCondition(ContextPlusCondition, Component[SourceMatchConditionConfig]):
    """Trigger the conversation after a specific source responds.

    Args:
        sources (List[str]): List of source names to trigger the conversation.

    Raises:
        ContextPlusException: If the trigger condition has already been reached.
    """

    component_config_schema = SourceMatchConditionConfig
    component_provider_override = "autogen_contextplus.conditions.SourceMatchCondition"

    def __init__(self, sources: List[str]) -> None:
        self._sources = sources
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    async def __call__(self, messages: Sequence[ContextMessage]) -> TriggerMessage | None:
        if self._triggered:
            raise ContextPlusException("Trigger condition has already been reached")

        _messages = [m for m in messages if isinstance(m, BaseContextMessageTypes)]

        if not _messages:
            return None
        for message in _messages:
            if message.source in self._sources:
                self._triggered = True
                return TriggerMessage(content=f"'{message.source}' answered", source="SourceMatchCondition")
        return None

    async def reset(self) -> None:
        self._triggered = False

    def _to_config(self) -> SourceMatchConditionConfig:
        return SourceMatchConditionConfig(sources=self._sources)

    @classmethod
    def _from_config(cls, config: SourceMatchConditionConfig) -> Self:
        return cls(sources=config.sources)


class TextMessageConditionConfig(BaseModel):
    """Configuration for the TextMessageCondition trigger condition."""

    source: str | None = None
    """The source of the text message to trigger the conversation."""


class TextMessageCondition(ContextPlusCondition, Component[TextMessageConditionConfig]):
    """Trigger the conversation if a :class:`~autogen_core.models.LLMMessage` and it's content type is `str` is received.

    This trigger condition checks for LLMMessage instances in the message sequence. When a LLMMessage is found,
    it trigger the conversation if either:
    - No source was specified (trigger on any text message)
    - The message source matches the specified source

    Args:
        source (str | None, optional): The source name to match against incoming messages. If None, matches any source.
            Defaults to None.
    """

    component_config_schema = TextMessageConditionConfig
    component_provider_override = "autogen_contextplus.conditions.TextMessageCondition"

    def __init__(self, source: str | None = None) -> None:
        self._triggered = False
        self._source = source

    @property
    def triggered(self) -> bool:
        return self._triggered

    async def __call__(self, messages: Sequence[ContextMessage]) -> TriggerMessage | None:
        if self._triggered:
            raise ContextPlusException("Trigger condition has already been reached")
        for message in messages:
            if (
                isinstance(message, BaseContextMessageTypes)
                and isinstance(message.content, str)
                and (self._source is None or message.source == self._source)
            ):
                self._triggered = True
                return TriggerMessage(
                    content=f"Text message received from '{message.source}'", source="TextMessageCondition"
                )
        return None

    async def reset(self) -> None:
        self._triggered = False

    def _to_config(self) -> TextMessageConditionConfig:
        return TextMessageConditionConfig(source=self._source)

    @classmethod
    def _from_config(cls, config: TextMessageConditionConfig) -> Self:
        return cls(source=config.source)


class FunctionCallConditionConfig(BaseModel):
    """Configuration for the :class:`FunctionCallCondition` trigger condition."""

    function_name: str


class FunctionCallCondition(ContextPlusCondition, Component[FunctionCallConditionConfig]):
    """Trigger the conversation if a :class:`~autogen_core.models.FunctionExecutionResult`
    with a specific name was received.

    Args:
        function_name (str): The name of the function to look for in the messages.

    Raises:
        ContextPlusException: If the trigger condition has already been reached.
    """

    component_config_schema = FunctionCallConditionConfig
    component_provider_override = "autogen_contextplus.conditions.FunctionCallCondition"

    def __init__(self, function_name: str) -> None:
        self._triggered = False
        self._function_name = function_name

    @property
    def triggered(self) -> bool:
        return self._triggered

    async def __call__(self, messages: Sequence[ContextMessage]) -> TriggerMessage | None:
        if self._triggered:
            raise ContextPlusException("Trigger condition has already been reached")
        for message in messages:
            if isinstance(message, FunctionExecutionResultMessage):
                for execution in message.content:
                    if execution.name == self._function_name:
                        self._triggered = True
                        return TriggerMessage(
                            content=f"Function '{self._function_name}' was executed.",
                            source="FunctionCallCondition",
                        )
        return None

    async def reset(self) -> None:
        self._triggered = False

    def _to_config(self) -> FunctionCallConditionConfig:
        return FunctionCallConditionConfig(
            function_name=self._function_name,
        )

    @classmethod
    def _from_config(cls, config: FunctionCallConditionConfig) -> Self:
        return cls(
            function_name=config.function_name,
        )
