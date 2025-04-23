from typing import List

from autogen_core.models import LLMMessage
from ...core._core import ContextPlusChatCompletionContext
from ...conditions import MaxMessageCondition
from ..modifier._buffered_cutoff_modifier import BufferedCutoffModifier

def buffered_cutoff_chat_completion_context_builder(
    buffer_count: int,
    max_messages: int | None = None,
    initial_messages: List[LLMMessage] | None = None,
) -> ContextPlusChatCompletionContext:
    """Build a buffered modified chat completion context.
    Args:
        buffer_count (int): The size of the buffer.
        trigger_count (int | None): The size of the trigger. When is None, the trigger count is set to the buffer count.
        initial_messages (List[LLMMessage] | None): The initial messages.
    Returns:
        ContextPlusChatCompletionContext: The buffered modified chat completion context.
    """

    if max_messages is None:
        max_messages = buffer_count

    return ContextPlusChatCompletionContext(
        modifier_func=BufferedCutoffModifier(buffer_count),
        modifier_condition=MaxMessageCondition(
            max_messages=max_messages,
        ),
        initial_messages=initial_messages,
    )
