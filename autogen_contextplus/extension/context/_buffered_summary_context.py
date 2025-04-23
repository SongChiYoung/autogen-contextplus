from typing import List

from autogen_core.models import LLMMessage, ChatCompletionClient
from ...core._core import ContextPlusChatCompletionContext
from ...conditions import MaxMessageCondition
from ..agent import BufferedSummaryAgent

def buffered_summary_chat_completion_context_builder(
    max_messages: int,
    model_client: ChatCompletionClient,
    system_message: str = "Summarize the conversation so far for your own memory",
    summary_format: str = "This portion of conversation has been summarized as follow: {summary}",
    summary_start: int = 0,
    summary_end: int = 0,
    initial_messages: List[LLMMessage] | None = None,
) -> ContextPlusChatCompletionContext:


    return ContextPlusChatCompletionContext(
        modifier_func=BufferedSummaryAgent(
            name="BufferedSummaryAgent",
            model_client=model_client,
            system_message=system_message,
            summary_format=summary_format,
            summary_start=summary_start,
            summary_end=summary_end,
        ),
        modifier_condition=MaxMessageCondition(
            max_messages=max_messages,
        ),
        initial_messages=initial_messages,
    )
