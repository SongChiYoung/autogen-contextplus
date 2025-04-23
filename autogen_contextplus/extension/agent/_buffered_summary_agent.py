from typing import List

from autogen_core.models import ChatCompletionClient, LLMMessage, UserMessage, SystemMessage
from autogen_core import CancellationToken
from ...base import BaseModifierAgent


class BufferedSummaryAgent(BaseModifierAgent):
    """
    A buffered summary agent that summarizes the conversation so far.
    It uses a summarizing function to summarize the conversation.
    The summarizing function is set at initialization.
    The summarizing condition is used to determine when to summarize the conversation.
    """
    component_provider_override = "autogen_contextplus.extension.agent.BufferedSummaryAgent"

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        system_message: str = "Summarize the conversation so far for your own memory",
        model_client_stream: bool = False,
        summary_format: str = "This portion of conversation has been summarized as follow: {summary}",
        summary_start: int = 0,
        summary_end: int = 0,
    ):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
            model_client_stream=model_client_stream,
            summary_format=summary_format,
            summary_start=summary_start,
            summary_end=summary_end,
        )

        self._summary_format = summary_format
        self._summary_start = summary_start
        self._summary_end = summary_end
        self._summary_message = [SystemMessage(content=f"{system_message}\n{summary_format}")]

    async def run(
        self,
        task: List[LLMMessage] | None = None,
        original_task: List[LLMMessage] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> List[LLMMessage]:
        """Run the summary agent."""
        if task is None:
            task = []
        if self._summary_start > 0 and self._summary_end < 0:
            prefix = task[:self._summary_start]
            suffix = task[self._summary_end:]
            task = task[self._summary_start:self._summary_end]
        elif self._summary_start > 0:
            prefix = task[:self._summary_start]
            suffix = []
            task = task[self._summary_start:]
        elif self._summary_end < 0:
            prefix = []
            suffix = task[self._summary_end:]
            task = task[:self._summary_end]
        else:
            prefix = []
            suffix = []

        task = self._summary_message + task
        task = BaseModifierAgent._get_compatible_context(
            self._model_client, task
        )

        result = await self._model_client.create(
            messages=task,
            cancellation_token=cancellation_token,
        )
        if isinstance(result.content, str):
            messages = prefix + [UserMessage(content=result.content,source="summary")] + suffix
        else:
            raise ValueError(
                f"Expected a string, but got {type(result.content)}"
            )

        return messages