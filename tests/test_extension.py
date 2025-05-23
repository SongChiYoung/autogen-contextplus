import pytest
from typing import List

from autogen_core import CancellationToken
from autogen_core.models import UserMessage, LLMMessage
from autogen_agentchat.base import TaskResult
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_contextplus import ContextPlusChatCompletionContext
from autogen_contextplus.extension.context import (
    buffered_cutoff_chat_completion_context_builder,
    buffered_summary_chat_completion_context_builder,
)


@pytest.mark.asyncio
async def test_buffered_cutoff_chat_completion_context_builder() -> None:
    """Test the buffered cutoff chat completion context builder."""
    buffer_count = 1
    max_messages = 1
    messages: List[LLMMessage] = [
        UserMessage(source="user", content="What is the capital of France?"),
        UserMessage(source="assistant", content="Paris"),
        UserMessage(source="user", content="What is the capital of Korea?"),
        UserMessage(source="assistant", content="Seoul"),
    ]

    context = buffered_cutoff_chat_completion_context_builder(
        buffer_count=buffer_count,
        max_messages=max_messages,
    )

    for message in messages:
        await context.add_message(message)

    assert len(context._messages) == 1  # type: ignore
    assert len(context._non_modified_messages) == 4  # type: ignore
    assert context._messages[0].content == "Seoul"  # type: ignore
    assert context._non_modified_messages[0].content == "What is the capital of France?"  # type: ignore
    assert context._non_modified_messages[1].content == "Paris"  # type: ignore
    assert context._non_modified_messages[2].content == "What is the capital of Korea?"  # type: ignore
    assert context._non_modified_messages[3].content == "Seoul"  # type: ignore


@pytest.mark.asyncio
async def test_buffered_cutoff_chat_completion_context_serialize() -> None:
    """Test the buffered cutoff chat completion context builder."""
    buffer_count = 1
    max_messages = 1

    context = buffered_cutoff_chat_completion_context_builder(
        buffer_count=buffer_count,
        max_messages=max_messages,
    )

    serialized_context = context.dump_component()
    assert serialized_context is not None

    deserialized_context = ContextPlusChatCompletionContext.load_component(serialized_context)
    assert deserialized_context is not None

    assert context._modifier_func.name == deserialized_context._modifier_func.name  # type: ignore

    messages: List[LLMMessage] = [
        UserMessage(source="user", content="What is the capital of France?"),
        UserMessage(source="assistant", content="Paris"),
        UserMessage(source="user", content="What is the capital of Korea?"),
        UserMessage(source="assistant", content="Seoul"),
    ]
    for message in messages:
        await deserialized_context.add_message(message)

    assert len(deserialized_context._messages) == 1  # type: ignore
    assert len(deserialized_context._non_modified_messages) == 4  # type: ignore
    assert deserialized_context._messages[0].content == "Seoul"  # type: ignore
    assert deserialized_context._non_modified_messages[0].content == "What is the capital of France?"  # type: ignore
    assert deserialized_context._non_modified_messages[1].content == "Paris"  # type: ignore
    assert deserialized_context._non_modified_messages[2].content == "What is the capital of Korea?"  # type: ignore
    assert deserialized_context._non_modified_messages[3].content == "Seoul"  # type: ignore


@pytest.mark.asyncio
async def test_buffered_summary_chat_completion_context_builder() -> None:
    """Test the buffered summary chat completion context builder."""
    max_messages = 4
    messages: List[LLMMessage] = [
        UserMessage(source="user", content="What is the capital of France?"),
        UserMessage(source="assistant", content="Paris"),
        UserMessage(source="user", content="What is the capital of Korea?"),
        UserMessage(source="assistant", content="Seoul"),
    ]

    client = ReplayChatCompletionClient(chat_completions=["SUMMARY1", "SUMMARY2"])

    context = buffered_summary_chat_completion_context_builder(
        max_messages=max_messages,
        model_client=client,
        system_message="Summarize the conversation so far for your own memory",
        summary_format="This portion of conversation has been summarized as follow: {summary}",
        summary_start=1,
        summary_end=-1,
    )

    for message in messages:
        await context.add_message(message)

    assert len(context._messages) == 3  # type: ignore
    assert len(context._non_modified_messages) == 4  # type: ignore
    assert context._messages[0].content == "What is the capital of France?"  # type: ignore
    assert context._messages[1].content == "SUMMARY1"  # type: ignore
    assert context._messages[2].content == "Seoul"  # type: ignore


@pytest.mark.asyncio
async def test_buffered_summary_chat_completion_context_serialize() -> None:
    """Test the buffered summary chat completion context builder."""
    max_messages = 4

    client = ReplayChatCompletionClient(chat_completions=["SUMMARY1", "SUMMARY2"])

    context = buffered_summary_chat_completion_context_builder(
        max_messages=max_messages,
        model_client=client,
        system_message="Summarize the conversation so far for your own memory",
        summary_format="This portion of conversation has been summarized as follow: {summary}",
        summary_start=1,
        summary_end=-1,
    )

    serialized_context = context.dump_component()
    assert serialized_context is not None

    print(f"[RESULTS] serialized_context: {serialized_context}")

    deserialized_context = ContextPlusChatCompletionContext.load_component(serialized_context)
    assert deserialized_context is not None

    assert context._modifier_func.name == deserialized_context._modifier_func.name  # type: ignore
    messages: List[LLMMessage] = [
        UserMessage(source="user", content="What is the capital of France?"),
        UserMessage(source="assistant", content="Paris"),
        UserMessage(source="user", content="What is the capital of Korea?"),
        UserMessage(source="assistant", content="Seoul"),
    ]
    for message in messages:
        await deserialized_context.add_message(message)
    assert len(deserialized_context._messages) == 3  # type: ignore
    assert len(deserialized_context._non_modified_messages) == 4  # type: ignore
    assert deserialized_context._messages[0].content == "What is the capital of France?"  # type: ignore
    assert deserialized_context._messages[1].content == "SUMMARY1"  # type: ignore
    assert deserialized_context._messages[2].content == "Seoul"  # type: ignore


@pytest.mark.asyncio
async def test_buffered_cutoff_chat_completion_with_agent() -> None:
    """Test the buffered cutoff chat completion context builder with agent."""
    buffer_count = 1
    max_messages = 1
    messages = [
        "What is the capital of France?",
        "What is the capital of Korea?",
    ]
    responses = [
        "Paris",
        "Seoul",
        "Paris",
        "Seoul",
    ]

    async def _buffered_cutoff(agnet: AssistantAgent) -> None:
        agent_responses: List[TaskResult] = []
        for message in messages:
            agent_responses.append(await agent.run(task=message))

        print(f"[RESULTS] agent_responses: {agent_responses}")
        print(f"[RESULTS] len(agent_responses): {len(agent_responses)}")

        assert len(agent_responses) == 2
        assert agent_responses[0].messages[1].to_text() == "Paris"
        assert agent_responses[1].messages[1].to_text() == "Seoul"

        assert len(context._messages) == 1  # type: ignore
        assert len(context._non_modified_messages) == 4  # type: ignore
        assert context._messages[0].content == "Seoul"  # type: ignore


    client = ReplayChatCompletionClient(
        chat_completions=responses
    )

    context = buffered_cutoff_chat_completion_context_builder(
        buffer_count=buffer_count,
        max_messages=max_messages,
    )

    agent = AssistantAgent(
        "tester",
        model_client=client,
        system_message="You are a helpful agent",
        model_context=context,
    )

    await _buffered_cutoff(agent)

    await agent.on_reset(cancellation_token=CancellationToken())
    serialized_agent = agent.dump_component()
    assert serialized_agent is not None

    deserialized_agent = AssistantAgent.load_component(serialized_agent)
    assert deserialized_agent is not None

    await _buffered_cutoff(deserialized_agent)