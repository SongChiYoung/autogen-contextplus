import asyncio
from pprint import pprint
from typing import List
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken 

from autogen_contextplus.conditions import (
    MaxMessageCondition,
)
from autogen_contextplus import (
    ContextPlusChatCompletionContext
)
from autogen_core.models import LLMMessage



def buffered_summary(
    messages: List[LLMMessage],
    non_summarized_messages: List[LLMMessage],
) -> List[LLMMessage]:
    """Summarize the last `buffer_count` messages."""
    if len(messages) > 3:
        return messages[-3:]
    return messages


async def main()-> None:
    client = ReplayChatCompletionClient(
        chat_completions=[
            "paris",
            "seoul",
            "paris",
            "seoul",
        ]
    )
    
    context = ContextPlusChatCompletionContext(
        modifier_func = buffered_summary,
        modifier_condition = MaxMessageCondition(max_messages=2)
    )
    agent = AssistantAgent(
        "helper",
        model_client=client,
        system_message="You are a helpful agent",
        model_context=context
    )
    
    await agent.run(task="What is the capital of France?")
    res = await context.get_messages()
    print(f"[RESULTS] res:")
    pprint(res)
    print(f"[RESULTS] len_context : {len(res)}, context_type: {type(context)}")

    await agent.run(task="What is the capital of Korea?")
    res = await context.get_messages()
    print(f"[RESULTS] res:")
    pprint(res)
    print(f"[RESULTS] len_context : {len(res)}, context_type: {type(context)}")

    print("==========================")

    cancellation_token = CancellationToken() 
    await agent.on_reset(cancellation_token=cancellation_token)
    test = agent.dump_component()
    agent = AssistantAgent.load_component(test)
    new_context = agent.model_context

    await agent.run(task="What is the capital of France?")
    res = await new_context.get_messages()
    print(f"[RESULTS] res:")
    pprint(res)
    print(f"[RESULTS] len_context : {len(res)}, context_type: {type(new_context)}")
    await agent.run(task="What is the capital of Korea?")
    res = await new_context.get_messages()
    print(f"[RESULTS] res:")
    pprint(res)
    print(f"[RESULTS] len_context : {len(res)}, context_type: {type(new_context)}")
    
if __name__ == "__main__":
    asyncio.run(main())