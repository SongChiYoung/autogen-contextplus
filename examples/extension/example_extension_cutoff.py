import asyncio
from pprint import pprint
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken 

from autogen_contextplus.extension.context import (
    buffered_cutoff_chat_completion_context_builder,
)


async def main()-> None:
    client = ReplayChatCompletionClient(
        chat_completions=[
            "paris",
            "seoul",
            "paris",
            "seoul",
        ]
    )
    
    context = buffered_cutoff_chat_completion_context_builder(
        buffer_count=2,
        max_messages=2,
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
