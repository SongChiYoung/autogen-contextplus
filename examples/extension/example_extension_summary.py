import asyncio
from pprint import pprint
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken 

from autogen_contextplus.extension.context import (
    buffered_summary_chat_completion_context_builder,
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
    client_summary = ReplayChatCompletionClient(
        chat_completions=[
            "SUMMARY",
        ]
    )

    context = buffered_summary_chat_completion_context_builder(
        max_messages=4,
        summary_start=1,
        summary_end=-1,
        model_client=client_summary,
        system_message="Summarize the conversation so far for your own memory",
        summary_format="This portion of conversation has been summarized as follow: {summary}",
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
