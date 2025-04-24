# 🧠 Autogen ContextPlus
[![GitHub stars](https://img.shields.io/github/stars/SongChiYoung/autogen-contextplus?style=social)](https://github.com/SongChiYoung/autogen-contextplus/stargazers)
[![Downloads](https://static.pepy.tech/personalized-badge/autogen-contextplus?period=week&units=international_system&left_color=gray&right_color=orange&left_text=Downloads/week)](https://pepy.tech/project/autogen-contextplus)

> Modular, customizable, and serializable context engine for [AutoGen](https://github.com/microsoft/autogen) — enabling structured message summarization, filtering, and rewriting logic with full compatibility.


🚀 **Try it? → Don’t forget to ⭐ the repo if useful!**


---

## ✨ What is ContextPlus?

**`autogen-contextplus`** provides a general-purpose context modifier system for AutoGen’s `model_context` layer. It supports:

- ✅ Condition-triggered message summarization
- ✅ Agent- or function-based message rewriting
- ✅ Component-based serialization / deserialization
- ✅ Full support for user-defined logic via AutoGen `FunctionTool` or custom agents

---

## 🔧 Installation

```bash
pip install autogen-contextplus
```

---

For development and type checking:
```bash
pip install -e ".[dev]"
```

---

## Example

```python
import asyncio
from pprint import pprint
from typing import List
from autogen_core.models import UserMessage, AssistantMessage
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken 
from autogen_core.model_context import BufferedChatCompletionContext

from autogen_contextplus.conditions import (
    MaxMessageCondition
)
from autogen_contextplus.base.types import (
    ModifierFunction,
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


async def main():
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
    context = agent.model_context

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
    
if __name__ == "__main__":
    asyncio.run(main())
```
