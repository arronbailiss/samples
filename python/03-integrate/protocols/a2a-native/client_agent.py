import asyncio

from strands import Agent
from strands.vended_tools import make_a2a_client

# initialize the A2A client tool, restricted to the local A2A server
a2a_client = make_a2a_client(allowed_endpoints=["http://localhost:9000"])

# initialize agent with tools
agent = Agent(tools=[a2a_client])
# you can also invoke the agent in a non-async context
# print(agent("pick an agent and make a sample call to test its functionality"))


# run the agent in an async context
async def main():
    await agent.invoke_async(
        "pick an agent and make a sample call to test its functionality"
    )


# run
asyncio.run(main())
