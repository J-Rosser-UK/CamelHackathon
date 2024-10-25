import json
import dotenv
import os

dotenv.load_dotenv()

from typing import List

from pydantic import BaseModel, Field

from camel.loaders import Firecrawl


from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage as bm
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.configs.openai_config import ChatGPTConfig
from camel.toolkits import SearchToolkit, FunctionTool

model = ModelFactory.create(
    model_type=ModelType.GPT_4O,
    model_platform=ModelPlatformType.OPENAI
)


def create_researcher_llm(researcher_name, researcher_description, question):
    agent = ChatAgent(
        model=model,
        system_message=bm.make_assistant_message('system', f"You are an agent representing {researcher_name}. You have to act like {researcher_name} and offer feedback on ideas. A short summary of {researcher_name} is provided below: {researcher_description}. Here's the research question: {question}"),
    )

    return agent


import random

import funding_finder

def simulate_conversation(researchers : dict, question : str, topic):
    url = funding_finder.generate_ukri_url(topic, 1)
    agents = {}
    for researcher, description in researchers.items():
        agent = create_researcher_llm(researcher, description + '\nHere is a relevant URL: ' + url, question)
        agents[researcher] = agent
    
    previous_responses = []

    for i in range(5):
        # Pick a random agent
        chosen_researcher = random.choice(list(agents.keys()))

        # Get the response
        response = agents[chosen_researcher].step('Here are the previous responses:' + '\n'.join([f"{r[0]}: {r[1]}" for r in previous_responses]) + '\n' + question)

        previous_responses.append((researcher, response.msg.content))

        print(f"{researcher}: {response.msg.content}")


import researchers

def main():
    topic = "Multimodal Agents"
    chosen_researchers = researchers.get_researchers_info(topic)
    print(chosen_researchers)

    question = "What are the key challenges in developing multimodal agents?"

    simulate_conversation(chosen_researchers, question, topic)
main()
