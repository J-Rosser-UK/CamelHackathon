import json
import dotenv
import os

dotenv.load_dotenv()

from typing import List

from pydantic import BaseModel, Field

from camel.loaders import Firecrawl

firecrawl = Firecrawl(os.getenv("FIRECRAWL_API_KEY"))

def crawl_website(url: str) -> List[str]:
    """
    Returns a Markdown representation of the website at the given URL.

    Args:
        url (str): The URL of the website to crawl.
    
    Returns:
        List[str]: A list of strings, each representing a section of the website.
    """
    print('Crawling website', url)
    try:
        return firecrawl.crawl(url)["data"][0]["markdown"]
    except Exception as e:
        print('Error:', e)
        return ['Crawling failed: ' + str(e)]

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

crawl_website_tool = FunctionTool(crawl_website)

agent = ChatAgent(
    model=model,
    system_message=bm.make_assistant_message('system', "You are a search and lookup agent. You can search the internet and crawl websites for information."),
    tools=SearchToolkit().get_tools() + [crawl_website_tool]
)

class ResearchListSchema(BaseModel):
    researcher_a : str = Field(title="Researcher A", description="The name of a researcher in the field.")
    researcher_b : str = Field(title="Researcher B", description="The name of a researcher in the field.")
    researcher_c : str = Field(title="Researcher C", description="The name of a researcher in the field.")

def lookup_info(name):
    researcher_info_agent = ChatAgent(
        model=model,
        system_message=bm.make_assistant_message('system', "You are a search and lookup agent. You can search the internet and provide summaries of the results to the user. The user can't open links, so you need to provide a summary of the information found."),
        tools=SearchToolkit().get_tools() + [crawl_website_tool]
    )

    response = researcher_info_agent.step(f"Provide a summary of the research work of {name}.")

    return response.msg.content

def get_researchers_info(topic):
    response = agent.step(f"I'm studying {topic} and I need to find the name of researchers that study this topic. Provide a list of well-known researchers in this field.")

    print(response.msg.content)

    researcher_response = agent.step(f'Now open each of the links and names you provided and compile a full list of researchers that study {topic}. Crawl 3 websites maximum.', response_format=ResearchListSchema)

    print(researcher_response.msg.content)

    researchers = list(json.loads(researcher_response.msg.content.replace("'", '"')).values())

    infos = {
        researcher: lookup_info(researcher) for researcher in researchers
    }

    return infos

#print(get_researchers_info("multimodal agents"))