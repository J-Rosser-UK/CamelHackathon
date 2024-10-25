import dotenv
import os

dotenv.load_dotenv()

from typing import List

from pydantic import BaseModel, Field

from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage as bm
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.configs.openai_config import ChatGPTConfig
from camel.toolkits import SearchToolkit, FunctionTool
from camel.loaders import Firecrawl

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

firecrawl = Firecrawl(os.getenv("FIRECRAWL_API_KEY"))

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

response = agent.step("I'm studying multimodal agents and I need to find the name of researchers that study this topic. Provide a list of well-known researchers in this field.")

print(response.text)