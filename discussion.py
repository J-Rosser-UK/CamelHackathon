import os
from getpass import getpass
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig
from camel.agents import ChatAgent
from camel.workforce import Workforce
from camel.tasks import Task


def get_api_key():
    """Prompt for the API key securely and set it in the environment."""
    api_key = getpass('Enter your Mistral API key: ')
    os.environ["MISTRAL_API_KEY"] = api_key
    return api_key


def create_chat_agent(template, proposal_title, funding_criteria, proposal_text, model):
    """Create a ChatAgent with the specified template and context."""
    system_message = template.format(proposal_title=proposal_title,
                                      funding_criteria=funding_criteria,
                                      proposal_text=proposal_text)
    return ChatAgent(system_message=system_message, model=model, message_window_size=12)


def setup_templates():
    """Define and return the various templates for the different roles."""
    researcher_template = """\
    You are the author of a research proposal titled "{proposal_title}". Your role is to represent and explain your research idea to a group, addressing any questions and defending your proposal against critique.

    **Your role**: Researcher (Author of the Proposal)
    **Your tone**: Neutral, professional, and receptive to feedback
    **Primary objectives**:
    - Provide a clear and thorough explanation of the research proposal when asked
    - Address questions or concerns from the other researchers, particularly from the critical and supportive researchers
    - Engage in a collaborative manner, listening to feedback and defending the merit of your proposal based on the criteria for funding

    **Funding Criteria**:
    The funding decision will be based on the following criteria: {funding_criteria}

    **Proposal Summary**:
    {proposal_text}
    """

    critical_template = """\
    You are a researcher critically evaluating a research proposal titled "{proposal_title}". Your role is to assess the proposal with a critical eye, identifying potential weaknesses or areas for improvement.

    **Your role**: Researcher (Critical Perspective)
    **Your tone**: Skeptical, questioning, and focused on identifying challenges
    **Primary objectives**:
    - Raise questions that probe the methodology, feasibility, or impact of the proposal
    - Point out potential weaknesses or limitations, such as gaps in research design, issues with resource allocation, or uncertainties in projected outcomes
    - Evaluate the proposal’s adherence to the funding criteria, questioning any aspects that may not meet them fully

    **Funding Criteria**:
    The funding decision will be based on the following criteria: {funding_criteria}

    **Proposal Summary**:
    {proposal_text}
    """

    supportive_template = """\
    You are a researcher who supports a proposal titled "{proposal_title}". Your role is to highlight the proposal's strengths, defend it against criticism, and emphasize its potential positive impacts.

    **Your role**: Researcher (Supportive Perspective)
    **Your tone**: Enthusiastic, supportive, and constructive
    **Primary objectives**:
    - Emphasize the strengths and innovative aspects of the proposal
    - Defend the proposal against any criticism, offering counterpoints that reinforce its merit
    - Show how the proposal aligns well with the funding criteria and highlight where it especially excels

    **Funding Criteria**:
    The funding decision will be based on the following criteria: {funding_criteria}

    **Proposal Summary**:
    {proposal_text}
    """

    reviewer_template = """\
    You are a reviewer representing the funding body evaluating a research proposal titled "{proposal_title}". Your role is to assess whether this proposal merits funding by evaluating its feasibility, impact, and alignment with the funding criteria.

    **Your role**: Reviewer (Funding Body Representative)
    **Your tone**: Objective, inquisitive, and professional
    **Primary objectives**:
    - Ask questions to clarify aspects of the proposal that may impact your funding decision
    - Evaluate the strengths and weaknesses based on input from the proposal author, critical researcher, and supportive researcher
    - Assess the proposal’s adherence to the funding criteria, making sure it meets the necessary standards in each category
    - Make a final funding decision or provide feedback on what would be needed for reconsideration

    **Funding Criteria**:
    The funding decision will be based on the following criteria: {funding_criteria}

    **Proposal Summary**:
    {proposal_text}
    """
    return researcher_template, critical_template, supportive_template, reviewer_template


def setup_proposal_details():
    """Define and return the proposal title, text, and funding criteria."""
    proposal_title = "Developing AI-Driven Diagnostic Tools for Early Detection of Rare Diseases"
    
    proposal_text = """\
    Overview: This proposal outlines a research project aimed at creating AI-driven diagnostic tools for the early detection of rare diseases. The project seeks to develop machine learning algorithms trained on large datasets to identify biomarkers and early symptoms that are typically missed in conventional diagnostics. By implementing these tools in clinical settings, this research aims to reduce the time to diagnosis for rare diseases, which often take years to identify due to the lack of visible symptoms and low prevalence.

    Objectives:
    1. Algorithm Development: Develop machine learning models capable of processing and analyzing large, diverse datasets to detect rare disease markers.
    2. Clinical Validation: Partner with hospitals to conduct pilot studies and validate the diagnostic tools on patient data.
    3. Scalability and Accessibility: Design a scalable solution that can be integrated into existing healthcare IT infrastructures, enabling broader adoption in clinics worldwide.

    Anticipated Impact: This project aims to reduce the average diagnostic period for rare diseases from years to a few weeks or months, potentially leading to improved patient outcomes and quality of life. Early detection also supports more effective treatment planning and helps reduce healthcare costs associated with late-stage disease management.
    """
    
    criteria = """\
    Scientific Merit and Innovation: The proposal should demonstrate a strong foundation in scientific principles, with an innovative approach that advances current methods.
    Feasibility and Methodology: The proposed project plan should be realistic and clearly outline steps, resources, and methodologies that support the achievement of its objectives.
    Potential Impact: The project should have the potential for significant positive outcomes, addressing a critical need or gap in the field.
    Collaboration and Resources: The proposal should involve appropriate partnerships and utilize available resources effectively, especially for clinical validation.
    Scalability and Broader Application: The project should have the potential for scaling beyond the initial study, allowing for broader impact in clinical practice or other research areas.
    Budget Justification: The proposal should provide a clear, well-reasoned budget that aligns with the project's scope, demonstrating prudent use of resources.
    """
    return proposal_title, proposal_text, criteria


def create_workforce(proposal_title, proposal_text, funding_criteria, model):
    """Setup the workforce and add agents for discussion."""
    researcher_template, critical_template, supportive_template, reviewer_template = setup_templates()

    # Create chat agents
    researcher = create_chat_agent(researcher_template, proposal_title, funding_criteria, proposal_text, model)
    critic = create_chat_agent(critical_template, proposal_title, funding_criteria, proposal_text, model)
    supporter = create_chat_agent(supportive_template, proposal_title, funding_criteria, proposal_text, model)
    reviewer = create_chat_agent(reviewer_template, proposal_title, funding_criteria, proposal_text, model)

    # Setup workforce for discussion
    workforce = Workforce(
        "Discussion on funding a research project",
        coordinator_agent_kwargs={"model": model},
        task_agent_kwargs={"model": model},
    )

    # Add agents to the workforce
    workforce.add_single_agent_worker("Researcher seeking feedback", worker=researcher) \
            .add_single_agent_worker("Critical researcher providing feedback", worker=critic) \
            .add_single_agent_worker("Supportive researcher providing feedback", worker=supporter) \
            .add_single_agent_worker("Reviewer deciding on funding", worker=reviewer)

    return workforce


def process_task(workforce):
    """Define and process the task."""
    task = Task(content="Have an actionable and constructive discussion about funding (or not funding) the research proposal.", id='0')
    return workforce.process_task(task)


def main():
    """Main entry point for running the proposal evaluation process."""
    get_api_key()
    proposal_title, proposal_text, funding_criteria = setup_proposal_details()

    # Model setup
    model = ModelFactory.create(
        model_platform=ModelPlatformType.MISTRAL,
        model_type=ModelType.MISTRAL_LARGE,
        model_config_dict=MistralConfig().as_dict()  # Optional: the config for model
    )

    workforce = create_workforce(proposal_title, proposal_text, funding_criteria, model)
    result = process_task(workforce)
    return result


if __name__ == "__main__":
    main()
