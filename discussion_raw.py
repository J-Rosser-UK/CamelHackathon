import os
from getpass import getpass

def main():

    # Prompt for the API key securely
    mistral_api_key = getpass('Enter your Mistral API key: ')
    os.environ["MISTRAL_API_KEY"] = mistral_api_key

    RESEARCHER_TEMPLATE = """
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

    As you engage, ensure your responses address the proposal's alignment with these funding criteria, aiming to clarify and strengthen the understanding of your proposal.
    """

    CRITICAL_TEMPLATE = """
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

    As you engage, focus on analyzing the proposal’s alignment with the funding criteria. Your questions should challenge the proposal where it may fall short or seem unclear in meeting these requirements.
    """

    SUPPORTIVE_TEMPLATE = """
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

    As you engage, focus on demonstrating the proposal's strong alignment with the funding criteria. Counter critiques by emphasizing areas where the proposal is particularly strong or impactful.
    """

    REVIEWER_TEMPLATE = """
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

    As you engage, focus on verifying the proposal’s alignment with the funding criteria. Use the input from other researchers to aid your evaluation, and make a well-informed funding decision based on these criteria.
    """

    PROPOSAL_TITLE = "Developing AI-Driven Diagnostic Tools for Early Detection of Rare Diseases"
    PROPOSAL_TEXT = """
    Overview: This proposal outlines a research project aimed at creating AI-driven diagnostic tools for the early detection of rare diseases. The project seeks to develop machine learning algorithms trained on large datasets to identify biomarkers and early symptoms that are typically missed in conventional diagnostics. By implementing these tools in clinical settings, this research aims to reduce the time to diagnosis for rare diseases, which often take years to identify due to the lack of visible symptoms and low prevalence.

    Objectives:

        Algorithm Development: Develop machine learning models capable of processing and analyzing large, diverse datasets to detect rare disease markers.
        Clinical Validation: Partner with hospitals to conduct pilot studies and validate the diagnostic tools on patient data.
        Scalability and Accessibility: Design a scalable solution that can be integrated into existing healthcare IT infrastructures, enabling broader adoption in clinics worldwide.

    Anticipated Impact: This project aims to reduce the average diagnostic period for rare diseases from years to a few weeks or months, potentially leading to improved patient outcomes and quality of life. Early detection also supports more effective treatment planning and helps reduce healthcare costs associated with late-stage disease management.
    """

    CRITERIA = """
    Scientific Merit and Innovation: The proposal should demonstrate a strong foundation in scientific principles, with an innovative approach that advances current methods.
    Feasibility and Methodology: The proposed project plan should be realistic and clearly outline steps, resources, and methodologies that support the achievement of its objectives.
    Potential Impact: The project should have the potential for significant positive outcomes, addressing a critical need or gap in the field.
    Collaboration and Resources: The proposal should involve appropriate partnerships and utilize available resources effectively, especially for clinical validation.
    Scalability and Broader Application: The project should have the potential for scaling beyond the initial study, allowing for broader impact in clinical practice or other research areas.
    Budget Justification: The proposal should provide a clear, well-reasoned budget that aligns with the project's scope, demonstrating prudent use of resources.
    """

    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, ModelType
    from camel.configs import MistralConfig

    # Define the model, here in this case we use gpt-4o-mini
    model = ModelFactory.create(
        model_platform=ModelPlatformType.MISTRAL,
        model_type=ModelType.MISTRAL_LARGE,
        model_config_dict=MistralConfig().as_dict(), # [Optional] the config for model
    )

    from camel.agents import ChatAgent
    researcher = ChatAgent(
        system_message=RESEARCHER_TEMPLATE.format(proposal_title=PROPOSAL_TITLE, funding_criteria=CRITERIA, proposal_text=PROPOSAL_TEXT),
        model=model,
        message_window_size=12, # [Optional] the length for chat memory
    )

    critic = ChatAgent(
        system_message=CRITICAL_TEMPLATE.format(proposal_title=PROPOSAL_TITLE, funding_criteria=CRITERIA, proposal_text=PROPOSAL_TEXT),
        model=model,
        message_window_size=12, # [Optional] the length for chat memory
    )

    supporter = ChatAgent(
        system_message=SUPPORTIVE_TEMPLATE.format(proposal_title=PROPOSAL_TITLE, funding_criteria=CRITERIA, proposal_text=PROPOSAL_TEXT),
        model=model,
        message_window_size=12, # [Optional] the length for chat memory
    )

    reviewer = ChatAgent(
        system_message=REVIEWER_TEMPLATE.format(proposal_title=PROPOSAL_TITLE, funding_criteria=CRITERIA, proposal_text=PROPOSAL_TEXT),
        model=model,
        message_window_size=12, # [Optional] the length for chat memory
    )

    from camel.workforce import Workforce

    workforce = Workforce(
        "Discussion on funding a research project",
        coordinator_agent_kwargs={"model": model},
        task_agent_kwargs={"model": model},
    )

    workforce.add_single_agent_worker(
        "A researcher looking for feedback on a research project", worker=researcher).add_single_agent_worker(
            "A critical researcher giving constructive feedback", worker=critic).add_single_agent_worker(
                "A supportive researcher giving constructive feedback", worker=supporter).add_single_agent_worker(
                    "A reviewer deciding if the project should get funded", worker=reviewer)

    from camel.tasks import Task

    task = Task(
        content="Have an actionable and constructive discussion about funding (or not funding) the research proposal.",
        id='0',
    )

    return workforce.process_task(task)
