from dotenv import load_dotenv
from anthropic import Anthropic
import tools.utils as ut
import tools.pubmed_search as ps
import asyncio

load_dotenv()

async def ask_ai_doctor_agent(
    question:str
):
    with open("data/metrics_only.csv", "r", encoding="utf-8") as f:
        csv_string = f.read()

    patient_info="""
        PATIENT CHARACTERISTICS
        sex: Male
        age: 3    new_new_messages = new_messages.copy()
        new_new_messages.append(
            {
                "role":"user", 
                "content":f"Here are the results of the PubMed research tool : {res} \n\n Formulate your final answer, **making sure to include all relevant article links**"
            },
        )

        # Call the Anthropic API
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0,
            system=system_prompt,
            messages=new_new_messages
        )

        print(response.content[0].text)2
        height: 177cm
        weight: 72KG
    """

    # Create Anthropic client
    client = Anthropic()
    # Define the tool in the format Anthropic expects
    pubmed_tool = {
        "name": "pubmed_search",
        "description": "A LLM chain that will intelligently run one or multiple searches for you, returning relevant articles as well as key findings from each.",
        "input_schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Here, you may include a list of research questions that you would like answered, in the context of your patient. Make the queries as specific as possible, with the objective of finding hyper-relevant solutions to the patient's problem."
                }
            },
            "required": ["query"]
        }
    }

    # Create a dictionary to map tool calls to actual functions
    tool_map = {
        "pubmed_search": ps.run_research_for_queries
    }

    system_prompt = """
    You are Concierge, an AI doctor specialized in longevity, providing users with evidence-based insights and advice.
    You answer questions in a clear manner, without omitting complex medical terminology, but providing detailed explanations.
    You never give medical information that isn't backed by sources
    You are a step-by step reasoning model, giving continual feedback on your reasoning to the user using structured output
    """

    context = patient_info + "\n\n" + csv_string

    user_prompt = f"""
    INSTRUCTIONS: 
    Given the CONTEXT, answer the given QUESTION. 
    Never give any information that isn't backed by clear sources. If you need sources, use the pubmed_search tool
    If you decide to use the tool, make sure to also explain what you're going to be searching and why.


    ---- CONTEXT: ----
    The user just recieved their updated blood test results, which are the following:

    {context}




    ---- QUESTION: ----
    {question}
    """

    # Create the message structure for Anthropic with tools
    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # Call the Anthropic API with tools
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4000,
        temperature=0,
        messages=messages,
        system=system_prompt,
        tools=[pubmed_tool]
    )

    assistant_response = response.content[0].text
    print(assistant_response)
    tool_use = response.content[-1]

    new_messages = messages.copy()
    new_messages.append(
        {"role":"assistant","content":assistant_response}
    )


    tool_calls = ut.search_tool_calls(response)
    if tool_calls['has_tool_use'] == True:
        for tool in tool_calls['tools']:
            function_name = tool['tool_name']
            params = tool['tool_input']
            if function_name == "pubmed_search":
                params = tool['tool_input']
                res = await ps.run_research_for_queries(params, context)

        new_new_messages = new_messages.copy()
        new_new_messages.append(
            {
                "role":"user", 
                "content":f"Here are the results of the PubMed research tool : {res} \n\n Formulate your final answer, **making sure to include all relevant article links**"
            },
        )

        # Call the Anthropic API
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0,
            system=system_prompt,
            messages=new_new_messages
        )

        print(response.content[0].text)

    return

async def main():
    """Runs the AI Doctor Agent as a command-line tool."""
    print("Welcome to the AI Doctor Agent. Type 'exit' or 'quit' to end.")
    while True:
        try:
            question = input("\nPlease ask a question: ")
            if question.lower() in ["exit", "quit"]:
                print("Exiting agent. Goodbye!")
                break
            await ask_ai_doctor_agent(question)
        except KeyboardInterrupt:
            print("\nExiting agent. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())