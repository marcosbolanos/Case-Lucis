from tools.utils import simple_research_pipeline
import httpx
import asyncio

async def run_research_for_queries(
    research_questions:dict[str,list[str]],
    context:str
):
    answer = ""
    for query in research_questions['queries']:
        results = await simple_research_pipeline(query, context)
        answer += "\n\n" + str(results)

    return answer