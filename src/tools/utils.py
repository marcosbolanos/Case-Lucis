import requests
import json
import httpx
import asyncio

from typing import Literal
from xml.etree import ElementTree as ET
from pydantic import BaseModel
from typing import Any
from anthropic import Anthropic



class PicoTerms(BaseModel):
    Population:list[str]
    Intervention:list[str]
    Comparator:list[str]
    Outcome:list[str]

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EMAIL = "marcosqbv@gmail.com"  # Email to comply with NCBI usage guidelines
API_KEY = "844017a24432a9d8abb84a670b50ef28f108"

# This has to be appended to a search that uses the "Systematic Review" doctype filter
systermatic_review_subset = "systematic [sb]"

# Process the response and handle any tool use blocks
def search_tool_calls(
        response:Any
    ) -> dict[str, bool] | dict[str, bool | list[dict[str, Any]]]:
    result = {}
    result["has_tool_use"] = False
    tool_list = []
    for block in response.content:
        if hasattr(block, 'type') and block.type == 'tool_use':
            result["has_tool_use"] = True
            tool = {}
            # Extract the tool details
            tool["tool_name"] = block.name
            tool["tool_input"] = block.input
            tool["tool_id"] = block.id
            tool_list.append(tool)
    if tool_list:
        result["tools"] = tool_list
    return result
    


def formulate_pico(
    research_question:str,
    context: str
) -> PicoTerms:
    client = Anthropic()
    system_prompt="""
    You are an AI medical doctor providing luxury care, currently conducting a literature search to dig deep into your patient's needs.
    """

    pico_prompt = f"""
    INSTRUCTIONS: 
    Given the conversation history, reformulate the research question into **keywords** using the PICO framework, taking into account all of the data you have on the patient.
    Try to put in keywords that are actually indexed in the MeSH database, they are usually one-word long. Include multiple synomics when you're not sure.

    Population: Be sure to be extremely specific about the patient's characteristics, so that the answers are relevant to them
    Intervention: Is the patient in search for a solution ? Are they considering a specific solution ? Something else ? This field can contain a few specific elements, or be left completely empty, depending on what you're looking to solve
    Comparator: The best comparator is usually a placebo, but again, if you are extremely certain that another keyword would produce better answers, feel free to do so
    Outcome: Is the patient searching to obtain some clinical benefit ? Are they wondering about efficacy, or worried about side effects ? Something else ? This is another field that can either contain a few specific elements, or be left empty.

    ---- THE PATIENT'S CONTEXT ----

    {context}

    ---- THE RESEARCH QUESTION ---- 

    Here's the research question you will focus on now:

    {research_question}

    Pass your response to the get_mesh_terms tool so that the PubMed API can give the exact search terms to use.
    """

    # Create the message structure for Anthropic
    messages = [
        {
            "role": "user",
            "content": pico_prompt
        }
    ]

    # Define the tool in the format Anthropic expects
    mesh_terms_tool = {
        "name": "get_mesh_terms",
        "description": "This tool will search the PubMed API for each of your specified keywords, and give you a set of results with definitions that will allow you to use the proper medical terminology",
        "input_schema": {
            "type": "object",
            "properties": {
                "population_keywords": {
                    "type":"array",
                    "items": {"type":"string"},
                    "description":"keywords for population"
                },
                    "intervetion_keywords": {
                    "type":"array",
                    "items": {"type":"string"},
                    "description":"keywords for intervention"
                },
                    "comparator_keywords": {
                    "type":"array",
                    "items": {"type":"string"},
                    "description":"keywords for comparator"
                },
                    "outcome_keywords": {
                    "type":"array",
                    "items": {"type":"string"},
                    "description":"keywords for outcome"
                },
            },
            "required": ["population_keywords", "intervetion_keywords", "comparator_keywords", "outcome_keywords"]
        }
    }

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4000,
        temperature=0,
        messages=messages,
        system=system_prompt,
        tools=[mesh_terms_tool]
    )

    tool_calls = search_tool_calls(response)
    raw_pico_terms = tool_calls["tools"][0]["tool_input"]

    return raw_pico_terms

# This function executes a query on PubMed and returns raw results. 
async def esearch_mesh(
    query: str,
):
    term = query

    params:dict[str, str | int] = {
        "db": "mesh",
        "term": term,
        "retmax": 6,
        "email": EMAIL,
        "usehistory":"n",
        "retmode":"json",
        "api_key": API_KEY
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(ESEARCH_URL, params=params)
    
    content_str = resp.content
    return content_str

async def run_parallel_mesh_queries(data_structure, max_workers=10):
    """Run queries for all keywords in parallel with a delay"""
    tasks = []
    
    # Create a list of tasks
    for category, keywords in data_structure.items():
        if isinstance(keywords, list):
            for kw in keywords:
                tasks.append((category, kw, esearch_mesh(kw)))

    delay = 0.5  # 500ms delay

    async def wrapper(task, i):
        await asyncio.sleep(i * delay)
        return await task

    # Run all tasks concurrently with a staggered start
    coros = [wrapper(task[2], i) for i, task in enumerate(tasks)]
    results_raw = await asyncio.gather(*coros, return_exceptions=True)
    
    results = {}
    for i, (category, keyword, _) in enumerate(tasks):
        if category not in results:
            results[category] = []
        
        result_content = results_raw[i]
        if isinstance(result_content, Exception):
            results[category].append({keyword: {"error": str(result_content)}})
        else:
            results[category].append({keyword: result_content})

    return results

def formulate_esearch_query(
    research_question:str,
    context:str,
    pico_terms:str,
    mesh_api_results:str,
):
    client = Anthropic()
    system_prompt="""
    You are an AI query formulator for PubMed. You formulate highly accurate PubMed queries from prompts
    Your outputs contain NOTHING except the PubMed queries themselves.
    """
    esearch_prompt=f"""
    INSTUCTIONS:
    Given the PATIENT'S CONTEXT, the chosen PICO terms, and the appropriate MESH TERMS that came from the MeSH term search tool, formulate a PubMed query to answer the RESEARCH QUESTION.
    Make sure to use boolean AND() OR() NOT() operators in a logical way to best capture the desired PICO.
    Remember that the goal is to answer your patient's question with rich, high-quality data that is higly relevant to their situation and needs.
    Do not output anything other than the query

    ---- PATIENT'S CONTEXT ----

    {context}

    ---- PICO TERMS ----

    {pico_terms}
    
    ---- APPROPRIATE MESH TERMS ----

    {mesh_api_results}

    ---- RESEARCH QUESTION ----

    {research_question}
    

    """

    # Create the message structure for Anthropic
    messages = [
        {
            "role": "user",
            "content": esearch_prompt
        }
    ]

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4000,
        temperature=0,
        messages=messages,
        system=system_prompt,
    )

    return response

# This function executes a query on PubMed and returns raw results. 
async def esearch_abstracts(
    query: str,
):
    term = query

    params:dict[str, str | int] = {
        "db": "pubmed",
        "term": term,
        "retmax": 15,
        "email": EMAIL,
        "usehistory":"y",
        "retmode":"json",
        "api_key": API_KEY
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(ESEARCH_URL, params=params)

    content = json.loads(resp.content)
    return content

def parse_efetch_query(content: bytes):
    """Parse PubMed efetch XML response with error handling"""
    try:
        # Check if content is empty
        if not content or len(content.strip()) == 0:
            print("⚠️ Empty response from PubMed API")
            return []
        
        # Try to decode content to check for error messages
        try:
            content_str = content.decode('utf-8')
            # Check for common PubMed error patterns
            if "error" in content_str.lower():
                print(f"⚠️ PubMed API error response: {content_str[:200]}...")
                return []
        except UnicodeDecodeError:
            pass
        
        # Parse XML
        root = ET.fromstring(content)
        
        # Check if we have any articles
        articles = root.findall(".//PubmedArticle")
        if not articles:
            print("⚠️ No articles found in PubMed response")
            return []
        
        results = []
        for article in articles:
            pmid = article.findtext(".//PMID") or "NotFound"
            title = article.findtext(".//ArticleTitle") or "NotFound"
            abstract = article.findtext(".//AbstractText") or "NotFound"
            start_page = article.findtext(".//StartPage") or ""
            end_page = article.findtext(".//EndPage") or ""
            pagination = f"{start_page}-{end_page}" if start_page and end_page else (start_page or end_page)

            # find DOI or default
            doi = next(
                (aid.text for aid in article.findall(".//ArticleId")
                 if aid.attrib.get("IdType") == "doi" and aid.text),
                "NotFound"
            )

            results.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "pagination": pagination,
                "doi": doi
            })
        
        print(f"✅ Successfully parsed {len(results)} articles from PubMed")
        return results
        
    except ET.ParseError as e:
        print(f"❌ XML parsing error: {str(e)}")
        print(f"Content preview: {content[:200]}...")
        return []
    except Exception as e:
        print(f"❌ Unexpected error parsing PubMed response: {str(e)}")
        print(f"Content preview: {content[:200]}...")
        return []

async def efetch_query_with_key(
    query_key:str,
    WebEnv:str
):
    """Fetch abstracts from PubMed with error handling"""
    params:dict[str, str | int]={
        "db":"pubmed",
        "query_key":query_key,
        "WebEnv":WebEnv,
        "retmax": 15,  # Updated to match the increased retmax
        "retmode":"xml",
        "email": EMAIL,
        "api_key": API_KEY
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:  # Add timeout
            resp = await client.get(EFETCH_URL, params=params)
        
        # Check HTTP status
        if resp.status_code != 200:
            print(f"❌ PubMed API returned status {resp.status_code}")
            return []
        
        content = resp.content
        results = parse_efetch_query(content)
        return results
        
    except httpx.TimeoutException:
        print("❌ Timeout while fetching from PubMed API")
        return []
    except Exception as e:
        print(f"❌ Error fetching from PubMed: {str(e)}")
        return []

def answer_from_abstracts(
    research_question: str,
    context: str,
    pico_terms: str,
    query_used: str,
    search_results: str
) -> dict:
    """
    Directly answer the research question from search results, focusing on the most relevant abstracts
    """
    client = Anthropic()
    
    system_prompt = """
    You are an AI medical research analyst. Your job is to analyze PubMed search results and provide a comprehensive answer to the research question for the specific patient context.
    
    Focus on finding the TOP 2-3 MOST RELEVANT abstracts that directly address the research question, even if the majority of results are not relevant. Acknowledge that many results may not be directly applicable, but extract valuable insights from the most relevant ones.
    
    Your response should be well-structured and include:
    1. A direct answer to the research question
    2. Evidence from the most relevant studies
    3. Specific recommendations for the patient
    4. Links to the most relevant articles
    5. A confidence assessment
    """
    
    answer_prompt = f"""
    INSTRUCTIONS:
    Analyze the PubMed search results below and provide a comprehensive answer to the research question for this specific patient. 
    
    IMPORTANT: Most abstracts may not be directly relevant - this is expected. Focus on identifying the TOP 2-3 MOST RELEVANT studies that address the research question and extract actionable insights from them.
    
    Your answer should include:
    1. A clear, direct answer to the research question
    2. Summary of findings from the most relevant studies (2-3 studies maximum)
    3. Specific, actionable recommendations for this patient
    4. Links to the most relevant articles for further reading
    5. Confidence level (high/moderate/low) based on evidence quality
    
    ---- PATIENT CONTEXT ----
    {context}
    
    ---- RESEARCH QUESTION ----
    {research_question}
    
    ---- PICO TERMS USED ----
    {pico_terms}
    
    ---- PUBMED QUERY USED ----
    {query_used}
    
    ---- SEARCH RESULTS (15 abstracts) ----
    {search_results}
    
    Please provide a structured response that directly answers the research question while acknowledging which studies were most relevant and why.
    """
    
    messages = [{"role": "user", "content": answer_prompt}]
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4000,
        temperature=0,
        messages=messages,
        system=system_prompt
    )
    
    return {
        "answer": response.content[0].text,
        "query_used": query_used,
        "total_abstracts_reviewed": 15
    }

async def simple_research_pipeline(
    research_question: str,
    context: str
) -> dict:
    """
    Simplified research pipeline that answers directly from first search results
    """
    try:
        # Step 1: Initial PICO formulation
        pico_terms = formulate_pico(research_question, context)
        
        # Step 2: Get MeSH terms
        mesh_results = await run_parallel_mesh_queries(pico_terms, max_workers=5)
        mesh_context = str(mesh_results)
        
        # Step 3: Query formulation
        query_response = formulate_esearch_query(research_question, context, str(pico_terms), mesh_context)
        query = query_response.content[0].text
        
        # Step 4: Execute search (retmax=15)
        abstracts_ids = await esearch_abstracts(query)
        
        if not abstracts_ids.get("esearchresult", {}).get("idlist"):
            return {
                "status": "no_results",
                "query": query
            }
        
        # Step 5: Fetch abstracts
        webenv = abstracts_ids["esearchresult"]["webenv"]
        query_key = abstracts_ids["esearchresult"]["querykey"]
        efetch_result = await efetch_query_with_key(query_key, webenv)
        
        # Check if we got any results from efetch
        if not efetch_result or len(efetch_result) == 0:
            return {
                "status": "no_results",
                "query": query,
                "error": "No abstracts could be retrieved from PubMed"
            }
        
        # Format results
        formatted_results = []
        for i, article in enumerate(efetch_result):
            if isinstance(article, dict) and article.get("pmid") != "NotFound":
                article["link"] = "https://pubmed.ncbi.nlm.nih.gov/" + article["pmid"]
                article.pop("pmid", None)
                article.pop("doi", None)
                formatted_results.append(str(article))
        
        if not formatted_results:
            return {
                "status": "no_results", 
                "query": query,
                "error": "No valid abstracts found in PubMed response"
            }
        
        search_results = "\n".join(formatted_results)
        
        # Step 6: Answer directly from results
        answer_result = answer_from_abstracts(
            research_question, context, str(pico_terms), query, search_results
        )
        
        return {
            "status": "success",
            "answer": answer_result["answer"],
            "query_used": answer_result["query_used"],
            "total_abstracts": answer_result["total_abstracts_reviewed"],
            "raw_results": formatted_results
        }
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }