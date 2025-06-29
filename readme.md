# Case Study : AI Doctor Agent

This is my solution for a software engineering case study. The objective was to build an AI agent that can answer questions regarding a lab test result, while searching PubMed for relevant litetature to provice advice backed with evidence. 

It was build using the `anthropic` python package, also sending requests to the PubMed API via a series of in-house integrations. 

My approach mostly included "naïve" prompting with little parsing of the data, focusing more on getting the right information bits into the model's context at the right time. 

Overall, I found this case study quite challenging, and it definitely took me longer than 4 hours, but it helped me gain a much deeper understanding of the problem at hand.

## The repo

### Required environment variables

You must set the following environment variables to run the project :
- `ANTHROPIC_API_KEY`
- `PUBMED_API_KEY` (no credits required)
- `EMAIL` (for the PubMed API rate limits)

### Project structure

The `/src` folder contains `ai_doctor_agent.py`, where you can find the main function to query the agent. It can be run as a CLI app, or imported into another script.

The `/notebooks` folder contains five jupyter notebooks where I explored the dataset and prototyped my functions. The fifth notebook contains the final result, with three example queries to the agent that I developed across the project.

The `/data` folder contains the raw dataset that was given to me, as well as a couple parsed versions that were generated from notebook 1.

## The agent

### Setup

The agent starts with appropriate system prompts and instructions, as well as a simplified version of the original json data, which contains only the biomarker names, values, and lab visit dates. 

The agent was also provided with mock information for the patient's age, sex, weight, and height.

Flattening the JSON into a table allowed to reduce the number of tokens for this context, while retaining information in a readable format. Sorting by name and date helped the LLM understand the chronological nature of the data. 

After being asked a question, the agent can choose whether to use a PubMed search tool. Having been instructed not to give any medical claims without evidence, it seemingly always choses to use the tool. 

### The PubMed search tool

The PubMed search tool takes in a list of research questions, uses a prompt chain to refine them into keywords, then uses the MeSH API to obtain the appropriate translations of each keyword for PubMed search. 

It then generates a search query using boolean operators, which integrates well with PubMed most of the time. The top 15 resulting articles are retrieved, and their abstracts are fetched.

The model is instructed to choose the top 2 to 3 most relevant articles to answer the research question. It does so for each research question that it originally asked, usually 3 or 4.

Finally, the tool outputs the resulting answers for each internal question, which enrich the model's context and help it answer the user's question.

## Discussion

### Quality of responses

The agent correctly reads the patient's data, and can properly summarize the main problems as well as the improvements in the patient's bloodwork. Since there are a lot of biomarkers at play here, being able to rapidly scan them is quite positive.

The agent's responses, however, remain superficial. While the PubMed search tool allows it to find relevant articles, it does not really ask questions about the link between different biomarkers in the context of the patient, and simply runs searches on each biomarker individually.

Furthermore, the agent currently can only read publication abstracts, which is a major gap in this implementation. Abstracts can give a summary of the study's conclusion and endpoints, but they are not sufficient to judge of a study's quality or relevance.

Perhaps most importantly, the resulting abstracts are rarely relevant to the patient. While the patient is a healthy subject, most articles are about Type 2 Diabetes and other diseases. 

In the future, it might be necessary to add hard-coded solutions to this problem, or thorough screening of hundreds of abstracts. 

### PubMed API rate limits

One of the main technical challenges I faced was the actual implementation of the PubMed API. While it is easy to run a single query, the back-and-forth cycle between the PubMed API and subsequent queries to Claude is a real speed bottleneck.

Naturally, one would think about parallelizing these I/O bound operations as much as possible. 

In practice, however, parallelizing these tasks resulted in many API errors related to rate-limits. I could have managed them better, but due to time constrains, I ultimately opted for running each research question sequentially. This is very slow and definitely not a good solution.

### Claude API costs

The current implementation uses Claude 3.7 Sonnet for every prompt. While this model can provide high-quality answers, it is quite expensive to run. One single query to the agent can sometimes use $0.50 or more, which puts into question the long-term sustainability of this strategy.

After adequatly optimizing the agent's performance, certain parts of its tool chains should be tested with lighter models to reduce costs.

## Conclusion

I am far from satisfied by this implementation, although it allowed me to get a first view on the main technical problems that this task will pose. 

I would start by exploring the different methods for allowing agents to retrieve PubMed articles. We need to make sure that they are highly relevant, but the screening process cannot be too costly.

One direction to explore would be to include more abstracts in the search, and use structured output to screen batches of 20-40 abstracts. I would try substituting Claude for an OpenAI model, as their structured output capabilities are more advanced.

Regardless, I believe that by iteratively improving on each aspect of my solution, it is definitely possible to obtain a working product.










