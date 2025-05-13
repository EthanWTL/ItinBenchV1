import os
import json
import ollama
import argparse
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.prompt import planner_no_route_agent, planner_route_OP_agent

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--OPENAI_API_KEY', type=str, help='Input OpenAI API Key', required=False)
    parser.add_argument('--MISTRAL_API_KEY', type=str, help='Input Mistral API Key', required=False)
    parser.add_argument('--GEMINI_API_KEY', type=str, help='Input Gemini API Key', required=False)
    parser.add_argument('--model', type=str, help='Input model name', required=False, default='gpt4o')
    parser.add_argument('--task', type=str, help='Input task name', required=False, default='filteredDataRouteOP')
    parser.add_argument('--numPlan', type=int, help='Input number of plans', required=False, default=5)

    args = parser.parse_args()
    return args

class SolePlanning:
    def __init__(self, args) -> None:
        self.planner_llm = args.model

        if self.planner_llm == 'gpt4o':
            self.llm = ChatOpenAI(temperature=0,
                        model_name='gpt-4o-2024-11-20',
                        openai_api_key=args.OPENAI_API_KEY)

        if self.planner_llm == 'mistral':
            self.llm = ChatMistralAI(
                model="mistral-large-2411",
                temperature=0,
                mistral_api_key = args.MISTRAL_API_KEY
            )
        
        if self.planner_llm == 'gemini':
            self.llm = ChatGoogleGenerativeAI(
                temperature=0,
                model="gemini-1.5-pro",
                google_api_key=args.GOOGLE_API_KEY
            )

    def createPlan(self, prompt):
        if self.planner_llm == 'gpt4o':
            request = self.llm.invoke([HumanMessage(prompt)]).content
            return request
        if self.planner_llm == 'mistral':
            request = self.llm.invoke(prompt).content
            return request

        if self.planner_llm == 'llama3.1':
            request = ollama.generate(model='llama3.1', prompt=prompt, options={'num_ctx': 70000})['response']
            return request
        
        if self.planner_llm == 'gemini':
            request = self.llm.invoke(prompt).content
            return request

if __name__ == '__main__':
    args = parse_args()

    #get task
    task = args.task

    #get data
    if 'all' in task:
        given_information = load_dataset("EthanWTL81/ItinBench", "data all", split="test")[0]['all_data']
    else:
        given_informations = load_dataset("EthanWTL81/ItinBench", "filtered data", split="test")

    #human query
    humanquerys = load_dataset("EthanWTL81/ItinBench", "human query", split="test")

    for i in range(args.numPlan):
        query = humanquerys[i]['query']

        if task == 'allDataNoRoute':
            prompt_agent = planner_no_route_agent
            prompt = prompt_agent.format(given_information = given_information, query = query)
            #print(prompt)
        if task == 'allDataRouteOP':
            prompt_agent = planner_route_OP_agent
            prompt = prompt_agent.format(given_information = given_information, query = query)
            #print(prompt)

        if task == 'filteredDataRouteOP':
            given_information = given_informations[i]['filtered_data']
            prompt_agent = planner_route_OP_agent
            prompt = prompt_agent.format(given_information = given_information, query = query)
            #print(prompt)

        #actual inference
        solePlanning = SolePlanning(args)
        plan = {"index": i+1, "plan": solePlanning.createPlan(prompt)}
        plans = [plan]

        with open (f'Output/{args.model}/plans/{args.task}.jsonl', 'a') as file:
            for plan in plans:
                file.write(json.dumps(plan) + '\n')
        if(i%20 == 0):
            print(f"we saved {i+1}th plans")
