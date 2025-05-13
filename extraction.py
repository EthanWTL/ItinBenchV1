import os
import json
import argparse
from openai import OpenAI
from pydantic import BaseModel
from utils.eval import PLAN_EXTRACTION_PROMPT

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--apikey', type=str, help='Input OpenAI API Key', required=False)
    parser.add_argument('--model', type=str, help='Input model name', required=False, default='gpt4o')
    parser.add_argument('--task', type=str, help='Input task name', required=False, default='filteredDataRouteOP')
    parser.add_argument('--numPlan', type=int, help='Input number of plans', required=False, default=1)

    args = parser.parse_args()
    return args

class breakfast(BaseModel):
    name: str
    address: str

class morning_attraction(BaseModel):
    name: str
    address: str

class lunch(BaseModel):
    name: str
    address: str

class afternoon_attraction(BaseModel):
    name: str
    address: str

class dinner(BaseModel):
    name: str
    address: str

class night_attraction(BaseModel):
    name: str
    address: str

class accommodation(BaseModel):
    name: str
    address: str

class OneDay(BaseModel):
    days: str
    breakfast: breakfast
    morning_attractions: list[morning_attraction]
    lunch: lunch
    afternoon_attractions: list[afternoon_attraction]
    dinner: dinner
    night_attractions: list[night_attraction]
    accommodation: accommodation

class WholePlan(BaseModel):
    itinerary: list[OneDay]

class ExtractPlan:
    def __init__(self, args):
        self.client = OpenAI(
            api_key = args.apikey
        )

    def parse(self, user_prompt):
        system_prompt = PLAN_EXTRACTION_PROMPT
        
        #generation into json format
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-11-20",
            temperature=0,
            messages=[
                {"role": "system","content": system_prompt,},
                {"role": "user","content": user_prompt,}
            ],
            response_format=WholePlan
        )
        output = json.loads(completion.choices[0].message.content)
        return output
    
def postProcess(plan_eval):
    hotel_list = []
    hotel_carriedOn = []
    current_hotel = {"name":"-", "address":"-"}

    for day in plan_eval['itinerary']:
        hotel_list.append(day['accommodation'])
        
    for item in hotel_list:
        if item != {"name":"-", "address":"-"}:
            current_hotel = item
        hotel_carriedOn.append(current_hotel)
    
    for i in range(len(plan_eval['itinerary'])):
        plan_eval['itinerary'][i]['accommodation'] = hotel_carriedOn[i]
    return plan_eval

if __name__ == '__main__':
    args = parse_args()
    #choose the model
    model = args.model
    #choose the task
    task  = args.task

    extractPlan = ExtractPlan(args)

    with open (f'Output/{args.model}/plans/{task}.jsonl', 'r') as f:
        plans = [json.loads(line.strip()) for line in f]
    
    extracted_plans = []
    for i in range(args.numPlan):
        extracted_plan = extractPlan.parse(plans[i]['plan'])
        #post process the plan: carry on hotels
        extracted_plan = postProcess(extracted_plan)
        extracted_plans=[{'index': i+1, 'plan': extracted_plan}]

        with open (f'Output/{args.model}/evals/{task}.jsonl', 'a') as file:
            for plan in extracted_plans:
                file.write(json.dumps(plan) + '\n')