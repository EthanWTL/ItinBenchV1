import os
import re
import importlib
import torch
import json
import argparse
from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from utils.prompt import zeroshot_react_agent_prompt
from typing import List, Dict, Any
from pandas import DataFrame
from transformers import pipeline

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--apikey', type=str, help='Input API Key', required=False)
    parser.add_argument('--model', type=str, help='Input model name', required=False, default='gpt4o')
    parser.add_argument('--numPlan', type=int, help='Input number of plans', required=False, default=1)

    args = parser.parse_args()
    return args

actionMapping = {"AccommodationSearch":"accommodations", "RestaurantSearch":"restaurants", "AttractionSearch":"attraction","BusinessClusterSearch":"nearby","Planner":"planner"}

class ReactAgent:
    def __init__(self, args, tools: List[str] = None, max_retries: int = 3) -> None: 
        model_map = {'gpt4o': 'gpt-4o-2024-11-20', 'mistral':'mistral-large-2411','llama318b':'meta-llama/Llama-3.1-8B','gemini':'gemini-1.5-pro'}
        self.planner_model_name = model_map[args.model]
        self.react_llm_name = model_map[args.model]
        self.react_name = model_map[args.model]
        self.answer = ''
        self.json_log = []
        self.notebook = []
        self.max_retries = max_retries
        self.last_actions = []
        self.current_observation = ''
        self.current_data = None
        self.tools = self.load_tools(tools, planner_model_name=self.planner_model_name)
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record['invalidAction'] = 0
        self.agent_prompt = zeroshot_react_agent_prompt

        if 'gpt-4o' in self.react_llm_name:
            stop_list = ['\n']
            #self.max_token_length = 15000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=256,
                     model_name=self.react_llm_name,
                     openai_api_key=args.apikey,
                     model_kwargs={"stop": stop_list}
                     )
        if 'mistral' in self.react_llm_name:
            self.llm = ChatMistralAI(
                    model="mistral-large-2411",
                    max_tokens=128,
                    temperature=0,
                    mistral_api_key = args.apikey,
                    model_kwargs={"stop": ['\n']}
                )
            
        if 'llama' in self.react_llm_name:
            self.llm = pipeline(
                    "text-generation", model="meta-llama/Llama-3.1-8B", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="cuda"
                )

        if 'mixtral' in self.react_llm_name:
            self.llm = ChatMistralAI(
                    model="open-mixtral-8x7b",
                    max_tokens=128,
                    temperature=0,
                    mistral_api_key = args.apikey,
                    model_kwargs={"stop": ['\n']}
                )
            
        if 'gemini' in self.react_llm_name:
            self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    max_tokens=128,
                    temperature=0,
                    google_api_key = args.apikey,
                    model_kwargs={"stop": ['\n']}
                )

        return

    def run(self,query,reset=True) -> None:
        self.query = query

        if reset:
            self.__reset_agent()

        while not self.is_finished():
            self.step()
        
        return self.answer, self.scratchpad, self.json_log

    def step(self) -> None:
        #record the log
        self.json_log.append({"step": self.step_n, "thought":"",
                              "action": "", "observation": "", "state":""})
        #thought
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent() #
        print(self.scratchpad.split('\n')[-1])
        self.json_log[-1]['thought'] = self.scratchpad.split('\n')[-1].replace(f'\nThought {self.step_n}:',"")
        
        
        #act
        self.scratchpad += f'\nAction {self.step_n}:'
       
        #get the action prompted
        action = self.prompt_agent()
        #print(action)
        ##

        if action == None or action == '' or action == '\n' or '[' not in action :
            self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            self.scratchpad += ' ' + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        self.json_log[-1]['action'] = self.scratchpad.split('\n')[-1].replace(f'\nAction {self.step_n}:',"")

        if len(self.last_actions) == 3:
            print("The same action has been repeated 3 times consecutively. So we stop here.")
            # self.log_file.write("The same action has been repeated 3 times consecutively. So we stop here.")
            self.json_log[-1]['state'] = 'same action 3 times repeated'
            self.finished = True
            return


        print(self.scratchpad.split('\n')[-1])
        
        
        #observation
        self.scratchpad += f'\nObservation {self.step_n}: '
        if action == None or action == '' or action == '\n' or '[' not in action:
            action_type = None 
            action_arg = None
            self.scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."
        else:
            action_type, action_arg = parse_action(action)
            #print(action_type)
            if action_type != "Planner":
                if action_type in actionMapping:
                    pending_action = actionMapping[action_type]
                elif action_type not in actionMapping:
                    pending_action = 'invalidAction'

                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"{pending_action} early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return # so if the max tries is reached, we stop the loop
                elif pending_action not in self.retry_record:
                    if self.retry_record['invalidAction'] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"invalidAction Early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"invalidAction early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"invalidAction early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

            if action_type == 'AccommodationSearch':
                #print('we are at acc search')
                try:
                    if validate_accommodation_parameters_format(action_arg):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['accommodations'].run(action_arg.split(',')[0],[p.strip() for p in action_arg.split('[')[1].strip('[]').split(',')])
                        self.current_observation = str(to_string(self.current_data))
                        self.scratchpad += 'AccommodationSearch Succeeded' #self.current_observation
                        self.notebook.append({'Description': 'Accommodation Choice', 'Content': self.current_data})
                        self.__reset_record()
                        self.json_log[-1]['state'] = 'Successful'
                        
                except ValueError as e:
                    print(e)
                    self.retry_record['accommodations'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. Parameter Error'
                except Exception as e:
                    print(e)
                    self.retry_record['accommodations'] += 1
                    self.current_observation = f'Illegal Accommodation Search. Please try again.'
                    self.scratchpad += f'Illegal Accommodation Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'AttractionSearch':
                try:
                    if validate_attraction_parameters_format(action_arg):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['attractions'].run(action_arg.split(',')[0],[action_arg.split(',')[1].strip()[1:][:-1]])
                        #print(self.current_data)
                        self.current_observation = str(to_string(self.current_data))
                        self.scratchpad += 'AttractionSearch Succeeded' #self.current_observation 
                        self.notebook.append({'Description': 'Attraction Choice', 'Content': self.current_data})
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                except ValueError as e:
                    print(e)
                    self.retry_record['attractions'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. Parameter Error'
                except Exception as e:
                    print(e)
                    self.retry_record['attractions'] += 1
                    self.current_observation = f'Illegal Attraction Search. Please try again.'
                    self.scratchpad += f'Illegal Attraction Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'RestaurantSearch': #action_arg = 'Cheap Budget, Indian, [Good Flavor, Good Value]'
                try:
                    if validate_restaurant_parameters_format(action_arg):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['restaurants'].run(action_arg.split('[')[0].split(',')[0].strip(),action_arg.split('[')[0].split(',')[1].strip(),[a.strip() for a in action_arg.split('[')[1].strip()[:-1].split(',')])
                        self.current_observation = str(to_string(self.current_data))
                        self.scratchpad += 'AttractionSearch Succeeded' #self.current_observation
                        self.notebook.append({'Description': 'Restaurant Choice', 'Content': self.current_data})
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                except ValueError as e:
                    print(e)
                    self.retry_record['restaurants'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. Parameter Error'
                except Exception as e:
                    print(e)
                    self.retry_record['restaurants'] += 1
                    self.current_observation = f'Illegal Restaurant Search. Please try again.'
                    self.scratchpad += f'Illegal Restaurant Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'BusinessClusterSearch': #action_arg = 'Cheap Budget, Indian, [Good Flavor, Good Value]'
                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_data = self.tools['nearby'].run(self.notebook)
                    self.current_observation = str(to_string(self.current_data))
                    self.scratchpad += 'BusinessClusterSearch Succeeded' #self.current_observation
                    self.notebook.append({'Description': 'Business Cluster Results', 'Content': self.current_data})
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'
                
                except Exception as e:
                    print(e)
                    self.retry_record['nearby'] += 1
                    self.current_observation = f'Illegal business cluster Search. Please try again.'
                    self.scratchpad += f'Illegal business cluster Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            #elif action_type == 'NotebookWrite':
            #    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),'Masked due to limited length. Make sure the data has been written in Notebook.')
            #    self.current_observation = str(self.tools['notebook'].write(self.current_data, action_arg))
            #    self.scratchpad  +=  self.current_observation
            #    self.json_log[-1]['state'] = f'Successful'

            elif action_type == 'Planner':
                self.current_observation = str(self.tools['planner'].run(self.notebook,action_arg))
                self.scratchpad  +=  self.current_observation
                self.answer = self.current_observation
                self.json_log[-1]['state'] = f'Successful'
            else:
                self.retry_record['invalidAction'] += 1
                self.current_observation = 'Invalid Action. Valid Actions are AccommodationSearch[Budget,Preference] / AttractionSearch[Budget, Preference] / RestaurantSearch[Budget, Cuisine, Preference]/ Planner[Query].'
                self.scratchpad += self.current_observation
                self.json_log[-1]['state'] = f'invalidAction'
            
            #print(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            # rite(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            #self.json_log[-1]['observation'] = self.current_observation
            #self.step_n += 1

            if action_type and action_type == 'Planner':
                self.finished = True
                self.answer = self.current_observation

        if action == None or action == '' or action == '\n' or '[' not in action:
            print(f'Observation {self.step_n}: ' + "No feedback from the environment due to the null action.")
            # write(f'Observation {self.step_n}: ' + "Your action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again.")
            self.json_log[-1]['observation'] = "No feedback from the environment due to the null action."
                #print(self.scratchpad)
                #print(self.json_log)
                #print(self.notebook) 
        else:
            #print(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            # rite(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            self.json_log[-1]['observation'] = self.current_observation
        
        self.step_n += 1
        if action_type and action_type == 'Planner' and self.retry_record['planner']==0:
            
            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return

    def is_finished(self) -> bool:
        return self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad: str = ''
        self.__reset_record()
        self.json_log = []
        self.current_observation = ''
        self.current_data = None
        self.last_actions = []
        self.notebook = []

    def prompt_agent(self) -> str:
        #print("we prompt the agents")
        while True:
            if self.react_name == 'gpt-4o-2024-11-20':
                request = format_step('gpt-4o-2024-11-20',self.llm.invoke(self._build_agent_prompt()).content)
            elif self.react_name == 'mistral-large-2411':
                request = format_step('mistral-large-2411',self.llm.invoke(self._build_agent_prompt()).content.split('\n')[0])
            elif self.react_name == 'open-mixtral-8x7b':
                request = format_step('open-mixtral-8x7b', self.llm.invoke(self._build_agent_prompt()).content.split('\n')[0])
            elif self.react_name == 'gemini-1.5-pro':
                #print('we are here')
                #print(self._build_agent_prompt())
                request = format_step('gemini-1.5-pro',self.llm.invoke(self._build_agent_prompt()).content.split('\n')[0])
            else:
                request = format_step('llama',self.llm(self._build_agent_prompt(), max_new_tokens = 256, return_full_text=False, do_sample=False)[0]['generated_text'].split('\n')[0])
            #print("here is the raw request: === ", request)
            return request  
        
    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record['invalidAction'] = 0

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                query=self.query,
                scratchpad=self.scratchpad)

    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}
        for tool_name in tools:
            module = importlib.import_module(f"tools.{tool_name}.apis") #
            
            if tool_name == 'planner' and planner_model_name is not None:
                tools_map[tool_name] = getattr(module, tool_name[0].upper()+tool_name[1:])(API_KEY=args.apikey, model_name=planner_model_name)
            elif tool_name == 'nearby':
                tools_map[tool_name] = getattr(module, tool_name[0].upper()+tool_name[1:])()
            else:
                tools_map[tool_name] = getattr(module, tool_name[0].upper()+tool_name[1:])(working_model = 'gpt4o')
        #print(tools_map)
        return tools_map

def format_step(model,step: str) -> str:
    #return step.strip('\n').strip().replace('\n', '')
    if model=='gemini-1.5-pro':
        response = step.split(':')[-1].strip()
        return response
    else:
        return step.strip('\n').strip().replace('\n', '')

def parse_action(string):
    if ('BusinessClusterSearch' not in string):
        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, string)
        if not match:
            raise ValueError('Please provide valid action from the list. Valid actions include: AccommodationSearch[Budget,Preference], AttractionSearch[Budget, Preference], RestaurantSearch[Budget, Cuisine, Preference], BusinessClusterSearch[], Planner[Query].')
        action_type = match.group(1)
        action_arg = match.group(2)
    else:
        action_type = 'BusinessClusterSearch'
        action_arg = ''
    #print(action_type,action_arg)
    return action_type,action_arg

#def parse action arg

def to_string(data) -> str:
    if data is not None:
        if type(data) == DataFrame:
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)
    
def validate_accommodation_parameters_format(action_arg):
    #print(action_arg)
    pattern = r"(.*\s*.*)\s*,\s*\[(.*)\]"
    match = re.match(pattern, action_arg)
    if not match:
        raise ValueError("Parameter format not match. Please try again. Valid Format: Budget, preference list.")
    budget = match.group(1).lower()
    preference_list = match.group(2)

    budget_accepted = ['cheap budget', 'moderate budget','expensive budget']
    budgetInRange = False
    if budget in budget_accepted:
        budgetInRange = True
    if not budgetInRange:
        raise ValueError("Wrong budget Input, valid ones include: cheap budget, moderate budget, and expensive budget. Please try again.")

    #preference
    preference = preference_list.split(',')
    preference_core = [p.lower().strip().split(' ')[-1].strip() for p in preference]
    preferenceInRange = True
    preferenceAccepted = ['location','service','safety','quality']
    for p in preference_core:
        if p not in preferenceAccepted:
            preferenceInRange = False

    if not preferenceInRange:
        raise ValueError("Wrong preference Input. Please try again.")
    return True
    
def validate_attraction_parameters_format(action_arg):
    pattern = r"(.*\s*.*)\s*,\s*\[(.*)\]"
    match = re.match(pattern, action_arg)
    if not match:
        raise ValueError("Parameter format not match. Please try again. Valid Format: Budget, Preference list.")
    budget = match.group(1).lower()
    #print(budget)
    preference_list = match.group(2)

    budget_accepted = ['cheap budget', 'moderate budget','expensive budget']
    budgetInRange = False
    if budget in budget_accepted:
        #print('budget in range')
        budgetInRange = True
    if not budgetInRange:
        raise ValueError("Wrong budget Input, valid ones include: cheap budget, moderate budget, and expensive budget. Please try again.")

    preference = preference_list.strip().split(',')
    if(len(preference) > 1 ):
        raise ValueError("Attraction only allows one preference. Please try again")
    if '-' in preference[0]:
        preference_core = preference[0].strip().split('-')[0].lower()
    else:
        preference_core = preference[0].strip().split(' ')[0].lower()
    #print(preference_core)
    preferenceAccepted = ["family","history","activity","nature","food","shopping"]
    preferenceIsInRange = False
    if(preference_core in preferenceAccepted):
        preferenceIsInRange = True
        #print('preference is in range')
    if not preferenceIsInRange:
        raise ValueError("Preference parameter invalid. Only family oriented / history oriented / activity oriented / nature oriented / food oriented / and shopping oriented are allowed. Please try again.")
    return True

def validate_restaurant_parameters_format(action_arg):
    pattern = r"(.*\s*.*),\s*(.*),\s*\[(.*)\]"
    match = re.match(pattern, action_arg)
    if not match:
        raise ValueError("Parameter format not match. Please try again. Valid Format: Budget, cuisine, preference list.")
    budget = match.group(1).lower()
    cuisine = match.group(2).lower()
    preference_list = match.group(3)

    budget_accepted = ['cheap budget', 'moderate budget','expensive budget']
    budgetInRange = False
    #print(budget)
    if budget in budget_accepted:
        budgetInRange = True
    if not budgetInRange:
        raise ValueError("Wrong budget Input, valid ones include: cheap budget, moderate budget and expensive budget. Please try again.")

    cuisine_accepted = ["us","mexican","irish","french","italian","greek","indian","chinese","japanese","korean","vietnamese","thai","asian fusion","middle eastern"]
    #print(cuisine)
    cuisineInRange = False
    if cuisine in cuisine_accepted:
        cuisineInRange = True
    if not cuisineInRange:
        raise ValueError("Cuisine not valid. Accepted cuisine is: US / Mexican / Irish / French / Italian / Greek / Indian / Chinese / Japanese / Korean / Vietnamese / Thai / Asian Fusion and Middle Eastern. Please try again.")

    preference_list = [p.lower().strip() for p in preference_list.split(',')]
    preference_core = [p.strip().split(' ')[-1] for p in preference_list]
    #print(preference_core)

    preferenceInRange = True
    preferenceAccepted = ['',"flavor","freshness","service","environment","value"]
    for p in preference_core:
        if p not in preferenceAccepted:
            preferenceInRange = False

    if not preferenceInRange:
        raise ValueError("Wrong preference Input. Accepted inputs are: good flavor / good freshness / good healthy/ good service / good environment / good value. Please try again.")    
    return True



if __name__ == "__main__":
    args = parse_args()
    tools_list = ["attractions","accommodations","restaurants","nearby","planner"]
    agent = ReactAgent(args, tools=tools_list)

    toolUsePlans = []
    toolUseScratchpads = []
    toolUseLogs = []

    with open ('Dataset/humanQuerys.jsonl') as file:
        humanquerys = [json.loads(line) for line in file]

    for i in range (args.numPlan):
        query = humanquerys[i]['query']
        planner_results, scratchpad, action_log  = agent.run(query)

        toolUsePlans=[{"index": i+1, "plan": planner_results}]
        toolUseScratchpads=[{"index": i+1, "scratchpad": scratchpad}]
        toolUseLogs=[{"index": i+1, "log": action_log}]
        
        if(i%20 == 0):
            print(f'done with plan: {i}')

        with open (f'Output/{args.model}/plans/toolUsePlans.jsonl', 'a') as file:
            for plan in toolUsePlans:
                json.dump(plan, file)
                file.write('\n')
        with open (f'Output/{args.model}/plans/toolUseScratchpads.jsonl', 'a') as file:
            for scratchpad in toolUseScratchpads:
                json.dump(scratchpad, file)
                file.write('\n')
        with open (f'Output/{args.model}/plans/toolUseLogs.jsonl', 'a') as file:
            for log in toolUseLogs:
                json.dump(log, file)
                file.write('\n')