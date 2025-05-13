from langchain.prompts import PromptTemplate # type: ignore

HUMAN_QUERY_GENERATION_PROMPT = """
Craft a a human like query for a travel plan given the following information. The input includes details such as trip duration, budget type, attractions types that the traveler wants to visit, dining preferences that they want to try, and accommodation requirements. Make sure each pairs of key words, like good environment, good location, are mentioned specifically.

----- Example Starts -----
Input: 
- general: 2 days, moderate budget, 
- attraction: history oriented, 
- restaurants: French, good environment, 
- hotel: good quality, good location

Output:
I want to go for a 2-day trip with a moderate budget. I want to visit some history-oriented attractions. Please find some good environment restaurants that provide French cuisine, I want to stay in a good quality hotel in a good location.
----- Example Ends -----

PromptInput: {input}

Output:"""

human_query_generation_agent = PromptTemplate(
                        input_variables=["input"],
                        template=HUMAN_QUERY_GENERATION_PROMPT,
                        )



PLANNER_NO_ROUTE_PROMPT = """

You are a proficient travel planner. Based on the given information and query, you will generate a travel plan like the following example. Ensure that all recommendations and their addresses are organized in chronological order for each day. Give exactly 4 attraction recommendations for each day. Be considerate, concise and well-structured.

----- Example Starts -----

Query: I am planning a 2-day trip with an expensive budget. I would like to visit some history-oriented attractions. Please recommend Japanese restaurants with a good environment. For accommodation, I am looking for a hotel with good location, good quality, and good service.

Travel Plan: 
Day X:
- Accommodation: 
  - Name: XXXX
    Address: XXXX, XXXX

- Breakfast: 
  - Name: XXXX 
    Address: XXXX, XXXX

- Morning Attraction: 
  - Name: XXXX 
    Address: XXXX, XXXX 

- Lunch: 
  - Name: XXXX 
    Address: XXXX, XXXX 

- Afternoon Attraction: 
  - Name: XXXX 
    Address: XXXX, XXXX 
  - Name: XXXX
    Address: XXXX, XXXX

- Dinner: 
  - Name: XXXX 
    Address: XXXX, XXXX 

- Night Attraction: 
  - Name: XXXX

----- Example Ends -----

Given Information: {given_information}

Query: {query}

Travel Plan: """

planner_no_route_agent = PromptTemplate(
                        input_variables=["given_information", "query"],
                        template=PLANNER_NO_ROUTE_PROMPT,
                        )


PLANNER_ROUTE_OP_PROMPT = """

You are a proficient travel planner. Based on the given information and query, you will generate a travel plan like the following example. Ensure that all recommendations and their addresses are organized in chronological order for each day. Give exactly 4 attraction recommendations for each day. Be considerate, concise and well-structured. Please also optimize the routes for the trip. For each day, find attractions that are close to each other for the recommendations.

----- Example Starts -----

Query: I am planning a 2-day trip with an expensive budget. I would like to visit some history-oriented attractions. Please recommend Japanese restaurants with a good environment. For accommodation, I am looking for a hotel with good location, good quality, and good service.

Travel Plan: 
Day X:
- Accommodation: 
  - Name: XXXX
    Address: XXXX, XXXX

- Breakfast: 
  - Name: XXXX 
    Address: XXXX, XXXX

- Morning Attraction: 
  - Name: XXXX 
    Address: XXXX, XXXX 

- Lunch: 
  - Name: XXXX 
    Address: XXXX, XXXX 

- Afternoon Attraction: 
  - Name: XXXX 
    Address: XXXX, XXXX 
  - Name: XXXX
    Address: XXXX, XXXX

- Dinner: 
  - Name: XXXX 
    Address: XXXX, XXXX 

- Night Attraction: 
  - Name: XXXX

----- Example Ends -----

Given Information: {given_information}

Query: {query}

Travel Plan: """

planner_route_OP_agent = PromptTemplate(
                        input_variables=["given_information", "query"],
                        template=PLANNER_ROUTE_OP_PROMPT,
                        )


ZEROSHOT_REACT_INSTRUCTION = """Collect information for a query plan using interleaving 'Thought', 'Action', and 'Observation' steps. Ensure you gather valid information related to transportation, dining, attractions, and accommodation. Note that the nested use of tools is prohibited. Don't include phrases like "Action: ", "Action 5", "Thought 1", or "Thought: " in your response. 'Thought' can reason about the current situation, and 'Action' can have 5 different types:

(1) AccommodationSearch[Budget,Preference]:
Description: Find the accommodation that matches the preference.
Parameters:
Preference: A list of preferences mentioned in the query.
Example: AccommodationSearch[Moderate Budget,[Good Location, Good Service]] would return the moderate price hotel that has a good or excellent location, as well as a good or excellent service.

(2) AttractionSearch[Budget, Preference]:
Description: Find the attractions that matches the preference.
Parameters:
Budget: The budget mentioned in the query.
Preference: A list of preferences mentioned in the query.
Example: AttractionSearch[Cheap budget,[Nature Oriented]] would return the cheap price and nature - oriented attractions.

(3) RestaurantSearch[Budget, Cuisine, Preference]:
Description: Find the restaurants that matches the preference.
Parameters:
Budget: The budget mentioned in the query.
Cuisine: The cuisine mentioned in the query.
Preference: A list of preferences mentioned in the query.
Example: RestaurantSearch[Expensive budget, Vietnamese, [Good Flavor, Good Value]] would return the expensive restaurants that offer Vietnamese cuisine, with good or excellent flavor and good or excellent value.

(4) BusinessClusterSearch[]:
Description: A tool that finds the number of business clusters given the information that you've collected. The tool will choose what business to be considered and return their spatial clustering information.
Example: BusinessClusterSearch[] would return you a list of business clusters among some attractions and hotels that you've collected. The businesses in the same cluster indicates that they are closer to each other and prefered to be arranged for the same day of the travel.

(5) Planner[Query]
Description: A smart planning tool that crafts detailed plans based on user input and the information stored in Notebook.
Parameters: 
Query: The query from user.
Example: Planner[Give me a 3-day trip plan in Philadelphia] would return a detailed 3-day trip plan.

You should use as many as possible steps to collect engough information to input to the Planner tool. 

Each action only calls one function once. Do not add any description in the action. Do not start action with "1. ", state the action directly.

Query: {query}{scratchpad}"""



zeroshot_react_agent_prompt = PromptTemplate(
                        input_variables=["query", "scratchpad"],
                        template=ZEROSHOT_REACT_INSTRUCTION,
                        )