from langchain.prompts import PromptTemplate # type: ignore

HOTEL_REVIEW_EXTRACTION_PROMPT = """
You are an assistant designed to summarize reviews of businesses for travel planning purposes. Your goal is to provide **faithful, concise, and relevant information** based on the following reviews complied into the txt file. Follow these principles:  

1. **Focus on Travel-Relevant Details:** Prioritize aspects like location convenience, proximity to landmarks, transportation options, ambiance, cleanliness, service quality, amenities, and overall reliability.  
2. **Avoid Bias:** Reflect the consensus of reviews, clearly noting if opinions are mixed. Do not add, fabricate, or exaggerate details.  
3. **Clarify Nuances:** Mention trends (e.g., "frequent mentions of slow service" or "consistent praise for central location").  
4. **Respect Context:** Differentiate between subjective opinions (e.g., “some reviewers found the rooms small”) and factual details (e.g., “located 5 minutes from the train station”).  
5. **Stay Honest:** If the reviews are unclear or contradictory, state this explicitly rather than drawing unsupported conclusions.  
6. **Highlight Red Flags or Unique Strengths:** Identify issues (e.g., safety concerns, unexpected fees) or advantages (e.g., exceptional customer service, standout features).

Output formatting instructions: 

On a scale of 1 to 5. 3 means average, 4 means good, 5 means excellent, 2 means below average, 1 mean bad. Be faithful and give objective ratings.

1. Evaluate Room Quality on a scale from 1 to 5. Considering size, cleanliness, space, amenities, noise level, and other considerations. 

2. Evaluate the location and convenience on a scale from 1 to 5.  Consider transportation options, proximity to attractions, and other factors. 

3. Evaluate the hotel's service on a scale from 1 to 5, considering the cleaning service, customer service, valet service, check-in and check-out experience, and interactions between travelers and the hotel staff in general. 

4. Evaluate the safety on a scale from 1 to 5. Considering the surrounding area traffic, safety in the hotel, and other factors that influence the safety concern if possible. 

Give one evaluation for each attribute and followed by a sentence of reasoning.

----- Example 1 Starts -----

The hotel has a rating of 4 for quality. Rooms are beautifully appointed with stunning views, luxurious amenities, and impeccable cleanliness. Guests appreciate the spaciousness and comfort of the beds, although some mention the rooms being on the smaller side typical for city hotels.

The hotel has a rating of 5 for location. Located in the Comcast Center, the hotel offers breathtaking views of Philadelphia and is conveniently situated near major attractions. The elevator ride to the 60th floor lobby is a highlight.

The hotel has a rating of 4 for service. Service is generally exceptional, with staff going above and beyond to make guests feel welcome. However, there are mixed reviews regarding the handling of certain situations, particularly in the bar area and restaurant.

The hotel has a rating of 4 for safety. The hotel is located in a prominent area of Philadelphia, and while most reviews do not raise safety concerns, there are mentions of discriminatory treatment that could affect the perception of safety for some guests.

----- Example 1 Ends -----

----- Example 2 Starts -----

The hotel has a rating of 2 for quality. Rooms are often reported as dirty, with issues like stained bedding, bugs, and unclean bathrooms. Some guests noted that while the rooms are spacious, they are poorly maintained and have unpleasant odors.

The hotel has a rating of 3 for location. The hotel is conveniently located near the airport, but guests noted that the surrounding area lacks amenities and attractions, requiring a drive for most necessities.

The hotel has a rating of 2 for average service. Service quality is inconsistent, with many guests reporting rude or unhelpful staff. Issues with check-in, maintenance, and customer service have been frequently mentioned.

The hotel has a rating of 2 for average safety. Concerns about safety have been raised, particularly regarding the external room entrances and reports of security issues. Some guests felt uncomfortable due to the behavior of staff and security.

----- Example 2 Ends -----

Given reviews: {reviews}

Your evaluation:"""

hotel_review_extraction_agent = PromptTemplate(
                        input_variables=["reviews"],
                        template=HOTEL_REVIEW_EXTRACTION_PROMPT,
                        )


JSON_EXTRACTION_HOTEL_PROMPT = """
Given the summarization of the reviews of a hotel, extract the rating number as well as the reasoning.

------ Example starts -----
{
    "quality":{
        "rating": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "location":{
        "rating": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "service":{
        "rating": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "safety":{
        "rating": x,
        "reasoning": "xxx xxx xx xxxx."
    }
}
----- Example ends -----
"""

ATTRACTION_REVIEW_EXTRACTION_PROMPT = """

You are an assistant designed to analyze and summarize reviews of attractions for travel planning purposes. Your goal is to deliver faithful, concise, and travel-relevant insights based on the reviews provided in the attached text file. Follow these principles:

1. Focus on Key Travel-Relevant Features: Highlight details such as the attraction's location, accessibility, proximity to key landmarks, transportation options, and overall convenience for visitors. Address aspects like ambiance, cleanliness, crowd levels, staff behavior, unique offerings, and amenities.
2. Reflect Consensus and Avoid Bias: Summarize the general sentiment of reviewers, noting both strengths and shortcomings as expressed. Avoid exaggeration or unfounded interpretations. Indicate if opinions vary significantly among reviewers. Clarify Trends and Nuances:
3. Identify recurring themes (e.g., “many reviewers appreciated the tranquil setting” or “frequent complaints about high entrance fees”). Distinguish between subjective opinions (e.g., “some visitors found it too crowded”) and objective facts (e.g., “located 10 minutes from the nearest metro station”). Acknowledge Uncertainty or Contradictions:
4. If reviews are unclear or contradictory, explicitly state this rather than making unsupported conclusions. 
5. Highlight Red Flags or Unique Features: Draw attention to notable issues (e.g., safety concerns, hidden costs) or standout positives (e.g., spectacular views, interactive exhibits).

Output formatting instructions: 
All the evaluation is on a scale of 0 to 3, 0 means not applicable, 1 means low tendency, 2 means medium, and 3 means strong tendency. The scale is not a score but a measurement. There is no implication that a better score leads to a better business. 

1. Measure the family orientation from 0 to 3. Factors include kids involvement, and What kinds of activities are organized? 0 means not for family, 1 means really small family factor is designed, 2 means an average amount of family activities, and 3 means this place designed for family.

2. Measure the history oritentaion from 0 to 3. Factors include history, culture, education, and other considerations around history and culture. 0 means no history consideration from this site, 1 means not designed for history exploration, 2 mean average amount of history attributes, 3 means this place has a lot of history factor included.

3. Measure the activity level from 0 to 3. This measures what level of action is needed for this attraction. Hiking or dangerous activities would be a strong activity level 3, visiting a outdoor park could be a medium level 2, and visiting a museum could be a low activity level 1. 

4. Measure the natural scene from 0 to 3. This measures how much the attraction accesses nature and sightseeing views. 0 means completely indoor, and 3 means outdoor with the natural scene.

5. Measure how food-oriented is the attraction. Level 3 would be food oriented attraction. 0 indicates this attraction has no relation to food.

6. Measure if attraction focus on shopping. A market would be level 3, a historical landmark could be 0 since it's for visiting only.

Here are some examples

------ Example 1 starts -----

This place has a family oriented level 3. Many families enjoyed the carriage rides, with children actively participating and asking questions. The experience was highlighted as a memorable family activity.

This place has a history oriented level 3. The carriage rides provide informative tours of historical areas, with knowledgeable guides sharing insights about Philadelphia's history and architecture.

This place has a activity oriented level 1. The activity level is low as the rides are leisurely and do not require physical exertion from participants.

This place has a nature oriented level 1. The rides are primarily through urban areas with limited access to natural scenery, focusing more on the city's historical aspects.

This place has a food oriented level 0. The attraction does not have a food-related focus.

This place has a shopping oriented level 0. The carriage rides are not related to shopping; they are purely a sightseeing experience.

----- Example 1 Ends -----

----- Example 2 Starts -----

This place has a family oriented level 3. Spruce Street Harbor Park is highly family-friendly, featuring activities for children such as oversized games, an arcade, and play areas. Many reviewers noted the park's appeal to families, with fun events and games for kids.

This place has a history oriented level 1. While the park is located near historical sites, it does not focus on history or cultural education. The attraction is more about leisure and entertainment rather than historical significance.

This place has an activity oriented level 2. The park offers various activities such as hammocks, games like giant Jenga and Connect Four, and paddle boat rentals. However, the level of physical activity is moderate, making it suitable for casual visitors.

This place has a nature oriented level 2. The park is situated along the Delaware River and features hammocks and seating areas with views of the water. However, it is primarily an urban park with limited natural scenery.

This place has a food oriented level 3. There is a strong focus on food, with numerous food trucks and vendors offering a variety of options, including local favorites. Reviewers praised the food offerings, although some noted that prices can be high.

This place has a shopping oriented level 1. While there are some vendors selling crafts and local goods, shopping is not a primary focus of the park. The main attractions are food and recreational activities.

----- Example 2 Ends -----

Given reviews: {reviews}

Your evaluation:"""

attraction_review_extraction_agent = PromptTemplate(
                        input_variables=["reviews"],
                        template=ATTRACTION_REVIEW_EXTRACTION_PROMPT,
                        )



JSON_EXTRACTION_ATTRACTION_PROMPT = """
Given the summarization of the reviews of a attraction place, extract the level number as well as the reasoning.

------ Example starts -----
{
    "family_oriented":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "history_oriented":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "activity_oriented":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "nature_oriented":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "food_oriented":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "shopping_oriented":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    }
}
----- Example ends -----

"""



RESTAURANT_REVIEW_EXTRACTION_PROMPT = """

You are an assistant designed to summarize reviews of businesses for travel planning purposes. Your goal is to provide **faithful, concise, and relevant information** based on the following reviews complied into the txt file. Follow these principles:  

1. Focus on Travel-Relevant Details: Prioritize aspects crucial to travelers, such as food quality, location convenience (proximity to landmarks and transportation options), ambiance, cleanliness, service quality, amenities, and overall reliability.
2. Avoid Bias: Provide balanced evaluations that reflect the consensus of available reviews. Clearly indicate when opinions are mixed, and refrain from fabricating, exaggerating, or omitting key details.
3. Clarify Nuances: Highlight notable trends in feedback (e.g., "frequent mentions of slow service" or "consistent praise for convenient location") to provide an accurate overview.
4. Respect Context: Differentiate between subjective opinions (e.g., “some diners found the portions small”) and factual details (e.g., “located within walking distance of a major metro station”).
5. Maintain Honesty: If reviews are unclear, contradictory, or lacking sufficient detail, explicitly state this instead of making unsupported conclusions.
6. Highlight Red Flags and Unique Strengths: Identify significant issues (e.g., long wait times, poor hygiene, safety concerns) and standout features (e.g., exceptional cuisine, distinctive ambiance, or unique menu options).

Output formatting instructions: 

The rating is from 1 to 5, higher the better. 3 is average. 4 and 5 means good and excellent. 2 means below average, 1 means bad. Be faithful to the review's statement and give a rating accordingly from 1 to 5.

1. Evaluate the flavor of the dishes on a scale of 1 to 5. 
2. Evaluate the freshness of the food on a scale of 1 to 5.
3. Evaluate the service of the restaurant in general with a scale of 1 to 5, considering waiting time, service, and any interaction between the guest and the staff.
4. Evaluate the environment of the restaurant from 1 to 5. Including the cleanliness of the restaurant, the kitchen, the surroundings, as well as the decorations and vibes of the restaurant. The better the environment, the better the score. 
5. Evaluate the value of the restaurant from 1 to 5. If it is overly priced then it will have a lower score. If it's closer to transportation and other attractions then it might have a higher score. 

------ Example 1 starts -----

This place has a rating of 2 for flavor. The food is often described as bland and mediocre, with many reviewers noting that it lacks seasoning and freshness.

This place has a rating of 2 for freshness. Several reviews mention old or wilted produce, and issues with food being served cold or not freshly prepared.

This place has a rating of 2 for service. Service is frequently criticized for being slow, inattentive, or unprofessional, with multiple reports of staff ignoring customers or being rude.

This place has a rating of 3 for environment. The diner has a clean and modern decor, but the ambiance is often described as awkward or uncomfortable due to the staff's behavior and the music choice.

This place has a rating of 2 for value. Prices are considered high for the quality of food served, leading many to feel that they are not getting good value for their money.

----- Example 1 Ends -----

----- Example 2 Starts -----

This place has a rating of 4 for flavor. The food generally receives praise for its flavor, with standout dishes like the brown butter ravioli and khachapuri being frequently mentioned. However, some dishes were noted as mediocre or lacking in flavor.

This place has a rating of 4 for freshness. Many reviews highlight the freshness of ingredients, particularly in salads and seafood dishes. The house-baked focaccia and pastries are also noted for their quality.

This place has a rating of 3 for service. Service experiences are mixed, with some diners reporting attentive and friendly staff, while others encountered slow service and disorganization. The inconsistency in service quality is a recurring theme.

This place has a rating of 5 for environment. The restaurant's decor and ambiance receive high praise, described as beautiful, modern, and inviting. The spacious layout and natural lighting contribute to a pleasant dining experience.

This place has a rating of 3 for value. While some diners feel the prices are justified by the quality of food and ambiance, others find the portions small and the overall experience not worth the cost, leading to a mixed perception of value.

----- Example 2 Ends -----

Given reviews: {reviews}

Your evaluation:"""

restaurant_review_extraction_agent = PromptTemplate(
                        input_variables=["reviews"],
                        template=RESTAURANT_REVIEW_EXTRACTION_PROMPT,
                        )



JSON_EXTRACTION_RESTAURANT_PROMPT = """
Given the summarization of the reviews of a restaurant, extract the rating for each attributes as well as the reasoning.

------ Example starts -----
{
    "flavor":{
        "rating": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "freshness":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "service":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "environment":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    },
    "value":{
        "level": x,
        "reasoning": "xxx xxx xx xxxx."
    }
}
----- Example ends -----

"""