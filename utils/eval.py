from langchain.prompts import PromptTemplate # type: ignore

PLAN_EXTRACTION_PROMPT = """

Extract the travel itinerary and parse the businesses' information into the JSON format as below. Be faithful and concise. Correctly document the right number of the attractions. Only write down the name and address of the businesses. If certain recommendations (like meals or accommodations) are not provided, replace the information with "-" for name and address. If recommendations for a session of attraction is not provided, replace the information as an empty array. 

----- Example Starts -----
{
    "itinerary":[
        {   
            "days": "x",
            "breakfast": {
                "name": "xxx",
                "address": "xxx"
            },
            "morning_attractions": [
                {
                    "name": "xxx",
                    "address": "xxx"
                },
                {
                    "name": "xxx",
                    "address": "xxx"
                }
            ],
            "lunch": {
                "name": "xxx",
                "address": "xxx"
            },
            "afternoon_attractions": [
                {
                    "name": "xxx",
                    "address": "xxx"
                },
                {
                    "name": "xxx",
                    "address": "xxx"
                }
            ],
            "dinner": {
                "name": "xxx",
                "address": "xxx"
            },
            "night_attractions": [
                {
                    "name": "xxx",
                    "address": "xxx"
                },
                {
                    "name": "xxx",
                    "address": "xxx"
                }
            ],
            "accommodation": {
                "name": "xxx",
                "address": "xxx"
            }
        },
        {
            "days": "x",
            "breakfast": {
                "name": "xxx",
                "address": "xxx"
            },
            "morning_attractions": [
                {
                    "name": "xxx",
                    "address": "xxx"
                },
                {
                    "name": "xxx",
                    "address": "xxx"
                }
            ],
            "lunch": {
                "name": "xxx",
                "address": "xxx"
            },
            "afternoon_attractions": [
                {
                    "name": "xxx",
                    "address": "xxx"
                },
                {
                    "name": "xxx",
                    "address": "xxx"
                }
            ],
            "dinner": {
                "name": "xxx",
                "address": "xxx"
            },
            "night_attractions": [
                {
                    "name": "xxx",
                    "address": "xxx"
                },
                {
                    "name": "xxx",
                    "address": "xxx"
                }
            ],
            "accommodation": {
                "name": "xxx",
                "address": "xxx"
            }
        }
    ]
}
----- Example Ends -----

"""

