import pandas as pd # type: ignore
from pandas import DataFrame # type: ignore
from typing import List
from difflib import get_close_matches
from datasets import load_dataset


class Attractions:
    def __init__(self, working_model):
        self.path = 'Dataset/Attractions.csv'
        self.data = pd.read_csv(self.path)
        #self.data = load_dataset("EthanWTL81/ItinBench", "attraction", split="test")
        #print("Accommodations loaded.")
    def run(self,
            budget: str,
            preference: List[str]
            ) -> DataFrame: #AttractionSearch[Moderate Budget, [Family Oriented]]
        #price
        price_map = {'cheap budget':['$','$$'],'moderate budget':['$','$$','$$$'],'expensive budget':['$$','$$$','$$$$']}
        price_limit = price_map[budget.lower()]
        result = self.data[self.data['price'].isin(price_limit)]
        
        #dealing with the preference, need to change the original dataset.
        # currently, we have family_oriented as the column name, but medium family oriented as the actual value, the llm might switch between
        col = preference[-1].lower()
        if col not in self.data.columns:
            col = get_close_matches(preference[0].lower(), self.data.columns, n=1, cutoff=0.6)
            if(col != []):
                col = col[0]
                #print(col)
                pref_list = ['medium ' + col, 'high ' + col]
                result = result[result[col].isin(pref_list)]
        else:
            pref_list = ['medium ' + col, 'high ' + col]
            result = result[result[col].isin(pref_list)]

        if len(result) == 0:
            return "There is no attractions that matches the preferences."
        #result.to_csv('attractions_csv.csv', index=False)

        result = result[['name','address','latitude','longitude']]
        return result