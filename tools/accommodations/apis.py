import pandas as pd # type: ignore
from pandas import DataFrame # type: ignore
from typing import List
from difflib import get_close_matches
from datasets import load_dataset

class Accommodations:
    def __init__(self, working_model):
        self.path = 'Dataset/Hotels.csv'
        self.data = pd.read_csv(self.path)
        #self.data = load_dataset("EthanWTL81/ItinBench", "hotel", split="test")
        #print("Accommodations loaded.")
    def run(self,
            budget: str,
            preference: List[str]
            ) -> DataFrame: #Cheap budget,[Good Location]
        #price
        price_map = {'cheap budget':['$','$$'],'moderate budget':['$','$$','$$$'],'expensive budget':['$$','$$$','$$$$']}
        price_limit = price_map[budget.lower()]
        result = self.data[self.data['price'].isin(price_limit)]
        
        #dealing with the preferences
        #print(preference)
        if preference != ['']: 
            for pref in preference:
                #print(pref)
                prefs = pref.split(' ')
                col = prefs[1].lower()
                #print(col)
                if col not in self.data.columns:
                    col = get_close_matches(col, self.data.columns, n=1, cutoff=0.6)
                    if(col != []):
                        col = col[0]
                        #print(col)
                        pref_list = ['good ' + col, 'excellent ' + col]
                        result = result[result[col].isin(pref_list)]
                else:
                    pref_list = ['good ' + col, 'excellent ' + col]
                    result = result[result[col].isin(pref_list)]

        result = result[['name','address','latitude','longitude']]
        
        if len(result) == 0:
            return "There is no Accommodation that matches the preferences."
        
        #result.to_csv('accommodation_csv.csv', index=False)
        return result
        